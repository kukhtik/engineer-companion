#include <jni.h>
#include <string>
#include <vector>
#include <cstdint>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ─────────────────────────────────────────────────────────────────────────────
// Native handle holding all llama.cpp resources for one model session.
// ─────────────────────────────────────────────────────────────────────────────
struct LlamaHandle {
    llama_model   * model   = nullptr;
    llama_context * ctx     = nullptr;
    // Sampler is per-generate; stored here so we can free it on crash/free
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: fill a batch with [n] tokens starting at position [pos0], all seq 0
// ─────────────────────────────────────────────────────────────────────────────
static void batch_fill(llama_batch & batch,
                       const llama_token * tokens, int32_t n,
                       int32_t pos0, bool last_logits) {
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token   [i]    = tokens[i];
        batch.pos     [i]    = pos0 + i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i]    = 0;
    }
    if (last_logits && n > 0) batch.logits[n - 1] = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// JNI: nativeLoadModel
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jlong JNICALL
Java_com_varian_engcomp_engine_LlamaEngine_nativeLoadModel(
        JNIEnv * env, jobject /* thiz */,
        jstring jpath, jint nCtx, jint nThreads) {

    llama_backend_init();

    const char * path = env->GetStringUTFChars(jpath, nullptr);
    LOGI("nativeLoadModel: path=%s nCtx=%d nThreads=%d", path, (int)nCtx, (int)nThreads);

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU-only
    llama_model * model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);

    if (!model) {
        LOGE("nativeLoadModel: llama_model_load_from_file returned null");
        return 0L;
    }

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t)nCtx;
    cparams.n_threads = (int32_t)nThreads;
    cparams.n_batch   = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("nativeLoadModel: llama_init_from_model returned null");
        llama_model_free(model);
        return 0L;
    }

    auto * handle = new LlamaHandle();
    handle->model = model;
    handle->ctx   = ctx;
    LOGI("nativeLoadModel: success, handle=%p", handle);
    return (jlong)(intptr_t)handle;
}

// ─────────────────────────────────────────────────────────────────────────────
// JNI: nativeGenerate  (streams tokens via TokenCallback.onToken)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jstring JNICALL
Java_com_varian_engcomp_engine_LlamaEngine_nativeGenerate(
        JNIEnv * env, jobject /* thiz */,
        jlong jhandle, jstring jprompt,
        jint maxTokens, jfloat temperature,
        jobject callback) {

    auto * h = reinterpret_cast<LlamaHandle *>((intptr_t)jhandle);
    if (!h || !h->model || !h->ctx) {
        LOGE("nativeGenerate: invalid handle");
        return env->NewStringUTF("");
    }

    // ── Resolve callback method ───────────────────────────────────────────────
    jclass   cbClass    = env->GetObjectClass(callback);
    jmethodID onTokenId = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    if (!onTokenId) {
        LOGE("nativeGenerate: GetMethodID(onToken) failed");
        return env->NewStringUTF("");
    }

    // ── Tokenize prompt ───────────────────────────────────────────────────────
    const char * prompt_str = env->GetStringUTFChars(jprompt, nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(h->model);

    // Count tokens
    int n_prompt = llama_tokenize(vocab, prompt_str, -1,
                                  nullptr, 0, /*add_special=*/true, /*parse_special=*/false);
    if (n_prompt < 0) n_prompt = -n_prompt;

    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt_str, -1,
                   prompt_tokens.data(), n_prompt,
                   /*add_special=*/true, /*parse_special=*/false);
    env->ReleaseStringUTFChars(jprompt, prompt_str);

    LOGI("nativeGenerate: %d prompt tokens, maxTokens=%d temp=%.2f",
         n_prompt, (int)maxTokens, (double)temperature);

    // ── Set up sampler ────────────────────────────────────────────────────────
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp((float)temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    llama_token eos_token = llama_vocab_eos(vocab);

    // ── Feed prompt ───────────────────────────────────────────────────────────
    llama_batch batch = llama_batch_init(512, 0, 1);
    {
        int32_t pos = 0;
        const int32_t step = 512;
        for (int32_t i = 0; i < n_prompt; i += step) {
            int32_t chunk = std::min(step, n_prompt - i);
            bool is_last  = (i + chunk >= n_prompt);
            batch_fill(batch, prompt_tokens.data() + i, chunk, pos, is_last);
            if (llama_decode(h->ctx, batch) != 0) {
                LOGE("nativeGenerate: llama_decode (prompt) failed at i=%d", i);
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                return env->NewStringUTF("[decode error]");
            }
            pos += chunk;
        }
    }

    // ── Generation loop ───────────────────────────────────────────────────────
    std::string result;
    char piece_buf[256];
    int32_t n_past = n_prompt;

    for (int i = 0; i < (int)maxTokens; ++i) {
        llama_token token = llama_sampler_sample(smpl, h->ctx, -1);
        llama_sampler_accept(smpl, token);

        if (token == eos_token) {
            LOGI("nativeGenerate: EOS at step %d", i);
            break;
        }

        int n_chars = llama_token_to_piece(vocab, token,
                                           piece_buf, sizeof(piece_buf),
                                           /*lstrip=*/0, /*special=*/false);
        if (n_chars < 0) n_chars = 0;

        std::string piece(piece_buf, (size_t)n_chars);
        result += piece;

        // Stream to Kotlin callback
        jstring jpiece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(callback, onTokenId, jpiece);
        env->DeleteLocalRef(jpiece);

        // Check for Java exception after callback
        if (env->ExceptionCheck()) {
            LOGE("nativeGenerate: exception in callback at step %d", i);
            env->ExceptionClear();
            break;
        }

        // Decode next token
        llama_token next_arr[1] = { token };
        batch_fill(batch, next_arr, 1, n_past, /*last_logits=*/true);
        if (llama_decode(h->ctx, batch) != 0) {
            LOGE("nativeGenerate: llama_decode (gen) failed at step %d", i);
            break;
        }
        ++n_past;
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);

    LOGI("nativeGenerate: done, %zu chars", result.size());
    return env->NewStringUTF(result.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// JNI: nativeFree
// ─────────────────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_varian_engcomp_engine_LlamaEngine_nativeFree(
        JNIEnv * /* env */, jobject /* thiz */, jlong jhandle) {
    auto * h = reinterpret_cast<LlamaHandle *>((intptr_t)jhandle);
    if (!h) return;
    if (h->ctx)   { llama_free(h->ctx);        h->ctx   = nullptr; }
    if (h->model) { llama_model_free(h->model); h->model = nullptr; }
    delete h;
    LOGI("nativeFree: done");
}
