#include <jni.h>
#include <string>
#include <android/log.h>

// Forward declarations from llama.h
struct llama_model;
struct llama_context;

// llama.cpp API (loaded from libllama.so)
extern "C" {
    llama_model* llama_load_model_from_file(const char* path, ...);
    llama_context* llama_new_context_with_model(llama_model* model, ...);
    bool llama_model_has_encoder(const llama_model* model);
    bool llama_model_has_decoder(const llama_model* model);
    int32_t llama_n_ctx(const llama_context* ctx);
    int32_t llama_tokenize(const llama_context* ctx, const char* text, int32_t text_len, int32_t* tokens, int32_t n_max_tokens, bool add_special, bool parse_special);
    const char* llama_token_to_piece(const llama_context* ctx, int32_t token, char* buf, int32_t length);
    bool llama_eval(llama_context* ctx, const int32_t* tokens, int32_t n_tokens, int32_t n_past);
    int32_t llama_sample(struct llama_context* ctx, int32_t idx);
    void llama_free(llama_context* ctx);
    void llama_free_model(llama_model* model);
}

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct LlamaState {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
};

static LlamaState g_state;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_varian_engcomp_LlamaEngine_loadModel(
    JNIEnv* env, jobject /*thiz*/, jstring model_path) {

    const char* path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model: %s", path);

    // Load model with default params
    g_state.model = llama_load_model_from_file(path, 0);
    env->ReleaseStringUTFChars(model_path, path);

    if (g_state.model == nullptr) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    // Create context with n_ctx=2048
    g_state.ctx = llama_new_context_with_model(g_state.model, nullptr);
    if (g_state.ctx == nullptr) {
        LOGE("Failed to create context");
        llama_free_model(g_state.model);
        g_state.model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_varian_engcomp_LlamaEngine_generate(
    JNIEnv* env, jobject /*thiz*/, jstring prompt, jint max_tokens, jfloat temperature) {

    if (g_state.ctx == nullptr || g_state.model == nullptr) {
        return env->NewStringUTF("");
    }

    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);

    // Tokenize prompt
    int n_tokens = llama_tokenize(
        g_state.ctx, prompt_str, -1, nullptr, 0, true, false);
    auto* tokens = new int32_t[n_tokens];
    llama_tokenize(
        g_state.ctx, prompt_str, -1, tokens, n_tokens, true, false);
    env->ReleaseStringUTFChars(prompt, prompt_str);

    std::string result;
    char piece_buf[64];
    int n_past = 0;

    for (int i = 0; i < max_tokens && i < n_tokens; i++) {
        if (!llama_eval(g_state.ctx, &tokens[i], 1, n_past++)) {
            LOGE("llama_eval failed at position %d", i);
            break;
        }
    }

    for (int i = 0; i < max_tokens; i++) {
        int32_t token = llama_sample(g_state.ctx, n_past);
        if (token == -1) break;

        llama_token_to_piece(g_state.ctx, token, piece_buf, sizeof(piece_buf));
        result += piece_buf;

        if (!llama_eval(g_state.ctx, &token, 1, n_past++)) break;

        // Check stop condition (newline or EOS)
        if (piece_buf[0] == '\n' && result.length() > 10) break;
    }

    delete[] tokens;
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_varian_engcomp_LlamaEngine_unloadModel(
    JNIEnv* /*env*/, jobject /*thiz*/) {

    if (g_state.ctx) {
        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
    if (g_state.model) {
        llama_free_model(g_state.model);
        g_state.model = nullptr;
    }
    LOGI("Model unloaded");
}
