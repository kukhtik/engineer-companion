package com.varian.engcomp.engine

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Kotlin interface to the llama.cpp JNI layer.
 *
 * Thread-safety: llama_decode is NOT thread-safe. All calls into native
 * code are serialized via [mutex].
 *
 * Usage:
 *   val engine = LlamaEngine()
 *   engine.loadModel("/data/user/0/com.varian.engcomp/files/models/gemma-3-4b-it-Q4_K_M.gguf")
 *   val answer = engine.generate(prompt, maxTokens=512, temperature=0.3f) { piece ->
 *       // streaming token callback
 *   }
 *   engine.free()
 */
class LlamaEngine {

    companion object {
        private const val TAG = "LlamaEngine"
        const val N_CTX     = 2048
        const val N_THREADS = 4

        init {
            System.loadLibrary("engineer_companion_jni")
        }
    }

    private val mutex = Mutex()
    private var handle: Long = 0L
    val isLoaded: Boolean get() = handle != 0L

    // ── JNI declarations ──────────────────────────────────────────────────────

    private external fun nativeLoadModel(path: String, nCtx: Int, nThreads: Int): Long
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        callback: TokenCallback
    ): String
    private external fun nativeFree(handle: Long)

    // ── Public API ────────────────────────────────────────────────────────────

    /** Load model from absolute [path]. Returns true on success. */
    suspend fun loadModel(
        path: String,
        nCtx: Int = N_CTX,
        nThreads: Int = N_THREADS
    ): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (handle != 0L) {
                Log.w(TAG, "loadModel called with model already loaded; freeing first")
                nativeFree(handle)
                handle = 0L
            }
            Log.i(TAG, "Loading model: $path (nCtx=$nCtx, nThreads=$nThreads)")
            handle = nativeLoadModel(path, nCtx, nThreads)
            val ok = handle != 0L
            Log.i(TAG, if (ok) "Model loaded, handle=$handle" else "Model load FAILED")
            ok
        }
    }

    /**
     * Generate text from [prompt], streaming each piece via [onToken].
     * Returns the full generated string.
     * Runs on [Dispatchers.IO] and holds [mutex] for the duration.
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.3f,
        onToken: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (handle == 0L) error("LlamaEngine: model not loaded")
            val cb = object : TokenCallback {
                override fun onToken(piece: String) { onToken(piece) }
            }
            nativeGenerate(handle, prompt, maxTokens, temperature, cb)
        }
    }

    /** Release native resources. Safe to call multiple times. */
    suspend fun free() = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (handle != 0L) {
                nativeFree(handle)
                handle = 0L
                Log.i(TAG, "Model freed")
            }
        }
    }

    /** Non-suspend cleanup for use from onCleared(). */
    fun freeSync() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }
}

/** Callback interface invoked by JNI for each generated token piece. */
interface TokenCallback {
    fun onToken(piece: String)
}
