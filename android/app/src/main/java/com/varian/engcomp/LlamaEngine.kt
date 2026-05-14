package com.varian.engcomp

import android.content.Context
import android.util.Log
import java.io.File

/**
 * JNI wrapper for llama.cpp inference.
 */
class LlamaEngine {

    companion object {
        private const val TAG = "LlamaEngine"
        private var isLoaded = false

        /**
         * Load GGUF model from internal storage.
         */
        fun loadModel(context: Context, modelPath: String = ""): Boolean {
            val path = if (modelPath.isNotEmpty()) modelPath
            else context.getDir("models", Context.MODE_PRIVATE).absolutePath +
                    "/gemma-3-4b-it-Q4_K_M.gguf"

            val modelFile = File(path)
            if (!modelFile.exists()) {
                Log.e(TAG, "Model not found: $path")
                return false
            }

            try {
                isLoaded = nativeLoadModel(path)
                Log.i(TAG, "Model loaded: $isLoaded")
                return isLoaded
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "JNI library not loaded: ${e.message}")
                return false
            }
        }

        /**
         * Generate answer for prompt. Blocks calling thread.
         */
        @Throws(IllegalStateException::class)
        fun generate(prompt: String, maxTokens: Int = 256, temperature: Float = 0.3f): String {
            if (!isLoaded) {
                throw IllegalStateException("Model not loaded")
            }
            return nativeGenerate(prompt, maxTokens, temperature) ?: ""
        }

        /**
         * Unload model and free memory.
         */
        fun unload() {
            if (isLoaded) {
                nativeUnload()
                isLoaded = false
            }
        }
    }

    // JNI declarations
    private external fun nativeLoadModel(modelPath: String): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int, temperature: Float): String?
    private external fun nativeUnload()

    init {
        try {
            System.loadLibrary("engineer_companion_jni")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Failed to load native library: ${e.message}")
        }
    }
}
