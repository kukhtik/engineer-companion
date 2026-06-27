package com.varian.engcomp.engine

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Copies large assets (ONNX models, binary index) from the APK asset tree to
 * [Context.filesDir] on first run.  ORT requires file-system paths, not streams.
 *
 * Idempotent: a file is skipped if it already exists and its size matches the
 * asset size.  Pass [force] = true to overwrite regardless.
 *
 * Usage:
 *   val paths = AssetCopier(context).copy { pct -> updateProgressUI(pct) }
 *   // paths.modelOnnx, paths.tokenizerOnnx, paths.chunksIndex
 */
class AssetCopier(private val ctx: Context, private val force: Boolean = false) {

    data class Paths(
        val modelOnnx: String,
        val tokenizerOnnx: String,
        val chunksIndex: String,
    )

    companion object {
        private const val TAG = "AssetCopier"

        private const val ASSET_MODEL   = "models/embedder_onnx/model.onnx"
        private const val ASSET_TOK     = "models/embedder_onnx/tokenizer.onnx"
        private const val ASSET_INDEX   = "db/chunks_index.bin"
    }

    /**
     * Copy assets to [Context.filesDir] if needed.
     * [onProgress] is called with a value in 0..100 as bytes are copied.
     * Runs synchronously — call from a background coroutine.
     */
    fun copy(onProgress: ((Int) -> Unit)? = null): Paths {
        val destDir = ctx.filesDir
        destDir.mkdirs()

        val assets = listOf(ASSET_MODEL, ASSET_TOK, ASSET_INDEX)
        val totalBytes = assets.sumOf { assetSize(it) }
        var copiedBytes = 0L

        for (assetPath in assets) {
            val destFile = File(destDir, assetPath)
            destFile.parentFile?.mkdirs()

            val assetSz = assetSize(assetPath)
            if (!force && destFile.exists() && destFile.length() == assetSz) {
                Log.d(TAG, "Skip (up-to-date): $assetPath (${assetSz / 1024} KB)")
                copiedBytes += assetSz
                onProgress?.invoke(((copiedBytes * 100) / totalBytes).toInt())
                continue
            }

            Log.i(TAG, "Copying $assetPath -> ${destFile.absolutePath} (${assetSz / 1024} KB)")
            ctx.assets.open(assetPath).use { inp ->
                FileOutputStream(destFile).use { out ->
                    val buf = ByteArray(256 * 1024)  // 256 KB chunks
                    var n: Int
                    while (inp.read(buf).also { n = it } != -1) {
                        out.write(buf, 0, n)
                        copiedBytes += n
                        if (totalBytes > 0) {
                            onProgress?.invoke(((copiedBytes * 100) / totalBytes).toInt())
                        }
                    }
                }
            }
            Log.i(TAG, "Done: $assetPath")
        }

        onProgress?.invoke(100)

        return Paths(
            modelOnnx    = File(destDir, ASSET_MODEL).absolutePath,
            tokenizerOnnx = File(destDir, ASSET_TOK).absolutePath,
            chunksIndex  = File(destDir, ASSET_INDEX).absolutePath,
        )
    }

    private fun assetSize(assetPath: String): Long {
        return try {
            ctx.assets.openFd(assetPath).use { it.length }
        } catch (e: Exception) {
            // openFd may fail for compressed assets; fall back to stream-read size estimate
            try {
                ctx.assets.open(assetPath).use { it.available().toLong() }
            } catch (e2: Exception) {
                0L
            }
        }
    }
}
