package com.varian.engcomp

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Kotlin adapter wrapping Chaquopy Python interpreter for vector search.
 *
 * Retrieval only — LLM runs via LlamaEngine (JNI).
 */
class PythonAdapter private constructor() {

    companion object {
        @Volatile
        private var instance: PythonAdapter? = null

        fun getInstance(context: Context): PythonAdapter {
            return instance ?: synchronized(this) {
                instance ?: PythonAdapter().also {
                    it.init(context)
                    instance = it
                }
            }
        }

        const val ASSET_DB_DIR = "db"

        fun dbPath(context: Context): String =
            context.getDir("db", Context.MODE_PRIVATE).absolutePath
    }

    private var python: Python? = null
    private var bridgeModule: PyObject? = null
    private var isReady: Boolean = false

    fun init(context: Context) {
        if (isReady) return

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        python = Python.getInstance()

        copyAssetsIfNeeded(context)

        // Add project root to Python path
        val repoRoot = context.filesDir.absolutePath
        python!!.getModule("sys")
            .get("path")
            .callAttr("insert", 0, repoRoot)

        bridgeModule = python!!.getModule("android.chaquopy.chaquopy_bridge")
        isReady = true
    }

    fun search(context: Context, query: String): List<SearchResult> {
        if (!isReady || bridgeModule == null) {
            return emptyList()
        }

        val dbPath = dbPath(context) + "/engineer.db"
        val jsonStr: String = bridgeModule!!.callAttr(
            "search",
            query,
            dbPath,
            "intfloat/multilingual-e5-small"
        ).toString()

        val json = JSONObject(jsonStr)
        if (!json.optBoolean("ok", false)) {
            val error = json.optString("error", "unknown")
            Log.e("PythonAdapter", "search error: $error")
            return emptyList()
        }

        val results = mutableListOf<SearchResult>()
        val arr = json.getJSONArray("results")
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            results.add(
                SearchResult(
                    text = obj.optString("text"),
                    source = obj.optString("source"),
                    page = obj.optInt("page"),
                    section = obj.optString("section"),
                    score = obj.optDouble("score", 0.0).toFloat()
                )
            )
        }
        return results
    }

    fun getDbInfo(context: Context): Pair<Int, String?> {
        val dbPath = dbPath(context) + "/engineer.db"
        val jsonStr: String = bridgeModule!!.callAttr("get_db_info", dbPath).toString()
        val json = JSONObject(jsonStr)
        val ok = json.optBoolean("ok", false)
        return if (ok) {
            Pair(json.optInt("row_count", 0), null)
        } else {
            Pair(0, json.optString("error"))
        }
    }

    private fun copyAssetsIfNeeded(context: Context) {
        val dbDir = context.getDir("db", Context.MODE_PRIVATE)
        val marker = File(context.filesDir, ".assets_copied")
        if (marker.exists()) return

        copyAssetDir(context, ASSET_DB_DIR, dbDir)
        marker.createNewFile()
    }

    private fun copyAssetDir(context: Context, assetDir: String, destDir: File) {
        val assets = context.assets.list(assetDir) ?: return
        for (name in assets) {
            try {
                val assetPath = "$assetDir/$name"
                val outFile = File(destDir, name)
                context.assets.open(assetPath).use { input ->
                    outFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                Log.w("PythonAdapter", "Failed to copy $name: ${e.message}")
            }
        }
    }
}
