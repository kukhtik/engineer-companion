package com.varian.engineercompanion

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin adapter wrapping Chaquopy Python interpreter.
 *
 * Responsibilities:
 * 1. Initialize Python on app startup.
 * 2. Copy APK assets (models + DB) to internal storage on first run.
 * 3. Expose search(query) to ViewModels — retrieval only, no LLM.
 * 4. JSON serialization for clean Kotlin-Python boundary.
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

        // Asset paths inside APK
        const val ASSET_DB_DIR = "db"
        const val ASSET_MODEL_DIR = "models"

        // Internal storage paths
        fun dbPath(context: Context): String =
            context.getDir("db", Context.MODE_PRIVATE).absolutePath

        fun modelsPath(context: Context): String =
            context.getDir("models", Context.MODE_PRIVATE).absolutePath
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

        // Ensure assets are on disk before Python tries to open them
        copyAssetsIfNeeded(context)

        // Add project root to Python path so `core/` and `android/` are importable
        val repoRoot = context.filesDir.absolutePath
        python!!.getModule("sys")
            .get("path")
            .callAttr("insert", 0, repoRoot)

        bridgeModule = python!!.getModule("android.chaquopy.chaquopy_bridge")
        isReady = true
    }

    /**
     * Search vector DB. Returns list of SearchResult or empty list on error.
     */
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
            android.util.Log.e("PythonAdapter", "search error: $error")
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

    /**
     * Return DB row count for diagnostics.
     */
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
        val modelDir = context.getDir("models", Context.MODE_PRIVATE)
        val marker = File(context.filesDir, ".assets_copied")
        if (marker.exists()) return

        copyAssetDir(context, ASSET_DB_DIR, dbDir)
        copyAssetDir(context, ASSET_MODEL_DIR, modelDir)
        marker.createNewFile()
    }

    private fun copyAssetDir(context: Context, assetDir: String, destDir: File) {
        val assets = context.assets.list(assetDir) ?: return
        for (name in assets) {
            val assetPath = "$assetDir/$name"
            val outFile = File(destDir, name)
            context.assets.open(assetPath).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
    }
}

data class SearchResult(
    val text: String,
    val source: String,
    val page: Int,
    val section: String,
    val score: Float
)
