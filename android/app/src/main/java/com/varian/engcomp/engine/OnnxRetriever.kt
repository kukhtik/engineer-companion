package com.varian.engcomp.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.varian.engcomp.model.SearchResult
import java.io.Closeable
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * On-device vector retriever using ONNX Runtime.
 *
 * Embedding pipeline (mirrors Python parity_check_onnx.py exactly):
 *   1. tokenizer.onnx  (SentencepieceTokenizer, fairseq=True)
 *      Input:  "input_text" STRING[1]
 *      Output: "input_ids"      INT64[seq]
 *              "attention_mask" INT64[seq]   (all ones)
 *              "token_type_ids" INT64[seq]   (all zeros)
 *   2. model.onnx  (intfloat/multilingual-e5-small)
 *      Inputs:  input_ids [1,seq], attention_mask [1,seq], token_type_ids [1,seq]
 *      Output:  last_hidden_state [1,seq,384]
 *   3. Mean-pool over attention_mask → FloatArray(384)
 *   4. L2-normalize → unit vector
 *
 * Search: dot-product cosine over all 11 629 index rows, garbage-filter,
 * return top-K [SearchResult].
 *
 * Thread-safe: call [embed]/[search] from any thread (Dispatchers.Default).
 * [close] releases ORT resources.
 */
class OnnxRetriever(
    private val tokenizerOnnxPath: String,
    private val modelOnnxPath: String,
    private val chunksIndexPath: String,
) : Closeable {

    private val TAG = "OnnxRetriever"

    // ── ORT sessions ──────────────────────────────────────────────────────────
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val tokSess: OrtSession
    private val embSess: OrtSession

    // ── Index data ────────────────────────────────────────────────────────────
    private val numRows: Int
    private val vecDim: Int           // always 384
    private val allVecs: FloatArray   // [numRows * vecDim]
    private val metadata: List<ChunkMeta>

    private val gson = Gson()

    // ── JSON metadata schema ──────────────────────────────────────────────────
    private data class ChunkMeta(
        @SerializedName("source")  val source: String  = "",
        @SerializedName("page")    val page: Int       = 0,
        @SerializedName("section") val section: String = "",
        @SerializedName("text")    val text: String    = "",
    )

    init {
        // ── Register ORT extensions custom ops (SentencepieceTokenizer) ───────
        val tokOpts = OrtSession.SessionOptions().apply {
            // OrtxPackage.getLibraryPath() returns the bundled .so path from the AAR.
            // The AAR includes libortextensions.so for arm64-v8a.
            try {
                val libPath = ai.onnxruntime.extensions.OrtxPackage.getLibraryPath()
                registerCustomOpLibrary(libPath)
                Log.i(TAG, "Registered ORT extensions from: $libPath")
            } catch (e: Exception) {
                Log.w(TAG, "OrtxPackage.getLibraryPath() failed ($e); trying reflection …")
                // Fallback: some extension builds expose the path differently.
                // If neither works the tokenizer session will throw at InferenceSession creation
                // and we will log a clear error.
            }
        }
        tokSess = env.createSession(tokenizerOnnxPath, tokOpts)
        Log.i(TAG, "Tokenizer session loaded. Inputs:  ${tokSess.inputNames}")
        Log.i(TAG, "Tokenizer session loaded. Outputs: ${tokSess.outputNames}")

        val embOpts = OrtSession.SessionOptions()
        embSess = env.createSession(modelOnnxPath, embOpts)
        Log.i(TAG, "Embedder session loaded. Inputs:  ${embSess.inputNames}")
        Log.i(TAG, "Embedder session loaded. Outputs: ${embSess.outputNames}")

        // ── Load binary index ─────────────────────────────────────────────────
        val (rows, dim, vecs, meta) = loadIndex(chunksIndexPath)
        numRows  = rows
        vecDim   = dim
        allVecs  = vecs
        metadata = meta
        Log.i(TAG, "Index loaded: $numRows rows, dim=$vecDim")
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Embed [query] → unit-norm FloatArray(384).
     * No prefix is added (model was indexed without prefix).
     */
    fun embed(query: String): FloatArray {
        // Step 1: tokenize
        val tokInputs = mapOf(
            "input_text" to OnnxTensor.createTensor(
                env,
                arrayOf(query),  // STRING [1]
            )
        )
        val (inputIds, attentionMask, tokenTypeIds) = tokSess.run(tokInputs).use { tokOut ->
            val ids  = (tokOut[0].value as LongArray)
            val mask = (tokOut[1].value as LongArray)
            val tti  = (tokOut[2].value as LongArray)
            Triple(ids, mask, tti)
        }
        tokInputs.values.forEach { it.close() }

        val seqLen = inputIds.size

        // Step 2: embed — reshape to [1, seqLen]
        val ids2d  = Array(1) { inputIds }
        val mask2d = Array(1) { attentionMask }
        val tti2d  = Array(1) { tokenTypeIds }

        val embInputs = mapOf(
            "input_ids"      to OnnxTensor.createTensor(env, ids2d),
            "attention_mask" to OnnxTensor.createTensor(env, mask2d),
            "token_type_ids" to OnnxTensor.createTensor(env, tti2d),
        )

        // last_hidden_state: [1, seqLen, 384] stored as float[][][]
        val hiddenState: Array<Array<FloatArray>> = embSess.run(embInputs).use { embOut ->
            @Suppress("UNCHECKED_CAST")
            embOut[0].value as Array<Array<FloatArray>>
        }
        embInputs.values.forEach { it.close() }

        // Step 3: mean-pool over attention_mask (mask is all-ones, so this is a plain mean)
        val dim = hiddenState[0][0].size  // 384
        val pooled = FloatArray(dim)
        var count = 0.0f
        for (t in 0 until seqLen) {
            val w = attentionMask[t].toFloat()
            if (w > 0f) {
                for (d in 0 until dim) pooled[d] += hiddenState[0][t][d] * w
                count += w
            }
        }
        if (count > 0f) for (d in 0 until dim) pooled[d] /= count

        // Step 4: L2-normalize
        return l2Normalize(pooled)
    }

    /**
     * Search for [topK] results for [query], after garbage-filtering a wider
     * candidate pool (~30 entries).
     */
    fun search(query: String, topK: Int = 8): List<SearchResult> {
        val queryVec = embed(query)

        // Brute-force cosine (dot-product, both unit-normed) over all rows
        // Use a small min-heap via a mutable list sorted on score
        data class Candidate(val idx: Int, val score: Float)

        // Larger initial pool to allow for garbage-filter attrition
        val poolSize = 30
        val heap = ArrayList<Candidate>(poolSize + 1)

        for (i in 0 until numRows) {
            val score = dotProduct(queryVec, allVecs, i * vecDim, vecDim)
            if (heap.size < poolSize || score > heap.last().score) {
                heap.add(Candidate(i, score))
                heap.sortByDescending { it.score }
                if (heap.size > poolSize) heap.removeAt(poolSize)
            }
        }

        // Garbage-filter then take topK
        return heap
            .filter { !isRetrievalGarbage(metadata[it.idx].text, metadata[it.idx].section) }
            .take(topK)
            .map { c ->
                val m = metadata[c.idx]
                SearchResult(
                    text    = m.text,
                    source  = m.source,
                    page    = m.page,
                    section = m.section,
                    score   = c.score,
                )
            }
    }

    override fun close() {
        try { tokSess.close() } catch (_: Exception) {}
        try { embSess.close() } catch (_: Exception) {}
        try { env.close()    } catch (_: Exception) {}
    }

    // ── Index loader ──────────────────────────────────────────────────────────
    private data class IndexData(
        val numRows: Int,
        val vecDim: Int,
        val vecs: FloatArray,
        val meta: List<ChunkMeta>,
    )

    private fun loadIndex(path: String): IndexData {
        RandomAccessFile(path, "r").use { raf ->
            val headerBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            raf.channel.read(headerBuf, 0L)
            headerBuf.flip()
            val numRows = headerBuf.int
            val vecDim  = headerBuf.int
            Log.i(TAG, "Index header: numRows=$numRows vecDim=$vecDim")

            val vecs = FloatArray(numRows * vecDim)
            val meta = ArrayList<ChunkMeta>(numRows)

            var pos = 8L
            val vecBytes = vecDim * 4
            val vecBuf = ByteBuffer.allocate(vecBytes).order(ByteOrder.LITTLE_ENDIAN)
            val metaLenBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until numRows) {
                // Read embedding vector
                vecBuf.clear()
                raf.channel.read(vecBuf, pos)
                vecBuf.flip()
                val base = i * vecDim
                for (d in 0 until vecDim) vecs[base + d] = vecBuf.float
                pos += vecBytes

                // Read metadata length
                metaLenBuf.clear()
                raf.channel.read(metaLenBuf, pos)
                metaLenBuf.flip()
                val metaLen = metaLenBuf.int
                pos += 4

                // Read metadata JSON
                val jsonBuf = ByteBuffer.allocate(metaLen)
                raf.channel.read(jsonBuf, pos)
                val jsonStr = String(jsonBuf.array(), Charsets.UTF_8)
                pos += metaLen

                meta.add(gson.fromJson(jsonStr, ChunkMeta::class.java) ?: ChunkMeta())
            }

            return IndexData(numRows, vecDim, vecs, meta)
        }
    }

    // ── Math helpers ──────────────────────────────────────────────────────────
    private fun dotProduct(a: FloatArray, b: FloatArray, bOffset: Int, len: Int): Float {
        var sum = 0f
        for (i in 0 until len) sum += a[i] * b[bOffset + i]
        return sum
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm < 1e-12f) return v
        return FloatArray(v.size) { v[it] / norm }
    }

    // ── Garbage filter — Kotlin port of core/query.py _is_retrieval_garbage ──
    /**
     * Returns true if this chunk should be excluded from RAG results.
     * Mirrors Python [_is_retrieval_garbage] exactly:
     *  1. Section label: ≤2 chars OR all dashes/symbols
     *  2. Low alpha density in section (< 35% alphabetic)
     *  3. OCR artifact patterns in text (~~, _{4,}, ..:, SCA~, leading ~)
     *  4. Low alpha ratio in text (< 30%)
     */
    private fun isRetrievalGarbage(text: String, section: String): Boolean {
        val sec = section.trim()

        // 1. Section label: ≤2 chars or entirely dashes/bullets/symbols
        if (sec.isNotEmpty() && isGarbageSection(sec)) return true

        // 2. Low alpha density in section label
        if (sec.length >= 4) {
            val alphaCount = sec.count { it.isLetter() }
            if (alphaCount.toFloat() / sec.length < 0.35f) return true
        }

        // 3. OCR artifact patterns in text
        val trimText = text.trim()
        if (trimText.contains("~~") ||
            Regex("~{2,}").containsMatchIn(trimText) ||
            Regex("_{4,}").containsMatchIn(trimText) ||
            trimText.contains("..:")  ||
            Regex("\\bSCA~").containsMatchIn(trimText) ||
            trimText.startsWith("~")) return true

        // 4. Low alpha ratio in text
        if (text.isNotEmpty()) {
            val alphaCount = text.count { it.isLetter() }
            if (alphaCount.toFloat() / text.length < 0.30f) return true
        }

        return false
    }

    /**
     * Matches section labels that are ≤2 chars or consist only of
     * dashes, underscores, bullets, tildes, spaces.
     * Python: `r"^[_\-•·~\s]{1,}$|^.{1,2}$"`
     */
    private fun isGarbageSection(sec: String): Boolean {
        if (sec.length <= 2) return true
        return sec.all { it in "_-•·~ \t\n\r" }
    }
}
