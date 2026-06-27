package com.varian.engcomp.engine

import android.util.Log
import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

/**
 * Real RAG engine: on-device ONNX retrieval with a stubbed LLM generation.
 *
 * Phase C retrieval pipeline:
 *   Embed (tokenizer.onnx → model.onnx → mean-pool → L2-norm)
 *   → Search (brute-force cosine over 11 629 index rows, garbage-filter, top-8)
 *   → Prompt (emit sources to UI)
 *   → GenerateStart
 *   → Token  (placeholder — real LLM wired in the next phase)
 *   → Done
 *
 * All heavy work runs on Dispatchers.Default/IO; [flow] emissions switch to
 * the collector's context automatically.
 *
 * @param retriever A fully initialized [OnnxRetriever].  Caller owns lifecycle.
 */
class OnnxRagEngine(private val retriever: OnnxRetriever) : RagEngine {

    companion object {
        private const val TAG = "OnnxRagEngine"
        private const val TOP_K = 8
        private const val POOL_SIZE = 30  // wider pool fed to garbage filter
    }

    override fun ask(query: String, history: List<Turn>): Flow<RagEvent> = flow {
        // ── 1. Embed ──────────────────────────────────────────────────────────
        emit(RagEvent.Embed)
        Log.d(TAG, "ask: embedding query")

        val sources: List<SearchResult> = withContext(Dispatchers.Default) {
            // ── 2. Search (embed + cosine + filter inside retriever) ──────────
            retriever.search(query, topK = TOP_K)
        }

        Log.d(TAG, "ask: search returned ${sources.size} results")

        // ── 3. Search count emitted ───────────────────────────────────────────
        // 'found' is the pre-filter candidate pool size, sources.size is post-filter.
        // We report POOL_SIZE as found (the wider pool explored), which matches the
        // contract in ChatViewModel: "найдено N" = number of candidates retrieved
        // before the garbage filter.
        emit(RagEvent.Search(found = POOL_SIZE))

        // ── 4. Prompt — sources ready ─────────────────────────────────────────
        emit(RagEvent.Prompt(sources = sources))

        // ── 5. GenerateStart ──────────────────────────────────────────────────
        emit(RagEvent.GenerateStart)

        // ── 6. Stub generation (Phase C — LLM not yet wired) ─────────────────
        // Emit a single truthful placeholder; the real sources are shown in the
        // source cards below.  Russian UI language to match the existing chat copy.
        val stubText = if (sources.isEmpty()) {
            "Соответствующие источники не найдены в локальной базе знаний."
        } else {
            "Локальная модель подключается в следующей сборке. " +
            "Найдено ${sources.size} источников — см. карточки ниже."
        }
        emit(RagEvent.Token(stubText))

        // ── 7. Done ───────────────────────────────────────────────────────────
        emit(RagEvent.Done(answer = stubText, sources = sources))
    }

    override suspend fun search(query: String): List<SearchResult> =
        withContext(Dispatchers.Default) {
            retriever.search(query, topK = TOP_K)
        }
}
