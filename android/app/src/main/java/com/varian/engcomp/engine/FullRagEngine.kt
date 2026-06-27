package com.varian.engcomp.engine

import android.util.Log
import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Full RAG engine: ONNX retrieval + llama.cpp on-device generation.
 *
 * Pipeline:
 *   Embed → Search(topK) → Prompt(sources) → GenerateStart
 *   → Token(piece)* [streaming] → Done(fullAnswer, sources)
 *
 * If the LlamaEngine is not loaded, emits a helpful error message instead of
 * crashing, so the UI degrades gracefully.
 *
 * Token streaming: JNI callback runs on the IO thread. A [Channel] bridges
 * the callback into the flow collector so each piece is emitted as
 * [RagEvent.Token] in real time.
 *
 * @param retriever Initialized OnnxRetriever.
 * @param llama     LlamaEngine instance (model already loaded via loadModel()).
 */
class FullRagEngine(
    private val retriever: OnnxRetriever,
    private val llama: LlamaEngine,
) : RagEngine {

    companion object {
        private const val TAG        = "FullRagEngine"
        private const val TOP_K      = 8
        private const val POOL_SIZE  = 30
        private const val MAX_TOKENS = 512
        private const val TEMPERATURE = 0.3f
    }

    private val promptBuilder = PromptBuilder()

    override fun ask(query: String, history: List<Turn>): Flow<RagEvent> = flow {
        // 1. Embed
        emit(RagEvent.Embed)

        // 2. Search
        val sources: List<SearchResult> = withContext(Dispatchers.Default) {
            retriever.search(query, topK = TOP_K)
        }
        Log.d(TAG, "ask: search returned ${sources.size} results")
        emit(RagEvent.Search(found = POOL_SIZE))

        // 3. Prompt
        emit(RagEvent.Prompt(sources = sources))

        // 4. Generate
        emit(RagEvent.GenerateStart)

        if (!llama.isLoaded) {
            val msg = "Модель не загружена. Установите gemma-3-4b-it-Q4_K_M.gguf через экран настройки."
            emit(RagEvent.Token(msg))
            emit(RagEvent.Done(answer = msg, sources = sources))
            return@flow
        }

        val prompt = promptBuilder.build(query, sources, history)

        // Bridge JNI callback → coroutine flow via Channel
        val tokenChannel = Channel<String>(capacity = Channel.UNLIMITED)

        try {
            val sb = StringBuilder()
            coroutineScope {
                // Launch generation on IO; tokens arrive via channel
                launch(Dispatchers.IO) {
                    try {
                        llama.generate(
                            prompt      = prompt,
                            maxTokens   = MAX_TOKENS,
                            temperature = TEMPERATURE,
                        ) { piece ->
                            tokenChannel.trySend(piece)
                        }
                    } finally {
                        tokenChannel.close()
                    }
                }

                for (piece in tokenChannel) {
                    sb.append(piece)
                    emit(RagEvent.Token(piece))
                }
            }

            val answer = PromptBuilder.stripSourceTags(sb.toString().trim())
            emit(RagEvent.Done(answer = answer, sources = sources))
        } catch (e: Exception) {
            Log.e(TAG, "Generation failed", e)
            tokenChannel.close()
            emit(RagEvent.Error("Ошибка генерации: ${e.message}"))
        }
    }

    override suspend fun search(query: String): List<SearchResult> =
        withContext(Dispatchers.Default) { retriever.search(query, topK = TOP_K) }
}
