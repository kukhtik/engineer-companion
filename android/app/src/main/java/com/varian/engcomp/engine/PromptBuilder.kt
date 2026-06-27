package com.varian.engcomp.engine

import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult

/**
 * Builds the Russian-language RAG prompt for the on-device LLM.
 *
 * Mirrors the Python core/query.py PromptBuilder logic:
 *  - System persona with strict rules (Russian only, source tags, no hallucination)
 *  - Context block: top-K search results formatted with [ИСТОЧНИК N] tags
 *  - History: last 3 Q/A turns, each capped at 300 chars, total capped at 1000 chars
 *  - Final question line + answer prompt
 */
class PromptBuilder {

    companion object {
        private const val MAX_HISTORY_TURNS  = 3
        private const val MAX_TURN_CHARS     = 300
        private const val MAX_HISTORY_CHARS  = 1000

        /**
         * Regex that matches source citation tags like "[ИСТОЧНИК 2] Название стр.47".
         * Used to strip raw citations from the generated answer before displaying.
         */
        val SOURCE_TAG_REGEX: Regex = Regex("""\[ИСТОЧНИК\s+\d+\][^\n]*""")

        /** Remove source citation tags from a generated answer string. */
        fun stripSourceTags(answer: String): String =
            SOURCE_TAG_REGEX.replace(answer, "").trim()

        private val SYSTEM_PERSONA = """
Ты — эксперт-ассистент, отвечающий строго по предоставленной документации.
ПРАВИЛА ОТВЕТА:
1. Используй ТОЛЬКО те источники из КОНТЕКСТА, которые реально относятся к вопросу. Нерелевантные фрагменты — полностью игнорируй, не упоминай и не суммируй их.
2. При цитировании указывай метку источника ТОЧНО так, как она написана в тексте контекста: например «[ИСТОЧНИК 2] Название документа стр.47». НЕ меняй название документа, не путай документы между собой, НЕ придумывай страницы.
3. Если ответ не найден ни в одном из релевантных фрагментов — честно скажи «не найдено в документации».
4. Отвечай на русском языке.
5. НЕ выдумывай расшифровку аббревиатур и фактов, которых нет в контексте. Если точного определения аббревиатуры или термина нет в предоставленных фрагментах — скажи явно: «В документации найдено только косвенное упоминание» и приведи то, что есть. Никогда не изобретай расшифровку на основе догадок.
КРИТИЧЕСКИ ВАЖНО: отвечай ТОЛЬКО на русском языке, даже если документация на английском. Весь текст ответа обязательно на русском; технические термины и названия моделей можно оставлять как есть.
        """.trimIndent()
    }

    /**
     * Build the full prompt string.
     *
     * @param query   The user's current question.
     * @param sources Retrieval results from OnnxRetriever (already ranked).
     * @param history Conversation history as [Turn] list (last 3 Q/A pairs used).
     */
    fun build(
        query: String,
        sources: List<SearchResult>,
        history: List<Turn>,
    ): String {
        val sb = StringBuilder()

        // System persona
        sb.append(SYSTEM_PERSONA)
        sb.append("\n\n")

        // Context block
        if (sources.isNotEmpty()) {
            sb.append("КОНТЕКСТ:\n")
            sources.forEachIndexed { idx, sr ->
                sb.append("[ИСТОЧНИК ${idx + 1}] Документ: «${sr.source}» стр.${sr.page} раздел: ${sr.section}\n")
                sb.append(sr.text)
                sb.append("\n\n")
            }
        }

        // History block (last N turns, capped per-turn and overall)
        val historyBlock = buildHistoryBlock(history)
        if (historyBlock.isNotEmpty()) {
            sb.append("ИСТОРИЯ ДИАЛОГА:\n")
            sb.append(historyBlock)
            sb.append("\n")
        }

        // Question + answer prompt
        sb.append("\nВОПРОС: $query\n\nОТВЕТ НА РУССКОМ ЯЗЫКЕ:")

        return sb.toString()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun buildHistoryBlock(history: List<Turn>): String {
        if (history.isEmpty()) return ""

        // Take last MAX_HISTORY_TURNS Q/A pairs (user+assistant = 2 entries each)
        // We work in pairs: (user, assistant)
        val pairs = mutableListOf<Pair<String, String>>()
        var i = 0
        while (i + 1 < history.size) {
            val roleA = history[i].role
            val roleB = history[i + 1].role
            if (roleA == "user" && roleB == "assistant") {
                pairs.add(history[i].content to history[i + 1].content)
                i += 2
            } else {
                i++
            }
        }

        val recentPairs = pairs.takeLast(MAX_HISTORY_TURNS)

        val sb = StringBuilder()
        var totalChars = 0

        for ((userContent, assistantContent) in recentPairs) {
            val userTrunc = userContent.take(MAX_TURN_CHARS)
            val assistantTrunc = assistantContent.take(MAX_TURN_CHARS)
            val entry = "Пользователь: $userTrunc\nАссистент: $assistantTrunc\n"

            if (totalChars + entry.length > MAX_HISTORY_CHARS) break
            sb.append(entry)
            totalChars += entry.length
        }

        return sb.toString()
    }
}
