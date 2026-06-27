package com.varian.engcomp.engine

import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

class FakeRagEngine : RagEngine {

    private val fakeSources = listOf(
        SearchResult(
            text = "Система обеспечивает точное позиционирование пучка с погрешностью менее 0.5 мм.",
            source = "technical_manual_v4.pdf",
            page = 47,
            section = "Система позиционирования",
            score = 0.94f
        ),
        SearchResult(
            text = "Процедура контроля качества включает ежедневную проверку изоцентра и дозиметрических параметров.",
            source = "qa_procedures_2024.pdf",
            page = 12,
            section = "Контроль качества",
            score = 0.87f
        ),
        SearchResult(
            text = "Коллиматор MLC содержит 120 листьев, обеспечивая формирование поля до 40×40 см².",
            source = "mlc_reference_guide.pdf",
            page = 8,
            section = "MLC — Технические характеристики",
            score = 0.82f
        )
    )

    private val fakeTokens = listOf(
        "Система ", "позиционирования ", "пучка ", "обеспечивает ", "высокую ", "точность — ",
        "менее ", "0.5 мм ", "от ", "изоцентра. ", "Для ", "контроля ", "качества ",
        "проводится ", "ежедневная ", "проверка ", "дозиметрических ", "параметров.",
        " MLC ", "поддерживает ", "120 ", "листьев ", "для ", "формирования ",
        "произвольной ", "формы ", "поля.", " Подробнее ", "см. ", "техническое ",
        "руководство ", "раздел ", "\"Система позиционирования\"."
    )

    override fun ask(query: String, history: List<Turn>): Flow<RagEvent> = flow {
        emit(RagEvent.Embed)
        delay(350)

        emit(RagEvent.Search(found = 20))
        delay(250)

        emit(RagEvent.Prompt(sources = fakeSources))
        delay(200)

        emit(RagEvent.GenerateStart)
        delay(150)

        val sb = StringBuilder()
        for (token in fakeTokens) {
            sb.append(token)
            emit(RagEvent.Token(text = token))
            delay(60L + (Math.random() * 80).toLong())
        }

        delay(100)
        emit(RagEvent.Done(answer = sb.toString(), sources = fakeSources))
    }

    override suspend fun search(query: String): List<SearchResult> = fakeSources
}
