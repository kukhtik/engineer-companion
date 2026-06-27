package com.varian.engcomp.engine

import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.flow.Flow

interface RagEngine {
    fun ask(query: String, history: List<Turn>): Flow<RagEvent>
    suspend fun search(query: String): List<SearchResult>
}
