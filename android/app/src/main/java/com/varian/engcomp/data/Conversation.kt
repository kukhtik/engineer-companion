package com.varian.engcomp.data

import com.varian.engcomp.model.SearchResult

data class Turn(
    val role: String,           // "user" or "assistant"
    val content: String,
    val sources: List<SearchResult> = emptyList(),
    val starred: Boolean = false
)

data class Conversation(
    val id: String,
    var title: String,
    val createdMillis: Long,
    val turns: MutableList<Turn> = mutableListOf()
)
