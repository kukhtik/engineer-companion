package com.varian.engcomp.engine

import com.varian.engcomp.model.SearchResult

sealed class RagEvent {
    object Embed : RagEvent()
    data class Search(val found: Int) : RagEvent()
    data class Prompt(val sources: List<SearchResult>) : RagEvent()
    object GenerateStart : RagEvent()
    data class Token(val text: String) : RagEvent()
    data class Done(val answer: String, val sources: List<SearchResult>) : RagEvent()
    data class Error(val message: String) : RagEvent()
}
