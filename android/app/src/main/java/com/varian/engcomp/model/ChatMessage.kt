package com.varian.engcomp.model

/**
 * Chat message model for UI display.
 */
data class ChatMessage(
    val role: String,  // "user" or "assistant"
    val content: String,
    val sources: List<SearchResult> = emptyList()
)
