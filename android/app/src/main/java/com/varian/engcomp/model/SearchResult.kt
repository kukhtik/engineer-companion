package com.varian.engcomp

/**
 * Search result data class from Python Chaquopy bridge.
 */
data class SearchResult(
    val text: String,
    val source: String,
    val page: Int,
    val section: String,
    val score: Float
)
