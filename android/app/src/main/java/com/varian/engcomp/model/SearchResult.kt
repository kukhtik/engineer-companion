package com.varian.engcomp.model

data class SearchResult(
    val text: String,
    val source: String,
    val page: Int,
    val section: String,
    val score: Float
)
