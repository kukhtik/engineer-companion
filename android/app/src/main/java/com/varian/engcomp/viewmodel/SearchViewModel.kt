package com.varian.engcomp.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class SearchViewModel : ViewModel() {
    private val _query = MutableStateFlow("")
    val query: StateFlow<String> = _query

    private val _sources = MutableStateFlow<List<com.varian.engcomp.model.SearchResult>>(emptyList())
    val sources: StateFlow<List<com.varian.engcomp.model.SearchResult>> = _sources

    private val _answer = MutableStateFlow("")
    val answer: StateFlow<String> = _answer

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error

    private val _dbInfo = MutableStateFlow("БД: не загружена (Compose shell)")
    val dbInfo: StateFlow<String> = _dbInfo

    fun onQueryChanged(text: String) { _query.value = text }
    fun search() { _answer.value = "[Shell mode — ядро Python/JNI не встроено]" }
}
