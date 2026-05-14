package com.varian.engcomp.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.varian.engcomp.LlamaEngine
import com.varian.engcomp.PythonAdapter
import com.varian.engcomp.SearchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class SearchViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "SearchViewModel"
        private const val SYSTEM_PROMPT = "Ты — сервисный инженер-эксперт " +
            "по медицинским линейным ускорителям Varian TrueBeam и VitalBeam. " +
            "Отвечай строго по предоставленной документации. " +
            "Если ответа нет в тексте — честно скажи «не найдено в документации». " +
            "Цитируй название документа и номер страницы. Отвечай на русском."
    }

    private val pythonAdapter = PythonAdapter.getInstance(application)
    private val llamaEngine = LlamaEngine()

    private val _query = MutableLiveData("")
    val query: LiveData<String> = _query

    private val _sources = MutableLiveData<List<SearchResult>>(emptyList())
    val sources: LiveData<List<SearchResult>> = _sources

    private val _answer = MutableLiveData("")
    val answer: LiveData<String> = _answer

    private val _isSearching = MutableLiveData(false)
    val isSearching: LiveData<Boolean> = _isSearching

    private val _error = MutableLiveData<String?>(null)
    val error: LiveData<String?> = _error

    private val _dbInfo = MutableLiveData("")
    val dbInfo: LiveData<String> = _dbInfo

    init {
        loadDbInfo()
    }

    private fun loadDbInfo() {
        viewModelScope.launch {
            try {
                val (count, err) = withContext(Dispatchers.IO) {
                    pythonAdapter.getDbInfo(getApplication())
                }
                _dbInfo.postValue(if (err != null) "Ошибка БД: $err" else "БД: $count чанков")
            } catch (e: Exception) {
                _dbInfo.postValue("Не удалось получить инфо о БД")
            }
        }
    }

    fun onQueryChanged(text: String) {
        _query.value = text
    }

    fun search() {
        val q = _query.value?.trim() ?: return
        if (q.isEmpty()) return

        _isSearching.postValue(true)
        _error.postValue(null)
        _answer.postValue("")
        _sources.postValue(emptyList())

        viewModelScope.launch {
            try {
                // Step 1: Retrieval via Chaquopy
                val results = withContext(Dispatchers.IO) {
                    pythonAdapter.search(getApplication(), q)
                }
                _sources.postValue(results)

                if (results.isEmpty()) {
                    _answer.postValue("Не найдено результатов в документации.")
                    _isSearching.postValue(false)
                    return@launch
                }

                // Step 2: Build prompt
                val prompt = buildPrompt(q, results)

                // Step 3: Generate via JNI llama.cpp
                val answerText = withContext(Dispatchers.IO) {
                    llamaEngine.generate(prompt, maxTokens = 256, temperature = 0.3f)
                }

                if (answerText.isEmpty()) {
                    _answer.postValue("[LLM не отвечает]")
                } else {
                    _answer.postValue(answerText)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Search failed", e)
                _error.postValue("Ошибка: ${e.message}")
            } finally {
                _isSearching.postValue(false)
            }
        }
    }

    private fun buildPrompt(query: String, hits: List<SearchResult>): String {
        val contextParts = hits.mapIndexed { i, h ->
            "[ИСТОЧНИК ${i + 1}] ${h.source} стр.${h.page} раздел: ${h.section}\n${h.text}"
        }
        val context = contextParts.joinToString("\n\n")
        return "$SYSTEM_PROMPT\n\nКОНТЕКСТ:\n$context\n\nВОПРОС: $query\n\nОТВЕТ:"
    }

    override fun onCleared() {
        super.onCleared()
        llamaEngine.unload()
    }
}
