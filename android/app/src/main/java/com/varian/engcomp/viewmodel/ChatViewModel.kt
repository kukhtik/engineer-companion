package com.varian.engcomp.viewmodel

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.varian.engcomp.data.Conversation
import com.varian.engcomp.data.ConversationStore
import com.varian.engcomp.data.Turn
import com.varian.engcomp.engine.FakeRagEngine
import com.varian.engcomp.engine.RagEngine
import com.varian.engcomp.engine.RagEvent
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ActivityStage(
    val label: String,
    val detail: String = ""
)

data class ChatUiState(
    val conversations: List<Conversation> = emptyList(),
    val currentConversationId: String = "",
    val currentTurns: List<Turn> = emptyList(),
    val isGenerating: Boolean = false,
    val streamingAnswer: String = "",
    val activityStage: ActivityStage? = null,
    val tokenCount: Int = 0,
    val elapsedMs: Long = 0L,
    val showFavoritesOnly: Boolean = false,
    val isDarkTheme: Boolean = true,
    val sourcesForCurrentGen: List<SearchResult> = emptyList()
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val store = ConversationStore(application)
    private val engine: RagEngine = FakeRagEngine()

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        val conv = store.getOrCreateEmpty()
        _uiState.value = ChatUiState(
            conversations = store.conversations,
            currentConversationId = conv.id,
            currentTurns = conv.turns.toList(),
            isDarkTheme = true
        )
    }

    fun send(query: String) {
        val convId = _uiState.value.currentConversationId
        if (query.isBlank() || _uiState.value.isGenerating) return

        // Add user turn
        val userTurn = Turn(role = "user", content = query)
        store.addTurn(convId, userTurn)

        // Add placeholder assistant turn
        val assistantPlaceholder = Turn(role = "assistant", content = "")
        store.addTurn(convId, assistantPlaceholder)

        val history = store.getConversation(convId)?.turns?.dropLast(1) ?: emptyList()

        _uiState.value = _uiState.value.copy(
            conversations = store.conversations,
            currentTurns = store.getConversation(convId)?.turns?.toList() ?: emptyList(),
            isGenerating = true,
            streamingAnswer = "",
            tokenCount = 0,
            elapsedMs = 0L,
            sourcesForCurrentGen = emptyList()
        )

        val startMs = System.currentTimeMillis()
        var tokenCount = 0

        viewModelScope.launch(Dispatchers.Default) {
            engine.ask(query, history).collect { event ->
                when (event) {
                    is RagEvent.Embed -> updateStage("Векторизация", "")
                    is RagEvent.Search -> updateStage("Поиск", "найдено ${event.found}")
                    is RagEvent.Prompt -> {
                        updateStage("Контекст", "${event.sources.size} источников")
                        _uiState.value = _uiState.value.copy(sourcesForCurrentGen = event.sources)
                    }
                    is RagEvent.GenerateStart -> updateStage("Генерация", "")
                    is RagEvent.Token -> {
                        tokenCount++
                        val newAnswer = _uiState.value.streamingAnswer + event.text
                        _uiState.value = _uiState.value.copy(
                            streamingAnswer = newAnswer,
                            tokenCount = tokenCount,
                            elapsedMs = System.currentTimeMillis() - startMs
                        )
                    }
                    is RagEvent.Done -> {
                        store.updateLastAssistantTurn(convId, event.answer, event.sources)
                        _uiState.value = _uiState.value.copy(
                            conversations = store.conversations,
                            currentTurns = store.getConversation(convId)?.turns?.toList() ?: emptyList(),
                            isGenerating = false,
                            streamingAnswer = "",
                            activityStage = null,
                            tokenCount = tokenCount,
                            elapsedMs = System.currentTimeMillis() - startMs,
                            sourcesForCurrentGen = emptyList()
                        )
                    }
                    is RagEvent.Error -> {
                        store.updateLastAssistantTurn(convId, "Ошибка: ${event.message}", emptyList())
                        _uiState.value = _uiState.value.copy(
                            conversations = store.conversations,
                            currentTurns = store.getConversation(convId)?.turns?.toList() ?: emptyList(),
                            isGenerating = false,
                            streamingAnswer = "",
                            activityStage = null
                        )
                    }
                }
            }
        }
    }

    private fun updateStage(label: String, detail: String) {
        _uiState.value = _uiState.value.copy(activityStage = ActivityStage(label, detail))
    }

    fun newChat() {
        val conv = store.getOrCreateEmpty()
        _uiState.value = _uiState.value.copy(
            conversations = store.conversations,
            currentConversationId = conv.id,
            currentTurns = conv.turns.toList(),
            isGenerating = false,
            streamingAnswer = "",
            activityStage = null,
            showFavoritesOnly = false
        )
    }

    fun openConversation(id: String) {
        val conv = store.getConversation(id) ?: return
        _uiState.value = _uiState.value.copy(
            currentConversationId = id,
            currentTurns = conv.turns.toList(),
            isGenerating = false,
            streamingAnswer = "",
            activityStage = null,
            showFavoritesOnly = false
        )
    }

    fun deleteConversation(id: String) {
        val currentId = _uiState.value.currentConversationId
        store.delete(id)
        if (id == currentId) {
            val conv = store.getOrCreateEmpty()
            _uiState.value = _uiState.value.copy(
                conversations = store.conversations,
                currentConversationId = conv.id,
                currentTurns = conv.turns.toList()
            )
        } else {
            _uiState.value = _uiState.value.copy(conversations = store.conversations)
        }
    }

    fun toggleFavorite(turnIndex: Int) {
        val convId = _uiState.value.currentConversationId
        store.toggleStarTurn(convId, turnIndex)
        _uiState.value = _uiState.value.copy(
            currentTurns = store.getConversation(convId)?.turns?.toList() ?: emptyList(),
            conversations = store.conversations
        )
    }

    fun toggleFavoritesFilter() {
        _uiState.value = _uiState.value.copy(showFavoritesOnly = !_uiState.value.showFavoritesOnly)
    }

    fun toggleTheme() {
        _uiState.value = _uiState.value.copy(isDarkTheme = !_uiState.value.isDarkTheme)
    }
}
