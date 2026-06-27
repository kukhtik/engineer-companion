package com.varian.engcomp.viewmodel

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.varian.engcomp.data.Conversation
import com.varian.engcomp.data.ConversationStore
import com.varian.engcomp.data.SettingsStore
import com.varian.engcomp.data.Turn
import com.varian.engcomp.engine.AssetCopier
import com.varian.engcomp.engine.FakeRagEngine
import com.varian.engcomp.engine.FullRagEngine
import com.varian.engcomp.engine.LlamaEngine
import com.varian.engcomp.engine.OnnxRagEngine
import com.varian.engcomp.engine.OnnxRetriever
import com.varian.engcomp.engine.RagEngine
import com.varian.engcomp.engine.RagEvent
import com.varian.engcomp.model.SearchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

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
    val sourcesForCurrentGen: List<SearchResult> = emptyList(),
    /** True while AssetCopier + OnnxRetriever are initializing on first run. */
    val engineLoading: Boolean = true,
    /** Non-null if the engine failed to initialize; UI can show FakeRagEngine fallback. */
    val engineError: String? = null,
    /** True when the GGUF model is installed and ready for generation. */
    val modelInstalled: Boolean = false,
    /** -1f = idle / not copying, 0f..1f = copy progress. */
    val modelCopyProgress: Float = -1f,
    /** True when the model is absent and the user needs to install it. */
    val showSetupScreen: Boolean = false,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "ChatViewModel"
        private const val GGUF_FILENAME = "gemma-3-4b-it-Q4_K_M.gguf"
        private const val COPY_CHUNK = 256 * 1024  // 256 KB
    }

    private val store = ConversationStore(application)
    private val settings = SettingsStore(application)

    // Start with FakeRagEngine so the UI is immediately usable while models load.
    private var engine: RagEngine = FakeRagEngine()
    private var retriever: OnnxRetriever? = null  // kept for close()
    private var llamaEngine: LlamaEngine? = null

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        val conv = store.getOrCreateEmpty()
        _uiState.value = ChatUiState(
            conversations = store.conversations,
            currentConversationId = conv.id,
            currentTurns = conv.turns.toList(),
            isDarkTheme = true,
            engineLoading = true,
        )
        // Async model init — runs once on first launch (asset copy ~23 MB + 448 MB).
        // Subsequent launches skip copy (files already present) and only parse the
        // ~23 MB index, which takes ~1–2 s on device.
        viewModelScope.launch(Dispatchers.IO) {
            initEngine()
        }
        viewModelScope.launch(Dispatchers.IO) {
            settings.isDarkTheme.collect { dark ->
                _uiState.value = _uiState.value.copy(isDarkTheme = dark)
            }
        }
    }

    private suspend fun initEngine() {
        try {
            Log.i(TAG, "initEngine: copying assets ...")
            updateStage("Загрузка моделей", "копирование файлов …")

            val paths = withContext(Dispatchers.IO) {
                AssetCopier(getApplication()).copy { pct ->
                    updateStage("Загрузка моделей", "$pct%")
                }
            }
            Log.i(TAG, "initEngine: assets ready, loading index ...")
            updateStage("Загрузка моделей", "индекс …")

            val r = withContext(Dispatchers.Default) {
                OnnxRetriever(
                    tokenizerOnnxPath = paths.tokenizerOnnx,
                    modelOnnxPath     = paths.modelOnnx,
                    chunksIndexPath   = paths.chunksIndex,
                )
            }
            retriever = r

            // Check if GGUF model is already present.
            // Resolve order: internal filesDir/models first (installed via SAF import),
            // then external files dir which adb can write to without root:
            //   /sdcard/Android/data/com.varian.engcomp/files/models/
            val app = getApplication<Application>()
            val internalGguf = File(app.filesDir, "models/$GGUF_FILENAME")
            val externalGguf = File(app.getExternalFilesDir("models"), GGUF_FILENAME)
            val ggufFile: File = when {
                internalGguf.exists() && internalGguf.length() > 0 -> internalGguf
                externalGguf.exists() && externalGguf.length() > 0 -> externalGguf
                else -> internalGguf  // canonical missing-file path (for error messages)
            }
            Log.i(TAG, "initEngine: GGUF probe — internal=${internalGguf.exists()} external=${externalGguf.exists()} → using ${ggufFile.absolutePath}")
            if (ggufFile.exists() && ggufFile.length() > 0) {
                Log.i(TAG, "initEngine: GGUF found at ${ggufFile.absolutePath}, loading ...")
                updateStage("Загрузка LLM", "инициализация …")
                val llama = LlamaEngine()
                val ok = llama.loadModel(ggufFile.absolutePath)
                if (ok) {
                    llamaEngine = llama
                    engine = FullRagEngine(r, llama)
                    Log.i(TAG, "initEngine: FullRagEngine ready")
                    _uiState.value = _uiState.value.copy(
                        engineLoading  = false,
                        modelInstalled = true,
                        showSetupScreen = false,
                        activityStage  = null,
                    )
                } else {
                    Log.w(TAG, "initEngine: GGUF found but load failed, falling back to OnnxRagEngine")
                    engine = OnnxRagEngine(r)
                    _uiState.value = _uiState.value.copy(
                        engineLoading  = false,
                        modelInstalled = false,
                        showSetupScreen = true,
                        activityStage  = null,
                    )
                }
            } else {
                Log.i(TAG, "initEngine: GGUF not found, using OnnxRagEngine (stub generation)")
                engine = OnnxRagEngine(r)
                _uiState.value = _uiState.value.copy(
                    engineLoading  = false,
                    modelInstalled = false,
                    showSetupScreen = true,
                    activityStage  = null,
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "initEngine failed: $e", e)
            // Fall back to FakeRagEngine; engine field already points to it
            _uiState.value = _uiState.value.copy(
                engineLoading = false,
                engineError = e.message ?: "Engine init failed",
                activityStage = null,
            )
        }
    }

    /**
     * Import the GGUF model from the given SAF [uri].
     * Copies the file to filesDir/models/gemma-3-4b-it-Q4_K_M.gguf in 256 KB chunks,
     * reporting progress via [ChatUiState.modelCopyProgress].
     * On success, loads the model into LlamaEngine and switches to FullRagEngine.
     */
    fun importModel(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val modelsDir = File(app.filesDir, "models")
            modelsDir.mkdirs()
            val destFile = File(modelsDir, GGUF_FILENAME)

            try {
                _uiState.value = _uiState.value.copy(modelCopyProgress = 0f)

                val cr = app.contentResolver
                val size = cr.openFileDescriptor(uri, "r")?.use { pfd ->
                    pfd.statSize
                } ?: -1L

                cr.openInputStream(uri)?.use { input ->
                    FileOutputStream(destFile).use { output ->
                        val buf = ByteArray(COPY_CHUNK)
                        var totalRead = 0L
                        var n: Int
                        while (input.read(buf).also { n = it } != -1) {
                            output.write(buf, 0, n)
                            totalRead += n
                            val progress = if (size > 0) totalRead.toFloat() / size.toFloat() else 0f
                            _uiState.value = _uiState.value.copy(
                                modelCopyProgress = progress.coerceIn(0f, 0.99f)
                            )
                        }
                    }
                } ?: run {
                    Log.e(TAG, "importModel: could not open input stream for $uri")
                    _uiState.value = _uiState.value.copy(
                        modelCopyProgress = -1f,
                        engineError = "Не удалось открыть файл",
                    )
                    return@launch
                }

                Log.i(TAG, "importModel: copy complete, loading model ...")
                _uiState.value = _uiState.value.copy(modelCopyProgress = 1f)

                // Load into LlamaEngine
                val llama = LlamaEngine()
                val ok = llama.loadModel(destFile.absolutePath)
                if (ok) {
                    val r = retriever
                    if (r != null) {
                        // Free old llama if any
                        llamaEngine?.freeSync()
                        llamaEngine = llama
                        engine = FullRagEngine(r, llama)
                        Log.i(TAG, "importModel: FullRagEngine switched in")
                    } else {
                        Log.w(TAG, "importModel: retriever not ready yet; model loaded but engine not switched")
                        llamaEngine = llama
                    }
                    _uiState.value = _uiState.value.copy(
                        modelInstalled    = true,
                        modelCopyProgress = -1f,
                        showSetupScreen   = false,
                    )
                } else {
                    Log.e(TAG, "importModel: model load failed after copy")
                    destFile.delete()
                    _uiState.value = _uiState.value.copy(
                        modelCopyProgress = -1f,
                        engineError = "Файл скопирован, но не удалось загрузить модель. Проверьте файл.",
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importModel: exception", e)
                destFile.delete()
                _uiState.value = _uiState.value.copy(
                    modelCopyProgress = -1f,
                    engineError = "Ошибка импорта: ${e.message}",
                )
            }
        }
    }

    /** Dismiss the setup screen and go to chat (model must already be installed). */
    fun onSetupDone() {
        _uiState.value = _uiState.value.copy(showSetupScreen = false)
    }

    override fun onCleared() {
        super.onCleared()
        try { retriever?.close() } catch (_: Exception) {}
        llamaEngine?.freeSync()
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
        val newDark = !_uiState.value.isDarkTheme
        _uiState.value = _uiState.value.copy(isDarkTheme = newDark)
        viewModelScope.launch(Dispatchers.IO) {
            settings.setDarkTheme(newDark)
        }
    }
}
