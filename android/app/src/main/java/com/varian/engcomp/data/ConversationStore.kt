package com.varian.engcomp.data

import android.content.Context
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import java.io.File
import java.util.UUID

class ConversationStore(context: Context) {
    private val file = File(context.filesDir, "conversations.json")
    private val gson: Gson = GsonBuilder().setPrettyPrinting().create()

    private val _conversations: MutableList<Conversation> = mutableListOf()
    val conversations: List<Conversation> get() = _conversations.toList()

    init {
        load()
        pruneEmpty()
    }

    private fun load() {
        if (!file.exists()) return
        try {
            val type = object : TypeToken<MutableList<Conversation>>() {}.type
            val loaded: MutableList<Conversation> = gson.fromJson(file.readText(), type) ?: mutableListOf()
            _conversations.clear()
            _conversations.addAll(loaded)
        } catch (_: Exception) { /* corrupt file — start fresh */ }
    }

    private fun pruneEmpty() {
        _conversations.removeAll { it.turns.isEmpty() }
    }

    private fun persist() {
        val tmp = File(file.parent, "conversations.json.tmp")
        tmp.writeText(gson.toJson(_conversations))
        tmp.renameTo(file)
    }

    /** Returns existing empty conversation or creates a new one. */
    fun getOrCreateEmpty(): Conversation {
        val existing = _conversations.firstOrNull { it.turns.isEmpty() }
        if (existing != null) return existing
        val conv = Conversation(
            id = UUID.randomUUID().toString(),
            title = "Новый диалог",
            createdMillis = System.currentTimeMillis()
        )
        _conversations.add(0, conv)
        persist()
        return conv
    }

    fun addTurn(conversationId: String, turn: Turn) {
        val conv = _conversations.find { it.id == conversationId } ?: return
        conv.turns.add(turn)
        if (conv.turns.size == 1 && turn.role == "user") {
            conv.title = turn.content.take(60).ifBlank { "Диалог" }
        }
        persist()
    }

    fun updateLastAssistantTurn(conversationId: String, content: String, sources: List<com.varian.engcomp.model.SearchResult>) {
        val conv = _conversations.find { it.id == conversationId } ?: return
        val lastAssIdx = conv.turns.indexOfLast { it.role == "assistant" }
        if (lastAssIdx >= 0) {
            conv.turns[lastAssIdx] = conv.turns[lastAssIdx].copy(content = content, sources = sources)
        }
        persist()
    }

    fun toggleStarTurn(conversationId: String, turnIndex: Int) {
        val conv = _conversations.find { it.id == conversationId } ?: return
        if (turnIndex < 0 || turnIndex >= conv.turns.size) return
        val t = conv.turns[turnIndex]
        conv.turns[turnIndex] = t.copy(starred = !t.starred)
        persist()
    }

    fun delete(id: String) {
        _conversations.removeAll { it.id == id }
        persist()
    }

    fun getConversation(id: String): Conversation? = _conversations.find { it.id == id }
}
