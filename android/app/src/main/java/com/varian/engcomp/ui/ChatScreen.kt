package com.varian.engcomp.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.LightMode
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.StarBorder
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberDrawerState
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.varian.engcomp.data.Conversation
import com.varian.engcomp.data.Turn
import com.varian.engcomp.model.SearchResult
import com.varian.engcomp.ui.theme.appColors
import com.varian.engcomp.viewmodel.ActivityStage
import com.varian.engcomp.viewmodel.ChatViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val colors = MaterialTheme.appColors
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }
    val context = LocalContext.current
    var selectedSource by remember { mutableStateOf<SearchResult?>(null) }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ConversationDrawer(
                conversations = uiState.conversations,
                currentId = uiState.currentConversationId,
                showFavoritesOnly = uiState.showFavoritesOnly,
                onNewChat = {
                    viewModel.newChat()
                    scope.launch { drawerState.close() }
                },
                onOpenConversation = { id ->
                    viewModel.openConversation(id)
                    scope.launch { drawerState.close() }
                },
                onDeleteConversation = { id -> viewModel.deleteConversation(id) },
                onToggleFavorites = { viewModel.toggleFavoritesFilter() },
                colors = colors
            )
        },
        scrimColor = colors.background.copy(alpha = 0.6f)
    ) {
        Scaffold(
            containerColor = colors.background,
            snackbarHost = { SnackbarHost(snackbarHostState) },
            topBar = {
                TopAppBar(
                    title = {
                        Text(
                            "Engineer Companion",
                            color = colors.textPrimary,
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 18.sp
                        )
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Default.Menu, contentDescription = "Меню", tint = colors.textPrimary)
                        }
                    },
                    actions = {
                        IconButton(onClick = { viewModel.toggleTheme() }) {
                            Icon(
                                if (uiState.isDarkTheme) Icons.Default.LightMode else Icons.Default.DarkMode,
                                contentDescription = "Тема",
                                tint = colors.accent
                            )
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = colors.surface,
                        titleContentColor = colors.textPrimary
                    )
                )
            },
            bottomBar = {
                ComposerBar(
                    isGenerating = uiState.isGenerating,
                    onSend = { query -> viewModel.send(query) },
                    colors = colors
                )
            }
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                // Activity strip (shown while generating)
                AnimatedVisibility(
                    visible = uiState.isGenerating || uiState.activityStage != null,
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    ActivityStrip(
                        stage = uiState.activityStage,
                        streamingAnswer = uiState.streamingAnswer,
                        tokenCount = uiState.tokenCount,
                        elapsedMs = uiState.elapsedMs,
                        colors = colors
                    )
                }

                // Thread
                val visibleTurns = if (uiState.showFavoritesOnly)
                    uiState.currentTurns.filter { it.role == "assistant" && it.starred }
                else
                    uiState.currentTurns

                val listState = rememberLazyListState()
                LaunchedEffect(uiState.currentTurns.size, uiState.streamingAnswer) {
                    if (visibleTurns.isNotEmpty()) {
                        listState.animateScrollToItem(visibleTurns.size - 1)
                    }
                }

                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    items(visibleTurns.withIndex().toList(), key = { (idx, _) -> idx }) { (idx, turn) ->
                        if (turn.role == "user") {
                            UserBubble(turn = turn, colors = colors)
                        } else {
                            // Show streaming answer in last assistant bubble while generating
                            val isLast = idx == visibleTurns.size - 1
                            val displayContent = if (isLast && uiState.isGenerating && uiState.streamingAnswer.isNotEmpty())
                                uiState.streamingAnswer
                            else
                                turn.content

                            AssistantBubble(
                                turn = turn,
                                displayContent = displayContent,
                                isStreaming = isLast && uiState.isGenerating,
                                turnIndex = idx,
                                onToggleStar = { viewModel.toggleFavorite(idx) },
                                onCopy = {
                                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                    clipboard.setPrimaryClip(ClipData.newPlainText("answer", displayContent))
                                    scope.launch { snackbarHostState.showSnackbar("Скопировано") }
                                },
                                onSourceClick = { src -> selectedSource = src },
                                colors = colors
                            )
                        }
                    }
                }
            }

            selectedSource?.let { src ->
                SourceDetailSheet(
                    source = src,
                    colors = colors,
                    onDismiss = { selectedSource = null }
                )
            }
        }
    }
}

@Composable
fun UserBubble(turn: Turn, colors: com.varian.engcomp.ui.theme.AppColors) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .clip(RoundedCornerShape(16.dp, 4.dp, 16.dp, 16.dp))
                .background(colors.accent)
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Text(
                text = turn.content,
                color = colors.accentInk,
                fontSize = 14.sp,
                lineHeight = 20.sp
            )
        }
    }
}

@Composable
fun AssistantBubble(
    turn: Turn,
    displayContent: String,
    isStreaming: Boolean,
    turnIndex: Int,
    onToggleStar: () -> Unit,
    onCopy: () -> Unit,
    onSourceClick: (SearchResult) -> Unit,
    colors: com.varian.engcomp.ui.theme.AppColors
) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(4.dp, 16.dp, 16.dp, 16.dp))
                .background(colors.surface)
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            Column {
                if (displayContent.isEmpty() && isStreaming) {
                    PulseDot(colors = colors)
                } else {
                    Text(
                        text = if (isStreaming) "$displayContent▌" else displayContent,
                        color = colors.textPrimary,
                        fontSize = 14.sp,
                        lineHeight = 22.sp
                    )
                }
            }
        }

        // Source cards
        if (turn.sources.isNotEmpty()) {
            Spacer(Modifier.height(4.dp))
            turn.sources.forEach { src ->
                SourceCard(source = src, onClick = { onSourceClick(src) }, colors = colors)
                Spacer(Modifier.height(2.dp))
            }
        }

        // Action row: star + copy
        Row(
            modifier = Modifier.padding(start = 4.dp, top = 2.dp),
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onToggleStar, modifier = Modifier.size(32.dp)) {
                Icon(
                    if (turn.starred) Icons.Default.Star else Icons.Default.StarBorder,
                    contentDescription = if (turn.starred) "Убрать из избранного" else "В избранное",
                    tint = if (turn.starred) colors.accent else colors.textMuted,
                    modifier = Modifier.size(16.dp)
                )
            }
            IconButton(onClick = onCopy, modifier = Modifier.size(32.dp)) {
                Icon(
                    Icons.Default.ContentCopy,
                    contentDescription = "Копировать",
                    tint = colors.textMuted,
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

@Composable
fun SourceCard(source: SearchResult, onClick: () -> Unit, colors: com.varian.engcomp.ui.theme.AppColors) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = colors.elevated),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.Description,
                contentDescription = null,
                tint = colors.textMuted,
                modifier = Modifier.size(14.dp)
            )
            Spacer(Modifier.width(6.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "источник стр.${source.page}",
                    color = colors.textSecondary,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    text = source.section,
                    color = colors.textMuted,
                    fontSize = 10.sp
                )
            }
        }
    }
}

@Composable
fun PulseDot(colors: com.varian.engcomp.ui.theme.AppColors) {
    val transition = rememberInfiniteTransition(label = "pulse")
    val alpha by transition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(500), RepeatMode.Reverse),
        label = "alpha"
    )
    Box(
        modifier = Modifier
            .size(8.dp)
            .clip(RoundedCornerShape(50))
            .background(colors.accent.copy(alpha = alpha))
    )
}

@Composable
fun ActivityStrip(
    stage: ActivityStage?,
    streamingAnswer: String,
    tokenCount: Int,
    elapsedMs: Long,
    colors: com.varian.engcomp.ui.theme.AppColors
) {
    val stages = listOf("Векторизация", "Поиск", "Контекст", "Генерация")
    val currentStageIndex = when (stage?.label) {
        "Векторизация" -> 0
        "Поиск" -> 1
        "Контекст" -> 2
        "Генерация" -> 3
        else -> -1
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(colors.elevated)
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    stages.forEachIndexed { idx, label ->
                        val isActive = idx == currentStageIndex
                        val isPast = idx < currentStageIndex
                        val detail = if (isActive) stage?.detail ?: "" else ""
                        val displayLabel = if (detail.isNotEmpty()) "$label · $detail" else label
                        Text(
                            text = displayLabel,
                            color = when {
                                isActive -> colors.accent
                                isPast -> colors.textSecondary
                                else -> colors.textMuted
                            },
                            fontSize = 11.sp,
                            fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal
                        )
                        if (idx < stages.size - 1) {
                            Text("→", color = colors.border, fontSize = 11.sp)
                        }
                    }
                }
                if (tokenCount > 0) {
                    Text(
                        "${tokenCount}т · ${elapsedMs / 1000.0}с",
                        color = colors.textMuted,
                        fontSize = 10.sp
                    )
                }
            }
        }
    }
}

@Composable
fun ComposerBar(
    isGenerating: Boolean,
    onSend: (String) -> Unit,
    colors: com.varian.engcomp.ui.theme.AppColors
) {
    var text by remember { mutableStateOf("") }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(colors.surface)
            .padding(horizontal = 12.dp, vertical = 8.dp)
            .imePadding(),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            placeholder = {
                Text(
                    "Уточните или задайте следующий вопрос…",
                    color = colors.textMuted,
                    fontSize = 13.sp
                )
            },
            modifier = Modifier.weight(1f),
            enabled = !isGenerating,
            maxLines = 4,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = {
                if (text.isNotBlank() && !isGenerating) {
                    onSend(text)
                    text = ""
                }
            }),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = colors.accent,
                unfocusedBorderColor = colors.border,
                focusedTextColor = colors.textPrimary,
                unfocusedTextColor = colors.textPrimary,
                disabledTextColor = colors.textMuted,
                cursorColor = colors.accent,
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent
            ),
            shape = RoundedCornerShape(12.dp)
        )
        IconButton(
            onClick = {
                if (text.isNotBlank() && !isGenerating) {
                    onSend(text)
                    text = ""
                }
            },
            enabled = !isGenerating && text.isNotBlank(),
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(if (!isGenerating && text.isNotBlank()) colors.accent else colors.elevated)
        ) {
            Icon(
                Icons.Default.Send,
                contentDescription = "Отправить",
                tint = if (!isGenerating && text.isNotBlank()) colors.accentInk else colors.textMuted,
                modifier = Modifier.size(20.dp)
            )
        }
    }
}

@Composable
fun ConversationDrawer(
    conversations: List<Conversation>,
    currentId: String,
    showFavoritesOnly: Boolean,
    onNewChat: () -> Unit,
    onOpenConversation: (String) -> Unit,
    onDeleteConversation: (String) -> Unit,
    onToggleFavorites: () -> Unit,
    colors: com.varian.engcomp.ui.theme.AppColors
) {
    var deleteTarget by remember { mutableStateOf<String?>(null) }

    if (deleteTarget != null) {
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Удалить диалог?", color = colors.textPrimary) },
            text = { Text("Это действие нельзя отменить.", color = colors.textSecondary) },
            confirmButton = {
                TextButton(onClick = {
                    deleteTarget?.let { onDeleteConversation(it) }
                    deleteTarget = null
                }) {
                    Text("Удалить", color = colors.danger)
                }
            },
            dismissButton = {
                TextButton(onClick = { deleteTarget = null }) {
                    Text("Отмена", color = colors.textSecondary)
                }
            },
            containerColor = colors.elevated
        )
    }

    ModalDrawerSheet(
        drawerContainerColor = colors.surface,
        modifier = Modifier.width(280.dp)
    ) {
        // Header
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.elevated)
                .padding(16.dp)
        ) {
            Text(
                "Engineer Companion",
                color = colors.accent,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
        }

        Spacer(Modifier.height(8.dp))

        // New chat button
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(colors.accent)
                .clickable(onClick = onNewChat)
                .padding(horizontal = 16.dp, vertical = 10.dp),
            contentAlignment = Alignment.Center
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(Icons.Default.Add, contentDescription = null, tint = colors.accentInk, modifier = Modifier.size(16.dp))
                Text("Новый чат", color = colors.accentInk, fontWeight = FontWeight.Medium)
            }
        }

        Spacer(Modifier.height(12.dp))

        // Favorites toggle
        NavigationDrawerItem(
            icon = {
                Icon(
                    if (showFavoritesOnly) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                    contentDescription = null,
                    tint = if (showFavoritesOnly) colors.accent else colors.textMuted,
                    modifier = Modifier.size(18.dp)
                )
            },
            label = { Text("Избранное", color = if (showFavoritesOnly) colors.accent else colors.textSecondary) },
            selected = showFavoritesOnly,
            onClick = onToggleFavorites,
            colors = NavigationDrawerItemDefaults.colors(
                selectedContainerColor = colors.accent.copy(alpha = 0.12f),
                unselectedContainerColor = Color.Transparent
            ),
            modifier = Modifier.padding(horizontal = 8.dp)
        )

        Divider(color = colors.border, modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp))

        // Conversations list header
        Text(
            "Диалоги",
            color = colors.textMuted,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        val listState = rememberLazyListState()
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            items(conversations, key = { it.id }) { conv ->
                ConversationItem(
                    conversation = conv,
                    isSelected = conv.id == currentId,
                    onClick = { onOpenConversation(conv.id) },
                    onLongClick = { deleteTarget = conv.id },
                    colors = colors
                )
            }
        }

        Divider(color = colors.border, modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp))

        // Bottom nav items
        NavigationDrawerItem(
            icon = { Icon(Icons.Default.Description, contentDescription = null, tint = colors.textMuted, modifier = Modifier.size(18.dp)) },
            label = { Text("Библиотека", color = colors.textSecondary) },
            selected = false,
            onClick = { /* TODO: navigate to library */ },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent),
            modifier = Modifier.padding(horizontal = 8.dp)
        )
        NavigationDrawerItem(
            icon = { Icon(Icons.Default.Settings, contentDescription = null, tint = colors.textMuted, modifier = Modifier.size(18.dp)) },
            label = { Text("Настройки", color = colors.textSecondary) },
            selected = false,
            onClick = { /* TODO: navigate to settings */ },
            colors = NavigationDrawerItemDefaults.colors(unselectedContainerColor = Color.Transparent),
            modifier = Modifier.padding(horizontal = 8.dp)
        )

        Spacer(Modifier.height(8.dp))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SourceDetailSheet(
    source: SearchResult,
    colors: com.varian.engcomp.ui.theme.AppColors,
    onDismiss: () -> Unit
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val scrollState = rememberScrollState()

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = colors.surface,
        contentColor = colors.textPrimary,
        dragHandle = null,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 16.dp)
        ) {
            // Title bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = source.source,
                    color = colors.textPrimary,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 16.sp,
                    modifier = Modifier.weight(1f)
                )
                IconButton(onClick = onDismiss) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Закрыть",
                        tint = colors.textMuted
                    )
                }
            }

            Spacer(Modifier.height(8.dp))
            Divider(color = colors.border)
            Spacer(Modifier.height(12.dp))

            // Metadata
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Column {
                    Text("Документ", color = colors.textMuted, fontSize = 10.sp)
                    Text(source.source, color = colors.textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                }
                Column {
                    Text("стр.", color = colors.textMuted, fontSize = 10.sp)
                    Text("${source.page}", color = colors.textSecondary, fontSize = 12.sp, fontWeight = FontWeight.Medium)
                }
            }

            if (source.section.isNotBlank()) {
                Spacer(Modifier.height(4.dp))
                Text("раздел: ${source.section}", color = colors.textMuted, fontSize = 11.sp)
            }

            Spacer(Modifier.height(16.dp))

            // Chunk text - scrollable and selectable
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f, fill = false)
                    .heightIn(min = 80.dp, max = 400.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(colors.elevated)
                    .padding(12.dp)
                    .verticalScroll(scrollState)
            ) {
                SelectionContainer {
                    Text(
                        text = source.text.ifBlank { "Текст фрагмента недоступен." },
                        color = colors.textPrimary,
                        fontSize = 13.sp,
                        lineHeight = 20.sp
                    )
                }
            }

            Spacer(Modifier.height(12.dp))

            // Footer note
            Text(
                text = "Полный просмотр страницы PDF доступен в десктоп-версии.",
                color = colors.textMuted,
                fontSize = 10.sp,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(Modifier.height(8.dp))
        }
    }
}

@OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun ConversationItem(
    conversation: Conversation,
    isSelected: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    colors: com.varian.engcomp.ui.theme.AppColors
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (isSelected) colors.accent.copy(alpha = 0.15f) else Color.Transparent)
            .combinedClickable(onClick = onClick, onLongClick = onLongClick)
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Text(
            text = conversation.title,
            color = if (isSelected) colors.accent else colors.textSecondary,
            fontSize = 13.sp,
            maxLines = 2
        )
    }
}
