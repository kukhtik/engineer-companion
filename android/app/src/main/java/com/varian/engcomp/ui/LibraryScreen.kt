package com.varian.engcomp.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.varian.engcomp.ui.theme.appColors

@Composable
fun LibraryScreen() {
    val colors = MaterialTheme.appColors
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {
        Text("Библиотека", color = colors.textPrimary, fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Spacer(Modifier.height(16.dp))
        Text(
            "Индекс встроен в приложение.",
            color = colors.textSecondary,
            fontSize = 14.sp
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "На устройстве хранится векторный индекс документации.\nКоличество документов и чанков будет доступно в следующей версии.",
            color = colors.textMuted,
            fontSize = 13.sp,
            lineHeight = 20.sp
        )
        Spacer(Modifier.height(24.dp))
        Text(
            "TODO: отображение doc/chunk counts из индекса.",
            color = colors.textMuted,
            fontSize = 11.sp
        )
    }
}
