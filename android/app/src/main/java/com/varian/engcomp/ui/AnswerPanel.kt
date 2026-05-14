package com.varian.engcomp.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.varian.engcomp.ui.theme.DribbbleDark

@Composable
fun AnswerPanel(
    answer: String,
    isLoading: Boolean,
    modifier: Modifier = Modifier
) {
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = DribbbleDark.bgElevated,
        modifier = modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        Column(
            modifier = Modifier
                .padding(12.dp)
                .fillMaxWidth()
        ) {
            if (isLoading) {
                Text(
                    text = "Генерация ответа...",
                    color = DribbbleDark.textMuted,
                    fontSize = 14.sp
                )
            } else if (answer.isEmpty()) {
                Text(
                    text = "Введите вопрос и нажмите Найти",
                    color = DribbbleDark.textMuted,
                    fontSize = 14.sp
                )
            } else {
                Text(
                    text = "Ответ:",
                    color = DribbbleDark.textMuted,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = answer,
                    color = DribbbleDark.textPrimary,
                    fontSize = 14.sp,
                    lineHeight = 20.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    }
}
