package com.varian.engcomp.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.varian.engcomp.ui.theme.appColors

@Composable
fun SettingsScreen(isDark: Boolean, onToggleTheme: () -> Unit) {
    val colors = MaterialTheme.appColors
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {
        Text("Настройки", color = colors.textPrimary, fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Spacer(Modifier.height(24.dp))

        Text("Интерфейс", color = colors.textMuted, fontSize = 11.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .clickable(onClick = onToggleTheme)
                .padding(horizontal = 4.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Тёмная тема", color = colors.textPrimary, fontSize = 14.sp)
                Text("Переключение между тёмным и светлым режимом", color = colors.textMuted, fontSize = 12.sp)
            }
            Switch(
                checked = isDark,
                onCheckedChange = { onToggleTheme() },
                colors = SwitchDefaults.colors(
                    checkedThumbColor = colors.accentInk,
                    checkedTrackColor = colors.accent,
                    uncheckedThumbColor = colors.textMuted,
                    uncheckedTrackColor = colors.border
                )
            )
        }

        Spacer(Modifier.height(24.dp))
        Divider(color = colors.border)
        Spacer(Modifier.height(24.dp))

        Text("Модель", color = colors.textMuted, fontSize = 11.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Text("Путь к модели: TODO", color = colors.textSecondary, fontSize = 13.sp)
        Spacer(Modifier.height(4.dp))
        Text("Языковая модель (llama.cpp) подключается в следующей фазе.", color = colors.textMuted, fontSize = 12.sp)

        Spacer(Modifier.height(24.dp))
        Divider(color = colors.border)
        Spacer(Modifier.height(24.dp))

        Text("О приложении", color = colors.textMuted, fontSize = 11.sp, fontWeight = FontWeight.Medium)
        Spacer(Modifier.height(8.dp))
        Text("Engineer Companion", color = colors.textPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
        Text("Версия 0.1.0 — Phase 1 (FakeRagEngine)", color = colors.textMuted, fontSize = 12.sp)
        Spacer(Modifier.height(4.dp))
        Text("Офлайн RAG-ассистент для технической документации.", color = colors.textSecondary, fontSize = 13.sp)
    }
}
