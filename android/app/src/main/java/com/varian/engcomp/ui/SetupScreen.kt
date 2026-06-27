package com.varian.engcomp.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Setup screen shown when the GGUF model is absent.
 *
 * Pure UI — all logic (copy progress, model loading) lives in ChatViewModel.
 *
 * @param copyProgress  -1f = not started / done, 0f..1f = copying in progress
 * @param installDone   true after the model has been successfully installed
 * @param onFileSelected called with the SAF URI when the user picks a file
 * @param onDone        called when the user taps "Готово — перейти в чат"
 */
@Composable
fun SetupScreen(
    copyProgress: Float,
    installDone: Boolean,
    onFileSelected: (Uri) -> Unit,
    onDone: () -> Unit,
) {
    val isCopying = copyProgress in 0f..0.9999f

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) onFileSelected(uri)
    }

    Scaffold { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "Установка модели",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )

            Text(
                text = "Для работы генерации ответов необходима языковая модель Gemma 3 4B (2.4 ГБ). " +
                       "Файл не входит в APK и должен быть установлен отдельно.",
                style = MaterialTheme.typography.bodyMedium,
            )

            // ADB instructions card
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Вариант 1 — через ADB:", fontWeight = FontWeight.SemiBold)
                    Text(
                        text = "adb push gemma-3-4b-it-Q4_K_M.gguf /sdcard/Download/",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Text("Затем нажмите «Импортировать файл» и выберите файл из папки Загрузки.")
                }
            }

            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text("Вариант 2 — прямой импорт:", fontWeight = FontWeight.SemiBold)
                    Text("Скопируйте файл .gguf на устройство (через USB, облако и т.д.) и выберите его кнопкой ниже.")
                }
            }

            Spacer(Modifier.height(8.dp))

            Button(
                onClick = { picker.launch(arrayOf("*/*")) },
                enabled = !isCopying && !installDone,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Импортировать файл")
            }

            if (isCopying) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    LinearProgressIndicator(
                        progress = { copyProgress },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        text = "Копирование… ${(copyProgress * 100).toInt()}%",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.align(Alignment.CenterHorizontally),
                    )
                }
            }

            if (installDone) {
                Text(
                    text = "Модель установлена успешно!",
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.SemiBold,
                )
            }

            Button(
                onClick = onDone,
                enabled = installDone,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Готово — перейти в чат")
            }
        }
    }
}
