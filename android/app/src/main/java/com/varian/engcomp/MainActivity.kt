package com.varian.engcomp

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.varian.engcomp.ui.AnswerPanel
import com.varian.engcomp.ui.SourcesList
import com.varian.engcomp.ui.theme.DribbbleDark
import com.varian.engcomp.viewmodel.SearchViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            EngineerCompanionTheme {
                SearchScreen()
            }
        }
    }
}

@Composable
fun EngineerCompanionTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = DribbbleDark.bgBase,
            surface = DribbbleDark.bgSurface,
            primary = DribbbleDark.accentCyan,
            secondary = DribbbleDark.accentGreen,
            onBackground = DribbbleDark.textPrimary,
            onSurface = DribbbleDark.textPrimary,
            error = DribbbleDark.accentRed,
        ),
        content = content
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(viewModel: SearchViewModel = viewModel()) {
    val query by viewModel.query.collectAsState()
    val sources by viewModel.sources.collectAsState()
    val answer by viewModel.answer.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()
    val error by viewModel.error.collectAsState()
    val dbInfo by viewModel.dbInfo.collectAsState()

    // Show errors as toast
    LaunchedEffect(error) {
        error?.let {
            Toast.makeText(this@MainActivity, it, Toast.LENGTH_LONG).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DribbbleDark.bgBase)
            .padding(12.dp)
    ) {
        // DB info bar
        Text(
            text = dbInfo,
            color = DribbbleDark.textMuted,
            fontSize = 11.sp,
            modifier = Modifier.padding(bottom = 4.dp)
        )

        // Search bar
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = { viewModel.onQueryChanged(it) },
                placeholder = {
                    Text(
                        "Вопрос по TrueBeam / VitalBeam...",
                        color = DribbbleDark.textMuted
                    )
                },
                modifier = Modifier.weight(1f),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DribbbleDark.accentCyan,
                    unfocusedBorderColor = DribbbleDark.borderColor,
                    focusedTextColor = DribbbleDark.textPrimary,
                    unfocusedTextColor = DribbbleDark.textPrimary,
                    cursorColor = DribbbleDark.accentCyan,
                ),
                shape = RoundedCornerShape(8.dp),
            )

            Button(
                onClick = { viewModel.search() },
                enabled = !isSearching && query.isNotBlank(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DribbbleDark.accentCyan,
                    contentColor = DribbbleDark.textPrimary
                ),
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(if (isSearching) "..." else "Найти")
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Answer area
        AnswerPanel(
            answer = answer,
            isLoading = isSearching
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Sources
        SourcesList(sources = sources)

        // Loading indicator
        if (isSearching) {
            LinearProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp),
                color = DribbbleDark.accentCyan,
                trackColor = DribbbleDark.borderColor,
            )
        }
    }
}
