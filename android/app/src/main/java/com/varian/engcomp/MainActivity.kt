package com.varian.engcomp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.varian.engcomp.ui.ChatScreen
import com.varian.engcomp.ui.theme.EngineerCompanionTheme
import com.varian.engcomp.viewmodel.ChatViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val chatViewModel: ChatViewModel = viewModel()
            val uiState by chatViewModel.uiState.collectAsState()
            EngineerCompanionTheme(darkTheme = uiState.isDarkTheme) {
                ChatScreen(viewModel = chatViewModel)
            }
        }
    }
}
