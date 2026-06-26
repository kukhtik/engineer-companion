package com.varian.engcomp.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Theme toggle state
// TODO(persist): replace remember with a ThemeViewModel backed by DataStore so the preference
//   survives process death. For Phase 1 a remembered in-memory toggle is sufficient.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

class ThemeState(initial: Boolean = true) {
    var isDark by mutableStateOf(initial)
    fun toggle() { isDark = !isDark }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// EngineerCompanionTheme — sets both Material3 ColorScheme and LocalAppColors
// ─────────────────────────────────────────────────────────────────────────────────────────────────

@Composable
fun EngineerCompanionTheme(
    darkTheme: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = if (darkTheme) darkM3Scheme() else lightM3Scheme()
    val appColors   = if (darkTheme) DarkAppColors  else LightAppColors

    CompositionLocalProvider(LocalAppColors provides appColors) {
        MaterialTheme(
            colorScheme = colorScheme,
            content = content,
        )
    }
}

// Convenience extension so any composable can reach the full token set without an import chain
val MaterialTheme.appColors: AppColors
    @Composable get() = LocalAppColors.current
