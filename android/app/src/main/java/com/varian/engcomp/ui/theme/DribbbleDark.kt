package com.varian.engcomp.ui.theme

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Raw color tokens — dual contrast palette matching the desktop (windows/main_window.py)
// ─────────────────────────────────────────────────────────────────────────────────────────────────

object DarkTokens {
    val Background  = Color(0xFF0A0A0A)
    val Surface     = Color(0xFF151515)
    val Elevated    = Color(0xFF1D1D1D)
    val Border      = Color(0xFF2A2A2A)
    val TextPrimary   = Color(0xFFF5F5F2)
    val TextSecondary = Color(0xFFB6B6AE)
    val TextMuted     = Color(0xFF8A8A82)
    val Accent      = Color(0xFFF5C518)   // yellow
    val AccentInk   = Color(0xFF0A0A0A)   // text on accent buttons
    val Danger      = Color(0xFFE5382B)
}

object LightTokens {
    val Background  = Color(0xFFFFFFFF)
    val Surface     = Color(0xFFF4F6F9)
    val Elevated    = Color(0xFFFFFFFF)
    val Border      = Color(0xFFD8DEE6)
    val TextPrimary   = Color(0xFF173A5E)
    val TextSecondary = Color(0xFF3A6FB0)
    val TextMuted     = Color(0xFF6B7B90)
    val Accent      = Color(0xFF1E88E5)   // blue
    val AccentInk   = Color(0xFFFFFFFF)   // text on accent buttons
    val Danger      = Color(0xFFD33A2C)
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// AppColors — full token set exposed via CompositionLocal for custom widgets (bubbles, cards, etc.)
// ─────────────────────────────────────────────────────────────────────────────────────────────────

@Immutable
data class AppColors(
    val background: Color,
    val surface: Color,
    val elevated: Color,
    val border: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textMuted: Color,
    val accent: Color,
    val accentInk: Color,
    val danger: Color,
    val isDark: Boolean,
)

val DarkAppColors = AppColors(
    background    = DarkTokens.Background,
    surface       = DarkTokens.Surface,
    elevated      = DarkTokens.Elevated,
    border        = DarkTokens.Border,
    textPrimary   = DarkTokens.TextPrimary,
    textSecondary = DarkTokens.TextSecondary,
    textMuted     = DarkTokens.TextMuted,
    accent        = DarkTokens.Accent,
    accentInk     = DarkTokens.AccentInk,
    danger        = DarkTokens.Danger,
    isDark        = true,
)

val LightAppColors = AppColors(
    background    = LightTokens.Background,
    surface       = LightTokens.Surface,
    elevated      = LightTokens.Elevated,
    border        = LightTokens.Border,
    textPrimary   = LightTokens.TextPrimary,
    textSecondary = LightTokens.TextSecondary,
    textMuted     = LightTokens.TextMuted,
    accent        = LightTokens.Accent,
    accentInk     = LightTokens.AccentInk,
    danger        = LightTokens.Danger,
    isDark        = false,
)

val LocalAppColors = staticCompositionLocalOf { DarkAppColors }

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Material3 ColorScheme mappings
// ─────────────────────────────────────────────────────────────────────────────────────────────────

private fun darkM3Scheme(): ColorScheme = darkColorScheme(
    primary            = DarkTokens.Accent,
    onPrimary          = DarkTokens.AccentInk,
    primaryContainer   = DarkTokens.Elevated,
    onPrimaryContainer = DarkTokens.TextPrimary,
    secondary          = DarkTokens.TextSecondary,
    onSecondary        = DarkTokens.Background,
    background         = DarkTokens.Background,
    onBackground       = DarkTokens.TextPrimary,
    surface            = DarkTokens.Surface,
    onSurface          = DarkTokens.TextPrimary,
    surfaceVariant     = DarkTokens.Elevated,
    onSurfaceVariant   = DarkTokens.TextSecondary,
    outline            = DarkTokens.Border,
    outlineVariant     = DarkTokens.Border,
    error              = DarkTokens.Danger,
    onError            = DarkTokens.TextPrimary,
)

private fun lightM3Scheme(): ColorScheme = lightColorScheme(
    primary            = LightTokens.Accent,
    onPrimary          = LightTokens.AccentInk,
    primaryContainer   = LightTokens.Surface,
    onPrimaryContainer = LightTokens.TextPrimary,
    secondary          = LightTokens.TextSecondary,
    onSecondary        = LightTokens.Background,
    background         = LightTokens.Background,
    onBackground       = LightTokens.TextPrimary,
    surface            = LightTokens.Surface,
    onSurface          = LightTokens.TextPrimary,
    surfaceVariant     = LightTokens.Elevated,
    onSurfaceVariant   = LightTokens.TextSecondary,
    outline            = LightTokens.Border,
    outlineVariant     = LightTokens.Border,
    error              = LightTokens.Danger,
    onError            = LightTokens.AccentInk,
)

// Convenience accessors kept for any existing call-sites that reference the old DribbbleDark object
@Deprecated("Use LocalAppColors.current instead", ReplaceWith("LocalAppColors.current"))
object DribbbleDark {
    val bgBase       get() = DarkTokens.Background
    val bgSurface    get() = DarkTokens.Surface
    val bgElevated   get() = DarkTokens.Elevated
    val textPrimary  get() = DarkTokens.TextPrimary
    val textSecondary get() = DarkTokens.TextSecondary
    val textMuted    get() = DarkTokens.TextMuted
    val accentCyan   get() = DarkTokens.Accent      // mapped to yellow accent; rename in follow-up
    val accentGreen  get() = DarkTokens.Accent
    val accentRed    get() = DarkTokens.Danger
    val accentOrange get() = DarkTokens.Accent
    val borderColor  get() = DarkTokens.Border
    val starYellow   get() = DarkTokens.Accent
}
