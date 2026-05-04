"""Design tokens: impeccable-ui dribbble-dark palette, Qt + CSS variants.

Usage:
    from design.tokens import DribbbleDarkQt, DribbbleDarkCSS
    palette = DribbbleDarkQt()
    widget.setStyleSheet(palette.as_stylesheet())
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DribbbleDarkQt:
    """OKLCH-based tokens mapped to hex for Qt Stylesheets."""
    bg_base: str = "#0a0a0a"
    bg_surface: str = "#151515"
    bg_elevated: str = "#202020"
    bg_row_hover: str = "#1a1a1a"
    bg_row_selected: str = "#262626"
    accent_red: str = "#BC2916"
    accent_green: str = "#7BB16F"
    accent_cyan: str = "#3EB8B5"
    accent_magenta: str = "#D64EC8"
    text_primary: str = "#F4F3F4"
    text_secondary: str = "#B2BAB6"
    text_muted: str = "#7A7A7A"
    border: str = "#2A2A2A"
    font_mono: str = '"SF Mono", "Consolas", "Fira Code", monospace'
    font_sans: str = '"Inter", "Segoe UI", system-ui, sans-serif'

    def as_stylesheet(self) -> str:
        return f"""
QMainWindow {{
    background-color: {self.bg_base};
    color: {self.text_primary};
    font-family: {self.font_sans};
}}
QWidget {{
    background-color: {self.bg_base};
    color: {self.text_primary};
}}
QLineEdit {{
    background-color: {self.bg_surface};
    color: {self.text_primary};
    border: 1px solid {self.border};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}}
QLineEdit:focus {{
    border: 2px solid {self.accent_cyan};
}}
QPushButton {{
    background-color: {self.bg_elevated};
    color: {self.text_primary};
    border: 1px solid {self.border};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {self.bg_row_hover};
}}
QPushButton:pressed {{
    background-color: {self.bg_row_selected};
}}
QListView, QListWidget {{
    background-color: {self.bg_surface};
    color: {self.text_primary};
    border: 1px solid {self.border};
    border-radius: 6px;
    outline: none;
    padding: 4px;
}}
QListView::item {{
    padding: 8px 12px;
    border-radius: 4px;
}}
QListView::item:hover {{
    background-color: {self.bg_row_hover};
}}
QListView::item:selected {{
    background-color: {self.bg_row_selected};
    border: 1px solid {self.accent_cyan};
}}
QLabel {{
    color: {self.text_primary};
}}
QLabel#muted {{
    color: {self.text_muted};
}}
QScrollBar:vertical {{
    background: {self.bg_base};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {self.border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {self.text_muted};
}}
QSplitter::handle {{
    background: {self.border};
}}
QTextEdit, QPlainTextEdit {{
    background-color: {self.bg_surface};
    color: {self.text_primary};
    border: 1px solid {self.border};
    border-radius: 6px;
    padding: 8px;
    font-family: {self.font_mono};
    font-size: 13px;
}}
"""


@dataclass(frozen=True)
class DribbbleDarkCSS:
    """CSS custom properties for web/Kivy/Android WebView."""
    bg_base: str = "oklch(6% 0.01 30)"
    bg_surface: str = "oklch(12% 0.01 30)"
    bg_elevated: str = "oklch(18% 0.01 30)"
    accent_red: str = "oklch(55% 0.22 25)"
    accent_green: str = "oklch(72% 0.14 145)"
    accent_cyan: str = "oklch(75% 0.15 210)"
    accent_magenta: str = "oklch(65% 0.18 330)"
    text_primary: str = "oklch(92% 0.005 30)"
    text_secondary: str = "oklch(72% 0.01 30)"
    text_muted: str = "oklch(55% 0.01 30)"
    border: str = "oklch(22% 0.01 30)"

    def as_css_block(self) -> str:
        return f""":root {{
  --bg-base: {self.bg_base};
  --bg-surface: {self.bg_surface};
  --bg-elevated: {self.bg_elevated};
  --accent-red: {self.accent_red};
  --accent-green: {self.accent_green};
  --accent-cyan: {self.accent_cyan};
  --accent-magenta: {self.accent_magenta};
  --text-primary: {self.text_primary};
  --text-secondary: {self.text_secondary};
  --text-muted: {self.text_muted};
  --border: {self.border};
  --font-mono: "SF Mono", "Consolas", "Fira Code", monospace;
  --font-sans: "Inter", "Segoe UI", system-ui, sans-serif;
}}"""
