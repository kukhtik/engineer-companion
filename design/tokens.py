"""Design tokens: two named themes (DARK and LIGHT) + QSS generation.

Backward-compat: existing code that does ``DribbbleDarkQt().as_stylesheet()``
continues to work.  The new API is::

    from design.tokens import DARK, LIGHT, as_stylesheet
    sheet = as_stylesheet(DARK)

Or via ThemeManager (windows/theme.py) which handles persistence + live reload.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Theme dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Theme:
    """Full colour + typography token set for one visual theme."""

    name: str

    # Backgrounds
    bg_base: str
    bg_surface: str
    bg_elevated: str
    bg_row_hover: str
    bg_row_selected: str

    # Borders
    border: str

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str

    # Accents
    accent_primary: str    # main interactive accent (yellow on dark, blue on light)
    accent_ink: str        # text colour drawn ON top of accent_primary fill
    accent_danger: str     # destructive / error accent (red on both themes)

    # Typography
    font_sans: str = '"Inter", "Segoe UI", system-ui, sans-serif'
    font_mono: str = '"SF Mono", "Consolas", "Fira Code", monospace'

    def as_stylesheet(self) -> str:
        """Return a full Qt stylesheet string for this theme."""
        return _build_qss(self)


# ---------------------------------------------------------------------------
# Built-in theme instances
# ---------------------------------------------------------------------------

DARK = Theme(
    name="dark",
    # Backgrounds — near-black base
    bg_base="#0a0a0a",
    bg_surface="#151515",
    bg_elevated="#1d1d1d",
    bg_row_hover="#1a1a1a",
    bg_row_selected="#262626",
    # Borders
    border="#2a2a2a",
    # Text
    text_primary="#F5F5F2",
    text_secondary="#B6B6AE",
    text_muted="#8a8a82",
    # Accents
    accent_primary="#F5C518",   # yellow
    accent_ink="#0a0a0a",       # dark text on yellow
    accent_danger="#E5382B",    # red
)

LIGHT = Theme(
    name="light",
    # Backgrounds — white base
    bg_base="#FFFFFF",
    bg_surface="#F4F6F9",
    bg_elevated="#FFFFFF",
    bg_row_hover="#EBF0F8",
    bg_row_selected="#D6E4F5",
    # Borders
    border="#D8DEE6",
    # Text
    text_primary="#173A5E",     # blue-ink, readable
    text_secondary="#3A6FB0",
    text_muted="#6B7B90",
    # Accents
    accent_primary="#1E88E5",   # blue
    accent_ink="#FFFFFF",       # white text on blue
    accent_danger="#D33A2C",    # red
)

# Map name -> Theme for ThemeManager look-ups
THEMES: dict[str, Theme] = {t.name: t for t in (DARK, LIGHT)}


# ---------------------------------------------------------------------------
# QSS builder
# ---------------------------------------------------------------------------

def _build_qss(t: Theme) -> str:
    """Build a comprehensive Qt stylesheet from a Theme instance."""
    return f"""
/* ---- Window / Widget base ---- */
QMainWindow {{
    background-color: {t.bg_base};
    color: {t.text_primary};
    font-family: {t.font_sans};
}}
QWidget {{
    background-color: {t.bg_base};
    color: {t.text_primary};
    font-family: {t.font_sans};
}}
QDialog {{
    background-color: {t.bg_base};
    color: {t.text_primary};
    font-family: {t.font_sans};
}}

/* ---- Inputs ---- */
QLineEdit {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}}
QLineEdit:focus {{
    border: 2px solid {t.accent_primary};
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {t.bg_row_hover};
    border-color: {t.accent_primary};
}}
QPushButton:pressed {{
    background-color: {t.bg_row_selected};
}}
QPushButton:disabled {{
    color: {t.text_muted};
    border-color: {t.border};
}}
QPushButton[destructive="true"] {{
    color: {t.accent_danger};
    border-color: {t.accent_danger};
}}
QPushButton[destructive="true"]:hover {{
    background-color: {t.accent_danger};
    color: #ffffff;
}}
QPushButton[primary="true"] {{
    background-color: {t.accent_primary};
    color: {t.accent_ink};
    border: 1px solid {t.accent_primary};
    font-weight: 700;
}}
QPushButton[primary="true"]:hover {{
    background-color: {t.accent_primary};
    border-color: {t.accent_primary};
}}
QPushButton[primary="true"]:pressed {{
    background-color: {t.accent_primary};
}}
QPushButton[primary="true"]:disabled {{
    background-color: {t.bg_elevated};
    color: {t.text_muted};
    border-color: {t.border};
    font-weight: normal;
}}

/* ---- Lists ---- */
QListView, QListWidget {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    outline: none;
    padding: 4px;
}}
QListView::item, QListWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
}}
QListView::item:hover, QListWidget::item:hover {{
    background-color: {t.bg_row_hover};
}}
QListView::item:selected, QListWidget::item:selected {{
    background-color: {t.bg_row_selected};
    border: 1px solid {t.accent_primary};
}}

/* ---- Labels ---- */
QLabel {{
    color: {t.text_primary};
}}
QLabel#muted {{
    color: {t.text_muted};
    font-size: 12px;
}}

/* ---- Scrollbars ---- */
QScrollBar:vertical {{
    background: {t.bg_base};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {t.border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t.text_muted};
}}
QScrollBar:horizontal {{
    background: {t.bg_base};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {t.border};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t.text_muted};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0;
}}

/* ---- Splitter ---- */
QSplitter::handle {{
    background: {t.border};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ---- Text areas ---- */
QTextBrowser, QTextEdit {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px;
    font-family: {t.font_sans};
    font-size: 14px;
    line-height: 1.5;
}}
QPlainTextEdit {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 8px;
    font-family: {t.font_mono};
    font-size: 13px;
}}

/* ---- Combo box ---- */
QComboBox {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
}}
QComboBox:focus {{
    border: 2px solid {t.accent_primary};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 4px;
    selection-background-color: {t.bg_row_selected};
}}

/* ---- CheckBox ---- */
QCheckBox {{
    color: {t.text_primary};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {t.border};
    border-radius: 3px;
    background: {t.bg_surface};
}}
QCheckBox::indicator:checked {{
    background: {t.accent_primary};
    border-color: {t.accent_primary};
}}

/* ---- SpinBoxes ---- */
QSpinBox, QDoubleSpinBox {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {t.accent_primary};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px;
    border: none;
    background: {t.bg_elevated};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background-color: {t.bg_surface};
    border: 1px solid {t.border};
    border-radius: 4px;
    text-align: center;
    color: {t.text_primary};
    font-size: 12px;
    height: 8px;
}}
QProgressBar::chunk {{
    background-color: {t.accent_primary};
    border-radius: 4px;
}}

/* ---- Table ---- */
QTableWidget, QTableView {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    gridline-color: {t.border};
    outline: none;
}}
QTableWidget::item, QTableView::item {{
    padding: 6px 10px;
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {t.bg_row_selected};
    color: {t.text_primary};
}}
QHeaderView::section {{
    background-color: {t.bg_elevated};
    color: {t.text_secondary};
    border: none;
    border-bottom: 1px solid {t.border};
    padding: 6px 10px;
    font-size: 12px;
}}

/* ---- Tabs ---- */
QTabWidget::pane {{
    border: 1px solid {t.border};
    border-radius: 6px;
    background-color: {t.bg_surface};
}}
QTabBar::tab {{
    background-color: {t.bg_elevated};
    color: {t.text_secondary};
    border: 1px solid {t.border};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 6px 14px;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background-color: {t.bg_surface};
    color: {t.text_primary};
    border-bottom-color: {t.bg_surface};
}}
QTabBar::tab:hover {{
    background-color: {t.bg_row_hover};
}}

/* ---- Menu / MenuBar ---- */
QMenuBar {{
    background-color: {t.bg_base};
    color: {t.text_primary};
    border-bottom: 1px solid {t.border};
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {t.bg_row_hover};
}}
QMenu {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 20px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {t.bg_row_selected};
}}
QMenu::separator {{
    height: 1px;
    background: {t.border};
    margin: 4px 8px;
}}

/* ---- ToolTip ---- */
QToolTip {{
    background-color: {t.bg_elevated};
    color: {t.text_primary};
    border: 1px solid {t.border};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}

/* ---- GroupBox ---- */
QGroupBox {{
    border: 1px solid {t.border};
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    color: {t.text_secondary};
    font-size: 12px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
}}

/* ---- Selection ---- */
QTextEdit, QPlainTextEdit, QTextBrowser, QLineEdit, QComboBox {{
    selection-background-color: {t.accent_primary};
    selection-color: {t.accent_ink};
}}

/* ---- Links ---- */
QLabel[openExternalLinks="true"] {{
    color: {t.accent_primary};
}}
"""


# ---------------------------------------------------------------------------
# Convenience top-level function (new preferred API)
# ---------------------------------------------------------------------------

def as_stylesheet(theme: Theme = DARK) -> str:
    """Return QSS for the given theme.  Defaults to DARK."""
    return theme.as_stylesheet()


# ---------------------------------------------------------------------------
# Backward-compat shim — DribbbleDarkQt().as_stylesheet() still works
# ---------------------------------------------------------------------------

class DribbbleDarkQt:
    """Legacy wrapper.  Returns DARK theme; as_stylesheet() works as before."""

    def __init__(self, **kwargs):
        # Accept (and ignore) any old-style keyword args
        pass

    def as_stylesheet(self) -> str:  # noqa: D102
        return DARK.as_stylesheet()

    # Forward attribute look-ups to DARK so old code like tokens.bg_base works
    def __getattr__(self, name: str):
        return getattr(DARK, name)


@dataclass(frozen=True)
class DribbbleDarkCSS:
    """CSS custom properties for web/Kivy/Android WebView (unchanged)."""

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
