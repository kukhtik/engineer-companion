"""ThemeManager: load/save the active theme and apply it to the QApplication.

Usage::

    from windows.theme import ThemeManager
    tm = ThemeManager()
    tm.apply(app)               # apply saved (or default) theme
    tm.set_theme("light", app)  # switch live
    tm.set_theme("dark", app)
    name = tm.get_theme()       # "dark" | "light"
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from design.tokens import DARK, THEMES, Theme, as_stylesheet
from windows.settings_dialog import SETTINGS_DIR, load_settings, save_settings

# Key used in the settings JSON file
_THEME_KEY = "theme"
_DEFAULT_THEME = "dark"


class ThemeManager:
    """Loads/saves the chosen theme name and applies QSS to the QApplication.

    The theme name is persisted inside the existing settings.json so no
    separate file is needed.
    """

    def __init__(self) -> None:
        self._settings = load_settings()
        self._current_name: str = self._settings.get(_THEME_KEY, _DEFAULT_THEME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_theme(self) -> str:
        """Return the active theme name ("dark" or "light")."""
        return self._current_name

    def get_theme_obj(self) -> Theme:
        """Return the active Theme dataclass instance."""
        return THEMES.get(self._current_name, DARK)

    def set_theme(self, name: str, app: QApplication | None = None) -> None:
        """Switch to *name* theme, persist the choice, and re-apply QSS.

        If *app* is provided the stylesheet is updated immediately (live re-theme
        without restart).  Pass ``QApplication.instance()`` from the call site.
        """
        if name not in THEMES:
            raise ValueError(f"Unknown theme {name!r}. Available: {list(THEMES)}")
        self._current_name = name
        # Persist
        self._settings[_THEME_KEY] = name
        save_settings(self._settings)
        # Apply live
        if app is not None:
            self._apply_to(app, THEMES[name])

    def apply(self, app: QApplication) -> None:
        """Apply the currently saved theme to *app* at startup."""
        self._apply_to(app, self.get_theme_obj())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_to(app: QApplication, theme: Theme) -> None:
        app.setStyleSheet(as_stylesheet(theme))
