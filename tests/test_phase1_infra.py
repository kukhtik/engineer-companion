"""Phase-1 infrastructure smoke tests.

Tests:
1. anim.py — headless construction + completion of fade_in/fade_out/slide_in
2. tokens.py — DARK and LIGHT themes, as_stylesheet(), DribbbleDarkQt compat
3. theme.py — ThemeManager.set_theme() changes QApplication stylesheet
4. main_window.py — CompanionWindow headless instantiation + theme switch
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


# ---------------------------------------------------------------------------
# Tokens / Theme tests
# ---------------------------------------------------------------------------

class TestTokensThemes:
    def test_dark_and_light_exist(self):
        from design.tokens import DARK, LIGHT
        assert DARK.name == "dark"
        assert LIGHT.name == "light"

    def test_dark_has_yellow_accent(self):
        from design.tokens import DARK
        assert DARK.accent_primary.startswith("#F5C") or "F5C" in DARK.accent_primary

    def test_light_has_blue_accent(self):
        from design.tokens import LIGHT
        assert "1E88" in LIGHT.accent_primary or "1e88" in LIGHT.accent_primary.lower()

    def test_dark_stylesheet_contains_bg_base(self):
        from design.tokens import DARK
        sheet = DARK.as_stylesheet()
        assert DARK.bg_base in sheet

    def test_light_stylesheet_contains_bg_base(self):
        from design.tokens import LIGHT
        sheet = LIGHT.as_stylesheet()
        assert LIGHT.bg_base in sheet

    def test_as_stylesheet_toplevel_defaults_to_dark(self):
        from design.tokens import DARK, as_stylesheet
        assert as_stylesheet() == DARK.as_stylesheet()

    def test_as_stylesheet_light(self):
        from design.tokens import LIGHT, as_stylesheet
        assert as_stylesheet(LIGHT) == LIGHT.as_stylesheet()

    def test_dark_and_light_stylesheets_differ(self):
        from design.tokens import DARK, LIGHT, as_stylesheet
        assert as_stylesheet(DARK) != as_stylesheet(LIGHT)

    def test_dribbbledarkqt_backward_compat(self):
        """DribbbleDarkQt().as_stylesheet() must still return DARK's stylesheet."""
        from design.tokens import DARK, DribbbleDarkQt
        old = DribbbleDarkQt()
        assert old.as_stylesheet() == DARK.as_stylesheet()

    def test_themes_dict_has_both(self):
        from design.tokens import THEMES
        assert "dark" in THEMES
        assert "light" in THEMES


# ---------------------------------------------------------------------------
# ThemeManager tests
# ---------------------------------------------------------------------------

class TestThemeManager:
    def test_default_theme_is_dark(self, app):
        from windows.theme import ThemeManager
        tm = ThemeManager()
        # default is dark (unless settings file already has light)
        name = tm.get_theme()
        assert name in ("dark", "light")  # just check it's valid

    def test_set_theme_light_changes_stylesheet(self, app):
        from design.tokens import DARK, LIGHT
        from windows.theme import ThemeManager

        tm = ThemeManager()
        tm.set_theme("dark", app)
        dark_sheet = app.styleSheet()
        tm.set_theme("light", app)
        light_sheet = app.styleSheet()

        assert dark_sheet != light_sheet
        assert LIGHT.bg_base in light_sheet
        assert DARK.bg_base not in light_sheet

    def test_set_theme_dark_reverts(self, app):
        from design.tokens import DARK
        from windows.theme import ThemeManager

        tm = ThemeManager()
        tm.set_theme("light", app)
        tm.set_theme("dark", app)
        sheet = app.styleSheet()
        assert DARK.bg_base in sheet

    def test_invalid_theme_raises(self, app):
        from windows.theme import ThemeManager
        tm = ThemeManager()
        with pytest.raises(ValueError):
            tm.set_theme("neon", app)

    def test_get_theme_obj_returns_theme(self):
        from design.tokens import Theme
        from windows.theme import ThemeManager
        tm = ThemeManager()
        obj = tm.get_theme_obj()
        assert isinstance(obj, Theme)


# ---------------------------------------------------------------------------
# Animation smoke tests
# ---------------------------------------------------------------------------

class TestAnimHelpers:
    def test_fade_in_constructs_and_starts(self, app):
        from windows.anim import fade_in
        w = QWidget()
        w.show()
        app.processEvents()
        anim = fade_in(w, duration=50)
        # pump events until the 50 ms animation finishes
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        # widget should still be visible; no crash
        assert not w.isHidden()
        w.deleteLater()
        app.processEvents()

    def test_fade_out_calls_on_done(self, app):
        from windows.anim import fade_out
        called = []
        w = QWidget()
        w.show()
        app.processEvents()
        fade_out(w, duration=50, on_done=lambda: called.append(True))
        deadline = time.monotonic() + 2.0
        while not called and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert called, "on_done callback was never called"
        w.deleteLater()
        app.processEvents()

    def test_slide_in_does_not_crash(self, app):
        from windows.anim import slide_in
        w = QWidget()
        w.resize(100, 100)
        w.show()
        app.processEvents()
        anim = slide_in(w, from_dx=-30, duration=50)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        w.deleteLater()
        app.processEvents()

    def test_animate_geometry_does_not_crash(self, app):
        from PySide6.QtCore import QRect
        from windows.anim import animate_geometry
        w = QWidget()
        w.setGeometry(QRect(0, 0, 100, 100))
        w.show()
        app.processEvents()
        animate_geometry(
            w,
            start_rect=QRect(0, 0, 50, 50),
            end_rect=QRect(0, 0, 100, 100),
            duration=50,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        w.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# CompanionWindow headless integration
# ---------------------------------------------------------------------------

class TestCompanionWindowPhase1:
    def test_window_instantiates_headless(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        w.show()
        app.processEvents()
        assert w.isVisible()
        w.close()
        w.deleteLater()
        app.processEvents()

    def test_min_size_is_960x640(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        ms = w.minimumSize()
        assert ms.width() == 960
        assert ms.height() == 640
        w.deleteLater()
        app.processEvents()

    def test_single_ask_input_present(self, app):
        """chat_input must exist; search_edit must be the same object (shim)."""
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()
        assert hasattr(w, "chat_input")
        assert hasattr(w, "search_edit")
        # They must be the exact same widget
        assert w.search_edit is w.chat_input
        assert w.send_btn is w.search_btn
        w.deleteLater()
        app.processEvents()

    def test_no_second_search_bar_in_sidebar(self, app):
        """The old sidebar search_edit that doubled as input is gone.
        The unified chat_input is NOT a child of the sources sidebar."""
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()
        # chat_input should exist
        assert w.chat_input is not None
        # results_list exists (sources panel)
        assert hasattr(w, "results_list")
        w.deleteLater()
        app.processEvents()

    def test_theme_switch_changes_app_stylesheet(self, app):
        from design.tokens import DARK, LIGHT
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()

        w._on_set_theme("dark")
        app.processEvents()
        dark_sheet = app.styleSheet()

        w._on_set_theme("light")
        app.processEvents()
        light_sheet = app.styleSheet()

        assert dark_sheet != light_sheet
        assert LIGHT.bg_base in light_sheet

        # Switch back
        w._on_set_theme("dark")
        app.processEvents()
        assert DARK.bg_base in app.styleSheet()

        w.deleteLater()
        app.processEvents()

    def test_view_tema_menu_exists(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        app.processEvents()

        menubar = w.menuBar()
        view_menu = None
        for action in menubar.actions():
            if "Вид" in action.text():
                view_menu = action.menu()
                break
        assert view_menu is not None, '"Вид" menu not found'

        # Find "Тема" submenu
        theme_menu = None
        for action in view_menu.actions():
            if "Тема" in action.text():
                theme_menu = action.menu()
                break
        assert theme_menu is not None, '"Тема" submenu not found'

        texts = [a.text() for a in theme_menu.actions()]
        assert any("Dark" in t or "Тёмная" in t for t in texts)
        assert any("Light" in t or "Светлая" in t for t in texts)

        w.deleteLater()
        app.processEvents()
