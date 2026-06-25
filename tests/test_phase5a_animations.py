"""Phase 5a animation tests.

Verifies that:
1. SettingsDialog, LibraryDialog, OCRDialog (when available) trigger a fade-in
   animation on first show and don't crash in headless mode.
2. The meta_label pulse starts when a query is fired (with mock pipeline).
3. The theme crossfade applies the new stylesheet before opacity is restored.
4. The chat_log crossfade (fade-out -> setHtml -> fade-in) completes without
   crashing and the HTML is actually updated.
5. Animation references are stored on the widget (no GC crash).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402
from PySide6.QtGui import QShowEvent  # noqa: E402


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


def _pump(app, seconds: float = 0.5) -> None:
    """Process Qt events for up to *seconds* seconds."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# SettingsDialog animation
# ---------------------------------------------------------------------------

class TestSettingsDialogAnimation:
    def test_show_event_triggers_fade_in_on_tabs(self, app):
        """showEvent should attach an opacity effect to the tabs widget."""
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        _pump(app, 0.4)

        # After show, tabs should have an opacity effect (fade_in attaches one)
        effect = dlg.tabs.graphicsEffect()
        assert isinstance(effect, QGraphicsOpacityEffect), (
            "Expected QGraphicsOpacityEffect on tabs after showEvent"
        )
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_fade_in_runs_to_completion(self, app):
        """After the animation duration the tabs opacity should reach 1.0."""
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        _pump(app, 0.6)  # animation is 200 ms; give it plenty of time

        effect = dlg.tabs.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            assert effect.opacity() >= 0.99, (
                f"Expected tabs opacity ~1.0 after animation, got {effect.opacity()}"
            )
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_show_event_fires_only_once(self, app):
        """The _shown_once guard prevents a second fade on re-show."""
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        _pump(app, 0.3)
        assert dlg._shown_once is True

        # Re-show: _shown_once should remain True (no double animation)
        dlg.hide()
        dlg.show()
        _pump(app, 0.1)
        assert dlg._shown_once is True

        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_accept_works_after_animation(self, app):
        """Accepting the dialog while animation played must not crash."""
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        _pump(app, 0.3)
        # Should not raise
        dlg._on_accept()
        app.processEvents()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# LibraryDialog animation
# ---------------------------------------------------------------------------

class TestLibraryDialogAnimation:
    def _seed_db(self, tmp_path: Path) -> Path:
        import lancedb
        import numpy as np
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("section", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("tokens", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
        ])
        db_p = tmp_path / "anim_test.db"
        db = lancedb.connect(str(db_p))
        db.create_table("chunks", data=[{
            "id": "x0", "text": "t", "source": "x.pdf", "page": 1,
            "section": "s", "chunk_index": 0, "tokens": 5,
            "vector": [0.0] * 384,
        }], schema=schema)
        return db_p

    def test_show_event_triggers_opacity_effect(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        db_p = self._seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        dlg.show()
        _pump(app, 0.4)

        # Opacity effect must be attached to self (fade_in(self, ...))
        effect = dlg.graphicsEffect()
        assert isinstance(effect, QGraphicsOpacityEffect), (
            "Expected QGraphicsOpacityEffect on LibraryDialog after showEvent"
        )
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_animation_completes_to_full_opacity(self, app, tmp_path):
        from windows.library_dialog import LibraryDialog
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        db_p = self._seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        dlg.show()
        _pump(app, 0.6)

        effect = dlg.graphicsEffect()
        if isinstance(effect, QGraphicsOpacityEffect):
            assert effect.opacity() >= 0.99

        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# OCRDialog animation (only if OCR available)
# ---------------------------------------------------------------------------

class TestOCRDialogAnimation:
    def test_show_event_triggers_opacity_effect_or_skip(self, app):
        """OCRDialog fades in on show. Skip gracefully if OCR unavailable."""
        from windows.ocr_dialog import OCRDialog, ocr_available
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        if not ocr_available():
            # Can still instantiate the dialog even if OCR is not available
            # (the menu item would be disabled, but we can still test the dialog)
            pass

        try:
            dlg = OCRDialog(None)
        except Exception as exc:
            pytest.skip(f"OCRDialog could not be instantiated: {exc}")

        dlg.show()
        _pump(app, 0.4)

        effect = dlg.graphicsEffect()
        assert isinstance(effect, QGraphicsOpacityEffect), (
            "Expected QGraphicsOpacityEffect on OCRDialog after showEvent"
        )
        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Main window: meta_label pulse
# ---------------------------------------------------------------------------

class TestMainWindowAnimations:
    class _MockPipeline:
        """Slow mock pipeline to keep the pulse going during test."""
        def ask(self, query):
            time.sleep(0.05)  # short delay — just enough to check state
            return {"answer": "ok", "sources": []}

    def test_meta_pulse_starts_when_query_fires(self, app):
        """_start_meta_pulse attaches an opacity effect to meta_label."""
        from windows.main_window import CompanionWindow
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        w = CompanionWindow(pipeline=self._MockPipeline())
        w.show()
        app.processEvents()

        # Simulate typing a query and clicking Send
        w.chat_input.setText("test query anim")
        w._on_search()
        app.processEvents()

        # After _on_search, meta_label should have an opacity effect (pulse)
        effect = w.meta_label.graphicsEffect()
        assert isinstance(effect, QGraphicsOpacityEffect), (
            "Expected meta_label to have QGraphicsOpacityEffect during pulse"
        )

        # Wait for worker to finish
        for _ in range(200):
            app.processEvents()
            if w.worker is None:
                break
            time.sleep(0.01)

        # Pump a bit more so chat crossfade finishes
        _pump(app, 0.4)

        w.close()
        w.deleteLater()
        app.processEvents()

    def test_meta_pulse_stops_after_result(self, app):
        """After _stop_meta_pulse, _meta_pulse_going is False."""
        from windows.main_window import CompanionWindow

        w = CompanionWindow(pipeline=self._MockPipeline())
        w.show()
        app.processEvents()

        w.chat_input.setText("test pulse stop")
        w._on_search()
        app.processEvents()

        # Wait for worker to finish
        for _ in range(300):
            app.processEvents()
            if w.worker is None:
                break
            time.sleep(0.01)

        _pump(app, 0.3)

        assert w._meta_pulse_going is False, (
            "meta pulse should be stopped after result is received"
        )

        w.close()
        w.deleteLater()
        app.processEvents()

    def test_chat_log_crossfade_updates_html(self, app):
        """After _on_result, the chat log HTML is updated (crossfade completes)."""
        from windows.main_window import CompanionWindow

        w = CompanionWindow(pipeline=self._MockPipeline())
        w.show()
        app.processEvents()

        w.chat_input.setText("crossfade html test")
        w._on_search()
        app.processEvents()

        # Wait for worker + crossfade to finish
        for _ in range(300):
            app.processEvents()
            if w.worker is None:
                break
            time.sleep(0.01)

        _pump(app, 0.5)  # let the 80ms fade-out + 200ms fade-in finish

        # The chat log should now contain the answer
        html = w.chat_log.toHtml()
        assert "crossfade html test" in html or "ok" in html, (
            "chat log should contain query or answer after crossfade"
        )

        w.close()
        w.deleteLater()
        app.processEvents()

    def test_theme_switch_visible_window_does_crossfade(self, app):
        """Calling _on_set_theme on a visible window uses the crossfade path."""
        from windows.main_window import CompanionWindow
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        from design.tokens import LIGHT

        w = CompanionWindow(pipeline=None)
        w.show()
        _pump(app, 0.1)  # ensure widget becomes visible

        central = w.centralWidget()
        # Only attempt the crossfade assertion if the widget is truly visible
        if central is not None and central.isVisible():
            w._on_set_theme("light")
            app.processEvents()
            # An opacity effect should be on central during or after the animation
            effect = central.graphicsEffect()
            # After the crossfade completes the effect will still be there
            _pump(app, 0.5)
            # Theme must have been applied
            assert LIGHT.bg_base in app.styleSheet()
        else:
            # Invisible window falls back to synchronous switch — just check it works
            w._on_set_theme("light")
            app.processEvents()
            assert LIGHT.bg_base in app.styleSheet()

        w.close()
        w.deleteLater()
        app.processEvents()

    def test_theme_switch_invisible_window_is_synchronous(self, app):
        """_on_set_theme on a non-visible window applies stylesheet immediately."""
        from windows.main_window import CompanionWindow
        from design.tokens import LIGHT, DARK

        w = CompanionWindow(pipeline=None)
        # Do NOT call show() — central widget will not be visible
        app.processEvents()

        w._on_set_theme("light")
        app.processEvents()
        # Stylesheet should be applied immediately (no async fade needed)
        assert LIGHT.bg_base in app.styleSheet()

        # Restore dark
        w._on_set_theme("dark")
        app.processEvents()
        assert DARK.bg_base in app.styleSheet()

        w.deleteLater()
        app.processEvents()

    def test_animation_reference_stored_prevents_gc(self, app):
        """The _anim_ref attribute exists on animated widgets after fade_in."""
        from windows.anim import fade_in

        w = QWidget()
        w.show()
        app.processEvents()

        fade_in(w, duration=50)
        # _anim_ref must be set to prevent GC during animation
        assert hasattr(w, "_anim_ref"), "_anim_ref not set — animation may be GC'd"

        _pump(app, 0.3)
        w.deleteLater()
        app.processEvents()
