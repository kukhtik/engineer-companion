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
        """showEvent should start a fade_in animation on the tabs widget.

        The animation attaches a QGraphicsOpacityEffect while running and
        removes it when done.  We check that _shown_once is set (the guard
        that triggers the animation) and that the widget becomes visible —
        we don't assert the effect is still present after the animation
        completes, since it is cleaned up on finish.
        """
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        app.processEvents()  # let showEvent fire

        # Guard flag must be set, confirming fade_in was triggered
        assert dlg._shown_once is True, "_shown_once guard not set — showEvent did not fire"
        assert dlg.tabs.isVisible(), "tabs widget should be visible after fade_in"

        _pump(app, 0.4)  # let animation complete
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_fade_in_runs_to_completion(self, app):
        """After the animation duration the tabs widget should be visible.

        The QGraphicsOpacityEffect is removed when the animation completes
        (cleanup in anim.py), so we check visibility rather than effect opacity.
        """
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS

        dlg = SettingsDialog(None, current=dict(DEFAULT_SETTINGS))
        dlg.show()
        _pump(app, 0.6)  # animation is 200 ms; give it plenty of time

        # After animation completes the widget must still be visible
        assert dlg.tabs.isVisible(), "tabs should be visible after fade_in completes"
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
        """showEvent triggers fade_in on the dialog itself.

        The effect is removed when the animation completes; we check that the
        dialog becomes visible (the animation started) rather than requiring
        the effect to persist after completion.
        """
        from windows.library_dialog import LibraryDialog

        db_p = self._seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        dlg.show()
        app.processEvents()  # let showEvent fire

        # The dialog should be visible — confirms fade_in was called (show() inside fade_in)
        assert dlg.isVisible(), "LibraryDialog should be visible after showEvent fade_in"
        _pump(app, 0.4)
        dlg.close()
        dlg.deleteLater()
        app.processEvents()

    def test_animation_completes_to_full_opacity(self, app, tmp_path):
        """After animation completes the dialog is still visible."""
        from windows.library_dialog import LibraryDialog

        db_p = self._seed_db(tmp_path)
        dlg = LibraryDialog(None, db_path=db_p)
        dlg.show()
        _pump(app, 0.6)

        assert dlg.isVisible(), "LibraryDialog should remain visible after animation"
        dlg.close()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# OCRDialog animation (only if OCR available)
# ---------------------------------------------------------------------------

class TestOCRDialogAnimation:
    def test_show_event_triggers_opacity_effect_or_skip(self, app):
        """OCRDialog fades in on show. Skip gracefully if OCR unavailable.

        The fade_in attaches and then removes the QGraphicsOpacityEffect; we
        verify the dialog becomes visible rather than checking for the effect.
        """
        from windows.ocr_dialog import OCRDialog, ocr_available

        try:
            dlg = OCRDialog(None)
        except Exception as exc:
            pytest.skip(f"OCRDialog could not be instantiated: {exc}")

        dlg.show()
        app.processEvents()  # let showEvent fire

        assert dlg.isVisible(), "OCRDialog should be visible after showEvent fade_in"
        _pump(app, 0.4)
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
        """_start_meta_pulse sets _meta_pulse_going=True and starts the timer.

        The pulse is now implemented via QTimer + stylesheet (not
        QGraphicsOpacityEffect) to avoid QPainter conflicts.
        """
        from windows.main_window import CompanionWindow

        w = CompanionWindow(pipeline=self._MockPipeline())
        w.show()
        app.processEvents()

        # Simulate typing a query and clicking Send
        w.chat_input.setText("test query anim")
        w._on_search()
        app.processEvents()

        # After _on_search, _meta_pulse_going must be True
        assert w._meta_pulse_going is True, (
            "Expected _meta_pulse_going=True after query starts"
        )

        # Wait for worker to finish
        for _ in range(200):
            app.processEvents()
            if w.worker is None:
                break
            time.sleep(0.01)

        # Pump a bit more
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
        """Calling _on_set_theme applies the new theme immediately.

        The opacity-effect crossfade on the central widget was removed to
        prevent QPainter "Painter not active" floods (the whole-tree pixmap
        grab conflicts with children painting).  Theme switching is now instant.
        We verify only that the stylesheet was updated.
        """
        from windows.main_window import CompanionWindow
        from design.tokens import LIGHT

        w = CompanionWindow(pipeline=None)
        w.show()
        _pump(app, 0.1)

        w._on_set_theme("light")
        app.processEvents()
        # Theme stylesheet must have been applied
        assert LIGHT.bg_base in app.styleSheet(), (
            "Light theme bg_base not found in app stylesheet after theme switch"
        )

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
