"""Tests for Phase C: rich activity panel and OCR progress display."""
import os
import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


class TestActivityPanelFullSequence:
    """Feed full event sequence, verify panel state."""

    def test_full_sequence_with_rerank(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()

        panel.on_embed()
        panel.on_search(20)
        assert panel._found_count == 20

        panel.on_rerank(20, 5)
        assert panel._has_rerank is True
        assert panel._rerank_from == 20
        assert panel._rerank_to == 5

        panel.on_prompt(5)
        assert panel._sources_k == 5

        panel.on_generate_start()
        for _ in range(12):
            panel.on_token()
        assert panel._token_count == 12

        summary = panel.on_done()
        assert "20" in summary
        assert "5" in summary
        assert "12" in summary
        assert not panel._strip_frame.isVisible()
        assert panel._summary_label.isVisible()

    def test_full_sequence_without_rerank(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()

        panel.on_embed()
        panel.on_search(15)
        panel.on_prompt(4)
        panel.on_generate_start()
        for _ in range(7):
            panel.on_token()
        summary = panel.on_done()

        assert panel._has_rerank is False
        assert "реранж" not in summary
        assert "15" in summary
        assert "4" in summary
        assert "7" in summary

    def test_rerank_row_hidden_without_rerank_event(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()
        # rerank row should be hidden after reset (no rerank event)
        _, _, row_w, _ = panel._stage_rows["rerank"]
        assert not row_w.isVisible()

    def test_rerank_row_shown_after_rerank_event(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()
        panel.on_embed()
        panel.on_search(10)
        panel.on_rerank(10, 3)
        _, _, row_w, _ = panel._stage_rows["rerank"]
        assert row_w.isVisible()

    def test_token_count_increments(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()
        panel.on_embed()
        panel.on_search(5)
        panel.on_prompt(3)
        panel.on_generate_start()
        for _ in range(5):
            panel.on_token()
        assert panel._token_count == 5

    def test_error_does_not_crash(self, app):
        from windows.main_window import _ActivityPanel
        panel = _ActivityPanel()
        panel.reset()
        panel.on_embed()
        panel.on_error("Something went wrong")
        app.processEvents()
        # Panel still alive
        assert panel is not None


class TestActivityPanelThemes:
    """Both themes instantiate without crash."""

    def test_dark_theme_no_crash(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        w._on_set_theme("dark")
        app.processEvents()
        assert w._activity_panel is not None
        w.deleteLater()
        app.processEvents()

    def test_light_theme_no_crash(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        w._on_set_theme("light")
        app.processEvents()
        assert w._activity_panel is not None
        w.deleteLater()
        app.processEvents()


class TestActivityPanelHeadless:
    """Full pipeline event sequence on real window, no crash."""

    def test_window_full_sequence_no_crash(self, app):
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        w.show()
        app.processEvents()

        try:
            panel = w._activity_panel
            assert panel is not None
            panel.reset()
            panel.on_embed()
            panel.on_search(20)
            panel.on_rerank(20, 5)
            panel.on_prompt(5)
            panel.on_generate_start()
            for _ in range(12):
                panel.on_token()
            panel.on_done()
            app.processEvents()
        finally:
            w.deleteLater()
            app.processEvents()

    def test_pipeline_events_via_on_pipeline_event(self, app):
        """_on_pipeline_event drives panel correctly."""
        from windows.main_window import CompanionWindow
        w = CompanionWindow(pipeline=None)
        w.show()
        app.processEvents()

        try:
            w._on_pipeline_event({"stage": "embed"})
            w._on_pipeline_event({"stage": "search", "found": 20})
            w._on_pipeline_event({"stage": "rerank", "from": 20, "to": 5})
            w._on_pipeline_event({"stage": "prompt", "sources": 5})
            w._on_pipeline_event({"stage": "generate_start"})
            for _ in range(3):
                w._on_pipeline_event({"stage": "token", "text": "word "})
            app.processEvents()

            panel = w._activity_panel
            assert panel._found_count == 20
            assert panel._has_rerank is True
            assert panel._sources_k == 5
            assert panel._token_count == 3
        finally:
            w.deleteLater()
            app.processEvents()


class TestOCRProgressLog:
    """OCR dialog progress log updates."""

    def test_ocr_progress_log_adds_lines(self, app):
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog()
        try:
            event = {
                'doc': 'test.pdf',
                'page_index': 0,
                'total_pages': 10,
                'source': 'native',
                'ocr_count': 0,
                'native_count': 1,
                'done_count': 1,
                'elapsed_s': 0.5,
            }
            dlg._on_progress(event)
            app.processEvents()
            assert len(dlg._page_log_lines) == 1
        finally:
            dlg.close()
            app.processEvents()

    def test_ocr_progress_log_caps_at_3(self, app):
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog()
        try:
            for i in range(5):
                event = {
                    'doc': 'test.pdf',
                    'page_index': i,
                    'total_pages': 10,
                    'source': 'ocr' if i % 2 == 0 else 'native',
                    'ocr_count': i,
                    'native_count': i,
                    'done_count': i + 1,
                    'elapsed_s': float(i + 1),
                }
                dlg._on_progress(event)
            app.processEvents()
            assert len(dlg._page_log_lines) <= 3
        finally:
            dlg.close()
            app.processEvents()

    def test_ocr_progress_native_label(self, app):
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog()
        try:
            event = {
                'doc': 'test.pdf',
                'page_index': 2,
                'total_pages': 10,
                'source': 'native',
                'ocr_count': 0,
                'native_count': 3,
                'done_count': 3,
                'elapsed_s': 1.0,
            }
            dlg._on_progress(event)
            app.processEvents()
            assert len(dlg._page_log_lines) >= 1
            assert 'нативный' in dlg._page_log_lines[-1].text()
        finally:
            dlg.close()
            app.processEvents()

    def test_ocr_progress_ocr_label(self, app):
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog()
        try:
            event = {
                'doc': 'test.pdf',
                'page_index': 1,
                'total_pages': 10,
                'source': 'ocr',
                'ocr_count': 1,
                'native_count': 0,
                'done_count': 1,
                'elapsed_s': 2.0,
            }
            dlg._on_progress(event)
            app.processEvents()
            assert 'OCR' in dlg._page_log_lines[-1].text()
        finally:
            dlg.close()
            app.processEvents()

    def test_ocr_page_label_uses_done_count(self, app):
        from windows.ocr_dialog import OCRDialog
        dlg = OCRDialog()
        try:
            event = {
                'doc': 'test.pdf',
                'page_index': 4,
                'total_pages': 20,
                'source': 'native',
                'ocr_count': 0,
                'native_count': 5,
                'done_count': 5,
                'elapsed_s': 3.0,
            }
            dlg._on_progress(event)
            app.processEvents()
            assert "5" in dlg.page_label.text()
            assert "20" in dlg.page_label.text()
        finally:
            dlg.close()
            app.processEvents()
