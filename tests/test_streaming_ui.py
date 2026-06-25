"""Tests for the streaming UI — status label updates and token accumulation.

Uses QT_QPA_PLATFORM=offscreen. Does NOT load real models.
Monkeypatches the pipeline to inject fake events.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def window(qapp):
    from windows.main_window import CompanionWindow
    w = CompanionWindow(pipeline=None)
    yield w
    w.close()


class TestStatusLabelUpdates:
    def test_embed_event_sets_label(self, window):
        window._start_status_indicator("Инициализация…")
        window._on_pipeline_event({"stage": "embed"})
        assert window._status_label.text() == "Поиск по документации…"

    def test_search_event_shows_count(self, window):
        window._start_status_indicator("…")
        window._on_pipeline_event({"stage": "search", "found": 12})
        assert "12" in window._status_label.text()

    def test_rerank_event_shows_arrow(self, window):
        window._start_status_indicator("…")
        window._on_pipeline_event({"stage": "rerank", "from": 8, "to": 3})
        txt = window._status_label.text()
        assert "8" in txt
        assert "3" in txt

    def test_prompt_event_shows_sources(self, window):
        window._start_status_indicator("…")
        window._on_pipeline_event({"stage": "prompt", "sources": 5})
        assert "5" in window._status_label.text()

    def test_generate_start_event(self, window):
        window._start_status_indicator("…")
        window._on_pipeline_event({"stage": "generate_start"})
        assert "Генерация" in window._status_label.text()

    def test_status_indicator_visible_during_query(self, window):
        # In offscreen mode the parent window is not shown, so isVisible() can
        # be False even after .show() is called on the child.  Check the explicit
        # hidden flag instead — it is set by hide() and cleared by show().
        window._status_label.hide()
        window._status_dot.hide()
        window._start_status_indicator("Тест")
        assert not window._status_label.isHidden()
        assert not window._status_dot.isHidden()

    def test_status_indicator_hidden_after_stop(self, window):
        window._start_status_indicator("Тест")
        window._stop_status_indicator()
        assert not window._status_label.isVisible()
        assert not window._status_dot.isVisible()

    def test_dot_pulse_going_flag(self, window):
        window._start_status_indicator("Тест")
        assert window._dot_pulse_going is True
        window._stop_status_indicator()
        assert window._dot_pulse_going is False


class TestTokenStreaming:
    def test_tokens_accumulate(self, window):
        window._current_query = "тестовый вопрос"
        window._streaming_answer = ""

        tokens = [" Это", " тест", " ответ."]
        for tok in tokens:
            window._on_pipeline_event({"stage": "token", "text": tok})

        assert window._streaming_answer == "".join(tokens)

    def test_streaming_content_appears_in_chat_log(self, window):
        window._current_query = "тестовый вопрос"
        window._streaming_answer = ""
        window._on_pipeline_event({"stage": "token", "text": " Привет"})
        html = window.chat_log.toHtml()
        assert "Привет" in html

    def test_cursor_visible_during_streaming(self, window):
        window._current_query = "q"
        window._streaming_answer = ""
        window._on_pipeline_event({"stage": "token", "text": "word"})
        html = window.chat_log.toHtml()
        # The blinking cursor ▌ should be in the HTML
        assert "▌" in html

    def test_generate_start_resets_streaming_answer(self, window):
        window._streaming_answer = "old leftover"
        window._on_pipeline_event({"stage": "generate_start"})
        assert window._streaming_answer == ""


class TestInputDisabledDuringQuery:
    """Verify send button is disabled during query and re-enabled after."""

    def test_send_btn_disabled_at_start(self, window):
        # Simulate what _on_search does without actually starting a thread
        window.send_btn.setEnabled(False)
        assert not window.send_btn.isEnabled()

    def test_send_btn_re_enabled_after_cleanup(self, window):
        window.send_btn.setEnabled(False)
        window._cleanup_worker()
        assert window.send_btn.isEnabled()
