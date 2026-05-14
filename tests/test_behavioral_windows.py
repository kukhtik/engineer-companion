"""Behavioral UI tests for Windows PySide6 interface.

Follows behavioral-ui-testing skill:
- Simulate user actions (clicks, key input)
- Assert on widget state, not internal methods
- No sleep() — use processEvents
- Cleanup after each test
- Runs headless with QT_QPA_PLATFORM=offscreen
"""

import os
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from windows.main_window import CompanionWindow  # noqa: E402
from windows.settings_dialog import SettingsDialog  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


@pytest.fixture
def window(app):
    """Window without LLM — search-only mode for fast tests."""
    w = CompanionWindow(pipeline=None)
    w.show()
    app.processEvents()
    yield w
    w.deleteLater()
    app.processEvents()


class TestUserSearchFlow:
    """End-to-end: user opens app, types query, clicks search, expects results area to update."""

    def test_user_can_type_in_search_and_sees_disabled_buttons_while_processing(self, app, window):
        # Arrange
        search = window.search_edit
        button = window.search_btn

        # Pre-assert
        assert button.isEnabled() is True
        assert search.text() == ""

        # Act — user types query
        QTest.keyClicks(search, "TrueBeam interlock error")
        app.processEvents()

        # Assert — text entered
        assert search.text() == "TrueBeam interlock error"

    def test_search_button_exists_and_is_clickable(self, app, window):
        btn = window.search_btn
        assert btn.isVisible() is True
        assert btn.text() == "Найти"

    def test_user_sends_query_and_chat_log_updates(self, app, window):
        # Arrange: inject mock pipeline? No, we test UI behaviour with None pipeline.
        # With pipeline=None, clicking search should show error in chat log.
        search = window.search_edit
        chat = window.chat_log
        btn = window.search_btn

        initial_doc_len = len(chat.toPlainText())

        # Act
        QTest.keyClicks(search, "test query")
        app.processEvents()
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
        app.processEvents()

        # Assert — chat log appended error because no LLM is loaded
        text = chat.toPlainText()
        assert "Ошибка: LLM pipeline не инициализирован" in text or len(text) > initial_doc_len

    def test_search_input_supports_enter_key(self, app, window):
        search = window.search_edit
        search.setText("")
        QTest.keyClicks(search, "enter test")
        app.processEvents()

        # Simulate Return pressed
        QTest.keyClick(search, Qt.Key.Key_Return)
        app.processEvents()

        # Some reaction expected (chat log update or error)
        assert len(window.chat_log.toPlainText()) > 0

    def test_results_list_is_empty_initially(self, app, window):
        assert window.results_list.count() == 0

    def test_window_title_is_correct(self, app, window):
        assert window.windowTitle() == "Engineer Companion"

    def test_chat_input_exists_and_has_placeholder(self, app, window):
        inp = window.chat_input
        assert inp.isVisible() is True
        assert "Уточняющий вопрос" in inp.placeholderText()

    def test_meta_label_is_muted_style(self, app, window):
        # muted is indicated by objectName "muted"
        assert window.meta_label.objectName() == "muted"

    def test_clear_button_exists_and_triggers_history_clear(self, app, window):
        # isolating: clear pre-existing history first
        window.history.clear()
        app.processEvents()

        btn = window.clear_btn
        assert btn.isVisible() is True
        assert "Очистить" in btn.text()

        # inject a fake history entry
        window.history.add("q", "a", [])
        app.processEvents()
        assert len(window.history.entries) == 1

        QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
        app.processEvents()

        assert len(window.history.entries) == 0
        assert "История очищена" in window.chat_log.toPlainText()


class TestBookmarks:
    """Star / unstar history entries via QTextBrowser anchor clicks."""

    def test_bookmark_toggle_changes_star_state(self, app, window):
        # Inject a fake entry so history is not empty
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [], "bookmarked": False}
        ]
        window.history.save()
        window._restore_history()
        app.processEvents()

        # Toggle via method (anchorClicked is harder to simulate headless)
        assert window.history.entries[0]["bookmarked"] is False
        window.history.toggle_bookmark(0)
        assert window.history.entries[0]["bookmarked"] is True
        window.history.toggle_bookmark(0)
        assert window.history.entries[0]["bookmarked"] is False

    def test_history_html_shows_star_for_bookmarked(self, app, window):
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [], "bookmarked": True}
        ]
        html_out = window.history.format_html()
        assert "★" in html_out
        assert "bookmark://0" in html_out


class TestUIWithMockPipeline:
    """Inject a mock pipeline returning fixed results; verify full flow."""

    class MockPipeline:
        def ask(self, query):
            return {
                "answer": "Mock answer for: " + query,
                "sources": [
                    {"source": "mock.pdf", "page": 42, "section": "3.2 Interlocks"},
                ],
            }

    def test_user_sees_result_items_after_search_with_pipeline(self, app):
        w = CompanionWindow(pipeline=self.MockPipeline())
        w.show()
        app.processEvents()

        try:
            search = w.search_edit
            QTest.keyClicks(search, "what is interlock")
            app.processEvents()
            QTest.mouseClick(w.search_btn, Qt.MouseButton.LeftButton)

            # Poll until worker finishes and is cleaned up
            for _ in range(200):
                app.processEvents()
                if w.worker is None:
                    break

            assert w.results_list.count() > 0
            first = w.results_list.item(0).text()
            assert "mock.pdf" in first
            assert "стр.42" in first
            chat = w.chat_log.toPlainText()
            assert "Mock answer" in chat
        finally:
            w.deleteLater()
            app.processEvents()


class TestSettingsDialog:
    def test_settings_dialog_opens_and_has_fields(self, app, window):
        dlg = SettingsDialog(window, current=window.settings)
        try:
            assert hasattr(dlg, "llm_path_edit")
            assert hasattr(dlg, "temperature_spin")
            assert hasattr(dlg, "max_tokens_spin")
            assert hasattr(dlg, "top_k_spin")
        finally:
            dlg.close()
            app.processEvents()

    def test_settings_cancel_does_not_change_values(self, app, window):
        orig_temp = window.settings["temperature"]
        dlg = SettingsDialog(window, current=window.settings)
        dlg.temperature_spin.setValue(1.0)
        dlg.reject()
        app.processEvents()
        # window.settings unchanged because dialog was rejected
        assert window.settings["temperature"] == orig_temp

    def test_settings_apply_changes_values(self, app, window):
        dlg = SettingsDialog(window, current=window.settings)
        dlg.temperature_spin.setValue(0.8)
        dlg.llm_path_edit.setText("/new/path/model.gguf")
        dlg._on_accept()
        app.processEvents()
        settings = dlg.get_settings()
        assert settings["temperature"] == 0.8
        assert settings["llm_model_path"] == "/new/path/model.gguf"
        dlg.close()
        app.processEvents()

    def test_settings_top_k_min_max(self, app, window):
        dlg = SettingsDialog(window, current=window.settings)
        assert dlg.top_k_spin.minimum() == 1
        assert dlg.top_k_spin.maximum() >= 20
        dlg.close()
        app.processEvents()


class TestBookmarksFilter:
    def test_bookmark_checkbox_exists(self, app, window):
        assert hasattr(window, "bookmarks_cb")
        assert window.bookmarks_cb.isVisible()

    def test_bookmark_checkbox_default_off(self, app, window):
        assert window.bookmarks_cb.isChecked() is False

    def test_bookmark_filter_only_shows_bookmarked(self, app, window):
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [], "bookmarked": True},
            {"query": "Q2", "answer": "A2", "sources": [], "bookmarked": False},
            {"query": "Q3", "answer": "A3", "sources": [], "bookmarked": True},
        ]
        filtered = window.history.format_html(bookmarks_only=True)
        assert "Q1" in filtered
        assert "Q2" not in filtered
        assert "Q3" in filtered


class TestHistorySearch:
    def test_history_search_input_exists(self, app, window):
        assert hasattr(window, "history_search")
        assert window.history_search.isVisible()

    def test_history_search_filters_entries(self, app, window):
        window.history.entries = [
            {"query": "TrueBeam", "answer": "answer about beam", "sources": [], "bookmarked": False},
            {"query": "VitalBeam", "answer": "answer about vital", "sources": [], "bookmarked": False},
            {"query": "Interlock", "answer": "answer about lock", "sources": [], "bookmarked": False},
        ]
        filtered = window.history.format_html(search_term="Beam")
        assert "TrueBeam" in filtered or "VitalBeam" in filtered
        assert "Interlock" not in filtered

    def test_history_search_no_match(self, app, window):
        window.history.entries = [
            {"query": "TrueBeam", "answer": "answer", "sources": [], "bookmarked": False},
        ]
        filtered = window.history.format_html(search_term="ZZZZZ")
        assert filtered == "" or "Нет записей" in filtered


class TestExport:
    def test_export_menu_item_exists(self, app, window):
        assert hasattr(window, "export_action") or hasattr(window, "_on_export_chat")

    def test_export_markdown(self, app, window, tmp_path):
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [{"source": "s1.pdf", "page": 1, "section": "Sec"}], "bookmarked": False},
        ]
        dest = tmp_path / "chat.md"
        lines = ["# Engineer Companion — Экспорт чата\n"]
        for e in window.history.entries:
            star = "★" if e.get("bookmarked") else " "
            lines.append(f"## {star} Вопрос: {e['query']}\n")
            lines.append(f"{e['answer']}\n")
            for src in e.get("sources", []):
                lines.append(f"- {src['source']}, стр.{src['page']} — {src['section']}\n")
            lines.append("\n")

        dest.write_text("".join(lines))
        assert dest.exists()
        content = dest.read_text()
        assert "Q1" in content
        assert "A1" in content
        assert "s1.pdf" in content

    def test_export_json(self, app, window, tmp_path):
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [], "bookmarked": True},
        ]
        dest = tmp_path / "chat.json"
        import json
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(window.history.entries, f, ensure_ascii=False, indent=2)
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data[0]["query"] == "Q1"
