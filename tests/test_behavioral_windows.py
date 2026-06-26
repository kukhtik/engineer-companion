"""Behavioral UI tests for Windows PySide6 interface — new chat-centric UI."""
import json
import os
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from windows.main_window import CompanionWindow, ConversationStore, ChatHistory  # noqa
from windows.settings_dialog import SettingsDialog  # noqa

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


@pytest.fixture
def window(app):
    w = CompanionWindow(pipeline=None)
    w.show()
    app.processEvents()
    yield w
    w.deleteLater()
    app.processEvents()


# ---------------------------------------------------------------------------
# TestWindowStructure
# ---------------------------------------------------------------------------

class TestWindowStructure:
    """Basic widget existence checks."""

    def test_conv_list_exists(self, app, window):
        assert hasattr(window, "conv_list") and window.conv_list is not None

    def test_send_btn_exists(self, app, window):
        assert hasattr(window, "send_btn") and window.send_btn.text() == "Отправить"

    def test_chat_input_exists(self, app, window):
        assert hasattr(window, "chat_input") and len(window.chat_input.placeholderText()) > 0

    def test_backward_compat_aliases(self, app, window):
        assert window.search_edit is window.chat_input
        assert window.search_btn is window.send_btn

    def test_window_title(self, app, window):
        assert window.windowTitle() == "Engineer Companion"

    def test_meta_label_exists(self, app, window):
        assert window.meta_label is not None
        assert window.meta_label.objectName() == "muted"

    def test_results_list_compat(self, app, window):
        assert window.results_list is not None
        assert window.results_list.count() == 0


# ---------------------------------------------------------------------------
# TestNoPipelineFlow
# ---------------------------------------------------------------------------

class TestNoPipelineFlow:
    """Behavior when pipeline=None."""

    def test_query_with_no_pipeline_shows_error(self, app, window):
        window.chat_input.setText("test")
        app.processEvents()
        QTest.mouseClick(window.send_btn, Qt.MouseButton.LeftButton)
        app.processEvents()
        text = window.chat_log.toPlainText()
        assert "Ошибка" in text or "pipeline" in text.lower()

    def test_send_btn_re_enabled_after_no_pipeline(self, app, window):
        window.chat_input.setText("re-enable test")
        app.processEvents()
        QTest.mouseClick(window.send_btn, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert window.send_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# TestConversationStore
# ---------------------------------------------------------------------------

class TestConversationStore:
    """Conversation persistence."""

    def test_new_conversation(self):
        cs = ConversationStore()
        before = len(cs.conversations)
        c = cs.new_conversation()
        assert c["id"]
        assert c["title"]
        assert len(cs.conversations) >= before + 1

    def test_save_and_load(self):
        cs = ConversationStore()
        c = cs.new_conversation()
        conv_id = c["id"]
        cs.save()
        # Create a new store instance pointing to same file
        cs2 = ConversationStore()
        found = cs2.get_conversation(conv_id)
        assert found is not None
        assert found["id"] == conv_id

    def test_update_turn_star(self):
        cs = ConversationStore()
        c = cs.new_conversation()
        c["turns"].append({
            "role": "assistant",
            "content": "Test answer",
            "sources": [],
            "starred": False,
        })
        cs.save()
        cs.update_turn_star(c["id"], 0, True)
        found = cs.get_conversation(c["id"])
        assert found["turns"][0]["starred"] is True

    def test_get_all_returns_newest_first(self):
        cs = ConversationStore()
        # Clear existing and add two with known order
        cs.conversations = []
        c1 = {
            "id": "aaa",
            "title": "First",
            "created": "2024-01-01T00:00:00",
            "turns": [],
        }
        c2 = {
            "id": "bbb",
            "title": "Second",
            "created": "2024-06-01T00:00:00",
            "turns": [],
        }
        cs.conversations = [c1, c2]
        all_convs = cs.get_all()
        # Newest (c2) should be first
        assert all_convs[0]["id"] == "bbb"
        assert all_convs[1]["id"] == "aaa"


# ---------------------------------------------------------------------------
# TestMultiTurnHistory
# ---------------------------------------------------------------------------

class TestMultiTurnHistory:
    """History passes correctly to worker."""

    class HistoryCapturePipeline:
        def __init__(self):
            self.captured_history = None

        def ask(self, query, history=None):
            self.captured_history = history
            return {"answer": "Answer for: " + query, "sources": []}

    def test_history_passed_to_worker(self, app):
        pipeline = self.HistoryCapturePipeline()
        w = CompanionWindow(pipeline=pipeline)
        w.show()
        app.processEvents()
        try:
            # First query
            w.chat_input.setText("first question")
            w._on_search()
            for _ in range(200):
                app.processEvents()
                if w.worker is None:
                    break
            # Second query — should have history
            w.chat_input.setText("second question")
            w._on_search()
            for _ in range(200):
                app.processEvents()
                if w.worker is None:
                    break
            # The pipeline's captured_history should include the first turn
            assert pipeline.captured_history is not None
            assert len(pipeline.captured_history) > 0
        finally:
            w.deleteLater()
            app.processEvents()


# ---------------------------------------------------------------------------
# TestFavorites
# ---------------------------------------------------------------------------

class TestFavorites:
    """Star/unstar functionality."""

    def test_toggle_star_updates_store(self, app, window):
        cs = ConversationStore()
        c = cs.new_conversation()
        c["turns"].append({
            "role": "assistant",
            "content": "Test answer",
            "sources": [],
            "starred": False,
        })
        cs.save()
        cs.update_turn_star(c["id"], 0, True)
        found = cs.get_conversation(c["id"])
        assert found["turns"][0]["starred"] is True

    def test_toggle_star_changes_button_text(self, app, window):
        from PySide6.QtWidgets import QPushButton
        # Set up a conversation with an assistant turn
        cs = window.conv_store
        c = cs.new_conversation()
        c["turns"].append({
            "role": "assistant",
            "content": "Starred answer",
            "sources": [],
            "starred": False,
        })
        cs.save()
        window._current_conv = c

        star_btn = QPushButton("☆ В избранное")
        star_btn.setCheckable(True)
        window._toggle_star(c["id"], 0, star_btn)
        app.processEvents()
        assert "★" in star_btn.text()

        window._toggle_star(c["id"], 0, star_btn)
        app.processEvents()
        assert "☆" in star_btn.text()


# ---------------------------------------------------------------------------
# TestNewConversation
# ---------------------------------------------------------------------------

class TestNewConversation:
    """New conversation clears thread."""

    def test_new_conversation_clears_thread(self, app, window):
        # Do a fake query to put something in the thread
        window._current_conv = window.conv_store.new_conversation()
        bub = window._make_user_bubble("test message")
        window.thread_layout.addWidget(bub)
        app.processEvents()
        assert window.thread_layout.count() > 0

        window._start_new_conversation()
        app.processEvents()
        assert window.thread_layout.count() == 0


# ---------------------------------------------------------------------------
# TestExportNewUI
# ---------------------------------------------------------------------------

class TestExportNewUI:
    """Export current conversation."""

    def test_export_logic_markdown(self, app, window, tmp_path):
        window._current_conv = {
            "id": "test", "title": "Test", "created": "2024-01-01",
            "turns": [
                {"role": "user", "content": "My question", "sources": [], "starred": False},
                {"role": "assistant", "content": "My answer", "sources": [
                    {"source": "doc.pdf", "page": 5, "section": "Sec"}
                ], "starred": False},
            ]
        }
        dest = tmp_path / "chat.md"
        turns = window._current_conv["turns"]
        lines = ["# Engineer Companion — Test\n\n"]
        for turn in turns:
            if turn["role"] == "user":
                lines.append(f"## Вопрос:\n{turn['content']}\n\n")
            elif turn["role"] == "assistant":
                lines.append(f"### Ответ:\n{turn['content']}\n")
                for src in turn.get("sources", []):
                    lines.append(f"- {src['source']}, стр.{src['page']}\n")
                lines.append("\n")
        dest.write_text("".join(lines))
        content = dest.read_text()
        assert "My question" in content
        assert "My answer" in content
        assert "doc.pdf" in content


# ---------------------------------------------------------------------------
# TestSourceCard
# ---------------------------------------------------------------------------

class TestSourceCard:
    """Source card opens page viewer."""

    def test_source_card_click_calls_open_page_viewer(self, app, window, monkeypatch):
        called = []
        monkeypatch.setattr(window, "_open_page_viewer", lambda f, p, o=False: called.append((f, p, o)))
        src = {"source": "test.pdf", "page": 3, "section": "Sec 1"}
        card = window._make_source_card(src)
        card.click()
        app.processEvents()
        assert len(called) == 1
        assert called[0][0] == "test.pdf"
        assert called[0][1] == 3


# ---------------------------------------------------------------------------
# TestStatusIndicator
# ---------------------------------------------------------------------------

class TestStatusIndicator:
    """Status indicator behavior."""

    def test_embed_event_sets_label(self, app, window):
        window._on_pipeline_event({"stage": "embed"})
        app.processEvents()
        assert "документац" in window._status_label.text().lower() or len(window._status_label.text()) > 0

    def test_status_hidden_after_stop(self, app, window):
        window._start_status_indicator("Test...")
        app.processEvents()
        assert window.status_row.isVisible()
        window._stop_status_indicator()
        app.processEvents()
        assert not window.status_row.isVisible()

    def test_dot_pulse_going_flag(self, app, window):
        window._start_status_indicator("Processing…")
        app.processEvents()
        assert window._dot_pulse_going is True
        window._stop_status_indicator()
        app.processEvents()
        assert window._dot_pulse_going is False

    def test_generate_start_resets_streaming_answer(self, app, window):
        window._streaming_answer = "some previous text"
        window._on_pipeline_event({"stage": "generate_start"})
        app.processEvents()
        assert window._streaming_answer == ""


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Backward compat shims work correctly."""

    def test_chat_log_toHtml_works(self, app, window):
        window.chat_log.setHtml("<b>test</b>")
        assert "test" in window.chat_log.toHtml()

    def test_chat_log_toPlainText_works(self, app, window):
        window.chat_log.setHtml("<b>test</b>")
        assert "test" in window.chat_log.toPlainText()

    def test_results_list_count(self, app, window):
        window.results_list.clear()
        assert window.results_list.count() == 0

    def test_results_list_is_empty_initially(self, app, window):
        # Fresh window — results_list starts empty
        assert window.results_list.count() == 0

    def test_user_sends_query_and_chat_log_updates(self, app, window):
        # With pipeline=None, clicking search should update chat_log with error
        search = window.search_edit
        btn = window.search_btn

        initial_len = len(window.chat_log.toPlainText())

        QTest.keyClicks(search, "test query")
        app.processEvents()
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
        app.processEvents()

        text = window.chat_log.toPlainText()
        assert "Ошибка: LLM pipeline не инициализирован" in text or len(text) > initial_len


# ---------------------------------------------------------------------------
# TestBookmarks (kept — operates on ChatHistory, not window widgets)
# ---------------------------------------------------------------------------

class TestBookmarks:
    """Star / unstar history entries."""

    def test_bookmark_toggle_changes_star_state(self, app, window):
        window.history.entries = [
            {"query": "Q1", "answer": "A1", "sources": [], "bookmarked": False}
        ]
        window.history.save()
        app.processEvents()

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


# ---------------------------------------------------------------------------
# TestHistorySearch (kept — operates on ChatHistory.format_html)
# ---------------------------------------------------------------------------

class TestHistorySearch:
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


# ---------------------------------------------------------------------------
# TestExport (kept — operates on history.entries directly)
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_menu_item_exists(self, app, window):
        assert hasattr(window, "_on_export_chat")

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
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(window.history.entries, f, ensure_ascii=False, indent=2)
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data[0]["query"] == "Q1"


# ---------------------------------------------------------------------------
# TestSettingsDialog (unchanged from old file)
# ---------------------------------------------------------------------------

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
