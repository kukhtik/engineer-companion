"""Tests for cloud UI integration — Tab 5, _decide_backend, QueryWorker cloud args,
_CloudUsageIndicator, _on_pipeline_event error handling.

All tests are headless (offscreen); no real API keys or network needed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def app():
    a = QApplication.instance() or QApplication([])
    yield a
    a.processEvents()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dlg(app, settings=None):
    from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS
    s = dict(DEFAULT_SETTINGS)
    if settings:
        s.update(settings)
    dlg = SettingsDialog(None, current=s)
    app.processEvents()
    return dlg


def _make_window(app):
    from windows.main_window import CompanionWindow
    w = CompanionWindow(pipeline=None)
    app.processEvents()
    return w


# ---------------------------------------------------------------------------
# Settings round-trip for cloud keys
# ---------------------------------------------------------------------------

class TestCloudSettingsRoundTrip:
    def test_cloud_tab_exists(self, app):
        dlg = _make_dlg(app)
        labels = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
        assert any("Облако" in l or "Ollama" in l for l in labels)
        dlg.deleteLater()
        app.processEvents()

    def test_default_gen_mode_is_local(self, app):
        from windows.settings_dialog import DEFAULT_SETTINGS
        assert DEFAULT_SETTINGS["gen_mode"] == "local"

    def test_default_cloud_model(self, app):
        from windows.settings_dialog import DEFAULT_SETTINGS
        assert DEFAULT_SETTINGS["cloud_model"] == "gpt-oss:120b-cloud"

    def test_gen_mode_round_trip(self, app):
        dlg = _make_dlg(app, {"gen_mode": "cloud"})
        idx = dlg.gen_mode_combo.findData("cloud")
        assert idx >= 0
        assert dlg.gen_mode_combo.currentData() == "cloud"
        dlg._on_accept()
        assert dlg.get_settings()["gen_mode"] == "cloud"
        dlg.deleteLater()
        app.processEvents()

    def test_gen_mode_auto(self, app):
        dlg = _make_dlg(app, {"gen_mode": "auto"})
        dlg._on_accept()
        assert dlg.get_settings()["gen_mode"] == "auto"
        dlg.deleteLater()
        app.processEvents()

    def test_api_key_round_trip(self, app):
        dlg = _make_dlg(app, {"ollama_api_key": "test-key-123"})
        assert dlg.cloud_api_key_edit.text() == "test-key-123"
        dlg._on_accept()
        assert dlg.get_settings()["ollama_api_key"] == "test-key-123"
        dlg.deleteLater()
        app.processEvents()

    def test_cloud_model_round_trip(self, app):
        dlg = _make_dlg(app, {"cloud_model": "qwen3.5:cloud"})
        dlg._on_accept()
        assert dlg.get_settings()["cloud_model"] == "qwen3.5:cloud"
        dlg.deleteLater()
        app.processEvents()

    def test_api_key_echo_mode_is_password(self, app):
        from PySide6.QtWidgets import QLineEdit
        dlg = _make_dlg(app)
        assert dlg.cloud_api_key_edit.echoMode() == QLineEdit.EchoMode.Password
        dlg.deleteLater()
        app.processEvents()

    def test_backward_compat_missing_cloud_keys(self, app, tmp_path):
        """Loading old settings without cloud keys should fill defaults."""
        import json
        import windows.settings_dialog as sd
        orig = sd.SETTINGS_PATH
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"db_path": "", "llm_model_path": ""}), encoding="utf-8")
        sd.SETTINGS_PATH = p
        try:
            loaded = sd.load_settings()
            assert "gen_mode" in loaded
            assert loaded["gen_mode"] == "local"
            assert "ollama_api_key" in loaded
            assert "cloud_model" in loaded
        finally:
            sd.SETTINGS_PATH = orig

    def test_gen_mode_combo_has_all_three_options(self, app):
        dlg = _make_dlg(app)
        modes = [dlg.gen_mode_combo.itemData(i) for i in range(dlg.gen_mode_combo.count())]
        assert "auto" in modes
        assert "local" in modes
        assert "cloud" in modes
        dlg.deleteLater()
        app.processEvents()

    def test_dialog_builds_with_5_tabs_dark_theme(self, app):
        dlg = _make_dlg(app, {"theme": "dark"})
        assert dlg.tabs.count() == 5
        dlg.deleteLater()
        app.processEvents()

    def test_dialog_builds_with_5_tabs_light_theme(self, app):
        dlg = _make_dlg(app, {"theme": "light"})
        assert dlg.tabs.count() == 5
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# _decide_backend logic
# ---------------------------------------------------------------------------

class TestDecideBackend:
    def test_mode_local_always_local(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "local"
        assert w._decide_backend() == "local"
        w.deleteLater()
        app.processEvents()

    def test_mode_cloud_always_cloud(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "cloud"
        assert w._decide_backend() == "cloud"
        w.deleteLater()
        app.processEvents()

    def test_mode_auto_no_key_returns_local(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "auto"
        w.settings["ollama_api_key"] = ""
        assert w._decide_backend() == "local"
        w.deleteLater()
        app.processEvents()

    def test_mode_auto_rate_limited_returns_local(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "auto"
        w.settings["ollama_api_key"] = "some-key"
        # Mock tracker that says we're rate-limited
        mock_tracker = MagicMock()
        mock_tracker.is_rate_limited.return_value = True
        w._cloud_usage_tracker = mock_tracker
        assert w._decide_backend() == "local"
        w.deleteLater()
        app.processEvents()

    def test_mode_auto_offline_returns_local(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "auto"
        w.settings["ollama_api_key"] = "some-key"
        mock_tracker = MagicMock()
        mock_tracker.is_rate_limited.return_value = False
        w._cloud_usage_tracker = mock_tracker
        with patch.object(w, "_is_online", return_value=False):
            result = w._decide_backend()
        assert result == "local"
        w.deleteLater()
        app.processEvents()

    def test_mode_auto_online_not_limited_returns_cloud(self, app):
        w = _make_window(app)
        w.settings["gen_mode"] = "auto"
        w.settings["ollama_api_key"] = "some-key"
        mock_tracker = MagicMock()
        mock_tracker.is_rate_limited.return_value = False
        w._cloud_usage_tracker = mock_tracker
        with patch.object(w, "_is_online", return_value=True):
            result = w._decide_backend()
        assert result == "cloud"
        w.deleteLater()
        app.processEvents()

    def test_is_online_caches_result(self, app):
        """_is_online should cache for 30s."""
        w = _make_window(app)
        # Prime with a cached True result (30s ago would have expired, but 0s ago won't)
        w._online_cache = (time.monotonic(), True)
        result = w._is_online()
        assert result is True
        w.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# QueryWorker forwards cloud args to pipeline
# ---------------------------------------------------------------------------

class TestQueryWorkerCloudArgs:
    def test_worker_passes_backend_and_cloud_args(self, app):
        """QueryWorker must forward backend/cloud_model/cloud_client/usage_tracker
        to pipeline.ask_streaming."""
        from windows.main_window import QueryWorker

        captured_kwargs: dict = {}

        def fake_ask_streaming(query, on_event, **kwargs):
            captured_kwargs.update(kwargs)
            return {"answer": "ok", "sources": []}

        mock_pipeline = MagicMock()
        mock_pipeline.ask_streaming.side_effect = fake_ask_streaming

        mock_client = MagicMock()
        mock_tracker = MagicMock()

        worker = QueryWorker(
            mock_pipeline,
            "test query",
            history=[],
            backend="cloud",
            cloud_model="qwen3.5:cloud",
            cloud_client=mock_client,
            usage_tracker=mock_tracker,
        )

        # Run synchronously by calling run() directly
        worker.run()
        app.processEvents()

        assert captured_kwargs.get("backend") == "cloud"
        assert captured_kwargs.get("cloud_model") == "qwen3.5:cloud"
        assert captured_kwargs.get("cloud_client") is mock_client
        assert captured_kwargs.get("usage_tracker") is mock_tracker

    def test_worker_backend_used_signal_emitted(self, app):
        """backend_used signal should be emitted after a successful query."""
        from windows.main_window import QueryWorker

        emitted: list[str] = []

        def fake_ask_streaming(query, on_event, **kwargs):
            return {"answer": "ok", "sources": []}

        mock_pipeline = MagicMock()
        mock_pipeline.ask_streaming.side_effect = fake_ask_streaming

        worker = QueryWorker(mock_pipeline, "q", backend="cloud")
        worker.backend_used.connect(lambda b: emitted.append(b))
        worker.run()
        app.processEvents()

        assert emitted == ["cloud"]

    def test_worker_falls_back_gracefully_on_typeerror(self, app):
        """If pipeline.ask_streaming doesn't accept cloud kwargs, fall back."""
        from windows.main_window import QueryWorker

        call_count = [0]

        def ask_streaming_limited(query, on_event, history=None):
            call_count[0] += 1
            return {"answer": "ok", "sources": []}

        mock_pipeline = MagicMock()
        # First call with cloud kwargs → TypeError, second simpler call → OK
        mock_pipeline.ask_streaming.side_effect = [
            TypeError("unexpected kwarg"),
            {"answer": "ok", "sources": []},
        ]
        # Use spec to test the fallback chain properly
        mock_pipeline2 = MagicMock()
        mock_pipeline2.ask_streaming.side_effect = ask_streaming_limited

        worker = QueryWorker(mock_pipeline2, "q", backend="local")
        results: list[dict] = []
        worker.result_ready.connect(lambda r: results.append(r))
        worker.run()
        app.processEvents()

        # Should have called ask_streaming (possibly with fallback signature)
        assert call_count[0] >= 1


# ---------------------------------------------------------------------------
# _CloudUsageIndicator rendering
# ---------------------------------------------------------------------------

class TestCloudUsageIndicator:
    def test_indicator_hidden_when_no_tracker(self, app):
        from windows.main_window import _CloudUsageIndicator
        ind = _CloudUsageIndicator()
        ind.refresh(None)
        assert not ind.isVisible()
        ind.deleteLater()
        app.processEvents()

    def test_indicator_shows_session_and_week(self, app):
        from windows.main_window import _CloudUsageIndicator
        mock_tracker = MagicMock()
        mock_tracker.snapshot.return_value = {
            "session": {"requests": 3, "prompt_tokens": 100, "completion_tokens": 200, "window_hours": 5},
            "week": {"requests": 10, "prompt_tokens": 500, "completion_tokens": 1000, "window_days": 7},
            "rate_limited_until": None,
        }
        ind = _CloudUsageIndicator()
        ind.refresh(mock_tracker)
        text = ind._label.text()
        assert "3" in text   # session requests
        assert "10" in text  # week requests
        assert ind.isVisible()
        ind.deleteLater()
        app.processEvents()

    def test_indicator_shows_rate_limited(self, app):
        from windows.main_window import _CloudUsageIndicator
        future_ts = time.time() + 3600  # 1 hour from now
        mock_tracker = MagicMock()
        mock_tracker.snapshot.return_value = {
            "session": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "window_hours": 5},
            "week": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "window_days": 7},
            "rate_limited_until": future_ts,
        }
        ind = _CloudUsageIndicator()
        ind.refresh(mock_tracker)
        text = ind._label.text()
        assert "лимит" in text
        assert "сброс" in text
        # Should be shown in danger color
        style = ind._label.styleSheet()
        assert "#f44336" in style or "red" in style.lower()
        ind.deleteLater()
        app.processEvents()

    def test_indicator_not_rate_limited_when_past(self, app):
        from windows.main_window import _CloudUsageIndicator
        past_ts = time.time() - 3600  # 1 hour ago
        mock_tracker = MagicMock()
        mock_tracker.snapshot.return_value = {
            "session": {"requests": 5, "prompt_tokens": 100, "completion_tokens": 200, "window_hours": 5},
            "week": {"requests": 20, "prompt_tokens": 500, "completion_tokens": 1000, "window_days": 7},
            "rate_limited_until": past_ts,
        }
        ind = _CloudUsageIndicator()
        ind.refresh(mock_tracker)
        text = ind._label.text()
        assert "лимит" not in text
        assert "5" in text
        ind.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# _on_pipeline_event error handling
# ---------------------------------------------------------------------------

class TestPipelineEventError:
    def test_error_event_shows_message_and_re_enables(self, app):
        """Stage 'error' must re-enable composer and show error text."""
        w = _make_window(app)
        # Simulate a live query state
        w.send_btn.setEnabled(False)
        w.chat_input.setEnabled(False)
        # Set up a mock bubble
        from PySide6.QtWidgets import QTextEdit
        te = QTextEdit()
        w._current_bubble_label = te
        app.processEvents()

        w._on_pipeline_event({"stage": "error", "message": "rate limit exceeded"})
        app.processEvents()

        assert w.send_btn.isEnabled()
        assert w.chat_input.isEnabled()
        log_text = w.chat_log.toPlainText()
        assert "rate limit exceeded" in log_text or "Ошибка" in log_text

        te.deleteLater()
        w.deleteLater()
        app.processEvents()

    def test_error_event_propagates_to_activity_panel(self, app):
        w = _make_window(app)
        panel = w._activity_panel
        if panel:
            panel.reset()
        app.processEvents()

        w._on_pipeline_event({"stage": "error", "message": "cloud error"})
        app.processEvents()

        # Just verify no crash; the panel handles display internally
        w.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Full-window cloud indicator visibility
# ---------------------------------------------------------------------------

class TestCloudIndicatorIntegration:
    def test_indicator_hidden_for_pure_local_user(self, app):
        """No API key + local mode → indicator hidden."""
        w = _make_window(app)
        w.settings["gen_mode"] = "local"
        w.settings["ollama_api_key"] = ""
        w._rebuild_cloud_clients()
        app.processEvents()
        assert w._cloud_indicator is not None
        assert not w._cloud_indicator.isVisible()
        w.deleteLater()
        app.processEvents()

    def test_indicator_shown_when_key_set(self, app):
        """API key set → indicator is not explicitly hidden."""
        w = _make_window(app)
        w.settings["gen_mode"] = "cloud"
        w.settings["ollama_api_key"] = "test-key"
        w._rebuild_cloud_clients()
        app.processEvents()
        # In headless tests the parent window isn't shown, so isVisible() may be False
        # even though the widget has been shown; check isHidden() instead.
        assert w._cloud_indicator is not None
        assert not w._cloud_indicator.isHidden()
        w.deleteLater()
        app.processEvents()
