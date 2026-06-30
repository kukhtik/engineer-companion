"""Tests for Phase 4 tabbed SettingsDialog.

Headless (offscreen) — no real models loaded.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

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


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestSettingsDialogStructure:
    def test_dialog_builds_with_5_tabs(self, app):
        dlg = _make_dlg(app)
        assert dlg.tabs.count() == 5
        dlg.deleteLater()
        app.processEvents()

    def test_tab_labels_are_russian(self, app):
        dlg = _make_dlg(app)
        labels = [dlg.tabs.tabText(i) for i in range(5)]
        assert any("Библиотека" in l for l in labels)
        assert any("Качество" in l for l in labels)
        assert any("Внешний" in l or "вид" in l.lower() for l in labels)
        assert any("Модел" in l for l in labels)
        assert any("Облако" in l or "Ollama" in l for l in labels)
        dlg.deleteLater()
        app.processEvents()

    def test_required_widgets_exist(self, app):
        dlg = _make_dlg(app)
        # Tab 2 widgets
        assert hasattr(dlg, "top_k_spin")
        assert hasattr(dlg, "rerank_cb")
        assert hasattr(dlg, "rerank_top_k_spin")
        assert hasattr(dlg, "temperature_spin")
        assert hasattr(dlg, "max_tokens_spin")
        assert hasattr(dlg, "n_ctx_spin")
        assert hasattr(dlg, "n_threads_spin")
        # Tab 3 widgets
        assert hasattr(dlg, "theme_combo")
        assert hasattr(dlg, "font_scale_spin")
        assert hasattr(dlg, "density_combo")
        # Tab 4 widgets
        assert hasattr(dlg, "llm_path_edit")
        # Tab 5 widgets
        assert hasattr(dlg, "gen_mode_combo")
        assert hasattr(dlg, "cloud_api_key_edit")
        assert hasattr(dlg, "cloud_model_combo")
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Rerank toggle
# ---------------------------------------------------------------------------

class TestRerankToggle:
    def test_rerank_off_produces_empty_rerank_model(self, app):
        """When rerank is disabled, get_settings() should have rerank_enabled=False."""
        dlg = _make_dlg(app, {"rerank_enabled": True})
        dlg.rerank_cb.setChecked(False)
        # Simulate accept
        dlg._on_accept()
        s = dlg.get_settings()
        assert s["rerank_enabled"] is False
        dlg.deleteLater()
        app.processEvents()

    def test_rerank_on_produces_enabled_true(self, app):
        dlg = _make_dlg(app, {"rerank_enabled": False})
        dlg.rerank_cb.setChecked(True)
        dlg._on_accept()
        s = dlg.get_settings()
        assert s["rerank_enabled"] is True
        dlg.deleteLater()
        app.processEvents()

    def test_rerank_top_k_disabled_when_rerank_off(self, app):
        dlg = _make_dlg(app)
        dlg.rerank_cb.setChecked(False)
        app.processEvents()
        assert not dlg.rerank_top_k_spin.isEnabled()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Pipeline wiring (monkeypatched)
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    def test_n_ctx_and_n_threads_flow_into_pipeline(self, app):
        """Verify that _rebuild_pipeline passes n_ctx/n_threads from settings."""
        from windows.main_window import CompanionWindow
        from windows.settings_dialog import DEFAULT_SETTINGS

        captured = {}

        def fake_pipeline(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        settings = dict(DEFAULT_SETTINGS)
        settings.update({
            "llm_model_path": "/fake/model.gguf",
            "n_ctx": 4096,
            "n_threads": 8,
            "rerank_enabled": False,
        })

        w = CompanionWindow(pipeline=None)
        w.settings = settings
        app.processEvents()

        with patch("windows.main_window.RAGQueryPipeline") as MockPipeline, \
             patch("pathlib.Path.exists", return_value=True):
            MockPipeline.side_effect = lambda **kw: MagicMock()
            # Call _rebuild_pipeline and capture kwargs
            call_kwargs = {}

            def capture(**kw):
                call_kwargs.update(kw)
                return MagicMock()

            MockPipeline.side_effect = capture
            w._rebuild_pipeline()

        assert call_kwargs.get("llm_n_ctx") == 4096
        assert call_kwargs.get("llm_n_threads") == 8
        w.deleteLater()
        app.processEvents()

    def test_rerank_disabled_passes_empty_rerank_model(self, app):
        """When rerank_enabled=False, _rebuild_pipeline passes rerank_model=''."""
        from windows.main_window import CompanionWindow
        from windows.settings_dialog import DEFAULT_SETTINGS

        settings = dict(DEFAULT_SETTINGS)
        settings.update({
            "llm_model_path": "/fake/model.gguf",
            "rerank_enabled": False,
        })

        w = CompanionWindow(pipeline=None)
        w.settings = settings
        app.processEvents()

        call_kwargs = {}

        with patch("windows.main_window.RAGQueryPipeline") as MockPipeline, \
             patch("pathlib.Path.exists", return_value=True):
            def capture(**kw):
                call_kwargs.update(kw)
                return MagicMock()
            MockPipeline.side_effect = capture
            w._rebuild_pipeline()

        assert call_kwargs.get("rerank_model") == ""
        w.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Theme in dialog
# ---------------------------------------------------------------------------

class TestThemeInDialog:
    def test_theme_combo_has_dark_and_light(self, app):
        dlg = _make_dlg(app)
        items = [dlg.theme_combo.itemData(i) for i in range(dlg.theme_combo.count())]
        assert "dark" in items
        assert "light" in items
        dlg.deleteLater()
        app.processEvents()

    def test_theme_persists_to_settings(self, app):
        dlg = _make_dlg(app, {"theme": "dark"})
        # Switch to light
        idx = dlg.theme_combo.findData("light")
        dlg.theme_combo.blockSignals(True)
        dlg.theme_combo.setCurrentIndex(idx)
        dlg.theme_combo.blockSignals(False)
        dlg._on_accept()
        assert dlg.get_settings()["theme"] == "light"
        dlg.deleteLater()
        app.processEvents()

    def test_changing_theme_calls_set_theme(self, app):
        mock_tm = MagicMock()
        from windows.settings_dialog import SettingsDialog, DEFAULT_SETTINGS
        s = dict(DEFAULT_SETTINGS)
        dlg = SettingsDialog(None, current=s, theme_manager=mock_tm)
        app.processEvents()

        # Trigger live change
        idx = dlg.theme_combo.findData("light")
        dlg.theme_combo.setCurrentIndex(idx)
        app.processEvents()

        mock_tm.set_theme.assert_called()
        dlg.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Font scale persistence
# ---------------------------------------------------------------------------

class TestFontScalePersistence:
    def test_font_scale_round_trips(self, app, tmp_path):
        """font_scale saves to JSON and reloads correctly."""
        from windows.settings_dialog import save_settings, load_settings, SETTINGS_PATH
        import windows.settings_dialog as sd

        orig_path = sd.SETTINGS_PATH
        test_path = tmp_path / "settings.json"
        sd.SETTINGS_PATH = test_path
        try:
            save_settings({"font_scale": 120, "theme": "dark"})
            loaded = load_settings()
            assert loaded["font_scale"] == 120
        finally:
            sd.SETTINGS_PATH = orig_path

    def test_load_settings_fills_missing_keys_with_defaults(self, app, tmp_path):
        from windows.settings_dialog import load_settings, DEFAULT_SETTINGS
        import windows.settings_dialog as sd

        orig_path = sd.SETTINGS_PATH
        test_path = tmp_path / "settings.json"
        # Write old-style settings without new keys
        test_path.write_text(json.dumps({"db_path": "", "llm_model_path": ""}), encoding="utf-8")
        sd.SETTINGS_PATH = test_path
        try:
            loaded = load_settings()
            # Should have all new keys with defaults
            assert "n_ctx" in loaded
            assert loaded["n_ctx"] == DEFAULT_SETTINGS["n_ctx"]
            assert "font_scale" in loaded
            assert "density" in loaded
        finally:
            sd.SETTINGS_PATH = orig_path


# ---------------------------------------------------------------------------
# apply_font_scale and apply_density smoke tests
# ---------------------------------------------------------------------------

class TestApplyHelpers:
    def test_apply_font_scale_changes_font(self, app):
        from windows.settings_dialog import apply_font_scale
        original_size = app.font().pointSize()
        apply_font_scale(app, 120)
        new_size = app.font().pointSize()
        # 120% of 10pt = 12pt
        assert new_size == 12
        # Restore
        apply_font_scale(app, 100)

    def test_apply_density_compact_injects_qss(self, app):
        from windows.settings_dialog import apply_density
        apply_density(app, "compact")
        sheet = app.styleSheet()
        assert "padding" in sheet.lower() or "density" in sheet.lower()

    def test_apply_density_comfortable_injects_qss(self, app):
        from windows.settings_dialog import apply_density
        apply_density(app, "comfortable")
        sheet = app.styleSheet()
        assert "QListWidget" in sheet
