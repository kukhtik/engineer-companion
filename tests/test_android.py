"""Unit tests for Android-specific wrapper code.

These test the Python-side Android helpers using plain Python (no emulator).
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pytest

from android.assets_loader import (
    get_android_files_dir,
    resolve_db_path,
    resolve_model_path,
    resolve_embedding_model,
    ensure_assets,
)


class TestAssetsLoader:
    def test_get_android_files_dir_none_on_desktop(self):
        assert get_android_files_dir() is None

    def test_get_android_files_dir_uses_kivy_env(self, monkeypatch):
        monkeypatch.setenv("ANDROID_ARGUMENT", "/data/data/com.varian.engcomp/files")
        result = get_android_files_dir()
        assert result is not None
        assert "varian.engcomp" in str(result)

    def test_get_android_files_dir_uses_chaquopy_env(self, monkeypatch):
        monkeypatch.setenv("ANDROID_FILES_DIR", "/data/local/tmp")
        result = get_android_files_dir()
        # Compare parts individually so assertion passes on Windows (backslash) and Linux (forward-slash)
        assert result is not None
        assert result.parts[-3:] == ("data", "local", "tmp")

    def test_resolve_db_path_desktop_fallback(self):
        path = resolve_db_path("engineer.db")
        # Use Path comparison to be OS-agnostic (avoids forward- vs back-slash mismatch)
        assert path.parts[-3:] == ("assets", "db", "engineer.db")

    def test_resolve_model_path_none_when_missing(self):
        path = resolve_model_path("nonexistent.gguf")
        assert path is None

    def test_resolve_model_path_with_env(self, monkeypatch, tmp_path):
        test_file = tmp_path / "models" / "test.gguf"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("dummy")
        monkeypatch.setenv("ANDROID_FILES_DIR", str(tmp_path))
        path = resolve_model_path("test.gguf")
        assert path is not None
        assert path.name == "test.gguf"

    def test_resolve_embedding_model_default(self):
        name = resolve_embedding_model()
        assert name == "intfloat/multilingual-e5-small"

    def test_ensure_assets_returns_dict_with_keys(self):
        result = ensure_assets()
        assert "db" in result
        assert "llm_model" in result

    def test_ensure_assets_db_path_is_pathlike(self):
        result = ensure_assets()
        assert result["db"] is None or isinstance(result["db"], Path)


class TestChaquopyBridgeSearch:
    """Test chaquopy_bridge functions without real DB."""

    def test_search_requires_init(self):
        """Calling search without _ensure_init should fail gracefully."""
        from android.chaquopy.chaquopy_bridge import search
        result = search("test", "/nonexistent")
        assert "\"ok\": false" in result or "error" in result

    def test_search_returns_json_string(self):
        from android.chaquopy.chaquopy_bridge import search
        result = search("test", "/nonexistent/db")
        assert isinstance(result, str)

    def test_search_error_contains_error_key(self):
        from android.chaquopy.chaquopy_bridge import search
        result = search("test", "/dev/null/db")
        assert "false" in result or "error" in result


class TestKivyAppHelpers:
    """Test functions from main.py that don't require Kivy runtime."""

    def test_main_module_imports(self):
        """Verify android/main.py can be parsed without Kivy runtime error."""
        import ast
        with open(Path(__file__).resolve().parents[1] / "android" / "main.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        # Check it's valid AST — no syntax errors
        assert tree.body, "main.py has parseable AST"
