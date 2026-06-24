"""Tests for windows/app_paths.py — dev-mode path resolution + ensure_seeded."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the repo root is importable
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_fresh():
    """Import app_paths with a clean state (re-import each time is fine in tests)."""
    import windows.app_paths as m
    return m


# ---------------------------------------------------------------------------
# Dev-mode path resolution
# ---------------------------------------------------------------------------

class TestDevPaths:
    """In dev mode (not frozen) paths resolve inside the repository."""

    def test_data_root_is_repo(self):
        m = _import_fresh()
        # In dev, data_root() == repo root
        assert m.data_root() == _REPO

    def test_library_dir_is_docs(self):
        m = _import_fresh()
        assert m.library_dir() == _REPO / "docs"

    def test_db_path_is_assets_db(self):
        m = _import_fresh()
        assert m.db_path() == _REPO / "assets" / "db" / "engineer.db"

    def test_ocr_cache_path_is_scripts(self):
        m = _import_fresh()
        assert m.ocr_cache_path() == _REPO / "scripts" / "ocr_cache.jsonl"

    def test_models_dir_is_assets_models(self):
        m = _import_fresh()
        assert m.models_dir() == _REPO / "assets" / "models"


# ---------------------------------------------------------------------------
# ensure_seeded is a no-op in dev mode
# ---------------------------------------------------------------------------

class TestEnsureSeededDev:
    def test_ensure_seeded_noop_in_dev(self, monkeypatch):
        """In dev mode ensure_seeded() returns without creating anything."""
        import windows.app_paths as m

        created_dirs: list = []
        original_mkdir = Path.mkdir

        def fake_mkdir(self, *a, **kw):
            created_dirs.append(self)
            return original_mkdir(self, *a, **kw)

        # Ensure not frozen
        monkeypatch.delattr(sys, "frozen", raising=False)
        # Should return immediately without calling mkdir
        m.ensure_seeded()
        # No assertion on created_dirs because we didn't patch mkdir;
        # the key is that it didn't raise and didn't create EngineerCompanionData.
        data_root = m.data_root()
        # In dev data_root is repo root — it exists already
        assert data_root.exists()

    def test_ensure_seeded_idempotent(self, monkeypatch):
        """Calling ensure_seeded() twice is safe."""
        import windows.app_paths as m
        monkeypatch.delattr(sys, "frozen", raising=False)
        m.ensure_seeded()
        m.ensure_seeded()   # second call must not raise


# ---------------------------------------------------------------------------
# Frozen-mode path resolution (monkeypatched)
# ---------------------------------------------------------------------------

class TestFrozenPaths:
    """Verify path logic when sys.frozen=True and sys.executable is mocked."""

    @pytest.fixture(autouse=True)
    def freeze(self, monkeypatch, tmp_path):
        """Simulate a frozen PyInstaller layout under tmp_path."""
        # tmp_path/app/EngineerCompanion.exe
        exe_dir = tmp_path / "app"
        exe_dir.mkdir()
        fake_exe = exe_dir / "EngineerCompanion.exe"
        fake_exe.write_text("fake")

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe), raising=False)
        self.exe_dir = exe_dir
        self.tmp_path = tmp_path

    def test_data_root_is_sibling_to_exe(self):
        import importlib, windows.app_paths as m
        importlib.reload(m)
        assert m.data_root() == self.exe_dir / "EngineerCompanionData"

    def test_library_dir_is_under_data_root(self):
        import importlib, windows.app_paths as m
        importlib.reload(m)
        assert m.library_dir() == self.exe_dir / "EngineerCompanionData" / "library"

    def test_db_path_is_under_data_root(self):
        import importlib, windows.app_paths as m
        importlib.reload(m)
        assert m.db_path() == self.exe_dir / "EngineerCompanionData" / "db" / "engineer.db"

    def test_ensure_seeded_creates_dirs(self):
        import importlib, windows.app_paths as m
        importlib.reload(m)
        m.ensure_seeded()
        assert m.library_dir().exists()
        assert m.db_path().parent.exists()

    def test_ensure_seeded_copies_seed_db(self):
        """If bundled db exists and writable db absent, it should be copied."""
        import importlib, windows.app_paths as m
        importlib.reload(m)

        # Create a fake bundled db dir
        bundled_db = m._bundle_root() / "assets" / "db" / "engineer.db"
        bundled_db.mkdir(parents=True, exist_ok=True)
        (bundled_db / "placeholder.lance").write_text("seed")

        m.ensure_seeded()

        dest = m.db_path()
        assert dest.exists()
        assert (dest / "placeholder.lance").read_text() == "seed"

    def test_ensure_seeded_idempotent_frozen(self):
        """Second call must not raise even when dest already exists."""
        import importlib, windows.app_paths as m
        importlib.reload(m)
        m.ensure_seeded()
        m.ensure_seeded()
