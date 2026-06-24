"""App-path resolver: writable user-data directory + first-run seeding.

In frozen (PyInstaller EXE) mode, the bundle is read-only; all mutable data
lives in a sibling directory ``EngineerCompanionData/`` next to the EXE.

In dev (source) mode every path resolves inside the repo so the existing
docs/ and assets/db/ continue to work unchanged.

Usage::

    from windows.app_paths import data_root, db_path, library_dir, ensure_seeded
    ensure_seeded()           # idempotent — call once at startup
    db = db_path()            # Path to writable engineer.db
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def _repo_root() -> Path:
    """Return the repository root (dev mode only)."""
    return Path(__file__).resolve().parents[1]


def _bundle_root() -> Path:
    """Return the ``_internal`` resources dir inside the PyInstaller bundle."""
    # PyInstaller sets sys._MEIPASS to the temporary extraction dir.
    mei = getattr(sys, "_MEIPASS", None)
    if mei:
        return Path(mei)
    # Fallback: sibling ``_internal`` next to the EXE (one-dir mode).
    return Path(sys.executable).parent / "_internal"


# ---------------------------------------------------------------------------
# Public path API
# ---------------------------------------------------------------------------

def data_root() -> Path:
    """Return the root writable data directory.

    - Frozen: ``<exe-dir>/EngineerCompanionData/``
    - Dev:    the repository root (so existing docs/ and assets/db/ are used
              directly without copying).
    """
    if _is_frozen():
        return Path(sys.executable).parent / "EngineerCompanionData"
    return _repo_root()


def library_dir() -> Path:
    """Directory where PDF files live (writable).

    - Frozen: ``data_root/library/``
    - Dev:    ``<repo>/docs/`` (the existing document folder)
    """
    if _is_frozen():
        return data_root() / "library"
    return _repo_root() / "docs"


def db_path() -> Path:
    """Path to the writable LanceDB store directory.

    - Frozen: ``data_root/db/engineer.db``
    - Dev:    ``<repo>/assets/db/engineer.db``
    """
    if _is_frozen():
        return data_root() / "db" / "engineer.db"
    return _repo_root() / "assets" / "db" / "engineer.db"


def ocr_cache_path() -> Path:
    """Path to the OCR cache JSONL file.

    - Frozen: ``data_root/ocr_cache.jsonl``
    - Dev:    ``<repo>/scripts/ocr_cache.jsonl``
    """
    if _is_frozen():
        return data_root() / "ocr_cache.jsonl"
    return _repo_root() / "scripts" / "ocr_cache.jsonl"


def models_dir() -> Path:
    """Read-only bundled models directory.

    Uses ``android.assets_loader.resolve_bundled_model`` convention:
    always points to ``assets/models/`` inside the bundle (or repo).
    This directory is NOT user-writable.
    """
    if _is_frozen():
        return _bundle_root() / "assets" / "models"
    return _repo_root() / "assets" / "models"


# ---------------------------------------------------------------------------
# First-run seeding
# ---------------------------------------------------------------------------

def ensure_seeded() -> None:
    """Create writable data directories and seed DB / PDFs on first frozen run.

    Safe to call every startup — all operations are idempotent:
    - directories are created only if missing
    - files are copied only if the destination does not already exist

    In dev mode this is a no-op (returns immediately).
    """
    if not _is_frozen():
        return  # dev: use repo files directly

    # ---- Ensure directory structure ----
    for d in (library_dir(), db_path().parent):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Seed DB: copy bundled engineer.db if writable copy absent ----
    dest_db = db_path()
    if not dest_db.exists():
        bundled_db = _bundle_root() / "assets" / "db" / "engineer.db"
        if bundled_db.exists():
            shutil.copytree(str(bundled_db), str(dest_db))

    # ---- Seed library PDFs: copy bundled docs/ into library_dir() ----
    lib = library_dir()
    bundled_docs = _bundle_root() / "docs"
    if bundled_docs.is_dir():
        for pdf in bundled_docs.glob("*.pdf"):
            dest_pdf = lib / pdf.name
            if not dest_pdf.exists():
                shutil.copy2(str(pdf), str(dest_pdf))
