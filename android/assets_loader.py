"""Shared asset path resolver for Android.

Works across Kivy, Chaquopy, and standalone Python contexts.
Resolves paths to internal storage where APK assets are copied on first launch.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_android_files_dir() -> Path | None:
    """Return internal storage root on Android. Detects via Kivy or environment."""
    # Kivy provides ANDROID_ARGUMENT env var with app private dir
    kivy_env = os.environ.get("ANDROID_ARGUMENT")
    if kivy_env:
        return Path(kivy_env)

    # Chaquopy / standard Android: filesDir passed via env from Kotlin
    chaq_env = os.environ.get("ANDROID_FILES_DIR")
    if chaq_env:
        return Path(chaq_env)

    # Fallback: assume current working dir is repo root (for desktop dev)
    return None


def resolve_db_path(db_name: str = "engineer.db") -> Path:
    """Return absolute path to LanceDB directory."""
    android_dir = get_android_files_dir()
    if android_dir is not None:
        return android_dir / "db" / db_name
    # Desktop fallback: repo/assets/db/
    repo = Path(__file__).resolve().parents[1]
    return repo / "assets" / "db" / db_name


def resolve_model_path(model_name: str = "qwen2.5-7b-instruct-q4_k_m.gguf") -> Path | None:
    """Return absolute path to GGUF model if it exists, else None."""
    android_dir = get_android_files_dir()
    if android_dir is not None:
        p = android_dir / "models" / model_name
        return p if p.exists() else None
    repo = Path(__file__).resolve().parents[1]
    p = repo / "assets" / "models" / model_name
    return p if p.exists() else None


def resolve_embedding_model(model_name: str = "intfloat/multilingual-e5-small") -> str:
    """Return HuggingFace model name or local path.

    On Android with Chaquopy, pre-downloaded model dir should be copied to assets
    and referenced by absolute path instead of HF download (no network on device).
    """
    android_dir = get_android_files_dir()
    if android_dir is not None:
        local = android_dir / "models" / model_name.replace("/", "_")
        if local.exists():
            return str(local)
    return model_name


def ensure_assets(android_files_dir: Path | None = None) -> dict[str, Path | None]:
    """Diagnostic: verify db and model exist."""
    return {
        "db": resolve_db_path(),
        "llm_model": resolve_model_path(),
        "embedding_model": Path(resolve_embedding_model()) if "/" not in resolve_embedding_model() else None,
    }
