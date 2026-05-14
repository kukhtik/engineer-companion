"""Build script: Windows standalone EXE via PyInstaller.

Run on Windows:
    python scripts/build_windows.py

Prerequisites:
    pip install pyinstaller PySide6 sentence-transformers llama-cpp-python lancedb
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EXCLUDES = [
    "torchvision", "torchaudio", "tensorflow", "keras",
    "matplotlib", "scipy", "pandas", "scikit_learn",
    "Cython", "setuptools", "pip", "wheel",
]

HIDDEN_IMPORTS = [
    "sentence_transformers",
    "llama_cpp",
    "lancedb",
    "PySide6",
    "structlog",
    "pymupdf",
    "numpy",
    "pyarrow",
    "pydantic",
    "tiktoken",
    "core.query",
    "core.indexer",
    "windows.main_window",
    "windows.settings_dialog",
    "design.tokens",
]

DATA_DIRS = ["assets", "core", "design", "windows", "docs"]


def build() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Install PyInstaller: pip install pyinstaller")
        sys.exit(1)

    cmd = [
        str(PYTHON), "-m", "PyInstaller",
        "--name", "EngineerCompanion",
        "--onedir",
        "--windowed",
        "--clean",
        "--noconfirm",
    ]

    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    for mod in EXCLUDES:
        cmd.extend(["--exclude-module", mod])

    for d in DATA_DIRS:
        src = REPO / d
        cmd.extend(["--add-data", f"{src}:{d}"])

    cmd.append(str(REPO / "windows" / "main_window.py"))

    print("Running PyInstaller...")
    print("  Command:", " ".join(str(c) for c in cmd))
    print()
    subprocess.run(cmd, cwd=REPO, check=True)

    out_dir = REPO / "dist" / "EngineerCompanion"
    print(f"\nBuild complete: {out_dir}")
    print("\nTo test:")
    print(f"  {out_dir / 'EngineerCompanion.exe'}")


if __name__ == "__main__":
    build()
