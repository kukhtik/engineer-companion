"""Build script: Windows standalone EXE via PyInstaller.

Usage:
    python scripts/build_windows.py
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDES = ["torchvision", "torchaudio", "tensorflow", "keras"]  # torch stays — sentence-transformers needs it


def build() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Install: pip install pyinstaller")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "EngineerCompanion",
        "--onedir",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--hidden-import", "sentence_transformers",
        "--hidden-import", "llama_cpp",
        "--hidden-import", "lancedb",
        "--hidden-import", "PySide6",
        "--hidden-import", "structlog",
        "--hidden-import", "pymupdf",
        "--hidden-import", "numpy",
        "--add-data", f"{REPO / 'assets'}:assets",
        "--add-data", f"{REPO / 'core'}:core",
        "--add-data", f"{REPO / 'design'}:design",
        "--add-data", f"{REPO / 'windows'}:windows",
        "--add-data", f"{REPO / 'docs'}:docs",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        str(REPO / "windows" / "main_window.py"),
    ]
    for ex in EXCLUDES:
        cmd.extend(["--exclude-module", ex])

    subprocess.run(cmd, cwd=REPO, check=True)
    print("Build complete: dist/EngineerCompanion")


if __name__ == "__main__":
    build()
