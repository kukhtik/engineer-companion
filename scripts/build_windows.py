"""Build script: Windows standalone EXE via PyInstaller.

Usage:
    python scripts/build_windows.py
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDES = ["torch", "torchvision", "torchaudio", "tensorflow", "keras"]  # keep binary small


def build() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Install: pip install pyinstaller")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "EngineerCompanion",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--hidden-import", "sentence_transformers",
        "--hidden-import", "llama_cpp",
        "--hidden-import", "lancedb",
        "--hidden-import", "PySide6",
        "--add-data", f"{REPO / 'assets'}:assets",
        "--add-data", f"{REPO / 'core'}:core",
        "--add-data", f"{REPO / 'design'}:design",
        "--add-data", f"{REPO / 'windows'}:windows",
        str(REPO / "windows" / "main_window.py"),
    ]
    for ex in EXCLUDES:
        cmd.extend(["--exclude-module", ex])

    subprocess.run(cmd, cwd=REPO, check=True)
    print("Build complete: dist/EngineerCompanion")


if __name__ == "__main__":
    build()
