"""Launch the app for 15 seconds and check for QPainter errors in stderr.

Usage:
    python scripts/check_qpainter_errors.py
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv-win" / "Scripts" / "python.exe"
APP = REPO / "engineer_companion.py"
STDERR_FILE = REPO / "scripts" / "_stderr_check.txt"

print("Launching app (no-LLM mode) for 15 seconds to capture stderr...")
env = {"PYTHONPATH": str(REPO), "QT_QPA_PLATFORM": "windows"}

proc = subprocess.Popen(
    [str(PYTHON), str(APP), "--no-llm"],
    stderr=open(STDERR_FILE, "w", encoding="utf-8"),
    stdout=subprocess.DEVNULL,
    env={**__import__("os").environ, **env},
)

time.sleep(15)
proc.terminate()
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()

# Check for QPainter errors
text = STDERR_FILE.read_text(encoding="utf-8", errors="replace")
qpainter_lines = [l for l in text.splitlines()
                  if "QPainter" in l or "Painter not active" in l or "paint device" in l.lower()]

print(f"\nTotal stderr lines: {len(text.splitlines())}")
if qpainter_lines:
    print(f"\n*** FOUND {len(qpainter_lines)} QPainter error lines: ***")
    for l in qpainter_lines[:20]:
        print(f"  {l}")
    sys.exit(1)
else:
    print("\nCLEAN: zero QPainter / 'Painter not active' lines in stderr.")
    sys.exit(0)
