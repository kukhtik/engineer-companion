#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${REPO}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: .venv not found. Run: uv venv .venv --python 3.11 && uv pip install -r requirements.txt" >&2
    exit 1
fi

# Suppress torch CPU warnings and speed up import on some systems
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

case "${1:-run}" in
    run)
        exec "$PYTHON" -m engineer_companion "${@:2}"
        ;;
    test)
        QT_QPA_PLATFORM=offscreen exec "$PYTHON" -m pytest tests/ -v "${@:2}"
        ;;
    index)
        exec "$PYTHON" -m core.indexer "${@:2}"
        ;;
    gui)
        exec "$PYTHON" -c "from windows.main_window import main; main()" "${@:2}"
        ;;
    build)
        exec "$PYTHON" scripts/build_windows.py "${@:2}"
        ;;
    *)
        echo "Usage: $0 {run|test|index|gui|build} [args...]"
        echo "  run   - Launch GUI (default)"
        echo "  test  - Run pytest suite"
        echo "  index - Rebuild vector DB from PDFs"
        echo "  gui   - Launch GUI directly"
        echo "  build - Build Windows EXE via PyInstaller"
        exit 1
        ;;
esac
