@echo off
cd /d E:\engineer-companion
echo Installing missing packages...
pip install llama-cpp-python sentence-transformers
echo.
echo Building EXE...
python scripts/build_windows.py --clean
pause
