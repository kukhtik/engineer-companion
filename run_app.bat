@echo off
rem Double-click launcher for the Engineer Companion desktop app (from source).
rem Uses the native-Windows venv .venv-win and the project root as PYTHONPATH.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
".venv-win\Scripts\python.exe" engineer_companion.py
if errorlevel 1 (
  echo.
  echo App exited with an error. Press any key to close.
  pause >nul
)
