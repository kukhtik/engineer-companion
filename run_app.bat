@echo off
rem Double-click launcher for the Engineer Companion desktop app (from source).
rem Uses the native-Windows venv .venv-win and the project root as PYTHONPATH.
rem Runs via pythonw.exe so NO console window appears for normal use.
rem For debugging (to see stderr), use run_app_debug.bat instead.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
start "" ".venv-win\Scripts\pythonw.exe" engineer_companion.py
