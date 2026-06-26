@echo off
rem Debug launcher — shows a console window with stderr so you can see errors.
rem For normal use (no console), use run_app.bat instead.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
".venv-win\Scripts\python.exe" engineer_companion.py
if errorlevel 1 (
  echo.
  echo App exited with an error. Press any key to close.
  pause >nul
)
