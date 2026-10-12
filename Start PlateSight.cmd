@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup.ps1 first to install PlateSight.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py --open-browser
if errorlevel 1 pause
