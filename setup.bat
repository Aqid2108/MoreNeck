@echo off
rem One-click setup: double-click or run from a terminal in this folder.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Setup FAILED - see the messages above.
)
pause
