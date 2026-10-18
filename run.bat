@echo off
rem Start the head: perception -> behavior -> neck, with the camera window.
rem   run.bat                    calm preset, moves the neck, shows the camera
rem   run.bat --preset curious   other presets: calm, curious, shy
rem   run.bat --dry-run          no servos: decide and print only
rem Keys in the window: q quit, c clear a neck fault. The neck rests and holds on exit.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set ARGS=%*
if "%ARGS%"=="" set ARGS=--preset calm --view
".venv\Scripts\python.exe" tools\behave.py %ARGS%
pause
