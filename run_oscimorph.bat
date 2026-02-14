@echo off
color 0B
title Oscimorph v1.0
echo Initializing phase shift ...
setlocal
set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%app"
if not exist "%APP_DIR%" (
  echo Missing app directory: "%APP_DIR%"
  pause
  exit /b 1
)
cd /d "%APP_DIR%"
set "PYTHONPATH=%APP_DIR%\src"
if not exist "debug" mkdir "debug"
if not exist "temp" mkdir "temp"

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python not found. Run install_dependencies.bat first.
  pause
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python 3.11+ is required. Run install_dependencies.bat first.
  pause
  exit /b 1
)

python -c "import importlib.util,sys;mods=['numpy','cv2','PIL','librosa','soundfile','moviepy','imageio','imageio_ffmpeg','PySide6'];missing=[m for m in mods if importlib.util.find_spec(m) is None];raise SystemExit(0 if not missing else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Missing required Python packages. Run install_dependencies.bat first.
  pause
  exit /b 1
)

start "" pythonw -m oscimorph > "debug\\oscimorph_run.log" 2>&1
echo App launched. Logs: app\debug\oscimorph_run.log
timeout /t 2 > nul
endlocal
