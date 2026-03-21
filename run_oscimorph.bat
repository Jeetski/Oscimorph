@echo off
color 0B
title Oscimorph alpha v0.1
echo Initializing phase shift ...
setlocal
set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%app"
set "VENV_DIR=%APP_DIR%\.venv"
set "VENV_PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
if not exist "%APP_DIR%" (
  echo Missing app directory: "%APP_DIR%"
  pause
  exit /b 1
)
cd /d "%APP_DIR%"
set "PYTHONPATH=%APP_DIR%\src"
if not exist "debug" mkdir "debug"
if not exist "temp" mkdir "temp"

if not exist "%VENV_PYTHONW%" (
  echo Project virtual environment not found. Run install_dependencies.bat first.
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ffmpeg not found on PATH. Run install_dependencies.bat first.
  pause
  exit /b 1
)

"%VENV_PYTHONW%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Project virtual environment is invalid. Run install_dependencies.bat first.
  pause
  exit /b 1
)

"%VENV_PYTHONW%" -c "import importlib.util,sys;mods=['numpy','cv2','PIL','librosa','soundfile','moviepy','imageio','imageio_ffmpeg','PySide6'];missing=[m for m in mods if importlib.util.find_spec(m) is None];raise SystemExit(0 if not missing else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Missing required packages in project virtual environment. Run install_dependencies.bat first.
  pause
  exit /b 1
)

start "" "%VENV_PYTHONW%" -m oscimorph > "debug\\oscimorph_run.log" 2>&1
echo App launched. Logs: app\debug\oscimorph_run.log
timeout /t 2 > nul
endlocal
