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
  echo Python not found. Attempting install via winget...
  where winget >nul 2>&1
  if %ERRORLEVEL% NEQ 0 (
    echo winget not available. Please install Python 3.11+ and rerun.
    pause
    exit /b 1
  )
  winget install -e --id Python.Python.3.11
  if %ERRORLEVEL% NEQ 0 (
    echo Python install failed. Please install Python 3.11+ manually.
    pause
    exit /b 1
  )
)

python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r "%APP_DIR%\requirements.txt"
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Failed to install requirements.
  pause
  exit /b 1
)

start "" pythonw -m oscimorph > "debug\\oscimorph_run.log" 2>&1
echo App launched. Logs: app\debug\oscimorph_run.log
timeout /t 2 > nul
endlocal
