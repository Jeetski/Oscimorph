@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Oscimorph Dependency Installer
color 0B

set "ROOT_DIR=%~dp0"
set "APP_DIR=%ROOT_DIR%app"
set "REQ_FILE=%APP_DIR%\requirements.txt"
set "OUTDATED_FILE=%TEMP%\oscimorph_outdated.txt"

if not exist "%APP_DIR%" (
  echo Missing app directory: "%APP_DIR%"
  exit /b 1
)
if not exist "%REQ_FILE%" (
  echo Missing requirements file: "%REQ_FILE%"
  exit /b 1
)

set "NEED_PY_INSTALL=0"
set "NEED_PY_UPGRADE=0"
set "CAN_AUTO_PY=0"
set "NEED_PIP_PACKAGES=0"
set "NEED_FFMPEG=0"
set "CAN_AUTO_FFMPEG=0"
set "CHECKS_FAILED=0"
set "OUTDATED_CHECK_OK=0"
set "MISSING_PKGS="
set "OUTDATED_PKGS="
set "MISSING_SET="
set "OUTDATED_SET="
set "PY_STATUS=satisfied"
set "PIP_STATUS=satisfied"
set "FFMPEG_STATUS=satisfied"

if exist "%OUTDATED_FILE%" del "%OUTDATED_FILE%" >nul 2>&1

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  set "NEED_PY_INSTALL=1"
  set "PY_STATUS=missing"
  set "PIP_STATUS=unknown (python missing)"
  where winget >nul 2>&1
  if %ERRORLEVEL% EQU 0 set "CAN_AUTO_PY=1"
) else (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
  if %ERRORLEVEL% NEQ 0 (
    set "NEED_PY_UPGRADE=1"
    set "PY_STATUS=outdated"
    set "PIP_STATUS=unknown (python outdated)"
    where winget >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "CAN_AUTO_PY=1"
  )
)

if %NEED_PY_INSTALL% EQU 0 if %NEED_PY_UPGRADE% EQU 0 (
  python -m pip --version >nul 2>&1
  if %ERRORLEVEL% NEQ 0 (
    set "CHECKS_FAILED=1"
    set "PIP_STATUS=missing"
  ) else (
    python -m pip list --outdated --format=freeze > "%OUTDATED_FILE%" 2>nul
    set "OUTDATED_CHECK_OK=1"
    if %ERRORLEVEL% NEQ 0 set "OUTDATED_CHECK_OK=0"

    for /f "usebackq tokens=1 delims==<>~! " %%P in (`findstr /R /V /C:"^[ ]*$" /C:"^[ ]*#" "%REQ_FILE%"`) do (
      python -m pip show "%%P" >nul 2>&1
      if !ERRORLEVEL! NEQ 0 (
        set "MISSING_PKGS=!MISSING_PKGS! %%P"
        set "MISSING_SET=!MISSING_SET!|%%P|"
        set "NEED_PIP_PACKAGES=1"
      ) else (
        if "!OUTDATED_CHECK_OK!"=="1" (
          findstr /I /R "^%%P==" "%OUTDATED_FILE%" >nul 2>&1
          if !ERRORLEVEL! EQU 0 (
            set "OUTDATED_PKGS=!OUTDATED_PKGS! %%P"
            set "OUTDATED_SET=!OUTDATED_SET!|%%P|"
            set "NEED_PIP_PACKAGES=1"
          )
        )
      )
    )
  )
)

where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  set "NEED_FFMPEG=1"
  set "FFMPEG_STATUS=missing"
  where winget >nul 2>&1
  if %ERRORLEVEL% EQU 0 set "CAN_AUTO_FFMPEG=1"
)

echo.
echo Dependency check summary:
echo -------------------------
echo - Python 3.11+: %PY_STATUS%
if %NEED_PY_INSTALL% EQU 1 if %CAN_AUTO_PY% EQU 1 echo   Action available: install via winget
if %NEED_PY_UPGRADE% EQU 1 if %CAN_AUTO_PY% EQU 1 echo   Action available: upgrade via winget
if %NEED_PY_INSTALL% EQU 1 if %CAN_AUTO_PY% EQU 0 echo   Action required: manual install
if %NEED_PY_UPGRADE% EQU 1 if %CAN_AUTO_PY% EQU 0 echo   Action required: manual upgrade

echo - pip: %PIP_STATUS%
if %CHECKS_FAILED% EQU 1 echo   Action required: repair pip for current Python install

echo - ffmpeg: %FFMPEG_STATUS%
if %NEED_FFMPEG% EQU 1 if %CAN_AUTO_FFMPEG% EQU 1 echo   Action available: install via winget
if %NEED_FFMPEG% EQU 1 if %CAN_AUTO_FFMPEG% EQU 0 echo   Action required: manual install

echo - Python packages:
set "ANY_PKG_ISSUE=0"
for /f "usebackq tokens=1 delims==<>~! " %%P in (`findstr /R /V /C:"^[ ]*$" /C:"^[ ]*#" "%REQ_FILE%"`) do (
  set "PKG_STATE=unknown (python/pip unavailable)"
  if %NEED_PY_INSTALL% EQU 0 if %NEED_PY_UPGRADE% EQU 0 if %CHECKS_FAILED% EQU 0 (
    python -m pip show "%%P" >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
      set "PKG_STATE=missing"
      set "ANY_PKG_ISSUE=1"
    ) else (
      set "PKG_STATE=satisfied"
      if "!OUTDATED_CHECK_OK!"=="1" (
        findstr /I /R "^%%P==" "%OUTDATED_FILE%" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
          set "PKG_STATE=outdated"
          set "ANY_PKG_ISSUE=1"
        )
      )
    )
  )
  echo   - %%P: !PKG_STATE!
)

set "ALL_OK=1"
if not "%NEED_PY_INSTALL%"=="0" set "ALL_OK=0"
if not "%NEED_PY_UPGRADE%"=="0" set "ALL_OK=0"
if not "%CHECKS_FAILED%"=="0" set "ALL_OK=0"
if not "%ANY_PKG_ISSUE%"=="0" set "ALL_OK=0"
if not "%NEED_FFMPEG%"=="0" set "ALL_OK=0"

if "%ALL_OK%"=="1" (
  echo - Everything required is already installed and up to date.
  if exist "%OUTDATED_FILE%" del "%OUTDATED_FILE%" >nul 2>&1
  exit /b 0
)

echo.
choice /C YN /M "Proceed with install/update tasks?"
if errorlevel 2 (
  echo Cancelled by user.
  if exist "%OUTDATED_FILE%" del "%OUTDATED_FILE%" >nul 2>&1
  exit /b 1
)

if %NEED_PY_INSTALL% EQU 1 (
  if %CAN_AUTO_PY% EQU 1 (
    winget install -e --id Python.Python.3.11
  ) else (
    echo Python 3.11+ must be installed manually, then rerun this installer.
    exit /b 1
  )
)

if %NEED_PY_UPGRADE% EQU 1 (
  if %CAN_AUTO_PY% EQU 1 (
    winget upgrade -e --id Python.Python.3.11
  ) else (
    echo Python 3.11+ upgrade must be done manually, then rerun this installer.
    exit /b 1
  )
)

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python still not available on PATH.
  exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Python version is still below 3.11.
  exit /b 1
)

python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
  echo Failed to upgrade pip.
  exit /b 1
)

if defined MISSING_PKGS (
  set "PKGS_TO_INSTALL=%MISSING_PKGS%"
)
if defined OUTDATED_PKGS (
  set "PKGS_TO_INSTALL=%PKGS_TO_INSTALL% %OUTDATED_PKGS%"
)
if not defined PKGS_TO_INSTALL (
  for /f "usebackq tokens=1 delims==<>~! " %%P in (`findstr /R /V /C:"^[ ]*$" /C:"^[ ]*#" "%REQ_FILE%"`) do (
    set "PKGS_TO_INSTALL=!PKGS_TO_INSTALL! %%P"
  )
)

if defined PKGS_TO_INSTALL (
  python -m pip install --upgrade %PKGS_TO_INSTALL%
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to install/update required Python packages.
    exit /b 1
  )
)

if %NEED_FFMPEG% EQU 1 (
  if %CAN_AUTO_FFMPEG% EQU 1 (
    winget install -e --id Gyan.FFmpeg
    if %ERRORLEVEL% NEQ 0 (
      echo ffmpeg install failed. Install manually if render export fails.
      exit /b 1
    )
  ) else (
    echo ffmpeg must be installed manually and added to PATH.
    exit /b 1
  )
)

if exist "%OUTDATED_FILE%" del "%OUTDATED_FILE%" >nul 2>&1
echo.
echo Dependency install/update completed.
exit /b 0
