@echo off
setlocal enabledelayedexpansion
REM ========================================
REM  CapNF Auto-Update Dashboard Service
REM ========================================

color 0B
title CapNF Auto-Update Service

cd /d "%~dp0"

REM Auto-detect Python installation
set "PYTHON_EXE="

REM Check common Anaconda locations first (most reliable)
if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
        goto :found_python
    )
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    "%USERPROFILE%\miniconda3\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
        goto :found_python
    )
)
if exist "C:\ProgramData\anaconda3\python.exe" (
    "C:\ProgramData\anaconda3\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
        goto :found_python
    )
)
if exist "C:\ProgramData\miniconda3\python.exe" (
    "C:\ProgramData\miniconda3\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=C:\ProgramData\miniconda3\python.exe"
        goto :found_python
    )
)

REM Check standard Python installations
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        goto :found_python
    )
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        goto :found_python
    )
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        goto :found_python
    )
)
if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        goto :found_python
    )
)

REM Try to find python in PATH (skip Windows Store stub)
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "TEMP_PYTHON=%%i"
    REM Skip Windows Store stub
    echo !TEMP_PYTHON! | findstr /i "WindowsApps" >nul
    if !errorlevel! neq 0 (
        "!TEMP_PYTHON!" --version >nul 2>&1
        if !errorlevel! == 0 (
            set "PYTHON_EXE=!TEMP_PYTHON!"
            goto :found_python
        )
    )
)

REM If Python not found
echo.
echo ❌ Python not found!
echo.
echo Please install Python or Anaconda.
echo Download from:
echo   - Anaconda: https://www.anaconda.com/download
echo   - Python: https://www.python.org/downloads/
echo.
pause
exit /b 1

:found_python

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║        CapNF Auto-Update Dashboard Service          ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  📊 Monitoring: Data\ folder
echo  🌐 Dashboard: https://Almo1990.github.io/capnf-dashboard/
echo.
echo  💡 How it works:
echo     • Drop new .tsv files into the Data\ folder
echo     • Pipeline runs automatically (takes 2-5 minutes)
echo     • Dashboard updates online automatically
echo.
echo  ⚠️  Keep this window OPEN for monitoring to continue
echo     Press Ctrl+C to stop
echo.
echo ══════════════════════════════════════════════════════════
echo.
echo ✅ Using Python: %PYTHON_EXE%
echo.

REM Start the auto-update service
"%PYTHON_EXE%" "%~dp0auto_update_dashboard.py"

echo.
echo Service stopped.
pause
