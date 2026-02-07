@echo off
setlocal
REM ========================================
REM  CapNF Auto-Update Dashboard Service
REM ========================================

color 0B
title CapNF Auto-Update Service

cd /d "%~dp0"

REM Set Python executable
set "PYTHON_EXE=C:\Users\Almohanad\anaconda3\python.exe"

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║        PWN CapNF Auto-Update Dashboard Service          ║
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

REM Check if Python exists
if not exist "%PYTHON_EXE%" (
    echo ❌ Python not found at: %PYTHON_EXE%
    pause
    exit /b 1
)

REM Start the auto-update service
"%PYTHON_EXE%" "%~dp0auto_update_dashboard.py"

echo.
echo Service stopped.
pause
