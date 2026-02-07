@echo off
REM ========================================
REM  CapNF Auto-Update Dashboard Service
REM ========================================
REM
REM This script starts a background service that monitors
REM the Data/ folder for new .tsv files and automatically:
REM   1. Runs the CapNF pipeline
REM   2. Generates updated dashboard HTML files  
REM   3. Deploys to GitHub Pages
REM
REM Keep this window open for auto-updates to work!
REM ========================================

color 0B
title CapNF Auto-Update Service

cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║        PWN CapNF Auto-Update Dashboard Service          ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  📊 Monitoring: Data\ folder
echo  🌐 Dashboard: https://YOUR_USERNAME.github.io/capnf-dashboard/
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

REM Check if watchdog is installed
C:\ProgramData\anaconda3\python.exe -c "import watchdog" 2>nul
if errorlevel 1 (
    echo 📦 Installing required package: watchdog...
    echo.
    C:\ProgramData\anaconda3\python.exe -m pip install watchdog
    echo.
    if errorlevel 1 (
        echo ❌ Failed to install watchdog
        echo Please run: C:\ProgramData\anaconda3\python.exe -m pip install watchdog
        pause
        exit /b 1
    )
)

REM Start the auto-update service
C:\ProgramData\anaconda3\python.exe auto_update_dashboard.py

echo.
echo Service stopped.
pause
