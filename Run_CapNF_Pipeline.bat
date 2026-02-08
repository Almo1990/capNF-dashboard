@echo off
setlocal enabledelayedexpansion
echo ===============================================================
echo     CapNF - Membrane Filtration Analytics Pipeline        
echo ===============================================================
echo.

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
echo ✅ Using Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" main.py

if %errorlevel% == 0 (
    echo.
    echo ===============================================================
    echo     Pipeline completed successfully!
    echo ===============================================================
    echo.
    
    REM Check if git is initialized
    git status >nul 2>&1
    if %errorlevel% == 0 (
        echo [Step 2/3] Deploying to GitHub Pages...
        echo.
        
        git add combined_data_plots\* outputs\*.json 2>nul
        git commit -m "Dashboard update %date% %time%" >nul 2>&1
        git push origin main >nul 2>&1
        
        if %errorlevel% == 0 (
            echo ✅ Deployed to GitHub Pages successfully!
            echo 🌐 Dashboard will be live at: https://Almo1990.github.io/capnf-dashboard/
            echo    (Wait ~1 minute for GitHub to process the update)
        ) else (
            REM Try master branch if main doesn't exist
            git push origin master >nul 2>&1
            if %errorlevel% == 0 (
                echo ✅ Deployed to GitHub Pages successfully!
                echo 🌐 Dashboard will be live at: https://Almo1990.github.io/capnf-dashboard/
            ) else (
                echo ⚠️  Git push skipped (no remote configured or not authenticated)
                echo    To enable auto-deployment, set up GitHub Pages (see GITHUB_SETUP.txt)
            )
        )
        echo.
    ) else (
        echo ⚠️  Git not initialized - skipping deployment
        echo    To enable auto-deployment, set up GitHub Pages (see GITHUB_SETUP.txt)
        echo.
    )
    
    echo [Step 3/3] Opening dashboard locally...
    timeout /t 2 /nobreak >nul
    start "" "combined_data_plots\index.html"
) else (
    echo.
    echo ===============================================================
    echo     Pipeline failed with error code: %errorlevel%
    echo ===============================================================
)

echo.
echo Press any key to close this window...
pause >nul
