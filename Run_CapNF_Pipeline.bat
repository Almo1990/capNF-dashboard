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
echo [ERROR] Python not found!
echo.
echo Please install Python or Anaconda.
echo Download from:
echo   - Anaconda: https://www.anaconda.com/download
echo   - Python: https://www.python.org/downloads/
echo.
pause
exit /b 1

:found_python
echo [OK] Using Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" main.py
if !errorlevel! neq 0 (
    echo.
    echo ===============================================================
    echo     Pipeline failed with error code: !errorlevel!
    echo ===============================================================
    goto :end
)

echo.
echo ===============================================================
echo     Pipeline completed successfully!
echo ===============================================================
echo.

REM [Step 2/3] Deploy to GitHub Pages
echo [Step 2/3] Deploying to GitHub Pages...
echo.

REM Auto-detect git: check PATH first, then MinGit in user home
set "GIT_EXE="
where git >nul 2>&1
if !errorlevel! == 0 (
    for /f "delims=" %%g in ('where git') do (
        "%%g" --version >nul 2>&1
        if !errorlevel! == 0 (
            set "GIT_EXE=%%g"
            goto :found_git
        )
    )
)
if exist "%USERPROFILE%\MinGit\cmd\git.exe" (
    set "GIT_EXE=%USERPROFILE%\MinGit\cmd\git.exe"
    set "PATH=%USERPROFILE%\MinGit\cmd;%PATH%"
    goto :found_git
)

echo    [SKIP] Git not found - skipping deployment
echo    Install Git or see GITHUB_SETUP.txt for MinGit setup
echo.
goto :open_dashboard

:found_git
echo    Using git: !GIT_EXE!

REM Deploy to gh-pages using the helper script
"%PYTHON_EXE%" deploy_to_pages.py --main-too
if !errorlevel! == 0 (
    echo.
    echo [OK] Deployed successfully!
    echo.
) else (
    echo.
    echo [WARNING] Deployment failed - check authentication or GITHUB_SETUP.txt
    echo.
)

:open_dashboard
echo [Step 3/3] Opening dashboard locally...
timeout /t 2 /nobreak >nul
start "" "combined_data_plots\index.html"

:end

echo.
echo Press any key to close this window...
pause >nul
