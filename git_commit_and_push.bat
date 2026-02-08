@echo off
REM Quick Git Commit and Push Script
REM Usage: git_commit_and_push.bat "Your commit message"

echo ================================================
echo Git Commit and Push Helper
echo ================================================
echo.

if "%~1"=="" (
    echo ERROR: Please provide a commit message
    echo Usage: git_commit_and_push.bat "Your commit message"
    exit /b 1
)

REM Check status
echo Checking git status...
C:\Users\abusa\MinGit\cmd\git.exe status
echo.

REM Add all changes
echo Adding all changes...
C:\Users\abusa\MinGit\cmd\git.exe add .
echo.

REM Commit with message
echo Committing with message: %~1
C:\Users\abusa\MinGit\cmd\git.exe commit -m "%~1"
echo.

REM Push to GitHub
echo Pushing to GitHub...
C:\Users\abusa\MinGit\cmd\git.exe push origin main
echo.

echo ================================================
echo Done! Changes pushed to GitHub
echo ================================================
pause
