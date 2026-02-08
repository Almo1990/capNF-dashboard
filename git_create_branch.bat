@echo off
REM Create Feature Branch Script
REM Usage: git_create_branch.bat "feature-name"

echo ================================================
echo Git Feature Branch Creator
echo ================================================
echo.

if "%~1"=="" (
    echo ERROR: Please provide a branch name
    echo Usage: git_create_branch.bat "feature-name"
    exit /b 1
)

REM Create branch
echo Creating branch %~1...
C:\Users\abusa\MinGit\cmd\git.exe branch %~1
echo.

REM Push branch to GitHub
echo Pushing branch to GitHub...
C:\Users\abusa\MinGit\cmd\git.exe push origin %~1
echo.

REM List all branches
echo All branches:
C:\Users\abusa\MinGit\cmd\git.exe branch -a
echo.

echo ================================================
echo Done! Branch %~1 created and pushed
echo To switch to this branch, run: git checkout %~1
echo ================================================
pause
