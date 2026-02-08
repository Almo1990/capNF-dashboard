@echo off
REM Create Version Tag Script
REM Usage: git_create_tag.bat "v1.1.0" "Release description"

echo ================================================
echo Git Version Tag Creator
echo ================================================
echo.

if "%~1"=="" (
    echo ERROR: Please provide a tag name
    echo Usage: git_create_tag.bat "v1.1.0" "Release description"
    exit /b 1
)

if "%~2"=="" (
    echo ERROR: Please provide a description
    echo Usage: git_create_tag.bat "v1.1.0" "Release description"
    exit /b 1
)

REM Create tag
echo Creating tag %~1 with message: %~2
C:\Users\abusa\MinGit\cmd\git.exe tag -a %~1 -m "%~2"
echo.

REM Push tag
echo Pushing tag to GitHub...
C:\Users\abusa\MinGit\cmd\git.exe push origin --tags
echo.

REM List all tags
echo All version tags:
C:\Users\abusa\MinGit\cmd\git.exe tag -l
echo.

echo ================================================
echo Done! Tag %~1 created and pushed to GitHub
echo ================================================
pause
