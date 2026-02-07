@echo off
echo ===============================================================
echo     PWN CapNF - Membrane Filtration Analytics Pipeline        
echo ===============================================================
echo.

cd /d "c:\Users\abusa\OneDrive - PWN\Documenten\R&D PWN-Almo\1 - Research\4 - Reserach places\1 - PWN\2 - Digitalization\Membrane filtration digitalization\CapNF"

C:\ProgramData\anaconda3\python.exe main.py

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
            echo 🌐 Dashboard will be live at: https://YOUR_USERNAME.github.io/capnf-dashboard/
            echo    (Wait ~1 minute for GitHub to process the update)
        ) else (
            REM Try master branch if main doesn't exist
            git push origin master >nul 2>&1
            if %errorlevel% == 0 (
                echo ✅ Deployed to GitHub Pages successfully!
                echo 🌐 Dashboard will be live at: https://YOUR_USERNAME.github.io/capnf-dashboard/
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
