# PowerShell Script to Run Auto-Update Dashboard Service
# For CapNF Project

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                          ║" -ForegroundColor Cyan
Write-Host "║        CapNF Auto-Update Dashboard Service          ║" -ForegroundColor Cyan  
Write-Host "║                                                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host " 📊 Monitoring: Data\ folder" -ForegroundColor Yellow
Write-Host " 🌐 Dashboard: https://Almo1990.github.io/capnf-dashboard/" -ForegroundColor Yellow
Write-Host ""
Write-Host " 💡 How it works:" -ForegroundColor Green
Write-Host "    • Drop new .tsv files into the Data\ folder"
Write-Host "    • Pipeline runs automatically (takes 2-5 minutes)"
Write-Host "    • Dashboard updates online automatically"
Write-Host ""
Write-Host " ⚠️  Keep this window OPEN for monitoring to continue" -ForegroundColor Red
Write-Host "    Press Ctrl+C to stop"
Write-Host ""
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Set Python executable path
$pythonExe = "c:\Users\Almohanad\OneDrive\Documents\Projects Python PWN\UF new\.conda\python-DESKTOP-S41CJGP.exe"

# Check if Python exists
if (!(Test-Path $pythonExe)) {
    Write-Host "❌ Error: Python executable not found at:" -ForegroundColor Red
    Write-Host "   $pythonExe" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please configure your Python environment."
    pause
    exit 1
}

# Run the auto-update service  
& $pythonExe "auto_update_dashboard.py"

Write-Host ""
Write-Host "Service stopped."
pause
