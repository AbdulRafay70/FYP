# UltraXAI - Local Startup Script
# Run this script to start both the FastAPI backend and the React frontend.

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  *** Starting UltraXAI Diagnostic System ***" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Start FastAPI Backend in the background
Write-Host ""
Write-Host "[1/2] Starting FastAPI Backend..." -ForegroundColor Yellow
$BackendJob = Start-Job -ScriptBlock {
    Set-Location -Path "$using:PWD\backend"
    
    # Check if local virtual environment exists and use it, otherwise fallback to global
    $VenvPath = "$using:PWD\.venv\Scripts\uvicorn.exe"
    if (Test-Path $VenvPath) {
        & $VenvPath app.main:app --host 0.0.0.0 --port 8000
    } else {
        uvicorn app.main:app --host 0.0.0.0 --port 8000
    }
}
Write-Host "  [+] Backend starting on http://localhost:8000" -ForegroundColor Green

# 2. Start React Frontend
Write-Host ""
Write-Host "[2/2] Starting React Frontend..." -ForegroundColor Yellow
Set-Location -Path "frontend"
npm run dev

# Note: The frontend command will block the terminal until the user stops it (Ctrl+C).
# After the frontend is stopped, we clean up the backend job.
Write-Host ""
Write-Host "Cleaning up background jobs..." -ForegroundColor Yellow
Stop-Job -Job $BackendJob
Remove-Job -Job $BackendJob
Write-Host "UltraXAI stopped successfully." -ForegroundColor Green
