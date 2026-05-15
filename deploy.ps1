#!/usr/bin/env pwsh
# One-click Fly.io deployment script for PowerShell

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     MLSD Project - Fly.io Deployment" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Fly CLI is installed
$flyctlExists = $null -ne (Get-Command flyctl -ErrorAction SilentlyContinue)

if (-not $flyctlExists) {
    Write-Host "[!] Fly CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Option 1: Visit https://fly.io/docs/hands-on/install-flyctl/" -ForegroundColor Yellow
    Write-Host "  Option 2: Run: choco install flyctl" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "[✓] Fly CLI found" -ForegroundColor Green
Write-Host ""
Write-Host "Logging into Fly.io..." -ForegroundColor Cyan
Write-Host ""

flyctl auth login

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Login failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[✓] Logged in successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Deploying to Fly.io..." -ForegroundColor Cyan
Write-Host ""

flyctl deploy --remote-only

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "[✓] DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your app is now live!" -ForegroundColor Green
    Write-Host ""
    
    flyctl info
    
    Write-Host ""
    Write-Host "Useful commands:" -ForegroundColor Cyan
    Write-Host "  View logs:  flyctl logs" -ForegroundColor Yellow
    Write-Host "  View URL:   flyctl info -a mlsd-project" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "[!] Deployment failed" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
