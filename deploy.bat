@echo off
REM One-click Fly.io deployment script
REM This script will deploy your ML system to Fly.io

echo.
echo ================================================
echo     MLSD Project - Fly.io Deployment
echo ================================================
echo.

REM Check if Fly CLI is installed
where flyctl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [!] Fly CLI not found. Installing...
    echo.
    echo Visit: https://fly.io/docs/hands-on/install-flyctl/
    echo Or run: choco install flyctl
    echo.
    pause
    exit /b 1
)

echo [✓] Fly CLI found
echo.
echo Logging into Fly.io...
flyctl auth login

if %ERRORLEVEL% NEQ 0 (
    echo [!] Login failed
    pause
    exit /b 1
)

echo.
echo [✓] Logged in successfully!
echo.
echo Deploying to Fly.io...
echo.

flyctl deploy --remote-only

if %ERRORLEVEL% EQ 0 (
    echo.
    echo ================================================
    echo [✓] DEPLOYMENT SUCCESSFUL!
    echo ================================================
    echo.
    echo Your app is now live!
    echo.
    flyctl info
    echo.
    echo View logs:     flyctl logs
    echo View URL:      flyctl info -a mlsd-project
    echo.
) else (
    echo [!] Deployment failed
)

pause
