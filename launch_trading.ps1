# AI Trading Brain Launcher
# Run: .\launch_trading.ps1
$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\point\projects\ai-trading-brain"
$env:PYTHONPATH = $ProjectRoot
Set-Location $ProjectRoot

# Kill any existing daemon on port 8420
$existing = netstat -ano | Select-String ":8420.*LISTENING"
if ($existing) {
    $parts = $existing -split '\s+'
    $pidStr = $parts[$parts.Length - 1]
    Write-Output "Killing existing daemon PID $pidStr..."
    Stop-Process -Id ([int]$pidStr) -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Create log directory
$logDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "daemon_$timestamp.log"
$errFile = Join-Path $logDir "daemon_error_$timestamp.log"

Write-Output "Starting AI Trading Brain daemon..."
Write-Output "  Project: $ProjectRoot"
Write-Output "  Log: $logFile"

$proc = Start-Process -FilePath "python" -ArgumentList "brain.py" -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError $errFile
Write-Output "  PID: $($proc.Id)"

Start-Sleep -Seconds 8

$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
    Write-Output "Status: RUNNING"
} else {
    Write-Output "Status: DIED - check logs:"
    Get-Content $logFile -Tail 20
    exit 1
}

# Check dashboard
Start-Sleep -Seconds 3
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8420/api/state" -TimeoutSec 10 -UseBasicParsing
    Write-Output "Dashboard: OK"
} catch {
    Write-Output "Dashboard: not responding yet (may still be starting)"
}
