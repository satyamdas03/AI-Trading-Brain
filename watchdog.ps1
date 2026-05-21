# AI Trading Brain Watchdog — auto-restarts daemon if it dies
# Run: powershell -ExecutionPolicy Bypass -File watchdog.ps1
# Stops: Ctrl+C or close window

param(
    [int]$Port = 8420,
    [int]$CheckIntervalSec = 30,
    [string]$ProjectDir = "C:\Users\point\projects\ai-trading-brain"
)

$ErrorActionPreference = "Continue"
$host.UI.RawUI.WindowTitle = "AI Trading Brain Watchdog"

Write-Host "=== AI Trading Brain Watchdog ===" -ForegroundColor Cyan
Write-Host "Port: $Port | Check interval: ${CheckIntervalSec}s | Dir: $ProjectDir" -ForegroundColor Gray
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

$restartCount = 0

while ($true) {
    $healthy = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $healthy = $true
        }
    } catch {
        # Health check failed
    }

    if (-not $healthy) {
        $restartCount++
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] Daemon DOWN — restarting (attempt #$restartCount)..." -ForegroundColor Red

        # Kill any existing python processes running brain.py on this port
        Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
            $proc = $_
            try {
                $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                if ($cmd -match "brain\.py") {
                    Write-Host "  Killing PID $($proc.Id)..." -ForegroundColor Yellow
                    $proc.Kill()
                }
            } catch {}
        }

        Start-Sleep -Seconds 3

        # Start daemon
        try {
            $proc = Start-Process python -ArgumentList "-u", "brain.py" `
                -WorkingDirectory $ProjectDir `
                -RedirectStandardOutput "$ProjectDir\logs\daemon_fixed.log" `
                -RedirectStandardError "$ProjectDir\logs\daemon_fixed_err.log" `
                -NoNewWindow -PassThru

            Write-Host "  Started PID $($proc.Id), waiting for health check..." -ForegroundColor Yellow

            # Wait up to 120s for daemon to become healthy
            $ready = $false
            for ($i = 0; $i -lt 24; $i++) {
                Start-Sleep -Seconds 5
                try {
                    $r = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 3 -UseBasicParsing
                    if ($r.StatusCode -eq 200) {
                        $ready = $true
                        break
                    }
                } catch {}
            }

            if ($ready) {
                Write-Host "  Daemon healthy! PID $($proc.Id)`n" -ForegroundColor Green
            } else {
                Write-Host "  WARNING: Daemon started but health check still failing after 120s`n" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  ERROR: Failed to start daemon: $_" -ForegroundColor Red
        }
    }

    Start-Sleep -Seconds $CheckIntervalSec
}
