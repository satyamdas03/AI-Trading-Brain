# AI Trading Brain — Quick Status Check
Write-Output "=== AI Trading Brain Status ==="
Write-Output "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Check daemon
$port = netstat -ano | Select-String ":8420.*LISTENING"
if ($port) {
    $pidStr = ($port -split '\s+')[-1]
    $proc = Get-Process -Id ([int]$pidStr) -ErrorAction SilentlyContinue
    if ($proc) {
        $uptime = [DateTime]::Now - $proc.StartTime
        Write-Output "Daemon: RUNNING (PID $pidStr, uptime $([int]$uptime.TotalHours)h $($uptime.Minutes)m)"
    }
} else {
    Write-Output "Daemon: DOWN"
}

# Check dashboards
try { $null = Invoke-WebRequest -Uri "http://localhost:8420/docs" -TimeoutSec 3 -UseBasicParsing; Write-Output "API docs: http://localhost:8420/docs" } catch { Write-Output "API docs: DOWN" }
try { $null = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 3 -UseBasicParsing; Write-Output "Dashboard: http://localhost:3000" } catch { Write-Output "Dashboard: DOWN" }

# Quick state snapshot
try {
    $r = Invoke-RestMethod -Uri "http://localhost:8420/api/state" -TimeoutSec 10
    Write-Output ("Last scoring: " + $r.last_scoring.Substring(0,19))
    Write-Output ("Signals: " + ($r.signals | ForEach-Object { "$($_.ticker)=$([math]::Round($_.composite_score,2))" }) -join ", ")
} catch { Write-Output "API state: error" }

# Restart daemon
Write-Output ""
Write-Output "Restart: cd C:\Users\point\projects\ai-trading-brain ; .\launch_trading.ps1"
Write-Output "Logs: tail -f C:\Users\point\projects\ai-trading-brain\logs\brain.log"
