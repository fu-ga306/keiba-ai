# Register the dashboard watchdog as a scheduled task.
#
# NOTE: this is OPTIONAL.
#   The existing watchdog task already calls dashboard_service.ensure() every
#   20 minutes (06:50-24:50, StartWhenAvailable=true), which covers reboot
#   recovery. This script only adds an at-logon trigger and an earlier start.
#   It requires administrator rights. Skip it unless you want the extra margin.
#
# Why this is needed:
#   The dashboard (Flask + ngrok) is what we sell, but neither was registered as a
#   scheduled task. Both were started by hand, so a reboot would leave them down.
#   The prediction system has a watchdog; the thing we sell did not.
#
# What the task does:
#   Runs `dashboard_service.py ensure` every 20 minutes.
#     Inside the service window (Fri 06:00 - Mon 09:00): start if down,
#       restart if the site is not reachable from outside.
#     Outside the window: stop it. Nobody visits on weekdays.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File register_dashboard_task.ps1
#
# To remove:
#   Unregister-ScheduledTask -TaskName 'keiba_dashboard_ensure' -Confirm:$false
#
# NOTE: keep this file ASCII-only.
#   PowerShell 5.1 reads a .ps1 without BOM as the ANSI codepage (cp932 here),
#   which mangles Japanese text and breaks parsing. Hit this on 2026-08-27.
#   Do not use backtick line continuations either; they broke on line endings.

$ErrorActionPreference = 'Stop'

$base = 'C:\Users\' + [char]0x5225 + [char]0x5E9C + [char]0x98DB + [char]0x6CB3 + '\OneDrive\' + [char]0x30C7 + [char]0x30B9 + [char]0x30AF + [char]0x30C8 + [char]0x30C3 + [char]0x30D7 + '\keiba_ai'
$py = 'C:\Users\' + [char]0x5225 + [char]0x5E9C + [char]0x98DB + [char]0x6CB3 + '\AppData\Local\Microsoft\WindowsApps\python3.11.exe'
$script = Join-Path $base 'dashboard_service.py'
$name = 'keiba_dashboard_ensure'

if (-not (Test-Path $py)) { Write-Output "  NG: python not found: $py"; exit 1 }
if (-not (Test-Path $script)) { Write-Output "  NG: script not found: $script"; exit 1 }

$arg = '"' + $script + '" ensure'
$act = New-ScheduledTaskAction -Execute $py -Argument $arg -WorkingDirectory $base

# At logon, plus every 20 minutes starting 06:00 daily.
$t1 = New-ScheduledTaskTrigger -AtLogOn
$t2 = New-ScheduledTaskTrigger -Once -At ([datetime]::Today.AddHours(6)) -RepetitionInterval (New-TimeSpan -Minutes 20)

# IgnoreNew: do not stack instances. StartWhenAvailable: catch up after sleep.
$set = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable

try {
    Register-ScheduledTask -TaskName $name -Action $act -Trigger $t1, $t2 -Settings $set -Force | Out-Null
    Write-Output "  OK: registered $name"
    $i = Get-ScheduledTaskInfo -TaskName $name
    Write-Output "      next run: $($i.NextRunTime)"
    Write-Output ""
    Write-Output "  Service window: Fri 06:00 - Mon 09:00"
    Write-Output "  Starts it if down, rebuilds it if not reachable from outside."
}
catch {
    Write-Output "  NG: failed to register: $($_.Exception.Message)"
    Write-Output ""
    Write-Output "  This may require administrator rights."
    Write-Output "  Run PowerShell as administrator and try again."
    exit 1
}
