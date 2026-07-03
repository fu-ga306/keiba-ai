# horse_scraper.py を実行している Python プロセスだけを停止する。
# 土日の当日予想（netkeibaアクセス）と血統スクレイピングの競合を避けるため、
# 予想開始前（早朝）にタスクスケジューラから呼ばれる。
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*horse_scraper*' } |
    ForEach-Object {
        try {
            Stop-Process -Id $_.ProcessId -Force
            "$(Get-Date -Format 'yyyy/MM/dd HH:mm') 血統スクレイピング停止 (PID: $($_.ProcessId))" |
                Out-File -FilePath "$PSScriptRoot\horse_scraper_stop.log" -Append -Encoding utf8
        } catch {}
    }
