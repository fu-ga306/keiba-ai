"""
weekly_update.py
────────────────
毎週月曜日に自動実行するスクリプト。
  1. result_tracker.py update（レース結果照合）
  2. prediction_record_v2.csv を GitHubにプッシュ
  3. サマリーをログに保存

タスクスケジューラ登録済み：毎週月曜 08:00
"""

import os
import sys
import subprocess
from datetime import datetime

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"
LOG_FILE = os.path.join(BASE_DIR, "weekly_update_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log("=" * 40)
    log("週次自動更新 開始")
    log("=" * 40)

    # ① result_tracker.py update
    log("レース結果照合中...")
    try:
        result = subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "update"],
            cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8"
        )
        log(result.stdout.strip() if result.stdout else "完了")
        if result.returncode != 0:
            log(f"エラー: {result.stderr}")
    except Exception as e:
        log(f"result_tracker エラー: {e}")

    # ② result_tracker.py summary
    log("サマリー生成中...")
    try:
        result = subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "result_tracker.py"), "summary"],
            cwd=BASE_DIR, capture_output=True, text=True, encoding="utf-8"
        )
        log(result.stdout.strip() if result.stdout else "完了")
    except Exception as e:
        log(f"サマリーエラー: {e}")

    # ③ GitHubにプッシュ
    log("GitHubにプッシュ中...")
    try:
        os.chdir(BASE_DIR)
        subprocess.run(["git", "add", "prediction_record_v2.csv"], cwd=BASE_DIR)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        if result.stdout.strip():
            date_str = datetime.now().strftime("%Y/%m/%d")
            subprocess.run(
                ["git", "commit", "-m", f"予想記録更新 {date_str}"],
                cwd=BASE_DIR
            )
            subprocess.run(["git", "push"], cwd=BASE_DIR)
            log("GitHubプッシュ完了")
        else:
            log("変更なし・スキップ")
    except Exception as e:
        log(f"Gitエラー: {e}")

    log("週次自動更新 完了")
    log("=" * 40)


if __name__ == "__main__":
    main()
