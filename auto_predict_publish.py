"""
auto_predict_publish.py
────────────────────────
当日予想の自動実行 + GitHub自動プッシュスクリプト。

動作：
  ① 午前7時  → 当日全レースを予想してtoday_predictions.csvに保存 → GitHubにプッシュ
  ② 各レース40分前 → 個別レースを再予想（オッズ確定後）→ GitHubにプッシュ

使い方:
    python auto_predict_publish.py        # スケジューラー起動
    python auto_predict_publish.py test   # 即時テスト実行（スケジューラーなし）
"""

import os
import sys
import re
import time
import subprocess
import pickle
import schedule
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"

JYO_NAMES = {
    1:"札幌", 2:"函館", 3:"福島", 4:"新潟",  5:"東京",
    6:"中山", 7:"中京", 8:"京都", 9:"阪神", 10:"小倉",
}

PYTHON = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"


# ── Git自動プッシュ ───────────────────────────────────────────────────────
def git_push(message: str):
    """today_predictions.csv と prediction_record_v2.csv をGitHubにプッシュ"""
    try:
        os.chdir(BASE_DIR)

        # 変更があるファイルだけ追加
        files = ["today_predictions.csv", "prediction_record_v2.csv"]
        for f in files:
            path = os.path.join(BASE_DIR, f)
            if os.path.exists(path):
                subprocess.run(["git", "add", f], cwd=BASE_DIR, check=True)

        # 変更がなければスキップ
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        if not result.stdout.strip():
            print("  [Git] 変更なし・スキップ")
            return

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=BASE_DIR, check=True
        )
        subprocess.run(
            ["git", "push"],
            cwd=BASE_DIR, check=True
        )
        print(f"  [Git] プッシュ完了: {message}")

    except subprocess.CalledProcessError as e:
        print(f"  [Git] エラー: {e}")
    except Exception as e:
        print(f"  [Git] 予期せぬエラー: {e}")


# ── 全レース予想（午前7時実行） ───────────────────────────────────────────
def run_morning_prediction():
    """当日の全レースを予想してCSVに保存・GitHubにプッシュ"""
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    print(f"\n{'='*50}")
    print(f"[{now}] 朝の一括予想開始")
    print(f"{'='*50}")

    # today_predictions.csv をリセット
    out_path = os.path.join(BASE_DIR, "today_predictions.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    # keiba_predict.py の today モードを実行
    try:
        result = subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "keiba_predict.py"), "today"],
            cwd=BASE_DIR,
            capture_output=False,  # ターミナルに出力
            text=True,
        )
        if result.returncode == 0:
            print(f"\n[{datetime.now().strftime('%H:%M')}] 朝の一括予想完了")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M')}] 予想エラー（returncode={result.returncode}）")
    except Exception as e:
        print(f"  予想実行エラー: {e}")
        return

    # GitHubにプッシュ
    date_str = datetime.now().strftime("%Y/%m/%d")
    git_push(f"当日予想更新 {date_str} 07:00")


# ── 個別レース予想（各レース40分前実行） ──────────────────────────────────
def run_race_prediction(race_id: str, race_time: str):
    """個別レースの予想を実行してGitHubにプッシュ"""
    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

    now = datetime.now().strftime("%H:%M")
    print(f"\n[{now}] {jyo_name} {race_no}R 予想開始（発走: {race_time}）")

    try:
        result = subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "keiba_predict.py"), race_id],
            cwd=BASE_DIR,
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print(f"  {jyo_name} {race_no}R 予想完了")
        else:
            print(f"  {jyo_name} {race_no}R 予想エラー")
            return
    except Exception as e:
        print(f"  予想実行エラー: {e}")
        return

    # GitHubにプッシュ
    git_push(f"{jyo_name} {race_no}R 予想更新 {datetime.now().strftime('%H:%M')}")


# ── スケジュール設定 ──────────────────────────────────────────────────────
def setup_schedule():
    """当日のレーススケジュールを設定する"""
    from keiba_auto import get_today_races

    print("当日レース一覧を取得中...")
    race_info = get_today_races()
    if not race_info:
        print("本日のレースが取得できませんでした")
        return 0

    now = datetime.now()
    scheduled = 0

    for race_id, race_time in sorted(race_info.items()):
        time_match = re.search(r"(\d{1,2}):(\d{2})", race_time)
        if not time_match:
            continue

        h = int(time_match.group(1))
        m = int(time_match.group(2)) - 40  # 40分前
        if m < 0:
            m += 60
            h -= 1

        notify_time  = f"{h:02d}:{m:02d}"
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if scheduled_dt < now:
            print(f"  スキップ（過去）: {race_id} {notify_time}")
            continue

        schedule.every().day.at(notify_time).do(
            run_race_prediction,
            race_id=race_id,
            race_time=race_time,
        )

        jyo_cd   = int(str(race_id)[4:6])
        race_no  = int(str(race_id)[10:12])
        jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
        print(f"  予約: {jyo_name} {race_no}R → {notify_time}（発走: {race_time}）")
        scheduled += 1

    return scheduled


# ── メイン ────────────────────────────────────────────────────────────────
def main():
    print(f"=== 自動予想・公開システム 起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}] ===\n")

    # 午前7時の一括予想をスケジュール
    schedule.every().day.at("07:00").do(run_morning_prediction)
    print("  [済] 朝7時の一括予想をスケジュール登録")

    # 当日レースの個別予想スケジュールを設定
    now = datetime.now()

    # 7時前なら起動時にスケジュール設定
    # 7時以降なら朝の予想完了後に設定
    if now.hour < 7:
        print("  7時前のため、7時の一括予想後にレーススケジュールを設定します")
        # 7時5分にスケジュール設定を行う
        schedule.every().day.at("07:05").do(setup_and_register)
    else:
        # すでに7時以降 → 今すぐスケジュール設定
        n = setup_schedule()
        print(f"  {n}レースをスケジュール登録")

    print(f"\n待機中... (Ctrl+Cで停止)\n")
    while True:
        schedule.run_pending()
        time.sleep(10)


def setup_and_register():
    """7時5分に個別レーススケジュールを設定（一括予想完了後）"""
    n = setup_schedule()
    print(f"  {n}レースをスケジュール登録（7:05）")
    return schedule.CancelJob  # 一度だけ実行


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # テスト：即時実行
        print("テストモード: 朝の一括予想を即時実行します")
        run_morning_prediction()
    elif len(sys.argv) > 1 and sys.argv[1] == "schedule":
        # スケジュールのみ設定（一括予想なし）
        n = setup_schedule()
        print(f"\n{n}レースをスケジュール登録。待機中...\n")
        while True:
            schedule.run_pending()
            time.sleep(10)
    else:
        main()
