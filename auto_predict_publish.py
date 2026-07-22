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


# ── ダッシュボード即時更新通知 ────────────────────────────────────────────
def notify_dashboard():
    """Flaskダッシュボードのキャッシュをクリアして最新データを即時反映する"""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000/api/refresh", timeout=3)
        print("  [Dashboard] キャッシュクリア → 最新データに更新")
    except Exception:
        pass  # Flask未起動の場合はスキップ（エラーにしない）


# ── Git自動プッシュ ───────────────────────────────────────────────────────
def git_push(message: str):
    """today_predictions.csv と prediction_record_v2.csv をGitHubにプッシュ"""
    try:
        os.chdir(BASE_DIR)

        # 変更があるファイルだけ追加
        files = ["today_predictions.csv", "prediction_record_v2.csv", "today_bets.csv",
                 "odds_history.csv"]   # オッズ変動特徴の蓄積データ（追記式・バックアップ用）
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

    # 開催日ガード: 非開催日は予想・プッシュをスキップ（get_today_racesが
    # 最小レース数チェック込みで空を返す→誤メール/誤予想を防止）。
    try:
        from keiba_auto import get_today_races
        if not get_today_races():
            print(f"[{now}] 本日は非開催（レース取得0件）→ 朝の一括予想をスキップ")
            return
    except Exception as e:
        print(f"  開催日チェックでエラー（続行）: {e}")

    # today_predictions.csv / today_bets.csv をリセット
    # （betsを消し忘れると前日レースの買い目が残存し、照合・ダッシュボードに混入する）
    for _f in ("today_predictions.csv", "today_bets.csv"):
        _p = os.path.join(BASE_DIR, _f)
        if os.path.exists(_p):
            os.remove(_p)

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

    # GitHubにプッシュ → ダッシュボード即時更新
    date_str = datetime.now().strftime("%Y/%m/%d")
    git_push(f"当日予想更新 {date_str} 07:00")
    notify_dashboard()


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

    # GitHubにプッシュ → ダッシュボード即時更新
    git_push(f"{jyo_name} {race_no}R 予想更新 {datetime.now().strftime('%H:%M')}")
    notify_dashboard()


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
def run_keiba_auto():
    """keiba_auto.py をサブプロセスで起動（メール送信・個別予想）"""
    print(f"\n[{datetime.now().strftime('%H:%M')}] keiba_auto.py 起動")
    try:
        result = subprocess.Popen(
            [PYTHON, os.path.join(BASE_DIR, "keiba_auto.py")],
            cwd=BASE_DIR,
        )
        print(f"  keiba_auto.py 起動完了 (PID: {result.pid})")
        # PIDを保存（停止時に使用）
        pid_file = os.path.join(BASE_DIR, "keiba_auto.pid")
        with open(pid_file, "w") as f:
            f.write(str(result.pid))
    except Exception as e:
        print(f"  keiba_auto.py 起動エラー: {e}")


def stop_keiba_auto():
    """keiba_auto.py を停止"""
    pid_file = os.path.join(BASE_DIR, "keiba_auto.pid")
    if not os.path.exists(pid_file):
        print("  keiba_auto.pid なし → スキップ")
        return
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        import signal, subprocess
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  keiba_auto.py 停止完了 (PID: {pid})")
            else:
                print(f"  keiba_auto.py 既に終了済み (PID: {pid})")
        except Exception:
            os.kill(pid, signal.SIGTERM)
            print(f"  keiba_auto.py 停止完了 (PID: {pid})")
    except (OSError, PermissionError) as e:
        print(f"  keiba_auto.py 既に終了済み (無視): {e}")
    except Exception as e:
        print(f"  keiba_auto.py 停止エラー: {e}")
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)


def main():
    print(f"=== 自動予想・公開システム 起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}] ===\n")

    # 多重起動ガード（2026-07-20）: タスクスケジューラ6:55の自動起動と手動起動が
    # 重なると全ジョブが二重になる（7/19発生）。既存プロセスがいれば即終了。
    from keiba_auto import ensure_single_instance
    ensure_single_instance("auto_predict_publish.py")

    # 午前7時の一括予想をスケジュール
    schedule.every().day.at("07:00").do(run_morning_prediction)
    print("  [済] 朝7時の一括予想をスケジュール登録")

    # keiba_auto.py の自動起動（6:58）・停止（21時）
    # keiba_auto.pyは各レース7分前にメール送信するため早めに起動
    schedule.every().day.at("06:58").do(run_keiba_auto)
    schedule.every().day.at("21:00").do(stop_keiba_auto)
    print("  [済] keiba_auto.py 自動起動(06:58)・停止(21:00)をスケジュール登録")

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