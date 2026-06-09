import schedule
import time
import subprocess
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
PYTHON   = sys.executable

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

def get_today_races():
    """今日のレースIDと発走時刻を取得"""
    today = datetime.now().strftime("%Y%m%d")
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={today}"

    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}

        for li in soup.find_all("li", class_=re.compile(r"RaceList_DataItem")):
            # race_id取得
            a = li.find("a", href=re.compile(r"race_id=\d{12}"))
            if not a:
                continue
            race_id = re.search(r'race_id=(\d{12})', a["href"]).group(1)

            # 発走時刻取得
            time_tag = li.find(class_=re.compile(r"RaceList_ItemTitle|RaceTime|time"))
            if time_tag:
                time_match = re.search(r'(\d{1,2}):(\d{2})', time_tag.get_text())
                if time_match:
                    race_info[race_id] = time_match.group(0)

        return race_info

    except Exception as e:
        print(f"取得エラー: {e}")
        return {}


def run_prediction(race_id):
    print(f"\n[{datetime.now().strftime('%H:%M')}] {race_id} 予測実行中...")
    subprocess.run([PYTHON, "notifier.py", race_id], cwd=BASE_DIR)


def main():
    print(f"起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}]")
    print("今日のレース取得中...")

    race_info = get_today_races()

    if not race_info:
        print("レースが見つかりませんでした")
        print("手動でrace_idを入力してください（カンマ区切り）:")
        ids = input().strip().split(",")
        times_input = {}
        for rid in ids:
            rid = rid.strip()
            t = input(f"{rid} の発走時刻（例 15:40）: ").strip()
            times_input[rid] = t
        race_info = times_input

    now = datetime.now()
    scheduled_count = 0

    for race_id, race_time in race_info.items():
        time_match = re.search(r'(\d{1,2}):(\d{2})', race_time)
        if not time_match:
            continue

        h = int(time_match.group(1))
        m = int(time_match.group(2))

        # 5分前に設定
        m -= 5
        if m < 0:
            m += 60
            h -= 1

        notify_time = f"{h:02d}:{m:02d}"

        # 過去の時刻はスキップ
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if scheduled_dt < now:
            print(f"スキップ（過去）: {race_id} {notify_time}")
            continue

        schedule.every().day.at(notify_time).do(run_prediction, race_id=race_id)
        print(f"予約済み: {race_id} → {notify_time} （発走: {race_time}）")
        scheduled_count += 1

    if scheduled_count == 0:
        print("予約できるレースがありませんでした")
        return

    print(f"\n{scheduled_count}件を予約しました。待機中... （Ctrl+Cで停止）\n")
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()