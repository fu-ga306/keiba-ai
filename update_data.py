"""
update_data.py
──────────────
定期実行用のデータ更新スクリプト。
以下を自動で実行する：
  1. 直近N週分のレース結果をスクレイピング
  2. cleaner.py でクリーニング
  3. 新規馬の父馬・母父馬情報を取得
  4. sire_stats.py で父馬産駒成績を再集計
  5. features.py で特徴量を再生成

推奨実行タイミング：
  - 毎週月曜日の朝（週末レース結果が確定した後）
  - 手動実行: python update_data.py
  - 週次自動化: Windowsタスクスケジューラに登録

使い方:
  python update_data.py          # 直近4週分を更新
  python update_data.py --weeks 8  # 直近8週分を更新
  python update_data.py --full     # 全期間を再取得（初回のみ）
"""

import os
import sys
import time
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

BASE_DIR   = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
LARGE_CSV  = os.path.join(BASE_DIR, "race_data_large.csv")
CLEAN_CSV  = os.path.join(BASE_DIR, "race_data_clean.csv")
HORSE_CSV  = os.path.join(BASE_DIR, "horse_master.csv")
SIRE_CSV   = os.path.join(BASE_DIR, "sire_stats.csv")
FEAT_CSV   = os.path.join(BASE_DIR, "race_features.csv")
LOG_FILE   = os.path.join(BASE_DIR, "update_log.txt")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

JYO_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ── ログ ──────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Seleniumドライバー生成 ────────────────────────────────────────────────
def make_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )


# ── Step1: 対象race_idを生成 ──────────────────────────────────────────────
def get_target_race_ids(weeks: int) -> list:
    """
    直近N週分の開催日から対象race_idを生成する。
    netkeiba のカレンダーページから実際の開催日を取得。
    """
    log(f"直近{weeks}週分の開催日を取得中...")

    today = datetime.now()
    target_dates = []

    # 直近N週の土日を収集
    for i in range(weeks * 2 + 4):
        d = today - timedelta(days=i)
        if d.weekday() in [5, 6]:  # 土日
            target_dates.append(d.strftime("%Y%m%d"))

    target_dates = sorted(set(target_dates))
    log(f"  対象日数: {len(target_dates)}日")

    # 各日付のrace_idを生成
    race_ids = []
    for date in target_dates:
        year = int(date[:4])
        for jyo in JYO_CODES:
            for kai in range(1, 6):
                for nichi in range(1, 9):
                    for race in range(1, 13):
                        race_ids.append(
                            f"{year}{jyo:02d}{kai:02d}{nichi:02d}{race:02d}"
                        )

    # 既存データのrace_idと照合してスキップ対象を特定
    existing_ids = set()
    if os.path.exists(LARGE_CSV):
        existing_df = pd.read_csv(LARGE_CSV, usecols=["race_id"])
        existing_ids = set(existing_df["race_id"].astype(str).unique())

    new_ids = [r for r in race_ids if r not in existing_ids]
    log(f"  新規取得対象: {len(new_ids)}件（既存スキップ: {len(race_ids)-len(new_ids)}件）")
    return new_ids


# ── Step2: レース結果スクレイピング ──────────────────────────────────────
def scrape_races(race_ids: list) -> int:
    """レース結果を取得してrace_data_large.csvに追記"""
    import re

    if not race_ids:
        log("新規取得対象なし")
        return 0

    log(f"レース結果取得開始: {len(race_ids)}件")
    new_data = []
    errors   = 0

    for i, race_id in enumerate(race_ids):
        try:
            url = f"https://db.netkeiba.com/race/{race_id}/"
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.encoding = "EUC-JP"
            soup = BeautifulSoup(r.text, "html.parser")

            race_info = {}
            race_data_div = soup.find("div", class_="data_intro")
            if race_data_div:
                shibari = race_data_div.find("span")
                if shibari:
                    info_text = shibari.get_text(strip=True)
                    dist_match = re.search(r"(\d{3,4})m", info_text)
                    race_info["距離"]     = int(dist_match.group(1)) if dist_match else None
                    race_info["馬場種別"] = "芝" if "芝" in info_text else "ダート"
                    race_info["馬場状態"] = None
                    for cond in ["不良", "稍重", "重", "良"]:
                        if cond in info_text:
                            race_info["馬場状態"] = cond
                            break
                cls_tag = race_data_div.find("p")
                if cls_tag:
                    race_info["レースクラス"] = cls_tag.get_text(strip=True)

            table = soup.find("table", class_="race_table_01")
            if table is None:
                continue

            headers_row = [
                th.get_text(strip=True)
                for th in table.find_all("tr")[0].find_all("th")
            ]
            rows      = []
            horse_ids = []
            for tr in table.find_all("tr")[1:]:
                cols = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not cols:
                    continue
                rows.append(cols)
                horse_id = None
                a_tag = tr.find("a", href=re.compile(r"/horse/\d+"))
                if a_tag:
                    m = re.search(r"/horse/(\d+)", a_tag["href"])
                    if m:
                        horse_id = m.group(1)
                horse_ids.append(horse_id)

            if not rows:
                continue

            df = pd.DataFrame(rows)
            if headers_row and len(headers_row) == df.shape[1]:
                df.columns = headers_row
            df["horse_id"] = horse_ids
            for key, val in race_info.items():
                df[key] = val
            df["race_id"] = race_id
            new_data.append(df)

            if (i + 1) % 50 == 0:
                log(f"  進捗: {i+1}/{len(race_ids)}件")

        except Exception as e:
            errors += 1

        time.sleep(1.5)

    if not new_data:
        log("新規レースデータなし")
        return 0

    new_df = pd.concat(new_data, ignore_index=True)

    if os.path.exists(LARGE_CSV):
        old_df = pd.read_csv(LARGE_CSV, low_memory=False)
        result = pd.concat([old_df, new_df], ignore_index=True)
        result = result.drop_duplicates(subset=["race_id", "馬番"])
    else:
        result = new_df

    result.to_csv(LARGE_CSV, index=False, encoding="utf-8-sig")
    log(f"  取得完了: 新規{len(new_df)}行 / エラー:{errors}件 → {LARGE_CSV}")
    return len(new_df)


# ── Step3: クリーニング ───────────────────────────────────────────────────
def run_cleaner():
    log("cleaner.py 実行中...")
    sys.path.insert(0, BASE_DIR)
    try:
        from cleaner import clean_race_data
        clean_race_data(input_csv=LARGE_CSV, output_csv=CLEAN_CSV)
        log(f"  クリーニング完了 → {CLEAN_CSV}")
    except Exception as e:
        log(f"  クリーニングエラー: {e}")


# ── Step4: 新規馬の父馬情報取得 ──────────────────────────────────────────
def run_horse_scraper():
    log("horse_scraper.py 実行中（新規馬のみ）...")
    sys.path.insert(0, BASE_DIR)
    try:
        from horse_scraper import build_horse_master
        build_horse_master()
        log(f"  父馬情報取得完了 → {HORSE_CSV}")
    except Exception as e:
        log(f"  horse_scraperエラー: {e}")


# ── Step5: 父馬産駒成績の再集計 ──────────────────────────────────────────
def run_sire_stats():
    log("sire_stats.py 実行中...")
    sys.path.insert(0, BASE_DIR)
    try:
        from sire_stats import build_sire_stats
        build_sire_stats()
        log(f"  産駒成績集計完了 → {SIRE_CSV}")
    except Exception as e:
        log(f"  sire_statsエラー: {e}")


# ── Step6: 特徴量再生成 ───────────────────────────────────────────────────
def run_features():
    log("features.py 実行中（時間がかかります）...")
    sys.path.insert(0, BASE_DIR)
    try:
        from features import build_features
        build_features(csv_path=CLEAN_CSV, out_path=FEAT_CSV)
        log(f"  特徴量再生成完了 → {FEAT_CSV}")
    except Exception as e:
        log(f"  featuresエラー: {e}")


# ── Step7: モデル再学習 ───────────────────────────────────────────────────
def run_model():
    log("model.py 実行中（3モデル: win/place2/place3）...")
    sys.path.insert(0, BASE_DIR)
    try:
        from model import train_all_targets
        train_all_targets(csv_path=FEAT_CSV)
        log("  モデル再学習完了 → model.pkl（win/place2/place3）")
    except Exception as e:
        log(f"  modelエラー: {e}")


# ── メイン ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="競馬AIデータ定期更新")
    parser.add_argument("--weeks",  type=int, default=4,
                        help="直近何週分を更新するか（デフォルト: 4）")
    parser.add_argument("--full",   action="store_true",
                        help="全期間を再取得（初回のみ推奨）")
    parser.add_argument("--skip-scrape",   action="store_true",
                        help="スクレイピングをスキップ")
    parser.add_argument("--skip-horse",    action="store_true",
                        help="父馬情報取得をスキップ")
    parser.add_argument("--skip-features", action="store_true",
                        help="特徴量再生成をスキップ")
    parser.add_argument("--skip-model",    action="store_true",
                        help="モデル再学習をスキップ")
    args = parser.parse_args()

    log("=" * 50)
    log("競馬AI 定期データ更新 開始")
    log("=" * 50)

    start = datetime.now()

    # Step1-2: スクレイピング
    if not args.skip_scrape:
        if args.full:
            log("【全期間モード】scraper.py を直接実行してください")
        else:
            race_ids = get_target_race_ids(args.weeks)
            scrape_races(race_ids)
    else:
        log("スクレイピング: スキップ")

    # Step3: クリーニング
    run_cleaner()

    # Step4: 父馬情報取得
    if not args.skip_horse:
        run_horse_scraper()
    else:
        log("父馬情報取得: スキップ")

    # Step5: 産駒成績集計
    if os.path.exists(HORSE_CSV):
        run_sire_stats()
    else:
        log("sire_stats: horse_master.csv がないためスキップ")

    # Step6: 特徴量再生成
    if not args.skip_features:
        run_features()
    else:
        log("特徴量再生成: スキップ")

    # Step7: モデル再学習
    if not args.skip_model:
        run_model()
    else:
        log("モデル再学習: スキップ")

    elapsed = datetime.now() - start
    log("=" * 50)
    log(f"全処理完了  所要時間: {elapsed}")
    log("=" * 50)


if __name__ == "__main__":
    main()