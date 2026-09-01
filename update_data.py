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
import re
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
# クラス名の抽出（2026-08-31）
#   長い順に見るのは「G1」が「G10」等に当たるのを防ぐため。
#   条件文（天候・発走を含む）はクラス名ではないので弾く。
_CLASS_WORDS = ("3勝クラス", "2勝クラス", "1勝クラス", "1600万下", "1000万下",
                "500万下", "1600万", "1000万", "500万",
                "未勝利", "新馬", "リステッド", "オープン",
                "JpnIII", "JpnII", "JpnI", "Jpn3", "Jpn2", "Jpn1",
                "GIII", "GII", "GI", "G3", "G2", "G1")


def _extract_class(text):
    """文字列からクラス名を取り出す。見つからなければ None。"""
    if not text:
        return None
    t = str(text)
    for w in _CLASS_WORDS:
        if w in t:
            return w
    return None


def _race_ids_of_dates(dates: list) -> list:
    """各日のレース一覧ページから、実在する race_id だけを拾う。

    一覧ページはJavaScriptで描画されるので requests では取れない。
    keiba_auto.get_today_races と同じくブラウザで開く。ブラウザは1回だけ
    起動して全日付を回す。
    非開催日は0件になる。取得に失敗しても総当たりには戻さない
    （総当たりはブロックの原因になるため）。
    """
    out = []
    driver = None
    try:
        driver = make_driver()
        for date in dates:
            try:
                driver.get(f"https://race.netkeiba.com/top/race_list.html"
                           f"?kaisai_date={date}")
                time.sleep(3)
                ids = sorted(set(re.findall(r"race_id=(\d{12})",
                                            driver.page_source)))
                log(f"    {date}: {len(ids)}レース")
                out.extend(ids)
            except Exception as e:
                log(f"    {date} の一覧取得に失敗（この日は飛ばします）: {str(e)[:60]}")
            time.sleep(2.0)
    except Exception as e:
        log(f"  ブラウザを起動できません: {str(e)[:80]}")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
    return out


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

    # 各日付の「レース一覧ページ」から実在するrace_idだけを取る（2026-08-09）。
    #   以前は 場10 × 回5 × 日8 × R12 = 4,800通りを総当たりで生成していた。
    #   実在するのは1日あたり36前後なので、97%が存在しないページへの
    #   アクセスになる。しかも日付ごとに同じIDを作り直すため同一URLを
    #   何度も叩いていた（2週指定で約9,000リクエスト）。
    #   7/27にCloudFrontで400を食らった原因はここだった可能性が高い。
    #   一覧ページを1日1回読めば、同じ結果が114リクエスト程度で済む。
    race_ids = sorted(set(_race_ids_of_dates(target_dates)))
    log(f"  実在レース: {len(race_ids)}件")
    if not race_ids:
        # 一覧が取れないまま総当たりに戻すとブロックを招くので、何もしない。
        log("  一覧からrace_idを取得できませんでした → 今回の取得は見送ります")
        return []

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
                # ⚠ 最初の <p> は条件文（2026-08-31に判明）
                #     <p><span>ダ右1700m / 天候 : 晴 / …</span></p>
                #     <p class="smalltxt">2026年8月30日 … サラ系3歳未勝利 …</p>
                #   最初の <p> を取っていたため、レースクラス欄に条件文が入り、
                #   クラス_num が 2026-07 以降 100% 欠損していた。
                #   クラス変化・距離×クラスが計算できず、学習にも本番にも影響。
                #
                #   DOMの形だけに頼らない。候補を順に見て、
                #   **クラス名として妥当なもの**を採る。
                race_info["レースクラス"] = None
                for _cand in ([t.get_text(" ", strip=True)
                               for t in race_data_div.find_all("p", class_="smalltxt")]
                              + [t.get_text(" ", strip=True)
                                 for t in race_data_div.find_all("p")]
                              + [soup.get_text(" ", strip=True)]):
                    _cls = _extract_class(_cand)
                    if _cls:
                        race_info["レースクラス"] = _cls
                        break

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
    """種牡馬成績を集計する。本番用と学習用の**両方**を作ること。

    ⚠ 2026-08-18まで build_sire_stats() を引数なしで1回だけ呼んでおり、
      本番用（全期間・sire_stats_father.csv）しか作られていなかった。
      学習用（≤2024・sire_stats_father_train.csv）は 2026-08-01 で止まっていた。

      学習用は features.py が use_train_snapshot=True で読む。つまり特徴量を
      作り直しても血統は古い集計のまま、という状態が続いていた。
      リーク防止のため年で切るのは正しいが、集計そのものは毎回やり直す必要がある
      （新しい馬の成績が ≤2024 の産駒成績にも効いてくるため）。
    """
    log("sire_stats.py 実行中...")
    sys.path.insert(0, BASE_DIR)
    try:
        from sire_stats import build_sire_stats, TRAIN_MAX_YEAR
        build_sire_stats(max_year=None, suffix="")            # 本番予測用（全期間）
        log(f"  本番用の集計完了 → {SIRE_CSV}")
        build_sire_stats(max_year=TRAIN_MAX_YEAR, suffix="_train")  # 学習・BT用
        log(f"  学習用の集計完了 → sire_stats_father_train.csv（≤{TRAIN_MAX_YEAR}）")
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