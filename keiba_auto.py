import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import smtplib
import re
import schedule
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from result_tracker import record_from_prediction
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ── 設定 ──────────────────────────────────────────────────────────────────
# 【セキュリティ修正】パスワード・パスは環境変数から取得
# 実行前に .env ファイルか OS の環境変数に設定してください:
#   GMAIL_ADDRESS=your@gmail.com
#   GMAIL_APP_PASS=xxxx xxxx xxxx xxxx
#   TO_ADDRESS=your@gmail.com
#   KEIBA_BASE_DIR=C:\Users\...\keiba_ai
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未インストール時は os.environ のみ参照

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS", "")
TO_ADDRESS    = os.environ.get("TO_ADDRESS", GMAIL_ADDRESS)
BASE_DIR      = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

JYO_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟",  5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}

# ── 戦略の期待値しきい値（案B: 生確率ベース） ──────────────────
# model.py の期待値スイープで判明した最適値（2026/06/15確定）。
# 生確率は正規化版より小さく出るが、スイープの結果 +0.3 が
# トータル利益最大（251回 回収率129.2%）と判明したため踏襲。
EV_THRESHOLD_MAIN  = 0.3   # 戦略A/C/F/H等の主要しきい値
EV_THRESHOLD_SUB   = 0.2   # 戦略D等のサブしきい値
EV_THRESHOLD_TAIKO = 0.1   # ○対抗の印選出しきい値

# model.py の FEATURE_COLS と完全一致させること（フォールバック用）
FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気", "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "体重増減_過去標準偏差", "体重増減_異常度",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順",
    "良馬場勝率", "重馬場勝率",
    "過去最速上り", "上り偏差", "距離別過去平均上り",
    "斤量変化", "乗り替わり", "連闘", "休み明け", "負担率",
    "レース内_過去勝率ランク", "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク", "レース内_騎手勝率ランク",
    "競馬場距離過去勝率", "競馬場距離過去平均着順",
    "競馬場過去勝率", "競馬場過去平均着順",
    "過去平均先行指数", "先行馬フラグ",
    "想定先行馬数", "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
    "開催月", "開催季節",
    "前走着順", "前走上り", "前走距離", "距離変化",
    "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
    "平均タイム差", "騎手競馬場勝率",
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率", "芝ダート×先行_過去勝率",
    "前走好走×人気薄", "前走着順×人気_乖離", "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
    "距離変化比率", "大幅延長フラグ", "大幅短縮フラグ",
    "距離延長幅", "長距離フラグ", "大幅延長×長距離", "距離延長×先行",
    "経験最長距離", "経験範囲超過", "延長×距離経験不足",
    "長距離複勝率", "前走余力", "延長×前走余力",
    "騎手直近勝率", "騎手直近複勝率", "騎手調子トレンド",
    "直近5走勝利数", "直近5走複勝数", "直近5走平均着順",
    "過去獲得賞金累計", "過去平均獲得賞金",
    "回り_num",
    "父系_今回距離適性", "母父系_今回距離適性",
    "父系_長距離勝率", "母父系_長距離勝率",
    "父系_芝ダ適性", "父系_複勝率",
]


# ── ユーティリティ ────────────────────────────────────────────────────────
def _make_chrome_driver():
    """ヘッドレス Chrome ドライバーを生成して返す"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def calc_place_prob_harvill(win_probs: np.ndarray) -> np.ndarray:
    """
    ハーヴィルモデルで複勝確率（3着以内）を近似計算する。
    win_probs: 正規化済みの勝ち確率配列
    """
    n = len(win_probs)
    place_probs = np.zeros(n)
    for i in range(n):
        pi = win_probs[i]
        # 自分が1着の確率
        place_probs[i] += pi
        # 自分が2着・3着になる確率（ハーヴィル近似）
        for j in range(n):
            if j == i:
                continue
            pj = win_probs[j]
            s_excl_j = 1.0 - pj
            if s_excl_j <= 0:
                continue
            # j が1着、残りから i が2着
            p2 = pj * (pi / s_excl_j)
            place_probs[i] += p2
            # j が1着、k が2着、i が3着
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = win_probs[k]
                s_excl_jk = s_excl_j - pk
                if s_excl_jk <= 0:
                    continue
                place_probs[i] += pj * (pk / s_excl_j) * (pi / s_excl_jk)
    return place_probs


def kelly_fraction(win_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    1/4 ケリー基準で推奨賭け率を返す（0〜1）。
    fraction=0.25 で破産リスクを抑えた保守的設定。
    """
    if odds <= 1 or win_prob <= 0:
        return 0.0
    b = odds - 1
    q = 1.0 - win_prob
    k = (b * win_prob - q) / b
    return max(0.0, k * fraction)


# ── Step1: 当日レース一覧取得 ──────────────────────────────────────────────
def get_today_races():
    print("当日レース一覧を取得中...")
    today = datetime.now().strftime("%Y%m%d")
    driver = _make_chrome_driver()
    try:
        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={today}"
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        race_info = {}
        for a in soup.find_all("a", href=re.compile(r"race_id=\d{12}")):
            race_id = re.search(r"race_id=(\d{12})", a["href"]).group(1)
            if race_id in race_info:
                continue
            parent = a.find_parent()
            text = parent.get_text(strip=True) if parent else ""
            time_match = re.search(r"(\d{1,2}:\d{2})", text)
            if time_match:
                race_info[race_id] = time_match.group(1)
        return race_info
    finally:
        driver.quit()


# ── Step2: 出馬表・オッズ取得 ──────────────────────────────────────────────
def get_race_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        # netkeibaがUTF-8に移行したため、固定指定ではなく自動判定にする
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}

        race_div = soup.find("div", class_="RaceData01")
        if race_div:
            text = race_div.get_text()
            print(f"  RaceData01: {text[:80]}")
            dist_match = re.search(r"(\d{3,4})m", text)
            race_info["距離"] = int(dist_match.group(1)) if dist_match else None
            race_info["馬場種別"] = "芝" if "芝" in text else "ダート"
            race_info["is_turf"] = 1.0 if "芝" in text else 0.0

            # 回り（右/左/直）を抽出 — モデルの学習データと一致させる
            m_turn = re.search(r"[芝ダ](右|左|直)", text)
            race_info["回り"] = m_turn.group(1) if m_turn else None
            race_info["回り_num"] = {"右": 1, "左": 2, "直": 3}.get(race_info["回り"])

            race_info["馬場状態"] = None
            full_text = "".join(
                elem for elem in race_div.parent.find_all(string=True)
            ).replace("\n", "").replace(" ", "").replace("\xa0", "")
            for condition in ["不良", "稍重", "重", "良"]:
                if condition in full_text:
                    race_info["馬場状態"] = condition
                    break
            race_info["馬場状態_num"] = {"良": 1, "稍重": 2, "重": 3, "不良": 4}.get(race_info["馬場状態"])
        else:
            print("  RaceData01: 見つからず")

        race_div2 = soup.find("div", class_="RaceData02")
        if race_div2:
            text2 = race_div2.get_text()
            race_info["レースクラス"] = None
            for cls in ["新馬", "未勝利", "1勝クラス", "2勝クラス",
                        "3勝クラス", "オープン", "G1", "G2", "G3"]:
                if cls in text2:
                    race_info["レースクラス"] = cls
                    break
            # クラス_num: モデル学習時(scraper_fixed.py)と同じ 1-8 スケール
            _cls_map_live = {
                "新馬": 1, "未勝利": 1,
                "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
                "オープン": 5, "G3": 6, "G2": 7, "G1": 8,
            }
            race_info["クラス_num"] = _cls_map_live.get(race_info["レースクラス"])

        table = soup.find("table", class_="Shutuba_Table")
        print(f"  Shutuba_Table: {table is not None}")
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr"):
            td = tr.find_all("td")
            if len(td) >= 8:
                # 馬名セル(td[3])のリンクから horse_id を取得（血統結合に使う）。
                # netkeibaのURL構造 /horse/数字 は文字コード変更後も不変。
                horse_id = None
                a_tag = td[3].find("a", href=re.compile(r"/horse/\d+"))
                if a_tag and a_tag.has_attr("href"):
                    m = re.search(r"/horse/(\d+)", a_tag["href"])
                    if m:
                        horse_id = m.group(1)
                rows.append({
                    "枠番":   td[0].get_text(strip=True),
                    "馬番":   td[1].get_text(strip=True),
                    "馬名":   td[3].get_text(strip=True),
                    "horse_id": horse_id,
                    "性齢":   td[4].get_text(strip=True),
                    "斤量":   td[5].get_text(strip=True),
                    "騎手":   td[6].get_text(strip=True),
                    "調教師": td[7].get_text(strip=True),
                    "馬体重": td[8].get_text(strip=True) if len(td) > 8 else "",
                })

        print(f"  取得行数: {len(rows)}")
        if not rows:
            return None

        df = pd.DataFrame(rows)
        for key, val in race_info.items():
            df[key] = val
        df["race_id"] = str(race_id)
        df["上り"] = np.nan

    except Exception as e:
        import traceback
        print(f"出馬表取得エラー: {race_id}")
        traceback.print_exc()
        return None

    # ── オッズ・人気取得（Selenium）──
    # ── オッズ取得（単勝・複勝・ワイド・馬連）────────────────────────────
    def fetch_odds_page(race_id, odds_type, sleep_sec=3):
        """netkeibaのオッズページをSeleniumで取得"""
        driver = _make_chrome_driver()
        try:
            url = (
                f"https://race.netkeiba.com/odds/index.html"
                f"?race_id={race_id}&type={odds_type}"
            )
            driver.get(url)
            time.sleep(sleep_sec)
            return BeautifulSoup(driver.page_source, "html.parser")
        finally:
            driver.quit()

    # ── 単勝・人気取得 ──
    try:
        soup_tan = fetch_odds_page(race_id, "b1")
        tables = soup_tan.find_all("table", class_="RaceOdds_HorseList_Table")
        odds_table = tables[0] if tables else None
        if odds_table:
            temp = []
            for tr in odds_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 6:
                    umaban = td[1].get_text(strip=True)
                    try:
                        odds_val = float(td[5].get_text(strip=True))
                    except ValueError:
                        odds_val = np.nan
                    temp.append((umaban, odds_val))
            valid = [(u, o) for u, o in temp if not np.isnan(o)]
            valid_sorted = sorted(valid, key=lambda x: x[1])
            ninki_map = {u: i+1 for i, (u, _) in enumerate(valid_sorted)}
            odds_map  = {u: (o, ninki_map.get(u, np.nan)) for u, o in temp}
            df["単勝オッズ"] = df["馬番"].astype(str).map(lambda x: odds_map.get(x, (np.nan, np.nan))[0])
            df["人気"]      = df["馬番"].astype(str).map(lambda x: odds_map.get(x, (np.nan, np.nan))[1])
            print(f"  単勝オッズ取得成功: {len(valid)}頭分")
        else:
            df["単勝オッズ"] = np.nan
            df["人気"]      = np.nan
    except Exception as e:
        print(f"  単勝オッズ取得エラー: {e}")
        df["単勝オッズ"] = np.nan
        df["人気"]      = np.nan

    # ── 複勝オッズ取得 ──
    try:
        soup_fuku = fetch_odds_page(race_id, "b2")
        tables = soup_fuku.find_all("table", class_="RaceOdds_HorseList_Table")
        fuku_table = tables[0] if tables else None
        fuku_map = {}
        if fuku_table:
            for tr in fuku_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 6:
                    umaban = td[1].get_text(strip=True)
                    # 複勝は最低・最高オッズの範囲で表示される場合あり
                    odds_text = td[5].get_text(strip=True)
                    try:
                        # "1.5 - 2.3" のような形式に対応
                        odds_vals = [float(x) for x in odds_text.replace("－", "-").split("-") if x.strip()]
                        fuku_min = min(odds_vals)
                        fuku_max = max(odds_vals)
                        fuku_map[umaban] = (fuku_min, fuku_max)
                    except:
                        fuku_map[umaban] = (np.nan, np.nan)
        df["複勝オッズ_min"] = df["馬番"].astype(str).map(lambda x: fuku_map.get(x, (np.nan, np.nan))[0])
        df["複勝オッズ_max"] = df["馬番"].astype(str).map(lambda x: fuku_map.get(x, (np.nan, np.nan))[1])
        print(f"  複勝オッズ取得成功: {len(fuku_map)}頭分")
    except Exception as e:
        print(f"  複勝オッズ取得エラー: {e}")
        df["複勝オッズ_min"] = np.nan
        df["複勝オッズ_max"] = np.nan

    # ── ワイド・馬連オッズ取得（組み合わせ形式） ──
    try:
        soup_wide = fetch_odds_page(race_id, "b4")
        # ワイドは馬番ペアごとのオッズ
        wide_map = {}
        wide_table = soup_wide.find("table", id="odds_wide_block")
        if wide_table is None:
            wide_tables = soup_wide.find_all("table", class_="RaceOdds_HorseList_Table")
            wide_table = wide_tables[0] if wide_tables else None
        if wide_table:
            for tr in wide_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 4:
                    try:
                        uma1 = td[0].get_text(strip=True)
                        uma2 = td[1].get_text(strip=True)
                        odds_text = td[3].get_text(strip=True)
                        odds_vals = [float(x) for x in odds_text.replace("－","-").split("-") if x.strip()]
                        wide_map[(uma1, uma2)] = (min(odds_vals), max(odds_vals))
                    except:
                        pass
        df["_wide_map"] = [wide_map] * len(df)
        print(f"  ワイドオッズ取得成功: {len(wide_map)}組み合わせ")
    except Exception as e:
        print(f"  ワイドオッズ取得エラー: {e}")
        df["_wide_map"] = [{}] * len(df)

    try:
        soup_umaren = fetch_odds_page(race_id, "b5")
        umaren_map = {}
        umaren_tables = soup_umaren.find_all("table", class_="RaceOdds_HorseList_Table")
        umaren_table = umaren_tables[0] if umaren_tables else None
        if umaren_table:
            for tr in umaren_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 4:
                    try:
                        uma1 = td[0].get_text(strip=True)
                        uma2 = td[1].get_text(strip=True)
                        odds_val = float(td[3].get_text(strip=True))
                        umaren_map[(uma1, uma2)] = odds_val
                    except:
                        pass
        df["_umaren_map"] = [umaren_map] * len(df)
        print(f"  馬連オッズ取得成功: {len(umaren_map)}組み合わせ")
    except Exception as e:
        print(f"  馬連オッズ取得エラー: {e}")
        df["_umaren_map"] = [{}] * len(df)

    # ── 馬単オッズ取得（順序あり：1着→2着） ──
    try:
        soup_umatan = fetch_odds_page(race_id, "b6")
        umatan_map = {}
        umatan_tables = soup_umatan.find_all("table", class_="RaceOdds_HorseList_Table")
        umatan_table = umatan_tables[0] if umatan_tables else None
        if umatan_table:
            for tr in umatan_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 4:
                    try:
                        uma1 = td[0].get_text(strip=True)
                        uma2 = td[1].get_text(strip=True)
                        odds_val = float(td[3].get_text(strip=True))
                        umatan_map[(uma1, uma2)] = odds_val  # (1着馬番, 2着馬番) -> オッズ
                    except:
                        pass
        df["_umatan_map"] = [umatan_map] * len(df)
        print(f"  馬単オッズ取得成功: {len(umatan_map)}組み合わせ")
    except Exception as e:
        print(f"  馬単オッズ取得エラー: {e}")
        df["_umatan_map"] = [{}] * len(df)

    return df


# ── Step3: 特徴量構築 ──────────────────────────────────────────────────────
def build_features(race_df, history_df):
    # ── 馬ごとの過去成績を集計 ──
    horse_stats = {}
    for horse, group in history_df.groupby("馬名"):
        valid = group.dropna(subset=["着順_num"])
        if len(valid) == 0:
            continue
        # 【バグ修正】race_id の最大値で最新レースを特定（ソート保証）
        valid = valid.sort_values("race_id")
        last3 = valid.tail(3)

        if "馬場状態_num" in valid.columns:
            goods = valid[valid["馬場状態_num"] == 1]
            heavy = valid[valid["馬場状態_num"] >= 3]
        else:
            goods = pd.DataFrame()
            heavy = pd.DataFrame()

        horse_stats[horse] = {
            "過去出走数":           len(valid),
            "過去平均着順":         valid["着順_num"].mean(),
            "過去勝率":             (valid["着順_num"] == 1).mean(),
            "過去複勝率":           (valid["着順_num"] <= 3).mean(),
            "過去平均上り":         valid["上り"].mean() if "上り" in valid.columns else np.nan,
            "直近3走平均着順":     last3["着順_num"].mean(),
            "過去平均タイム秒":     valid["タイム秒"].mean() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均タイム秒": last3["タイム秒"].mean() if "タイム秒" in last3.columns else np.nan,
            "過去最速タイム秒":     valid["タイム秒"].min()  if "タイム秒" in valid.columns else np.nan,
            "直近3走平均上り":     last3["上り"].mean() if "上り" in last3.columns else np.nan,
            "過去平均体重増減":     valid["体重増減"].mean() if "体重増減" in valid.columns else np.nan,
            "過去最速上り":         valid["上り"].min() if "上り" in valid.columns else np.nan,
            "上り偏差":             valid["上り"].std() if "上り" in valid.columns else np.nan,
            "良馬場勝率":           (goods["着順_num"] == 1).mean() if len(goods) > 0 else 0.0,
            "重馬場勝率":           (heavy["着順_num"] == 1).mean() if len(heavy) > 0 else 0.0,
            # 追加特徴量用
            "直近斤量":             valid["斤量"].iloc[-1] if "斤量" in valid.columns else np.nan,
            "直近騎手":             str(valid["騎手"].iloc[-1]) if "騎手" in valid.columns else "",
            "valid_df":             valid,
        }

    # ── 騎手・調教師の累積成績 ──
    jockey_stats  = {}
    trainer_stats = {}
    for _, row in history_df.sort_values("race_id").iterrows():
        j = str(row.get("騎手", ""))
        t = str(row.get("調教師", ""))
        chakujun = row.get("着順_num", np.nan)
        if pd.isna(chakujun):
            continue
        for stats, key in [(jockey_stats, j), (trainer_stats, t)]:
            if key not in stats:
                stats[key] = {"runs": 0, "wins": 0, "top3": 0}
            stats[key]["runs"] += 1
            if chakujun == 1:
                stats[key]["wins"] += 1
            if chakujun <= 3:
                stats[key]["top3"] += 1

    # ── 各馬の特徴量を構築 ──
    rows = []
    for _, row in race_df.iterrows():
        horse   = str(row.get("馬名", ""))
        jockey  = str(row.get("騎手", ""))
        trainer = str(row.get("調教師", ""))
        current_dist = pd.to_numeric(row.get("距離", np.nan), errors="coerce")

        feat = {
            "馬名":       horse,
            "race_id":     row.get("race_id", ""),
            "単勝オッズ": pd.to_numeric(row.get("単勝オッズ", np.nan), errors="coerce"),
            "人気":       pd.to_numeric(row.get("人気", np.nan), errors="coerce"),
            "枠番":       pd.to_numeric(row.get("枠番", np.nan), errors="coerce"),
            "馬番":       pd.to_numeric(row.get("馬番", np.nan), errors="coerce"),
            "斤量":       pd.to_numeric(row.get("斤量", np.nan), errors="coerce"),
            "距離":       current_dist,
            "is_turf":    1 if row.get("馬場種別") == "芝" else 0,
            "馬場状態":   row.get("馬場状態", None),
        }

        baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
        feat["馬場状態_num"] = baba_map.get(row.get("馬場状態"), np.nan)

        class_map = {
            "新馬": 1, "未勝利": 1,
            "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
            "オープン": 5, "G3": 6, "G2": 7, "G1": 8,
        }
        feat["クラス_num"] = class_map.get(row.get("レースクラス"), np.nan)
        feat["回り_num"] = row.get("回り_num")

        seire = str(row.get("性齢", ""))
        feat["is_male"]      = 1 if "牡" in seire else 0
        feat["is_female"]    = 1 if "牝" in seire else 0
        feat["is_castrated"] = 1 if "セ" in seire else 0
        age_match = re.search(r"(\d+)", seire)
        feat["年齢"] = int(age_match.group(1)) if age_match else np.nan

        weight_raw = str(row.get("馬体重", ""))
        w_match = re.search(r"(\d+)", weight_raw)
        feat["馬体重"] = int(w_match.group(1)) if w_match else np.nan
        d_match = re.search(r"\(([+-]?\d+)\)", weight_raw)
        feat["体重増減"] = int(d_match.group(1)) if d_match else np.nan

        # 馬の過去成績
        hs = horse_stats.get(horse, {})
        for key in [
            "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
            "過去平均上り", "直近3走平均着順", "過去平均タイム秒",
            "直近3走平均タイム秒", "過去最速タイム秒",
            "直近3走平均上り", "過去平均体重増減", "良馬場勝率", "重馬場勝率",
            "過去最速上り", "上り偏差",
        ]:
            feat[key] = hs.get(key, np.nan)

        # 【バグ修正】同距離過去成績：feature.py と同じく ±200m でマッチ
        feat["同距離過去勝率"]     = 0.0
        feat["同距離過去平均着順"] = np.nan
        if "valid_df" in hs and pd.notna(current_dist):
            v_df = hs["valid_df"]
            if "距離" in v_df.columns:
                same_dist_df = v_df[
                    (pd.to_numeric(v_df["距離"], errors="coerce") >= current_dist - 200) &
                    (pd.to_numeric(v_df["距離"], errors="coerce") <= current_dist + 200)
                ]
                if len(same_dist_df) > 0:
                    feat["同距離過去勝率"]     = (same_dist_df["着順_num"] == 1).mean()
                    feat["同距離過去平均着順"] = same_dist_df["着順_num"].mean()

        # 【バグ修正】前走間隔：ソート済み valid_df の最終行を使う
        if "valid_df" in hs and "race_id" in hs["valid_df"].columns and len(hs["valid_df"]) > 0:
            try:
                current_date = datetime.strptime(
                    str(row.get("race_id", ""))[:8], "%Y%m%d"
                )
                last_race_id = str(hs["valid_df"]["race_id"].astype(str).max())[:8]
                last_date    = datetime.strptime(last_race_id, "%Y%m%d")
                feat["前走間隔"] = (current_date - last_date).days / 7
            except Exception:
                feat["前走間隔"] = np.nan
        else:
            feat["前走間隔"] = np.nan

        # ── 距離変化系特徴量（学習時features.pyと同じロジック）──
        prev_dist = np.nan
        if "valid_df" in hs and len(hs["valid_df"]) > 0 and "距離" in hs["valid_df"].columns:
            try:
                prev_dist = pd.to_numeric(
                    hs["valid_df"]["距離"].iloc[-1], errors="coerce"
                )
            except Exception:
                prev_dist = np.nan
        feat["前走距離"] = prev_dist
        if pd.notna(current_dist) and pd.notna(prev_dist):
            feat["距離変化"]     = current_dist - prev_dist
            feat["距離変化比率"] = current_dist / prev_dist if prev_dist > 0 else np.nan
            feat["大幅延長フラグ"] = 1.0 if (current_dist - prev_dist) > 200 else 0.0
            feat["大幅短縮フラグ"] = 1.0 if (current_dist - prev_dist) < -200 else 0.0
            feat["距離延長幅"]   = max(current_dist - prev_dist, 0)
        else:
            feat["距離変化"]     = np.nan
            feat["距離変化比率"] = np.nan
            feat["大幅延長フラグ"] = np.nan
            feat["大幅短縮フラグ"] = np.nan
            feat["距離延長幅"]   = np.nan
        feat["長距離フラグ"] = 1.0 if (pd.notna(current_dist) and current_dist >= 2001) else 0.0
        feat["大幅延長×長距離"] = (
            (feat["大幅延長フラグ"] if pd.notna(feat["大幅延長フラグ"]) else 0)
            * feat["長距離フラグ"]
        )
        # 距離延長×先行は先行馬フラグ確定後に計算（下流で補完）

        prev_jockey    = hs.get("直近騎手", "")
        feat["乗り替わり"] = 0 if (prev_jockey == "" or prev_jockey == current_jockey) else 1

        prev_kinryo    = hs.get("直近斤量", np.nan)
        current_kinryo = feat["斤量"]
        feat["斤量変化"] = (
            current_kinryo - prev_kinryo
            if pd.notna(prev_kinryo) and pd.notna(current_kinryo)
            else np.nan
        )

        interval = feat.get("前走間隔", np.nan)
        feat["連闘"]   = 1 if pd.notna(interval) and interval <= 1  else 0
        feat["休み明け"] = 1 if pd.notna(interval) and interval >= 12 else 0
        feat["負担率"] = (
            current_kinryo / feat["馬体重"]
            if pd.notna(current_kinryo) and pd.notna(feat.get("馬体重")) and feat["馬体重"] > 0
            else np.nan
        )

        # 騎手・調教師成績
        js = jockey_stats.get(jockey, {})
        feat["騎手勝率"]   = js["wins"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        feat["騎手複勝率"] = js["top3"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        ts = trainer_stats.get(trainer, {})
        feat["調教師勝率"]   = ts["wins"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan
        feat["調教師複勝率"] = ts["top3"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan

        rows.append(feat)

    pdf = pd.DataFrame(rows)
    pdf["出走頭数"]    = len(pdf)
    pdf["馬体重_相対"] = pdf["馬体重"] - pdf["馬体重"].mean()
    pdf["斤量_相対"]   = pdf["斤量"]   - pdf["斤量"].mean()
    pdf["人気順位"]    = pdf["人気"].rank()
    pdf["人気_inv"]    = 1 / pdf["人気"].clip(lower=1)

    race_id_str = str(pdf["race_id"].iloc[0])
    pdf["競馬場cd"]   = int(race_id_str[4:6])
    pdf["レース番号"] = int(race_id_str[10:12])
    pdf["距離カテゴリ"] = pd.cut(
        pdf["レース番号"], bins=[0, 4, 8, 12], labels=[1, 2, 3]
    ).astype(float)
    pdf["距離別過去平均着順"] = np.nan
    pdf["上り"] = np.nan

    # 距離別過去平均上り（horse_statsから距離カテゴリ別に取得）
    def get_dist_avg_agari(row):
        horse = row["馬名"]
        hs = horse_stats.get(horse, {})
        v_df = hs.get("valid_df", pd.DataFrame())
        if len(v_df) == 0 or "上り" not in v_df.columns:
            return np.nan
        dist_cat = row.get("距離カテゴリ", np.nan)
        if pd.isna(dist_cat):
            return np.nan
        # 距離カテゴリに対応するレース番号範囲でフィルタ
        bins = {1.0: (1, 4), 2.0: (5, 8), 3.0: (9, 12)}
        rng = bins.get(float(dist_cat))
        if rng is None:
            return np.nan
        if "レース番号" in v_df.columns:
            same = v_df[
                (pd.to_numeric(v_df["レース番号"], errors="coerce") >= rng[0]) &
                (pd.to_numeric(v_df["レース番号"], errors="coerce") <= rng[1])
            ]
            return same["上り"].mean() if len(same) > 0 else np.nan
        return np.nan

    pdf["距離別過去平均上り"] = pdf.apply(get_dist_avg_agari, axis=1)

    # レース内ランク特徴量
    pdf["レース内_過去勝率ランク"] = pdf["過去勝率"].rank(
        ascending=False, method="min"
    )
    pdf["レース内_直近3走平均着順ランク"] = pdf["直近3走平均着順"].rank(
        ascending=True, method="min"
    )
    pdf["レース内_過去平均上りランク"] = pdf["過去平均上り"].rank(
        ascending=True, method="min"
    )
    pdf["レース内_騎手勝率ランク"] = pdf["騎手勝率"].rank(
        ascending=False, method="min"
    )

    # 全 FEATURE_COLS を数値型に統一
    for col in FEATURE_COLS:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    return pdf


# ── Step4: メール本文作成 ──────────────────────────────────────────────────
def make_email_body(race_id, pdf):
    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
    now      = datetime.now().strftime("%Y/%m/%d %H:%M")

    lines = [f"【競馬AI予測】{jyo_name} {race_no}R  {now}\n"]
    lines.append("=" * 40)

    baba = "不明"
    if "馬場状態" in pdf.columns:
        val = pdf["馬場状態"].iloc[0]
        if pd.notna(val) and str(val).strip() not in ("", "None"):
            baba = str(val).strip()

    dist = (
        int(pdf["距離"].iloc[0])
        if "距離" in pdf.columns and pd.notna(pdf["距離"].iloc[0])
        else "不明"
    )
    turf = "芝" if pdf["is_turf"].iloc[0] == 1 else "ダート"
    lines.append(f"馬場: {turf} {dist}m  状態: {baba}\n")

    for mark, label in [("◎", "◎本命"), ("○", "○対抗"), ("▲", "▲穴馬")]:
        row_match = pdf[pdf["印"] == mark]
        if row_match.empty:
            continue
        row = row_match.iloc[0]

        odds     = row.get("単勝オッズ", np.nan)
        pop      = row.get("人気", np.nan)
        weight   = row.get("馬体重", np.nan)
        weight_d = row.get("体重増減", np.nan)
        win_p    = row.get("勝ち確率", np.nan)
        place_p  = row.get("複勝確率", np.nan)
        ev_win   = row.get("単勝期待値", np.nan)
        ev_place = row.get("複勝期待値", np.nan)
        winrate  = row.get("過去勝率", np.nan)
        runs     = int(row.get("過去出走数", 0)) if pd.notna(row.get("過去出走数")) else 0
        strategy = row.get("該当戦略", "")
        kelly    = row.get("推奨賭け率", np.nan)

        odds_str     = f"{odds}倍" if pd.notna(odds) and odds > 0 else "未確定"
        pop_str      = f"{int(pop)}番人気" if pd.notna(pop) and pop > 0 else "未確定"
        weight_str   = (
            f"{int(weight)}kg({weight_d:+.0f})"
            if pd.notna(weight) and pd.notna(weight_d)
            else "当日発表待ち"
        )
        win_p_str    = f"{win_p*100:.1f}%"   if pd.notna(win_p)    else "-"
        place_p_str  = f"{place_p*100:.1f}%" if pd.notna(place_p)  else "-"
        ev_win_str   = f"{ev_win:+.2f}"       if pd.notna(ev_win)   else "-"
        ev_place_str = f"{ev_place:+.2f}"     if pd.notna(ev_place) else "-"
        winrate_str  = (
            f"{winrate*100:.0f}%({runs}走)"
            if pd.notna(winrate) and runs > 0
            else "データなし"
        )
        kelly_str    = f"{kelly*100:.1f}%" if pd.notna(kelly) and kelly > 0 else "-"

        lines.append(f"\n{label} {row['馬名']}")
        lines.append(f"  オッズ: {odds_str}  {pop_str}")
        lines.append(f"  馬体重: {weight_str}")
        lines.append(f"  過去勝率: {winrate_str}")
        lines.append(f"  AI勝ち確率: {win_p_str}  複勝確率: {place_p_str}")
        lines.append(f"  単勝期待値: {ev_win_str}  複勝期待値: {ev_place_str}")
        lines.append(f"  推奨賭け率（1/4Kelly）: {kelly_str}")

        if strategy:
            verdict = f"🔥【高回収率バックテスト該当】{strategy}"
        elif pd.notna(ev_win):
            if win_p < 0.03:
                verdict = "❌ 見送り（超大穴・ノイズ判定）"
            elif ev_win >= 1.0 and win_p >= 0.12:
                verdict = "✅ 強く買い推奨"
            elif ev_win >= 0.5 and win_p >= 0.06:
                verdict = "🟡 買い推奨"
            elif ev_win >= 0.2 and win_p >= 0.04:
                verdict = "🟢 買い検討"
            else:
                verdict = "❌ 見送り"
        else:
            verdict = "⚠️ オッズ確定後に判断"

        lines.append(f"  【判定】{verdict}")

    lines.append("\n" + "=" * 40)

    valid_pdf = pdf[pdf["勝ち確率"] >= 0.03]

    lines.append("\n📊 単勝期待値ランキング TOP3（勝率3%以上）")
    for _, row in valid_pdf.sort_values("単勝期待値", ascending=False).head(3).iterrows():
        ev    = row.get("単勝期待値", np.nan)
        odds  = row.get("単勝オッズ", np.nan)
        win_p = row.get("勝ち確率", np.nan)
        if pd.notna(ev):
            lines.append(
                f"  {row['馬名']}: 期待値{ev:+.2f} "
                f"(確率{win_p*100:.1f}% × {odds}倍)"
            )

    lines.append("\n📊 複勝期待値ランキング TOP3（勝率3%以上）")
    for _, row in valid_pdf.sort_values("複勝期待値", ascending=False).head(3).iterrows():
        ev_p    = row.get("複勝期待値", np.nan)
        place_p = row.get("複勝確率", np.nan)
        est_odds = row.get("複勝推定オッズ", np.nan)
        if pd.notna(ev_p):
            lines.append(
                f"  {row['馬名']}: 期待値{ev_p:+.2f} "
                f"(確率{place_p*100:.1f}% × {est_odds:.2f}倍推定)"
            )

    lines.append("\n※複勝オッズはハーヴィルモデルで推定。実際と異なる場合があります。")

    # ── 馬券種別 買い目提案 ──────────────────────────────────────────
    lines.append("\n" + "=" * 40)
    lines.append("🎯 買い目提案")
    lines.append("=" * 40)

    # 単勝推奨
    lines.append("\n【単勝】")
    strategy_horses = pdf[pdf["該当戦略"] != ""].sort_values("単勝期待値", ascending=False)
    if not strategy_horses.empty:
        for _, row in strategy_horses.head(2).iterrows():
            ev  = row.get("単勝期待値", np.nan)
            odd = row.get("単勝オッズ", np.nan)
            kelly = row.get("推奨賭け率", np.nan)
            kelly_s = f"{kelly*100:.1f}%" if pd.notna(kelly) and kelly > 0 else "-"
            lines.append(
                f"  ✅ {row['馬名']} {odd}倍 "
                f"期待値{ev:+.2f} Kelly:{kelly_s} "
                f"({row['該当戦略']})"
            )
    else:
        lines.append("  （戦略該当馬なし）")

    # 複勝推奨
    lines.append("\n【複勝】")
    valid_fuku = pdf[pdf["複勝確率"] >= 0.35].copy()
    if "複勝期待値_実" in pdf.columns:
        valid_fuku = valid_fuku[valid_fuku["複勝期待値_実"] >= 0.1]
        valid_fuku = valid_fuku.sort_values("複勝期待値_実", ascending=False)
        col = "複勝期待値_実"
    else:
        valid_fuku = valid_fuku.sort_values("複勝確率", ascending=False)
        col = "複勝確率"
    if not valid_fuku.empty:
        for _, row in valid_fuku.head(3).iterrows():
            fuku_min = row.get("複勝オッズ_min", np.nan)
            fuku_max = row.get("複勝オッズ_max", np.nan)
            if pd.notna(fuku_min):
                odds_s = f"{fuku_min}〜{fuku_max}倍"
            else:
                odds_s = "オッズ確定前"
            pp = row["複勝確率"]
            ev = row.get(col, np.nan)
            lines.append(
                f"  ✅ {row['馬名']} {odds_s} "
                f"複勝確率{pp*100:.1f}% 期待値{ev:+.2f}"
            )
    else:
        lines.append("  （推奨馬なし）")

    # ワイド買い目
    wide_bets = pdf["_wide_bets"].iloc[0] if "_wide_bets" in pdf.columns else []
    if wide_bets:
        lines.append("\n【ワイド】")
        sorted_wide = sorted(wide_bets, key=lambda x: x["ワイド期待値"], reverse=True)
        for bet in sorted_wide[:3]:
            u1 = bet["馬番1"]
            u2 = bet["馬番2"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["ワイド期待値"]
            mark = "✅" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} × {n2} "
                f"{bet['ワイドオッズ_min']}〜{bet['ワイドオッズ_max']}倍 "
                f"的中率{bet['ワイド的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    # 馬連買い目
    umaren_bets = pdf["_umaren_bets"].iloc[0] if "_umaren_bets" in pdf.columns else []
    if umaren_bets:
        lines.append("\n【馬連】")
        sorted_umaren = sorted(umaren_bets, key=lambda x: x["馬連期待値"], reverse=True)
        for bet in sorted_umaren[:3]:
            u1 = bet["馬番1"]
            u2 = bet["馬番2"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["馬連期待値"]
            mark = "✅" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} × {n2} "
                f"{bet['馬連オッズ']}倍 "
                f"的中率{bet['馬連的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    # 馬単買い目
    umatan_bets = pdf["_umatan_bets"].iloc[0] if "_umatan_bets" in pdf.columns else []
    if umatan_bets:
        lines.append("\n【馬単】")
        sorted_umatan = sorted(umatan_bets, key=lambda x: x["馬単期待値"], reverse=True)
        for bet in sorted_umatan[:3]:
            u1 = bet["馬番1着"]
            u2 = bet["馬番2着"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["馬単期待値"]
            mark = "✅" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} → {n2} "
                f"{bet['馬単オッズ']}倍 "
                f"的中率{bet['馬単的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    lines.append("\n※AIによる予測です。投資は自己責任でお願いします。")
    return "\n".join(lines)


# ── Step5: メール送信 ──────────────────────────────────────────────────────
def send_email(subject, body):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        print("  メール設定が未完了（環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS を設定）")
        return
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = TO_ADDRESS
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.send_message(msg)
        print(f"  メール送信完了 → {TO_ADDRESS}")
    except Exception as e:
        print(f"  メール送信エラー: {e}")


# ── Step6: 1レース実行 ────────────────────────────────────────────────────
def run_single_race(race_id, models, use_cols, history_df, mf_models=None, mf_cols=None,
                    place2_pack=None, place3_pack=None):
    print(f"\n[{datetime.now().strftime('%H:%M')}] {race_id} 予測開始...")

    print("出馬表取得中...")
    race_df = get_race_data(race_id)
    if race_df is None:
        print(f"  出馬表取得失敗: {race_id}")
        return

    print(f"  出馬表取得成功: {len(race_df)}頭")

    print("特徴量構築中...")
    try:
        # 一本化: 学習(features.py)と同じパイプラインで特徴量を作る。
        # これにより血統・距離適性・騎手トレンド・好調度が学習と完全一致で計算される。
        # 失敗時は従来の予測専用 build_features にフォールバック（安全策）。
        try:
            import features as _features
            pdf = _features.build_features_for_prediction(race_df, history_df)
            # 必須列が揃っているか簡易チェック（揃わなければ従来法へ）
            if pdf is None or len(pdf) == 0 or "馬名" not in pdf.columns:
                raise ValueError("一本化パイプラインの出力が不正")
            print(f"  特徴量構築成功(一本化): {len(pdf)}行")
        except Exception as e_uni:
            print(f"  一本化パイプライン不可({e_uni}) → 従来法にフォールバック")
            pdf = build_features(race_df, history_df)
            print(f"  特徴量構築成功(従来): {len(pdf)}行")
    except Exception as e:
        import traceback
        print(f"  特徴量構築エラー: {e}")
        traceback.print_exc()
        return

    print("予測中...")
    try:
        # 学習時と同じ列・順序で入力（use_cols は model.pkl から取得）
        X = pdf.reindex(columns=use_cols)
        preds = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
        pdf["予測着順スコア"] = preds
        pdf["予測順位"] = pdf["予測着順スコア"].rank(ascending=False)
        print("  予測成功")
    except Exception as e:
        import traceback
        print(f"  予測エラー: {e}")
        traceback.print_exc()
        return

    # 勝ち確率はレース内で正規化（合計100%・FUJIAI型No-Odds方式）。
    # 生確率は校正されているがレース内合計が100%にならず期待値が過小になるため、
    # レース内で正規化して期待値計算・表示を適正にする。
    raw = np.clip(np.nan_to_num(pdf["予測着順スコア"].values, nan=0.0), 0, 1)
    raw_sum = raw.sum()
    win_probs = raw / raw_sum if raw_sum > 0 else np.ones(len(raw)) / len(raw)
    pdf["勝ち確率"]   = win_probs
    pdf["勝ち確率_生"] = raw

    # ── 連対率・複勝率（3モデルがあれば独立予想、なければハーヴィル） ──
    use_multi = place2_pack is not None and place3_pack is not None
    if use_multi:
        try:
            p2_models, p2_cols = place2_pack
            p3_models, p3_cols = place3_pack
            X_p2 = pdf.reindex(columns=p2_cols)
            X_p3 = pdf.reindex(columns=p3_cols)
            p2_raw = np.mean([m.predict_proba(X_p2)[:, 1] for m in p2_models], axis=0)
            p3_raw = np.mean([m.predict_proba(X_p3)[:, 1] for m in p3_models], axis=0)
            p2_raw = np.clip(np.nan_to_num(p2_raw, nan=0.0), 0, 1)
            p3_raw = np.clip(np.nan_to_num(p3_raw, nan=0.0), 0, 1)
            # 正規化（FUJIAI型）: 連対率は合計≒200%、複勝率は合計≒300%。少頭数は枠数で頭打ち。
            n_runners = len(pdf)
            p2_sum = p2_raw.sum()
            p3_sum = p3_raw.sum()
            target2 = min(2, n_runners)
            target3 = min(3, n_runners)
            place2 = (p2_raw / p2_sum * target2) if p2_sum > 0 else np.full(n_runners, target2 / n_runners)
            place3 = (p3_raw / p3_sum * target3) if p3_sum > 0 else np.full(n_runners, target3 / n_runners)
            place2 = np.clip(place2, 0, 1)
            place3 = np.clip(place3, 0, 1)
            # 包含関係を保証: 勝率 ≤ 連対率 ≤ 複勝率
            place2 = np.maximum(place2, win_probs)
            place3 = np.clip(np.maximum(place3, place2), 0, 1)
            pdf["連対確率"] = place2
            pdf["複勝確率"] = place3
            pdf["連対順位"] = pd.Series(place2, index=pdf.index).rank(ascending=False)
            pdf["複勝順位"] = pd.Series(place3, index=pdf.index).rank(ascending=False)
            print("  連対率・複勝率を独立モデルで予想（正規化・包含関係保証）")
        except Exception as e:
            print(f"  3モデル予測エラー→ハーヴィルにフォールバック: {e}")
            use_multi = False

    if not use_multi:
        # 【従来】ハーヴィルモデルによる複勝確率（生の勝率ベース）
        fuku = calc_place_prob_harvill(win_probs)
        fuku = np.maximum(fuku, win_probs)
        pdf["複勝確率"] = fuku
        pdf["連対確率"] = np.maximum(pdf["複勝確率"] * 0.7, win_probs)  # 簡易近似
        pdf["連対順位"] = pdf["連対確率"].rank(ascending=False)
        pdf["複勝順位"] = pdf["複勝確率"].rank(ascending=False)

    # 複勝推定オッズ（ハーヴィル確率ベース、最低1.0倍）
    pdf["複勝推定オッズ"] = (1.0 / pdf["複勝確率"].clip(lower=0.01)).clip(upper=30.0)

    # 期待値（単一定義）
    pdf["単勝期待値"] = pdf["勝ち確率"] * pdf["単勝オッズ"] - 1
    pdf["複勝期待値"] = pdf["複勝確率"] * pdf["複勝推定オッズ"] - 1

    # 【改善】1/4ケリー基準による推奨賭け率
    pdf["推奨賭け率"] = pdf.apply(
        lambda r: kelly_fraction(r["勝ち確率"], r["単勝オッズ"]),
        axis=1,
    )

    # ── 複勝期待値（実オッズベース） ──────────────────────────────────
    if "複勝オッズ_min" in pdf.columns:
        # 実オッズが取れた場合は実オッズで計算
        pdf["複勝期待値_実"] = pdf["複勝確率"] * pdf["複勝オッズ_min"] - 1
    else:
        pdf["複勝期待値_実"] = pdf["複勝期待値"]  # ハーヴィル推定値を使用

    # ── ワイド・馬連の期待値と買い目提案 ────────────────────────────
    wide_map   = pdf["_wide_map"].iloc[0]   if "_wide_map"   in pdf.columns else {}
    umaren_map = pdf["_umaren_map"].iloc[0] if "_umaren_map" in pdf.columns else {}

    # 上位3頭の馬番を取得
    top3 = pdf.sort_values("勝ち確率", ascending=False).head(3)
    top3_umaban = top3["馬番"].astype(str).tolist()

    # ワイド期待値計算（◎×○ / ◎×▲ / ○×▲）
    wide_bets = []
    if wide_map and len(top3_umaban) >= 2:
        for i in range(len(top3_umaban)):
            for j in range(i+1, len(top3_umaban)):
                u1, u2 = top3_umaban[i], top3_umaban[j]
                key1 = (u1, u2)
                key2 = (u2, u1)
                wide_odds = wide_map.get(key1, wide_map.get(key2, (np.nan, np.nan)))
                if isinstance(wide_odds, tuple) and pd.notna(wide_odds[0]):
                    # ワイド的中確率 = 2頭が共に3着以内に入る確率（近似）
                    r1 = pdf[pdf["馬番"].astype(str) == u1]
                    r2 = pdf[pdf["馬番"].astype(str) == u2]
                    if len(r1) > 0 and len(r2) > 0:
                        p1 = float(r1["複勝確率"].iloc[0])
                        p2 = float(r2["複勝確率"].iloc[0])
                        # 近似：両方3着以内の確率
                        wide_prob = p1 * p2 * 1.5  # ハーヴィル近似係数
                        wide_prob = min(wide_prob, 0.95)
                        wide_ev = wide_prob * wide_odds[0] - 1
                        wide_bets.append({
                            "馬番1": u1, "馬番2": u2,
                            "ワイドオッズ_min": wide_odds[0],
                            "ワイドオッズ_max": wide_odds[1] if len(wide_odds) > 1 else wide_odds[0],
                            "ワイド的中確率": wide_prob,
                            "ワイド期待値": wide_ev,
                        })

    # 馬連期待値計算
    umaren_bets = []
    if umaren_map and len(top3_umaban) >= 2:
        for i in range(len(top3_umaban)):
            for j in range(i+1, len(top3_umaban)):
                u1, u2 = top3_umaban[i], top3_umaban[j]
                key1 = (u1, u2)
                key2 = (u2, u1)
                umaren_odds = umaren_map.get(key1, umaren_map.get(key2, np.nan))
                if pd.notna(umaren_odds):
                    r1 = pdf[pdf["馬番"].astype(str) == u1]
                    r2 = pdf[pdf["馬番"].astype(str) == u2]
                    if len(r1) > 0 and len(r2) > 0:
                        p1 = float(r1["勝ち確率"].iloc[0])
                        p2 = float(r2["勝ち確率"].iloc[0])
                        # 馬連的中確率 = どちらかが1着でもう一方が2着
                        umaren_prob = p1 * (p2 / (1 - p1 + 1e-9)) + p2 * (p1 / (1 - p2 + 1e-9))
                        umaren_prob = min(umaren_prob, 0.95)
                        umaren_ev = umaren_prob * umaren_odds - 1
                        umaren_bets.append({
                            "馬番1": u1, "馬番2": u2,
                            "馬連オッズ": umaren_odds,
                            "馬連的中確率": umaren_prob,
                            "馬連期待値": umaren_ev,
                        })

    # 馬単（順序あり：上位2頭の正逆2パターン）
    umatan_map = pdf["_umatan_map"].iloc[0] if "_umatan_map" in pdf.columns else {}
    umatan_bets = []
    if umatan_map and len(top3_umaban) >= 2:
        for i in range(len(top3_umaban)):
            for j in range(len(top3_umaban)):
                if i == j:
                    continue
                u1, u2 = top3_umaban[i], top3_umaban[j]  # u1=1着, u2=2着
                umatan_odds = umatan_map.get((u1, u2), np.nan)
                if pd.notna(umatan_odds):
                    r1 = pdf[pdf["馬番"].astype(str) == u1]
                    r2 = pdf[pdf["馬番"].astype(str) == u2]
                    if len(r1) > 0 and len(r2) > 0:
                        p1 = float(r1["勝ち確率"].iloc[0])
                        p2 = float(r2["勝ち確率"].iloc[0])
                        # 馬単的中確率 = u1が1着 × u2が(u1以外で)2着
                        umatan_prob = p1 * (p2 / (1 - p1 + 1e-9))
                        umatan_prob = min(umatan_prob, 0.95)
                        umatan_ev = umatan_prob * umatan_odds - 1
                        umatan_bets.append({
                            "馬番1着": u1, "馬番2着": u2,
                            "馬単オッズ": umatan_odds,
                            "馬単的中確率": umatan_prob,
                            "馬単期待値": umatan_ev,
                        })

    pdf["_wide_bets"]   = [wide_bets]   * len(pdf)
    pdf["_umaren_bets"] = [umaren_bets] * len(pdf)
    pdf["_umatan_bets"] = [umatan_bets] * len(pdf)

    # 市場フリーモデルによる乖離スコア計算
    pdf["MF予測順位"] = np.nan
    pdf["乖離スコア"] = np.nan
    if mf_models is not None and mf_cols is not None:
        try:
            X_mf = pdf.reindex(columns=mf_cols)
            mf_preds = np.mean([m.predict_proba(X_mf)[:, 1] for m in mf_models], axis=0)
            pdf["MF勝ち確率"] = mf_preds
            pdf["MF予測順位"] = pd.Series(mf_preds).rank(ascending=False).values
            # 乖離スコア：通常順位 - MF順位（プラスほど市場が過小評価）
            pdf["乖離スコア"] = pdf["予測順位"] - pdf["MF予測順位"]
            print("  市場フリー予測成功")
        except Exception as e:
            print(f"  市場フリー予測エラー（スキップ）: {e}")

    # バックテスト戦略のリアルタイム判定
    pdf["該当戦略"] = ""
    for idx, row in pdf.iterrows():
        strategies = []
        if (row["予測順位"] == 1
                and row["単勝期待値"] >= 0.3
                and 1.5 <= row["単勝オッズ"] <= 20):
            strategies.append("戦略A")
            if row["人気"] != 1:
                strategies.append("戦略A-2(超穴妙味)")
        # 戦略B廃止（回収率92.8%で赤字のため）
        if (row["人気"] >= 3
                and row["予測順位"] == 1
                and row["単勝期待値"] >= 0.3):
            strategies.append("戦略C(市場乖離)")
        if (pd.notna(row.get("前走間隔"))
                and 2 <= row["前走間隔"] <= 4
                and row["予測順位"] == 1
                and row["単勝期待値"] >= 0.2):
            strategies.append("戦略D(好ローテ)")
        # 戦略E：一時無効化（交互作用特徴量追加後、乖離スコアの有効性が崩壊したため）
        # 旧: 乖離スコア≥5 × MF予測1位（回収率168.1%→0.0%に悪化）
        # if (pd.notna(row.get("乖離スコア"))
        #         and row.get("乖離スコア", 0) >= 5
        #         and row.get("MF予測順位", 99) == 1):
        #     strategies.append("戦略E(市場見落とし)")
        # 戦略F：中京・東京 × 予測1位 × 期待値≥0.3（回収率197.2%）
        # 競馬場cd: 5=東京, 7=中京
        jyo_cd = int(str(row.get("race_id", "000000000000"))[4:6])
        if (jyo_cd in [5, 7]
                and row["予測順位"] == 1
                and row["単勝期待値"] >= 0.3
                and 1.5 <= row["単勝オッズ"] <= 20):
            strategies.append("戦略F(東京・中京)")
        # 戦略H：中山・小倉 × 予測1位 × 期待値≥0.3（回収率128.9%）
        # 競馬場cd: 6=中山, 10=小倉
        if (jyo_cd in [6, 10]
                and row["予測順位"] == 1
                and row["単勝期待値"] >= 0.3
                and 1.5 <= row["単勝オッズ"] <= 20):
            strategies.append("戦略H(中山・小倉)")
        # 戦略FG：中京・東京 × 短距離(〜1400m) × 予測1位 × 期待値≥0.3（回収率265.0%）
        if (jyo_cd in [5, 7]
                and pd.notna(row.get("距離"))
                and row.get("距離", 9999) <= 1400
                and row["予測順位"] == 1
                and row["単勝期待値"] >= 0.3
                and 1.5 <= row["単勝オッズ"] <= 20):
            strategies.append("戦略FG(東京・中京×短距離)")
        if strategies:
            pdf.at[idx, "該当戦略"] = " / ".join(strategies)

    # 印の選出（◎○▲）
    # C-4: MFモデルを主力化（回収率 MF:142.6% > 通常:112.5%）
    pdf["印"] = ""
    USE_MF_FOR_MARKS = True
    rank_col = (
        "MF勝ち確率"
        if USE_MF_FOR_MARKS
        and "MF勝ち確率" in pdf.columns
        and pdf["MF勝ち確率"].notna().any()
        else "勝ち確率"
    )

    # ◎本命：期待値0.3以上の馬の中でMF勝ち確率最大
    honmei_cands = pdf[pdf["単勝期待値"] >= 0.3].sort_values(rank_col, ascending=False)
    idx_honmei = (
        honmei_cands.index[0]
        if not honmei_cands.empty
        else pdf.sort_values(rank_col, ascending=False).index[0]
    )
    pdf.at[idx_honmei, "印"] = "◎"

    # ○対抗：本命以外で期待値0.1以上×MF勝ち確率上位
    taiko_cands = pdf[
        (pdf["単勝期待値"] >= 0.1) & (pdf.index != idx_honmei)
    ].sort_values(rank_col, ascending=False)
    if not taiko_cands.empty:
        idx_taiko = taiko_cands.index[0]
    else:
        remaining = pdf[pdf.index != idx_honmei].sort_values(rank_col, ascending=False)
        idx_taiko = remaining.index[0] if not remaining.empty else idx_honmei
    if idx_taiko != idx_honmei:
        pdf.at[idx_taiko, "印"] = "○"

    # ▲穴馬：本命・対抗以外で期待値0.3以上×オッズ5倍以上の最高期待値馬
    ana_cands = pdf[
        (pdf["単勝期待値"] >= 0.3)
        & (pdf["単勝オッズ"] >= 5.0)
        & (~pdf.index.isin([idx_honmei, idx_taiko]))
    ].sort_values("単勝期待値", ascending=False)
    if not ana_cands.empty:
        idx_ana = ana_cands.index[0]
    else:
        remaining = pdf[~pdf.index.isin([idx_honmei, idx_taiko])].sort_values(
            "単勝期待値", ascending=False
        )
        idx_ana = remaining.index[0] if not remaining.empty else idx_honmei
    if idx_ana not in [idx_honmei, idx_taiko]:
        pdf.at[idx_ana, "印"] = "▲"

    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

    # 記録保存（prediction_record_v2.csv のみに一元化）
    # result_tracker.record_from_prediction で update_results/dashboard と
    # 互換性のあるカラム形式（jyo, race, honmei, hit, honmei_actual等）で保存する
    try:
        record_from_prediction(race_id, pdf)
    except Exception as e:
        print(f"  記録保存エラー: {e}")

    # メール送信
    body    = make_email_body(race_id, pdf)
    subject = f"【競馬AI】{jyo_name} {race_no}R 予測"
    print(body)
    send_email(subject, body)


# ── Step7: スケジューラー ─────────────────────────────────────────────────
def main():
    print(f"=== 競馬AI自動予測 起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}] ===\n")

    print("モデル読み込み中...")
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    models   = saved["models"]
    use_cols = saved["use_cols"]

    # 3モデル（win/place2/place3）。新形式なら独立予想、旧形式ならハーヴィル
    place2_pack = None
    place3_pack = None
    if saved.get("format") == "multi_v1" and saved.get("place2") and saved.get("place3"):
        place2_pack = (saved["place2"]["models"], saved["place2"]["use_cols"])
        place3_pack = (saved["place3"]["models"], saved["place3"]["use_cols"])
        print("  3モデル構成(win/place2/place3)を検出 → 独立予想を使用")
    else:
        print("  旧モデル構成 → 連対率・複勝率はハーヴィル変換を使用")

    # 市場フリーモデルを読み込む（存在する場合のみ）
    mf_models = None
    mf_cols   = None
    mf_path   = os.path.join(BASE_DIR, "model_mf.pkl")
    if os.path.exists(mf_path):
        try:
            with open(mf_path, "rb") as f:
                mf_saved = pickle.load(f)
            mf_models = mf_saved["models"]
            mf_cols   = mf_saved["use_cols"]
            print("市場フリーモデル読み込み完了 → 戦略E有効")
        except Exception as e:
            print(f"市場フリーモデル読み込みスキップ: {e}")
    else:
        print("market_free_model.pkl なし → 戦略Eスキップ")

    print("履歴データ読み込み中（しばらくかかります）...")
    history_df = pd.read_csv(
        os.path.join(BASE_DIR, "race_data_clean.csv"), low_memory=False
    )
    print(f"読み込み完了: {len(history_df)}行\n")

    race_info = get_today_races()
    if not race_info:
        print("本日のレースが取得できませんでした")
        return

    print(f"\n{len(race_info)}レースをスケジュール登録中...")
    now = datetime.now()
    scheduled = 0

    for race_id, race_time in sorted(race_info.items()):
        time_match = re.search(r"(\d{1,2}):(\d{2})", race_time)
        if not time_match:
            continue

        h = int(time_match.group(1))
        m = int(time_match.group(2)) - 7
        if m < 0:
            m += 60
            h -= 1

        notify_time  = f"{h:02d}:{m:02d}"
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if scheduled_dt < now:
            print(f"  スキップ（過去）: {race_id} {notify_time}")
            continue

        schedule.every().day.at(notify_time).do(
            run_single_race,
            race_id=race_id,
            models=models,
            use_cols=use_cols,
            history_df=history_df,
            mf_models=mf_models,
            mf_cols=mf_cols,
            place2_pack=place2_pack,
            place3_pack=place3_pack,
        )
        jyo_cd   = int(str(race_id)[4:6])
        race_no  = int(str(race_id)[10:12])
        jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
        print(
            f"  予約: {jyo_name} {race_no}R ({race_id}) "
            f"→ {notify_time}（発走: {race_time}）"
        )
        scheduled += 1

    if scheduled == 0:
        print("予約できるレースがありませんでした")
        return

    print(f"\n{scheduled}件を予約しました。待機中... (Ctrl+Cで停止)\n")
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    # 自動スケジュール待機モード（各レース発走7分前に予想・メール送信）
    main()