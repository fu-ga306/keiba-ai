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

# model.py の FEATURE_COLS と完全一致させること
FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気",
    "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順", "良馬場勝率", "重馬場勝率",
    # 上がり関連特徴量
    "過去最速上り", "上り偏差", "距離別過去平均上り",
    # 回収率向上：追加特徴量
    "斤量変化", "乗り替わり", "連闘", "休み明け", "負担率",
    # レース内ランク
    "レース内_過去勝率ランク",
    "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク",
    "レース内_騎手勝率ランク",
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
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}

        race_div = soup.find("div", class_="RaceData01")
        if race_div:
            text = race_div.get_text()
            print(f"  RaceData01: {text[:80]}")
            dist_match = re.search(r"(\d{3,4})m", text)
            race_info["距離"] = int(dist_match.group(1)) if dist_match else None
            race_info["馬場種別"] = "芝" if "芝" in text else "ダート"

            race_info["馬場状態"] = None
            full_text = "".join(
                elem for elem in race_div.parent.find_all(string=True)
            ).replace("\n", "").replace(" ", "").replace("\xa0", "")
            for condition in ["不良", "稍重", "重", "良"]:
                if condition in full_text:
                    race_info["馬場状態"] = condition
                    break
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

        table = soup.find("table", class_="Shutuba_Table")
        print(f"  Shutuba_Table: {table is not None}")
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr"):
            td = tr.find_all("td")
            if len(td) >= 8:
                rows.append({
                    "枠番":   td[0].get_text(strip=True),
                    "馬番":   td[1].get_text(strip=True),
                    "馬名":   td[3].get_text(strip=True),
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
    # 【バグ修正】finally で必ず quit() するよう変更
    odds_driver = _make_chrome_driver()
    try:
        odds_url = (
            f"https://race.netkeiba.com/odds/index.html"
            f"?race_id={race_id}&type=b1"
        )
        odds_driver.get(odds_url)
        time.sleep(4)
        odds_soup = BeautifulSoup(odds_driver.page_source, "html.parser")
    finally:
        odds_driver.quit()

    try:
        odds_tables = odds_soup.find_all("table", class_="RaceOdds_HorseList_Table")
        odds_table = odds_tables[0] if odds_tables else None

        if odds_table:
            temp = []
            for tr in odds_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 6:
                    umaban   = td[1].get_text(strip=True)
                    odds_str = td[5].get_text(strip=True)
                    try:
                        odds_val = float(odds_str)
                    except ValueError:
                        odds_val = np.nan
                    temp.append((umaban, odds_val))

            valid = [(u, o) for u, o in temp if not np.isnan(o)]
            valid_sorted = sorted(valid, key=lambda x: x[1])
            ninki_map = {u: i + 1 for i, (u, _) in enumerate(valid_sorted)}
            odds_map  = {u: (o, ninki_map.get(u, np.nan)) for u, o in temp}

            df["単勝オッズ"] = df["馬番"].astype(str).map(
                lambda x: odds_map.get(x, (np.nan, np.nan))[0]
            )
            df["人気"] = df["馬番"].astype(str).map(
                lambda x: odds_map.get(x, (np.nan, np.nan))[1]
            )
            print(f"  オッズ取得成功: {len(valid)}頭分")
        else:
            df["単勝オッズ"] = np.nan
            df["人気"]      = np.nan

    except Exception as e:
        import traceback
        print(f"オッズ解析エラー: {race_id}")
        traceback.print_exc()
        df["単勝オッズ"] = np.nan
        df["人気"]      = np.nan

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
            "新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
            "3勝クラス": 5, "オープン": 6, "G3": 7, "G2": 8, "G1": 9,
        }
        feat["クラス_num"] = class_map.get(row.get("レースクラス"), np.nan)

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

        # 追加特徴量
        current_jockey = jockey
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
    lines.append("※AIによる予測です。投資は自己責任でお願いします。")
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
def run_single_race(race_id, models, use_cols, history_df, mf_models=None, mf_cols=None):
    print(f"\n[{datetime.now().strftime('%H:%M')}] {race_id} 予測開始...")

    print("出馬表取得中...")
    race_df = get_race_data(race_id)
    if race_df is None:
        print(f"  出馬表取得失敗: {race_id}")
        return

    print(f"  出馬表取得成功: {len(race_df)}頭")

    print("特徴量構築中...")
    try:
        pdf = build_features(race_df, history_df)
        print(f"  特徴量構築成功: {len(pdf)}行")
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

    # 勝ち確率の正規化
    raw = np.nan_to_num(pdf["予測着順スコア"].values, nan=0.0)
    raw = np.clip(raw, 0, None)
    win_probs = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / len(raw)
    pdf["勝ち確率"] = win_probs

    # 【改善】ハーヴィルモデルによる複勝確率
    pdf["複勝確率"] = calc_place_prob_harvill(win_probs)

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

    # 市場フリーモデルによる乖離スコア計算
    pdf["MF予測順位"] = np.nan
    pdf["乖離スコア"] = np.nan
    if mf_models is not None and mf_cols is not None:
        try:
            X_mf = pdf.reindex(columns=mf_cols)
            mf_preds = np.mean([m.predict_proba(X_mf)[:, 1] for m in mf_models], axis=0)
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
        # 戦略E：市場乖離スコア≥3 × MF予測1位（回収率133.3%）
        if (pd.notna(row.get("乖離スコア"))
                and row.get("乖離スコア", 0) >= 3
                and row.get("MF予測順位", 99) == 1):
            strategies.append("戦略E(市場見落とし)")
        if strategies:
            pdf.at[idx, "該当戦略"] = " / ".join(strategies)

    # 印の選出（◎○▲）
    pdf["印"] = ""

    # ◎本命：期待値0.3以上の馬の中で勝ち確率最大
    # → 期待値がプラスで実力も伴う馬を本命に（2着多発問題への対策）
    honmei_cands = pdf[pdf["単勝期待値"] >= 0.3].sort_values("勝ち確率", ascending=False)
    idx_honmei = (
        honmei_cands.index[0]
        if not honmei_cands.empty
        else pdf.sort_values("勝ち確率", ascending=False).index[0]
    )
    pdf.at[idx_honmei, "印"] = "◎"

    # ○対抗：本命以外で期待値0.1以上×勝ち確率上位
    taiko_cands = pdf[
        (pdf["単勝期待値"] >= 0.1) & (pdf.index != idx_honmei)
    ].sort_values("勝ち確率", ascending=False)
    if not taiko_cands.empty:
        idx_taiko = taiko_cands.index[0]
    else:
        remaining = pdf[pdf.index != idx_honmei].sort_values("勝ち確率", ascending=False)
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
    try:
        uma_honmei = pdf[pdf["印"] == "◎"].iloc[0]
        uma_taiko  = pdf[pdf["印"] == "○"].iloc[0] if not pdf[pdf["印"] == "○"].empty else uma_honmei
        uma_ana    = pdf[pdf["印"] == "▲"].iloc[0] if not pdf[pdf["印"] == "▲"].empty else uma_honmei

        csv_path_v2 = os.path.join(BASE_DIR, "prediction_record_v2.csv")
        write_header = not os.path.exists(csv_path_v2) or os.path.getsize(csv_path_v2) == 0
        with open(csv_path_v2, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "race_id,jyo,race,honmei,taiko,ana,"
                    "honmei_win_p,taiko_win_p,ana_win_p\n"
                )
            f.write(
                f"{race_id},{jyo_name},{race_no},"
                f"{uma_honmei['馬名']},{uma_taiko['馬名']},{uma_ana['馬名']},"
                f"{uma_honmei['勝ち確率']:.3f},"
                f"{uma_taiko['勝ち確率']:.3f},"
                f"{uma_ana['勝ち確率']:.3f}\n"
            )
        print("  [記録保存成功]")
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
        os.path.join(BASE_DIR, "race_features.csv"), low_memory=False
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
    # 自動スケジュール待機モード
    # main()

    # 特定日のテスト実行
    print("モデル読み込み中...")
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    models   = saved["models"]
    use_cols = saved["use_cols"]

    mf_models = None
    mf_cols   = None
    mf_path   = os.path.join(BASE_DIR, "model_mf.pkl")
    if os.path.exists(mf_path):
        try:
            with open(mf_path, "rb") as f:
                mf_saved = pickle.load(f)
            mf_models = mf_saved["models"]
            mf_cols   = mf_saved["use_cols"]
            print("市場フリーモデル読み込み完了")
        except Exception as e:
            print(f"市場フリーモデル読み込みスキップ: {e}")

    print("履歴データ読み込み中...")
    history_df = pd.read_csv(
        os.path.join(BASE_DIR, "race_features.csv"), low_memory=False
    )
    print(f"読み込み完了: {len(history_df)}行")

    target_day = "2026050211"
    race_ids   = [f"{target_day}{str(r).zfill(2)}" for r in range(1, 13)]

    for race_id in race_ids:
        run_single_race(race_id, models, use_cols, history_df, mf_models, mf_cols)
        time.sleep(2)