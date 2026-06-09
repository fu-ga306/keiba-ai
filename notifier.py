import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

# ── メール設定 ──────────────────────────
GMAIL_ADDRESS  = "hyu.1999.3.6@gmail.com"  # ← 書き換え
GMAIL_APP_PASS = "pzwy zefw clly cwgt"                  # ← アプリパスワード16文字
TO_ADDRESS     = "hyu.1999.3.6@gmail.com"

FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気", "上り", "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
]

# 競馬場コード → 名前
JYO_NAMES = {
    1:"札幌", 2:"函館", 3:"福島", 4:"新潟", 5:"東京",
    6:"中山", 7:"中京", 8:"京都", 9:"阪神", 10:"小倉"
}

def get_odds(race_id):
    """単勝オッズと人気を取得"""
    url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", class_="RaceOdds_HorseList_Table")
        if table is None:
            return {}

        temp = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) >= 6:
                umaban  = tds[1].get_text(strip=True)
                odds_str = tds[5].get_text(strip=True)
                try:
                    odds_val = float(odds_str)
                except:
                    odds_val = np.nan
                temp.append((umaban, odds_val))

        # 人気順を計算
        valid = [(u, o) for u, o in temp if not np.isnan(o)]
        valid_sorted = sorted(valid, key=lambda x: x[1])
        ninki_map = {u: i+1 for i, (u, o) in enumerate(valid_sorted)}

        odds_map = {}
        for umaban, odds_val in temp:
            odds_map[umaban] = (odds_val, ninki_map.get(umaban, np.nan))

        return odds_map

    except Exception as e:
        print(f"オッズ取得エラー: {race_id} → {e}")
        return {}

def get_race_result(race_id):
    # 出馬表URL（レース前はこちらを使う）
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}

        # レース情報取得
        race_div = soup.find("div", class_="RaceData01")
        if race_div:
            text = race_div.get_text()
            dist_match = re.search(r'(\d{3,4})m', text)
            race_info["距離"] = int(dist_match.group(1)) if dist_match else None
            race_info["馬場種別"] = "芝" if "芝" in text else "ダート"
            race_info["馬場状態"] = None
            for condition in ["不良", "重", "稍重", "良"]:
                if condition in text:
                    race_info["馬場状態"] = condition
                    break

        race_div2 = soup.find("div", class_="RaceData02")
        if race_div2:
            text2 = race_div2.get_text()
            race_info["レースクラス"] = None
            for cls in ["新馬", "未勝利", "1勝クラス", "2勝クラス",
                        "3勝クラス", "オープン", "G1", "G2", "G3"]:
                if cls in text2:
                    race_info["レースクラス"] = cls
                    break

        # 出馬表テーブル取得
        table = soup.find("table", class_="Shutuba_Table")
        if table is None:
            # 別クラス名も試す
            table = soup.find("table", id="shutuba_table")
        if table is None:
            tables = soup.find_all("table")
            for t in tables:
                if len(t.find_all("tr")) > 5:
                    table = t
                    break
        if table is None:
            print(f"テーブルが見つかりませんでした: {race_id}")
            return None

        rows = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 8:
                row = {
                    "枠番":   tds[0].get_text(strip=True),
                    "馬番":   tds[1].get_text(strip=True),
                    "馬名":   tds[3].get_text(strip=True),
                    "性齢":   tds[4].get_text(strip=True),
                    "斤量":   tds[5].get_text(strip=True),
                    "騎手":   tds[6].get_text(strip=True),
                    "調教師": tds[7].get_text(strip=True),
                    "単勝オッズ": tds[9].get_text(strip=True) if len(tds) > 9 else np.nan,
                    "人気":   tds[10].get_text(strip=True) if len(tds) > 10 else np.nan,
                    "馬体重": tds[8].get_text(strip=True) if len(tds) > 8 else np.nan,
                }
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # 列名を手動で付与（出馬表の列順）
        shutuba_cols = ["枠番", "馬番", "馬名", "性齢", "斤量", "騎手",
                        "厩舎", "馬体重", "単勝オッズ", "人気",
                        "調教師", "馬主", "馬齢"]
        if len(shutuba_cols) >= df.shape[1]:
            df.columns = shutuba_cols[:df.shape[1]]
        
        for key, val in race_info.items():
            df[key] = val
        df["race_id"] = str(race_id)
        df["上り"] = np.nan
        df["調教師"] = df.get("調教師", "")
        return df

    except Exception as e:
        print(f"エラー: {race_id} → {e}")
        return None


def build_features(race_df, history_df):
    """予測用特徴量を構築"""
    horse_stats = {}
    for horse, group in history_df.groupby("馬名"):
        valid = group.dropna(subset=["着順_num"])
        if len(valid) == 0:
            continue
        last3 = valid.tail(3)
        horse_stats[horse] = {
            "過去出走数":           len(valid),
            "過去平均着順":         valid["着順_num"].mean(),
            "過去勝率":             (valid["着順_num"] == 1).mean(),
            "過去複勝率":           (valid["着順_num"] <= 3).mean(),
            "過去平均上り":         valid["上り"].mean(),
            "直近3走平均着順":     last3["着順_num"].mean(),
            "過去平均タイム秒":     valid["タイム秒"].mean() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均タイム秒": last3["タイム秒"].mean() if "タイム秒" in last3.columns else np.nan,
            "過去最速タイム秒":     valid["タイム秒"].min() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均上り":     last3["上り"].mean(),
            "過去平均体重増減":     valid["体重増減"].mean(),
        }

    jockey_stats = {}
    trainer_stats = {}
    for _, row in history_df.iterrows():
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

    rows = []
    for _, row in race_df.iterrows():
        horse   = str(row.get("馬名", ""))
        jockey  = str(row.get("騎手", ""))
        trainer = str(row.get("調教師", ""))

        feat = {
            "馬名":    horse,
            "race_id": row.get("race_id", ""),
            "単勝オッズ": pd.to_numeric(row.get("単勝オッズ", np.nan), errors="coerce"),
            "人気":    pd.to_numeric(row.get("人気", np.nan), errors="coerce"),
            "枠番":    pd.to_numeric(row.get("枠番", np.nan), errors="coerce"),
            "馬番":    pd.to_numeric(row.get("馬番", np.nan), errors="coerce"),
            "斤量":    pd.to_numeric(row.get("斤量", np.nan), errors="coerce"),
            "距離":    pd.to_numeric(row.get("距離", np.nan), errors="coerce"),
            "is_turf": 1 if row.get("馬場種別") == "芝" else 0,
        }

        baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
        feat["馬場状態_num"] = baba_map.get(row.get("馬場状態"), np.nan)

        class_map = {"新馬": 1, "未勝利": 2, "1勝クラス": 3, "2勝クラス": 4,
                     "3勝クラス": 5, "オープン": 6, "G3": 7, "G2": 8, "G1": 9}
        feat["クラス_num"] = class_map.get(row.get("レースクラス"), np.nan)

        seire = str(row.get("性齢", ""))
        feat["is_male"]      = 1 if "牡" in seire else 0
        feat["is_female"]    = 1 if "牝" in seire else 0
        feat["is_castrated"] = 1 if "セ" in seire else 0
        age_match = re.search(r'(\d+)', seire)
        feat["年齢"] = int(age_match.group(1)) if age_match else np.nan

        weight_raw = str(row.get("馬体重", ""))
        w_match = re.search(r'(\d+)', weight_raw)
        feat["馬体重"] = int(w_match.group(1)) if w_match else np.nan
        d_match = re.search(r'\(([+-]?\d+)\)', weight_raw)
        feat["体重増減"] = int(d_match.group(1)) if d_match else np.nan

        hs = horse_stats.get(horse, {})
        for key in ["過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
                    "過去平均上り", "直近3走平均着順", "過去平均タイム秒",
                    "直近3走平均タイム秒", "過去最速タイム秒",
                    "直近3走平均上り", "過去平均体重増減"]:
            feat[key] = hs.get(key, np.nan)

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

    return pdf


def predict_race(race_id, models, history_df):
    """1レースの予測を実行してDataFrameを返す"""
    race_df = get_race_result(race_id)
    if race_df is None:
        return None

    pdf = build_features(race_df, history_df)
    X = pdf[[c for c in FEATURE_COLS if c in pdf.columns]]
    preds = np.mean([m.predict(X) for m in models], axis=0)
    pdf["予測着順スコア"] = preds
    pdf["予測順位"] = pdf["予測着順スコア"].rank()
    return pdf


def make_email_body(results):
    """予測結果をメール本文に整形"""
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    lines = [f"【競馬AI予測】{now}\n"]
    lines.append("=" * 40)

    recommend_count = 0

    for race_id, pdf in results.items():
        jyo_cd = int(str(race_id)[4:6])
        race_no = int(str(race_id)[10:12])
        jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

        lines.append(f"\n■ {jyo_name} {race_no}R  (race_id: {race_id})")

        top3 = pdf.sort_values("予測順位").head(3)
        for _, row in top3.iterrows():
            mark = "◎" if row["予測順位"] == 1 else ("○" if row["予測順位"] == 2 else "▲")
            odds = row.get("単勝オッズ", "?")
            pop  = row.get("人気", "?")
            lines.append(f"  {mark} {row['馬名']}  オッズ{odds}倍  {pop}番人気")

        # 戦略4（中穴）推奨馬
        buy = pdf[
            (pdf["予測順位"] == 1) &
            (pdf["単勝オッズ"] >= 3) &
            (pdf["単勝オッズ"] <= 10)
        ]
        if len(buy) > 0:
            horse = buy.iloc[0]
            lines.append(f"\n  🎯 推奨買い: {horse['馬名']}  {horse['単勝オッズ']}倍")
            recommend_count += 1

    lines.append(f"\n{'=' * 40}")
    lines.append(f"推奨買い合計: {recommend_count}レース")
    lines.append("\n※このメールはAIによる予測です。投資は自己責任でお願いします。")
    return "\n".join(lines)


def send_email(subject, body):
    """Gmailでメール送信"""
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = TO_ADDRESS
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.send_message(msg)
        print(f"メール送信完了 → {TO_ADDRESS}")
    except Exception as e:
        print(f"メール送信エラー: {e}")


def run(race_ids):
    """
    メイン処理
    race_ids: 予測したいレースIDのリスト
    """
    print("モデル・履歴データ読み込み中...")
    with open("model.pkl", "rb") as f:
        models = pickle.load(f)
    history_df = pd.read_csv("race_features.csv")

    results = {}
    for race_id in race_ids:
        print(f"予測中: {race_id}", end=" ")
        pdf = predict_race(race_id, models, history_df)
        if pdf is not None:
            results[race_id] = pdf
            print("✓")
        else:
            print("スキップ")
        time.sleep(1.5)

    if not results:
        print("予測できたレースがありませんでした")
        return

    body = make_email_body(results)
    print("\n" + body)

    subject = f"【競馬AI予測】{datetime.now().strftime('%m/%d')} {len(results)}レース"
    send_email(subject, body)


if __name__ == "__main__":
    # ここに今日予測したいrace_idを入力
    race_ids = [
        "202605021101",
        
        # 必要なだけ追加
    ]
    run(race_ids)