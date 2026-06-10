import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}

BASE_DIR    = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
# prediction_record_v2.csv を正式ファイルとして使用
RECORD_FILE = os.path.join(BASE_DIR, "prediction_record_v2.csv")

JYO_NAMES = {
    1:"札幌", 2:"函館", 3:"福島", 4:"新潟",  5:"東京",
    6:"中山", 7:"中京", 8:"京都", 9:"阪神", 10:"小倉"
}


def get_race_result(race_id):
    """レース結果をSeleniumで取得"""
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table", class_="RaceTable01")
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) >= 5:
                rows.append([td.get_text(strip=True) for td in tds])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        col_names = ["着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム"]
        n = df.shape[1]
        df.columns = (
            col_names[:n]
            if n <= len(col_names)
            else col_names + list(range(n - len(col_names)))
        )
        df["着順_num"] = pd.to_numeric(df["着順"], errors="coerce")
        df["race_id"]  = race_id
        return df

    except Exception as e:
        print(f"  結果取得エラー: {race_id} → {e}")
        return None
    finally:
        driver.quit()


def load_records():
    if os.path.exists(RECORD_FILE):
        return pd.read_csv(RECORD_FILE)
    return pd.DataFrame(columns=[
        "日付", "race_id", "jyo", "race",
        "honmei", "honmei_win_p", "taiko", "taiko_win_p",
        "ana", "ana_win_p",
        "honmei_odds", "honmei_ninki", "honmei_ev",
        "honmei_actual", "hit", "taiko_actual", "ana_actual",
    ])


def save_record(record):
    df = load_records()
    # race_id重複は更新（同じレースを複数回予想した場合は上書き）
    if "race_id" in df.columns and len(df) > 0:
        df = df[df["race_id"].astype(str) != str(record["race_id"])]
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(RECORD_FILE, index=False, encoding="utf-8-sig")
    print(f"記録保存 → {RECORD_FILE}")


def record_from_prediction(race_id, pdf):
    """
    予測結果DataFrameからレコードを作成して保存。
    keiba_auto.py の run_single_race から呼び出す。
    update_results / show_summary / dashboard.py と同じカラム形式で保存する。
    """
    top3  = pdf.sort_values("予測順位").head(3)
    top1  = top3.iloc[0]
    top2  = top3.iloc[1] if len(top3) > 1 else None
    top3h = top3.iloc[2] if len(top3) > 2 else None

    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

    record = {
        "日付":          datetime.now().strftime("%Y/%m/%d"),
        "race_id":       race_id,
        "jyo":           jyo_name,
        "race":          race_no,
        "honmei":        top1["馬名"],
        "honmei_win_p":  float(top1.get("勝ち確率", np.nan)) if pd.notna(top1.get("勝ち確率")) else None,
        "taiko":         top2["馬名"] if top2 is not None else None,
        "taiko_win_p":   float(top2.get("勝ち確率", np.nan)) if (top2 is not None and pd.notna(top2.get("勝ち確率"))) else None,
        "ana":           top3h["馬名"] if top3h is not None else None,
        "ana_win_p":     float(top3h.get("勝ち確率", np.nan)) if (top3h is not None and pd.notna(top3h.get("勝ち確率"))) else None,
        "honmei_odds":   top1.get("単勝オッズ", np.nan),
        "honmei_ninki":  top1.get("人気", np.nan),
        "honmei_ev":     float(top1.get("単勝期待値", np.nan)) if pd.notna(top1.get("単勝期待値")) else None,
        "honmei_actual": np.nan,
        "hit":           np.nan,
        "taiko_actual":  np.nan,
        "ana_actual":    np.nan,
    }
    save_record(record)
    print(f"  記録保存: {jyo_name} {race_no}R → {top1['馬名']}")
    return record


def update_results(date_str=None):
    """
    prediction_record.csv の hit が未設定のレースを対象に結果を取得・更新する。
    date_str は使用しない（race_id に日付が含まれないため）。
    """
    if not os.path.exists(RECORD_FILE):
        print(f"記録ファイルが見つかりません: {RECORD_FILE}")
        return

    df = pd.read_csv(RECORD_FILE)

    # 結果列がなければ追加
    for col in ["honmei_actual", "hit", "taiko_actual", "ana_actual"]:
        if col not in df.columns:
            df[col] = np.nan

    # hit が未設定の行のみ対象
    target = df[df["hit"].isna()].copy()

    if len(target) == 0:
        print("未更新レースはありません")
        return

    print(f"未更新 {len(target)}レース分の結果を取得中...")

    for idx, row in target.iterrows():
        race_id   = str(row["race_id"])
        result_df = get_race_result(race_id)

        if result_df is None:
            print(f"  {race_id}: 結果未取得（レース未了の可能性）")
            continue

        # ◎本命
        honmei = row["honmei"]
        match1 = result_df[result_df["馬名"] == honmei]
        if len(match1) > 0:
            actual1 = match1["着順_num"].iloc[0]
            hit     = 1 if actual1 == 1 else 0
        else:
            actual1 = np.nan
            hit     = np.nan

        df.at[idx, "honmei_actual"] = actual1
        df.at[idx, "hit"]           = hit

        # ○対抗・▲穴馬
        for col, name_col in [("taiko_actual", "taiko"), ("ana_actual", "ana")]:
            horse = row.get(name_col, np.nan)
            if pd.notna(horse):
                match = result_df[result_df["馬名"] == str(horse)]
                if len(match) > 0:
                    df.at[idx, col] = match["着順_num"].iloc[0]

        hit_str  = "✅ 的中" if hit == 1 else ("❌ 外れ" if hit == 0 else "⚠️ 未判定")
        t_actual = df.at[idx, "taiko_actual"]
        a_actual = df.at[idx, "ana_actual"]
        print(
            f"  {row['jyo']} {row['race']}R: "
            f"◎{honmei}→{actual1}着 {hit_str}  "
            f"○{row['taiko']}→{t_actual}着  "
            f"▲{row['ana']}→{a_actual}着"
        )
        time.sleep(1)

    df.to_csv(RECORD_FILE, index=False, encoding="utf-8-sig")
    print("\n記録更新完了 → " + RECORD_FILE)


def show_summary(date_str=None):
    """サマリーを表示。date_str を指定するとその日だけ集計"""
    if not os.path.exists(RECORD_FILE):
        print(f"記録ファイルが見つかりません: {RECORD_FILE}")
        return

    df = pd.read_csv(RECORD_FILE)

    if "hit" not in df.columns:
        print("結果がまだ更新されていません。先に update を実行してください。")
        return

    df_done = df.dropna(subset=["hit"])
    # date_str は無視（race_id に日付が含まれないため全件集計）

    if len(df_done) == 0:
        print("集計できる記録がありません")
        return

    total    = len(df_done)
    hits     = int(df_done["hit"].sum())
    hit_rate = hits / total * 100

    label = f"（{date_str or '全期間'}）"
    print("\n" + "="*40)
    print(f"📊 予測実績サマリー {label}")
    print("="*40)
    print(f"ベット数:   {total}回")
    print(f"的中数:     {hits}回（◎本命1着）")
    print(f"的中率:     {hit_rate:.1f}%")

    # ◎○▲それぞれの平均着順
    for col, label2 in [
        ("honmei_actual", "◎本命"),
        ("taiko_actual",  "○対抗"),
        ("ana_actual",    "▲穴馬"),
    ]:
        if col in df_done.columns:
            avg = pd.to_numeric(df_done[col], errors="coerce").mean()
            print(f"{label2}平均着順: {avg:.1f}着" if pd.notna(avg) else f"{label2}平均着順: -")

    # 競馬場別
    print("\n── 競馬場別成績 ──")
    for jyo, group in df_done.groupby("jyo"):
        g_hits = int(group["hit"].sum())
        g_rate = g_hits / len(group) * 100
        print(f"  {jyo}: {g_hits}/{len(group)}回  的中率{g_rate:.1f}%")

    # レースごとの一覧
    print("\n── レース別結果 ──")
    for _, row in df_done.iterrows():
        h = int(row["hit"]) if pd.notna(row["hit"]) else "-"
        mark = "✅" if h == 1 else "❌"
        print(
            f"  {row['jyo']} {row['race']}R  "
            f"◎{row['honmei']}({row.get('honmei_actual','-')}着) "
            f"○{row['taiko']}({row.get('taiko_actual','-')}着) "
            f"▲{row['ana']}({row.get('ana_actual','-')}着) {mark}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd  = sys.argv[1]
        date = sys.argv[2] if len(sys.argv) > 2 else None
        if cmd == "update":
            update_results(date)
        elif cmd == "summary":
            show_summary(date)
        else:
            print("使い方: result_tracker.py [update|summary] [日付 例:2026/06/06]")
    else:
        today = datetime.now().strftime("%Y%m%d")
        update_results(today)
        show_summary(today)