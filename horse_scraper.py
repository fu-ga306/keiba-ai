"""
horse_scraper.py
────────────────
race_data_clean.csv の horse_id をもとに
netkeiba の馬ページから父馬・母父馬情報を取得し
horse_master.csv に保存する。

使い方:
    python horse_scraper.py
"""

from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_DIR     = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
INPUT_CSV    = os.path.join(BASE_DIR, "race_data_clean.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "horse_master.csv")


def get_horse_profile(horse_id: str) -> dict | None:
    """
    netkeiba の馬ページから父馬・母父馬情報を取得する。
    Selenium を使用（requests では 403 になるため）。
    blood_table の構造:
      td[0]=父馬  td[1]=父父馬
      td[2]=母馬  td[3]=母父馬
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    # horse_id の .0 を除去して整数文字列に変換
    horse_id_str = str(horse_id).replace(".0", "").strip()
    url = f"https://db.netkeiba.com/horse/{horse_id_str}/"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    try:
        driver.get(url)
        time.sleep(3.0)
        soup = BeautifulSoup(driver.page_source, "html.parser")
    finally:
        driver.quit()

    try:
        # 404・エラーページの検出（「このページは動作していません」など）
        h1 = soup.find("h1")
        if h1:
            h1_text = h1.get_text(strip=True)
            if "動作していません" in h1_text or "見つかりません" in h1_text or "404" in h1_text:
                # 存在しないページはNoneを返さずスキップ（horse_master.csvに保存しない）
                return None

        # テーブルが0件の場合もエラーページと判断してスキップ
        if len(soup.find_all("table")) == 0:
            return None

        profile = {
            "horse_id": horse_id_str,
            "父馬":    None,
            "母父馬":  None,
            "馬名":    None,
        }

        # 馬名取得
        if h1:
            profile["馬名"] = h1.get_text(strip=True)

        # blood_table から父馬・母父馬を取得
        # 構造: td[0]=父馬  td[1]=父父馬  td[2]=母馬  td[3]=母父馬
        blood_table = soup.find("table", class_="blood_table")
        if blood_table:
            tds = blood_table.find_all("td")
            if len(tds) >= 1:
                a = tds[0].find("a")
                val = a.get_text(strip=True) if a else tds[0].get_text(strip=True)
                profile["父馬"] = val if val else None
            if len(tds) >= 4:
                a = tds[3].find("a")
                val = a.get_text(strip=True) if a else tds[3].get_text(strip=True)
                profile["母父馬"] = val if val else None

        # 父馬が取得できなかった場合もスキップ（地方馬・障害馬等）
        if profile["父馬"] is None:
            return None

        return profile

    except Exception as e:
        print(f"  パースエラー: horse_id={horse_id_str} → {e}")
        return None


def build_horse_master():
    """
    race_data_clean.csv から horse_id を収集し
    馬ページから父馬・母父馬情報を取得して horse_master.csv に保存する。
    """
    print("race_data_clean.csv を読み込み中...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    if "horse_id" not in df.columns:
        print("エラー: horse_id 列がありません。")
        print("scraper.py を最新版に更新して再スクレイピングしてください。")
        return

    # ユニークな horse_id を取得
    horse_ids = df[["馬名", "horse_id"]].dropna(subset=["horse_id"])
    horse_ids = horse_ids.drop_duplicates(subset=["horse_id"])
    print(f"ユニーク馬数: {len(horse_ids)}頭")

    # 既存の horse_master.csv があればスキップ（.0を除去して統一）
    existing_ids = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        existing_ids = set(
            existing_df["horse_id"].astype(str)
            .str.replace(".0", "", regex=False).str.strip().unique()
        )
        print(f"取得済み: {len(existing_ids)}頭 → スキップします")

    # horse_idsのhorse_idも.0除去して比較
    horse_ids["horse_id_clean"] = (
        horse_ids["horse_id"].astype(str)
        .str.replace(".0", "", regex=False).str.strip()
    )
    target = horse_ids[~horse_ids["horse_id_clean"].isin(existing_ids)]
    print(f"新規取得対象: {len(target)}頭")

    if len(target) == 0:
        print("新規取得対象なし")
        return

    results = []
    errors  = 0

    for i, (_, row) in enumerate(target.iterrows()):
        horse_id = str(row["horse_id"])
        uma_name = str(row["馬名"])
        print(f"[{i+1}/{len(target)}] {uma_name} ({horse_id})", end=" ")

        profile = get_horse_profile(horse_id)
        if profile and profile.get("父馬") is not None:
            results.append(profile)
            print(f"✓ 父:{profile['父馬']}  母父:{profile['母父馬']}")
        else:
            # 404・ページなし・父馬なし → 再アクセス防止のためhorse_idだけ記録
            horse_id_clean = str(row["horse_id"]).replace(".0", "").strip()
            results.append({
                "horse_id": horse_id_clean,
                "父馬":    None,
                "母父馬":  None,
            })
            print("- (ページなし・地方馬等)")

        time.sleep(3.0)

        # 200件ごとに中間保存
        if len(results) % 200 == 0 and results:
            _save(results, append=True)
            results = []
            print(f"--- 中間保存済み ---")

    # 残りを保存
    if results:
        _save(results, append=True)

    print(f"\n完了！エラー: {errors}件")
    print(f"保存先: {OUTPUT_CSV}")


def _save(records: list, append: bool = True):
    new_df = pd.DataFrame(records)
    # horse_id の .0 を除去
    new_df["horse_id"] = new_df["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    # 必要列のみ保持（父馬Noneも「取得済み」として保存して再アクセスを防ぐ）
    new_df = new_df[["horse_id", "父馬", "母父馬"]].copy()

    if len(new_df) == 0:
        return

    if append and os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["horse_id"])
        combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    build_horse_master()