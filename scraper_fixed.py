# -*- coding: utf-8 -*-
"""
scraper.py 修正版（クラス取得の修正・賞金追加・horse_id維持）
─────────────────────────────────────────
変更点:
  1. クラス取得を修正
     旧: h1 をそのまま「レースクラス」に入れていた
        （古いデータはこの処理自体が無く、コース文字列が混入していた）
     新: p.smalltxt の条件 + h1 のグレードから
        「レースクラス」(名称) と「クラス_num」(序列1-8) を生成
  2. 賞金(万円) をテーブルから取得（下位馬は0埋め）
  3. horse_id は従来どおり取得（変更なし・既に正常動作）
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}


def parse_race_class(smalltxt, h1):
    """
    smalltxt 例: "2024年11月30日 5回中山1日目 2歳未勝利  (混)[指](馬齢)"
    h1 例:       "第91回東京優駿(GI)" / "2歳未勝利"
    戻り値: (クラス名, クラス_num 1-8)

    序列:
      1 新馬・未勝利 / 2 1勝 / 3 2勝 / 4 3勝
      5 オープン特別・L / 6 G3 / 7 G2 / 8 G1
    """
    text = smalltxt or ""
    h1 = h1 or ""

    # グレードは h1 の () 内から判定（smalltxtは全部「オープン」なので区別不可）
    grade = None
    if "GⅠ" in h1 or "(GI)" in h1:
        grade = "G1"
    if "GⅡ" in h1 or "(GII)" in h1:
        grade = "G2"
    if "GⅢ" in h1 or "(GIII)" in h1:
        grade = "G3"

    if grade:
        return grade, {"G3": 6, "G2": 7, "G1": 8}[grade]
    if "未勝利" in text or "新馬" in text:
        return "新馬・未勝利", 1
    if "1勝クラス" in text or "500万" in text:
        return "1勝クラス", 2
    if "2勝クラス" in text or "1000万" in text:
        return "2勝クラス", 3
    if "3勝クラス" in text or "1600万" in text:
        return "3勝クラス", 4
    if "オープン" in text or "リステッド" in text or "(L)" in h1 or "（L）" in h1:
        return "オープン特別・L", 5
    return None, None


def get_race_result(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        # ── レース情報（距離・馬場種別・馬場状態・クラス）──────────
        race_info = {}
        race_data_div = soup.find("div", class_="data_intro")
        if race_data_div:
            shibari = race_data_div.find("span")
            if shibari:
                info_text = shibari.get_text(strip=True)
                dist_match = re.search(r"(\d{3,4})m", info_text)
                race_info["距離"] = int(dist_match.group(1)) if dist_match else None
                race_info["馬場種別"] = "芝" if "芝" in info_text else "ダート"
                race_info["馬場状態"] = None
                for condition in ["不良", "稍重", "重", "良"]:
                    if condition in info_text:
                        race_info["馬場状態"] = condition
                        break
                # 回り（右/左/直）も同じ文字列から無料で取れる
                turn = None
                m_turn = re.search(r"[芝ダ](右|左|直)", info_text)
                if m_turn:
                    turn = m_turn.group(1)
                race_info["回り"] = turn

        # ── クラス取得（修正の核心）──────────────────────────
        # h1: レース名/クラス名、smalltxt: 統一フォーマットの条件
        h1_text = None
        racedata = soup.find("dl", class_="racedata fc")
        if racedata:
            h1 = racedata.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)

        smalltxt = soup.find("p", class_="smalltxt")
        smalltxt_text = smalltxt.get_text(strip=True) if smalltxt else None

        class_name, class_num = parse_race_class(smalltxt_text, h1_text)
        race_info["レースクラス"] = class_name       # 旧来の列名を維持（中身は正しいクラス名に）
        race_info["クラス_num"] = class_num
        race_info["レース名"] = h1_text               # 参考: 元のh1も保持

        # ── 結果テーブル ────────────────────────────────
        table = soup.find("table", class_="race_table_01")
        if table is None:
            print(f"  テーブルが見つかりません: {race_id}")
            return None

        header_cells = table.find_all("tr")[0].find_all("th")
        headers = [th.get_text(strip=True) for th in header_cells]

        # 賞金列の位置を特定（"賞金" を含むヘッダー）
        prize_idx = None
        for idx, hname in enumerate(headers):
            if "賞金" in hname:
                prize_idx = idx
                break

        rows = []
        horse_ids = []
        prizes = []

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            cols = [td.get_text(strip=True) for td in tds]
            if not cols:
                continue
            rows.append(cols)

            # horse_id（馬名リンクから）
            horse_id = None
            a_tag = tr.find("a", href=re.compile(r"/horse/\d+"))
            if a_tag:
                m = re.search(r"/horse/(\d+)", a_tag["href"])
                if m:
                    horse_id = m.group(1)
            horse_ids.append(horse_id)

            # 賞金（着順上位馬のみ値が入る。無い馬は0）
            prize = 0.0
            if prize_idx is not None and prize_idx < len(cols):
                raw = cols[prize_idx].replace(",", "").strip()
                if raw and raw not in ("-", "nan"):
                    try:
                        prize = float(raw)
                    except ValueError:
                        prize = 0.0
            prizes.append(prize)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        if headers and len(headers) == df.shape[1]:
            df.columns = headers

        df["horse_id"] = horse_ids
        df["賞金"] = prizes

        for key, val in race_info.items():
            df[key] = val

        df["race_id"] = race_id
        return df

    except Exception as e:
        print(f"  エラー: {race_id} → {e}")
        return None


if __name__ == "__main__":
    def generate_race_ids(years, jyo_codes, kai_range, nichi_range, race_range):
        ids = []
        for year in years:
            for jyo in jyo_codes:
                for kai in kai_range:
                    for nichi in nichi_range:
                        for race in race_range:
                            ids.append(f"{year}{jyo:02d}{kai:02d}{nichi:02d}{race:02d}")
        return ids

    race_ids = generate_race_ids(
        years=list(range(2019, 2027)),
        jyo_codes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        kai_range=range(1, 6),
        nichi_range=range(1, 9),
        race_range=range(1, 13),
    )

    print(f"取得予定: {len(race_ids)}レース")

    OUT = "race_data_large.csv"
    existing_ids = set()
    if os.path.exists(OUT):
        existing_df = pd.read_csv(OUT, usecols=["race_id"])
        existing_ids = set(existing_df["race_id"].astype(str).unique())
        print(f"取得済み: {len(existing_ids)}レース → スキップします")

    all_data = []
    errors = 0
    skipped = 0

    for i, race_id in enumerate(race_ids):
        if race_id in existing_ids:
            skipped += 1
            continue

        print(f"[{i+1}/{len(race_ids)}] {race_id}", end=" ")
        df = get_race_result(race_id)
        if df is not None:
            all_data.append(df)
            print(f"OK {len(df)}頭")
        else:
            print("スキップ")
            errors += 1
        time.sleep(2)

        if len(all_data) % 100 == 0 and all_data:
            tmp = pd.concat(all_data, ignore_index=True)
            tmp.to_csv(f"race_data_new_tmp_{len(all_data)}.csv",
                       index=False, encoding="utf-8-sig")
            print(f"--- 中間保存: {len(all_data)}件 ---")

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        if os.path.exists(OUT):
            old_df = pd.read_csv(OUT, low_memory=False)
            result = pd.concat([old_df, new_df], ignore_index=True)
            result = result.drop_duplicates(subset=["race_id", "馬番"])
        else:
            result = new_df
        result.to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"\n完了！合計 {len(result)}行 / 新規:{len(new_df)}行 / エラー:{errors}件")
    else:
        print(f"\n新規データなし（既存スキップ: {skipped}件）")
