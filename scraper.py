import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
}


def get_race_result(race_id):
    url = f"https://db.netkeiba.com/race/{race_id}/"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "html.parser")

        # レース情報（距離・馬場・クラス）
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

            race_class_tag = race_data_div.find("p")
            if race_class_tag:
                race_info["レースクラス"] = race_class_tag.get_text(strip=True)

        table = soup.find("table", class_="race_table_01")
        if table is None:
            print(f"  テーブルが見つかりません: {race_id}")
            return None

        # ヘッダー取得
        headers = [
            th.get_text(strip=True)
            for th in table.find_all("tr")[0].find_all("th")
        ]

        rows = []
        horse_ids = []  # 【追加】horse_id リスト

        for tr in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not cols:
                continue
            rows.append(cols)

            # 【追加】馬名セルのリンクから horse_id を取得
            horse_id = None
            a_tag = tr.find("a", href=re.compile(r"/horse/\d+"))
            if a_tag:
                m = re.search(r"/horse/(\d+)", a_tag["href"])
                if m:
                    horse_id = m.group(1)
            horse_ids.append(horse_id)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        if headers and len(headers) == df.shape[1]:
            df.columns = headers

        # 【追加】horse_id 列を付与
        df["horse_id"] = horse_ids

        # レース情報を全行に付与
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
                            ids.append(
                                f"{year}{jyo:02d}{kai:02d}{nichi:02d}{race:02d}"
                            )
        return ids

    race_ids = generate_race_ids(
        years=list(range(2019, 2027)),
        jyo_codes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        kai_range=range(1, 6),
        nichi_range=range(1, 9),
        race_range=range(1, 13),
    )

    print(f"取得予定: {len(race_ids)}レース（約{len(race_ids)*2//3600}時間）")

    # 取得済みのrace_idをスキップ
    existing_ids = set()
    if os.path.exists("race_data_large.csv"):
        existing_df = pd.read_csv("race_data_large.csv", usecols=["race_id"])
        existing_ids = set(existing_df["race_id"].astype(str).unique())
        print(f"取得済み: {len(existing_ids)}レース → スキップします")

    all_data = []
    errors   = 0
    skipped  = 0

    for i, race_id in enumerate(race_ids):
        if race_id in existing_ids:
            skipped += 1
            continue

        print(f"[{i+1}/{len(race_ids)}] {race_id}", end=" ")
        df = get_race_result(race_id)
        if df is not None:
            all_data.append(df)
            print(f"✓ {len(df)}頭")
        else:
            print("スキップ")
            errors += 1
        time.sleep(2)

        # 100件ごとに中間保存
        if len(all_data) % 100 == 0 and all_data:
            tmp = pd.concat(all_data, ignore_index=True)
            tmp.to_csv(
                f"race_data_new_tmp_{len(all_data)}.csv",
                index=False, encoding="utf-8-sig",
            )
            print(f"--- 中間保存: {len(all_data)}件 ---")

    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)

        if os.path.exists("race_data_large.csv"):
            old_df = pd.read_csv("race_data_large.csv", low_memory=False)
            result = pd.concat([old_df, new_df], ignore_index=True)
            result = result.drop_duplicates(subset=["race_id", "馬番"])
        else:
            result = new_df

        result.to_csv("race_data_large.csv", index=False, encoding="utf-8-sig")
        print(
            f"\n完了！合計 {len(result)}行 / "
            f"新規:{len(new_df)}行 / エラー:{errors}件"
        )
    else:
        print(f"\n新規データなし（既存スキップ: {skipped}件）")