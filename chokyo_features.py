# -*- coding: utf-8 -*-
"""坂路調教(chokyo_hc.csv)からレース単位の調教特徴量を作る。

特徴量（無料サイトの最終追い切り情報でも再現できる共通仕様を軸に設計）:
  chk_last4f   : 最終追い切りの坂路4Fタイム（秒）
  chk_last1f   : 最終追い切りのラスト1F（秒）＝終いの伸び
  chk_days     : 最終追い切りからレースまでの日数
  chk_n14      : レース前14日間の坂路本数（乗り込み量）※要JRA-VAN継続 or 代替
  chk_best4f   : レース前14日間のベスト4F
出力: chokyo_features.csv (race_id, 馬名, 上記5列)
レース日はJVのRAレコード(data/jv/RACE_RA.txt)から取得。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def build_race_date_map():
    """RAレコード → race_id(12桁) : 開催日(datetime) のマップ"""
    rows = []
    with open(os.path.join(BASE, "data", "jv", "RACE_RA.txt"), encoding="utf-8") as f:
        for line in f:
            if not line.startswith("RA"):
                continue
            year, mmdd = line[11:15], line[15:19]
            jyo, kai, day, rno = line[19:21], line[21:23], line[23:25], line[25:27]
            if not (year + mmdd + jyo + kai + day + rno).isdigit():
                continue
            rows.append((year + jyo + kai + day + rno, year + mmdd))
    df = pd.DataFrame(rows, columns=["race_id", "ymd"]).drop_duplicates("race_id")
    df["date"] = pd.to_datetime(df["ymd"], format="%Y%m%d")
    print(f"race日付マップ: {len(df)}レース ({df['ymd'].min()}〜{df['ymd'].max()})")
    return dict(zip(df["race_id"], df["date"]))


def main():
    date_map = build_race_date_map()

    print("調教データ読み込み中...")
    ck = pd.read_csv(os.path.join(BASE, "chokyo_hc.csv"),
                     dtype={"horse_id": str, "調教日": str})
    ck["date"] = pd.to_datetime(ck["調教日"], format="%Y%m%d")
    ck = ck.dropna(subset=["time4f"]).sort_values(["horse_id", "date"])
    print(f"  坂路調教: {len(ck)}行 / {ck['horse_id'].nunique()}頭")

    # 馬ごとの調教配列（日付昇順）
    horse_data = {}
    for hid, g in ck.groupby("horse_id"):
        horse_data[hid] = (g["date"].values, g["time4f"].values, g["lap1f"].values)

    print("レース×馬の紐付け読み込み中...")
    rd = pd.read_csv(os.path.join(BASE, "race_data_clean.csv"), low_memory=False,
                     usecols=["race_id", "馬名", "horse_id"])
    rd["race_id"] = rd["race_id"].astype(str)
    rd["horse_id"] = rd["horse_id"].astype(str)
    rd = rd.drop_duplicates(["race_id", "馬名"])
    rd["race_date"] = rd["race_id"].map(date_map)
    n_nodate = rd["race_date"].isna().sum()
    rd = rd.dropna(subset=["race_date"])
    print(f"  対象: {len(rd)}行（日付不明で除外 {n_nodate}）")

    out = np.full((len(rd), 5), np.nan)
    win14 = np.timedelta64(14, "D")
    for i, (hid, rdate) in enumerate(zip(rd["horse_id"].values, rd["race_date"].values)):
        hd = horse_data.get(hid)
        if hd is None:
            continue
        dates, t4, l1 = hd
        idx = np.searchsorted(dates, rdate)          # レース日より前の調教まで
        if idx == 0:
            continue
        last = idx - 1
        days = (rdate - dates[last]) / np.timedelta64(1, "D")
        if days <= 0:
            last -= 1
            if last < 0:
                continue
            days = (rdate - dates[last]) / np.timedelta64(1, "D")
        lo = np.searchsorted(dates, rdate - win14)
        seg_t4 = t4[lo:idx]
        out[i, 0] = t4[last]                          # chk_last4f
        out[i, 1] = l1[last]                          # chk_last1f
        out[i, 2] = days                              # chk_days
        out[i, 3] = idx - lo                          # chk_n14
        out[i, 4] = np.nanmin(seg_t4) if len(seg_t4) else np.nan   # chk_best4f

    res = rd[["race_id", "馬名"]].copy()
    for j, c in enumerate(["chk_last4f", "chk_last1f", "chk_days", "chk_n14", "chk_best4f"]):
        res[c] = out[:, j]
    cov = res["chk_last4f"].notna().mean() * 100
    res.to_csv(os.path.join(BASE, "chokyo_features.csv"), index=False, encoding="utf-8-sig")
    print(f"保存: chokyo_features.csv ({len(res)}行, 坂路調教カバー率 {cov:.1f}%)")
    print(res.dropna().tail(3).to_string())


if __name__ == "__main__":
    main()
