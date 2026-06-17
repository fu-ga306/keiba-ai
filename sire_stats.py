"""
sire_stats.py
─────────────
horse_master.csv + race_data_clean.csv から
父馬・母父馬の産駒成績を集計して sire_stats.csv に保存する。

使い方:
    python sire_stats.py
"""

import pandas as pd
import numpy as np
import os

BASE_DIR       = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
RACE_CSV       = os.path.join(BASE_DIR, "race_data_clean.csv")
HORSE_CSV      = os.path.join(BASE_DIR, "horse_master.csv")
OUTPUT_CSV     = os.path.join(BASE_DIR, "sire_stats.csv")


def build_sire_stats():
    print("race_data_clean.csv を読み込み中...")
    race_df = pd.read_csv(RACE_CSV, low_memory=False)
    print(f"  {len(race_df)}行")

    print("horse_master.csv を読み込み中...")
    horse_df = pd.read_csv(HORSE_CSV)
    print(f"  {len(horse_df)}頭")

    # horse_id で結合して父馬・母父馬を付与（.0除去で結合キーをそろえる）
    race_df["horse_id"]  = race_df["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    horse_df["horse_id"] = horse_df["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df = race_df.merge(horse_df[["horse_id", "父馬", "母父馬"]], on="horse_id", how="left")

    print(f"結合後: {len(df)}行  父馬情報あり: {df['父馬'].notna().sum()}行")

    # 着順_num がなければ生成
    if "着順_num" not in df.columns:
        df["着順_num"] = pd.to_numeric(df.get("着順"), errors="coerce")

    # 集計関数
    def calc_sire_stats(df, sire_col, prefix):
        result_rows = []
        for sire, group in df.groupby(sire_col):
            valid = group.dropna(subset=["着順_num"])
            if len(valid) < 5:  # 5走未満は除外
                continue

            # 全体成績
            total     = len(valid)
            wins      = (valid["着順_num"] == 1).sum()
            top2      = (valid["着順_num"] <= 2).sum()
            top3      = (valid["着順_num"] <= 3).sum()

            # 芝成績
            turf      = valid[valid["is_turf"] == 1]
            turf_wins = (turf["着順_num"] == 1).sum() if len(turf) > 0 else 0

            # ダート成績
            dirt      = valid[valid["is_turf"] == 0]
            dirt_wins = (dirt["着順_num"] == 1).sum() if len(dirt) > 0 else 0

            # 距離別成績（短距離〜長距離）
            short  = valid[valid["距離"] <= 1400] if "距離" in valid.columns else pd.DataFrame()
            mid    = valid[(valid["距離"] > 1400) & (valid["距離"] <= 2000)] if "距離" in valid.columns else pd.DataFrame()
            long_  = valid[valid["距離"] > 2000] if "距離" in valid.columns else pd.DataFrame()

            # 馬場状態別成績
            ryou   = valid[valid["馬場状態_num"] == 1] if "馬場状態_num" in valid.columns else pd.DataFrame()
            omoi   = valid[valid["馬場状態_num"] >= 3] if "馬場状態_num" in valid.columns else pd.DataFrame()

            row = {
                f"{prefix}名":              sire,
                f"{prefix}_産駒数":         total,
                f"{prefix}_勝率":           wins  / total,
                f"{prefix}_連対率":         top2  / total,
                f"{prefix}_複勝率":         top3  / total,
                f"{prefix}_芝勝率":         turf_wins / len(turf) if len(turf) > 0 else np.nan,
                f"{prefix}_ダート勝率":     dirt_wins / len(dirt) if len(dirt) > 0 else np.nan,
                f"{prefix}_短距離勝率":     (short["着順_num"] == 1).sum() / len(short) if len(short) > 0 else np.nan,
                f"{prefix}_中距離勝率":     (mid["着順_num"]   == 1).sum() / len(mid)   if len(mid)   > 0 else np.nan,
                f"{prefix}_長距離勝率":     (long_["着順_num"] == 1).sum() / len(long_) if len(long_) > 0 else np.nan,
                f"{prefix}_良馬場勝率":     (ryou["着順_num"]  == 1).sum() / len(ryou)  if len(ryou)  > 0 else np.nan,
                f"{prefix}_重馬場勝率":     (omoi["着順_num"]  == 1).sum() / len(omoi)  if len(omoi)  > 0 else np.nan,
                f"{prefix}_平均着順":       valid["着順_num"].mean(),
            }
            result_rows.append(row)

        return pd.DataFrame(result_rows)

    print("父馬産駒成績を集計中...")
    sire_df  = calc_sire_stats(df, "父馬",  "父")
    print(f"  父馬: {len(sire_df)}頭分")

    print("母父馬産駒成績を集計中...")
    bms_df   = calc_sire_stats(df, "母父馬", "母父")
    print(f"  母父馬: {len(bms_df)}頭分")

    # 保存
    sire_df.to_csv(OUTPUT_CSV.replace(".csv", "_father.csv"),
                   index=False, encoding="utf-8-sig")
    bms_df.to_csv(OUTPUT_CSV.replace(".csv", "_bms.csv"),
                  index=False, encoding="utf-8-sig")

    # 統合版も保存
    sire_df["種別"] = "父"
    bms_df["種別"]  = "母父"
    sire_df = sire_df.rename(columns={"父名": "種牡馬名"})
    bms_df  = bms_df.rename(columns={"母父名": "種牡馬名"})
    combined = pd.concat([sire_df, bms_df], ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n保存完了:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_CSV.replace('.csv', '_father.csv')}")
    print(f"  {OUTPUT_CSV.replace('.csv', '_bms.csv')}")

    # サンプル表示
    print("\n父馬成績サンプル（勝率上位5頭・産駒20頭以上）:")
    top = sire_df[sire_df["父_産駒数"] >= 20].sort_values("父_勝率", ascending=False).head(5)
    print(top[["種牡馬名", "父_産駒数", "父_勝率", "父_芝勝率", "父_ダート勝率"]].to_string())

    return sire_df, bms_df


if __name__ == "__main__":
    build_sire_stats()
