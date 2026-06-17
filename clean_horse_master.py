"""
clean_horse_master.py
─────────────────────
horse_master.csv の不正データを削除して再取得対象を明確にする。

削除対象：
  ① 父馬が空・None・NaN の行
  ② 「このページは動作していません」が混入している行
  ③ 馬名が空の行
"""

import pandas as pd
import os

BASE_DIR  = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
HORSE_CSV = os.path.join(BASE_DIR, "horse_master.csv")

if not os.path.exists(HORSE_CSV):
    print("horse_master.csv が見つかりません")
    exit()

df = pd.read_csv(HORSE_CSV, header=None,
                 names=["horse_id", "父馬", "母父馬", "備考"])

print(f"クリーニング前: {len(df)}行")

# 念のためバックアップを作成
backup_path = HORSE_CSV.replace(".csv", "_backup.csv")
df.to_csv(backup_path, index=False, header=False, encoding="utf-8-sig")
print(f"  バックアップ作成 → horse_master_backup.csv")

# ① 「このページは動作していません」が含まれる行を除外
error_mask = df.apply(
    lambda row: row.astype(str).str.contains(
        "このページは動作していません", na=False
    ).any(),
    axis=1,
)
print(f"  エラーページ行: {error_mask.sum()}件 → 削除")
df = df[~error_mask]

# ② 父馬が空・NaN の行を除外
empty_mask = df["父馬"].isna() | (df["父馬"].astype(str).str.strip() == "")
print(f"  父馬空白行: {empty_mask.sum()}件 → 削除")
df = df[~empty_mask]

# ③ horse_id が NaN の行を除外
id_mask = df["horse_id"].isna()
print(f"  horse_id空白行: {id_mask.sum()}件 → 削除")
df = df[~id_mask]

# horse_id の .0 を除去
df["horse_id"] = df["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()

# 備考列は不要なので削除
df = df[["horse_id", "父馬", "母父馬"]].copy()

# 重複削除
before_dedup = len(df)
df = df.drop_duplicates(subset=["horse_id"])
print(f"  重複削除: {before_dedup - len(df)}件")

print(f"クリーニング後: {len(df)}行")

df.to_csv(HORSE_CSV, index=False, encoding="utf-8-sig")
print(f"保存完了 → {HORSE_CSV}")
print("\nサンプル（先頭5件）:")
print(df.head().to_string())