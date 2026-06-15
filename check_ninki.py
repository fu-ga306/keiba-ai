# -*- coding: utf-8 -*-
"""
新モデル(生確率)が予測1位に選んだ馬の人気分布を確認する。
以前の「1番人気を97%選ぶ（人気追従）」問題が改善したか検証する。
model_result.csv（model.py実行時に生成・2025年データの予測結果）を使う。
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "model_result.csv")

df = pd.read_csv(path)
top1 = df[df["予測順位"] == 1]
n = len(top1)

print(f"予測1位の人気分布（{n}レース）")
print("=" * 40)

bands = [
    ("1番人気",      (top1["人気"] == 1)),
    ("2-3番人気",    (top1["人気"] >= 2) & (top1["人気"] <= 3)),
    ("4-6番人気",    (top1["人気"] >= 4) & (top1["人気"] <= 6)),
    ("7番人気以下",  (top1["人気"] >= 7)),
]
for label, mask in bands:
    cnt = mask.sum()
    pct = mask.mean() * 100
    print(f"  {label:10}: {cnt:4d}回  ({pct:5.1f}%)")

print("=" * 40)
# 予測1位が実際に勝った割合も人気帯別に
print("\n予測1位の的中率（人気帯別）")
print("=" * 40)
for label, mask in bands:
    sub = top1[mask]
    if len(sub) == 0:
        continue
    win = (sub["着順_num"] == 1).mean() * 100
    print(f"  {label:10}: {len(sub):4d}回  勝率{win:5.1f}%")

# 1番人気以外を1位にした割合（独自性の指標）
non_fav = (top1["人気"] != 1).mean() * 100
print("\n" + "=" * 40)
print(f"1番人気以外を予測1位にした割合: {non_fav:.1f}%")
print("  （高いほど人気に追従せず独自予想している）")
