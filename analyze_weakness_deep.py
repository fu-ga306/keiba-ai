# -*- coding: utf-8 -*-
"""
analyze_weakness_deep.py
────────────────────────
MFモデルの「外し方」を深掘りする。弱点（長距離・多頭数）で、
どういう馬を・どういうときに外しているかを特定し、
作るべき特徴量を精密に狙う。

入力:
  model_mf_result.csv
  race_features.csv（脚質・前走・実績などの詳細列を結合）

分析:
  A. 長距離(2001m〜)でMF複勝1位が飛んだ(4着以下)ケースの特徴
     - 脚質別（逃げ/先行/差し/追込）に外し率
     - 前走距離との差（距離延長/短縮）別
     - 距離経験(同距離過去出走数)別
  B. 多頭数(13頭〜)で外したケースの特徴
     - 枠順(内/中/外)別の外し率
     - 脚質別の外し率
  C. 共通: MFが高評価だが飛んだ馬 vs 当たった馬 の差を比較

使い方:
  python analyze_weakness_deep.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MF_CSV   = os.path.join(BASE_DIR, "model_mf_result.csv")
FEAT_CSV = os.path.join(BASE_DIR, "race_features.csv")


def safe_rate(mask_hit, mask_all):
    n = mask_all.sum()
    if n == 0:
        return None, 0
    return mask_hit.sum() / n * 100, int(n)


def main():
    if not os.path.exists(MF_CSV):
        print("model_mf_result.csv がありません。market_free_model.py を先に実行。")
        return

    mf = pd.read_csv(MF_CSV)
    mf = mf.dropna(subset=["着順_num"])
    mf["race_id"] = mf["race_id"].astype(str)

    # 詳細列を結合
    detail_cols = ["race_id", "馬名", "距離", "距離カテゴリ", "出走頭数", "枠番", "馬番",
                   "先行馬フラグ", "過去平均先行指数", "前走距離", "距離変化",
                   "同距離過去勝率", "同距離過去平均着順", "過去出走数",
                   "前走着順", "馬体重", "体重増減", "年齢", "斤量"]
    feat = pd.read_csv(FEAT_CSV, low_memory=False)
    feat["race_id"] = feat["race_id"].astype(str)
    use = [c for c in detail_cols if c in feat.columns]
    df = mf.merge(feat[use].drop_duplicates(subset=["race_id", "馬名"]),
                  on=["race_id", "馬名"], how="left")

    # MF複勝1位だけ抽出（この馬たちが当たり/外れの対象）
    top = df[df["MF複勝順位"] == 1].copy()
    top["的中"] = (top["着順_num"] <= 3).astype(int)
    top["飛んだ"] = (top["着順_num"] >= 4).astype(int)

    print("=" * 58)
    print("MFモデル 外し方の深掘り分析")
    print("=" * 58)
    print(f"MF複勝1位の総数: {len(top)}  全体複勝率: {top['的中'].mean()*100:.1f}%")

    # ── A. 長距離での外し方 ──
    print("\n" + "=" * 58)
    print("A. 長距離(2001m〜)での外し方")
    print("=" * 58)
    lng = top[top["距離"] >= 2001].copy()
    print(f"  長距離のMF複勝1位: {len(lng)}頭  複勝率{lng['的中'].mean()*100:.1f}%")

    # 脚質別（先行指数で簡易分類）
    if "過去平均先行指数" in lng.columns:
        print("\n  [脚質別の複勝率]（先行指数が低い=逃げ先行、高い=差し追込）")
        lng["脚質帯"] = pd.qcut(lng["過去平均先行指数"].rank(method="first"),
                              4, labels=["逃げ・先行", "好位", "中団", "後方"])
        for q, g in lng.groupby("脚質帯", observed=True):
            r, n = safe_rate(g["的中"] == 1, g["的中"] >= 0)
            if r is not None:
                mark = "🔴" if r < 50 else "🟡" if r < 58 else "🟢"
                print(f"    {mark} {q:<10} {n:>4}頭  複勝率{r:>5.1f}%")

    # 距離延長/短縮別
    if "距離変化" in lng.columns:
        print("\n  [前走からの距離変化別]")
        lng["距離変化帯"] = pd.cut(lng["距離変化"], [-9999, -1, 0, 200, 9999],
                                labels=["距離短縮", "同距離", "延長(〜200)", "延長(200〜)"])
        for q, g in lng.groupby("距離変化帯", observed=True):
            r, n = safe_rate(g["的中"] == 1, g["的中"] >= 0)
            if r is not None and n >= 10:
                mark = "🔴" if r < 50 else "🟡" if r < 58 else "🟢"
                print(f"    {mark} {q:<12} {n:>4}頭  複勝率{r:>5.1f}%")

    # 距離経験別
    if "同距離過去勝率" in lng.columns:
        print("\n  [同距離(長距離)の経験別]")
        lng["距離経験"] = np.where(lng["同距離過去勝率"].fillna(0) > 0, "経験あり好走", "未勝利/未経験")
        for q, g in lng.groupby("距離経験"):
            r, n = safe_rate(g["的中"] == 1, g["的中"] >= 0)
            if r is not None:
                mark = "🔴" if r < 50 else "🟡" if r < 58 else "🟢"
                print(f"    {mark} {q:<14} {n:>4}頭  複勝率{r:>5.1f}%")

    # ── B. 多頭数での外し方 ──
    print("\n" + "=" * 58)
    print("B. 多頭数(13頭〜)での外し方")
    print("=" * 58)
    many = top[top["出走頭数"] >= 13].copy()
    print(f"  多頭数のMF複勝1位: {len(many)}頭  複勝率{many['的中'].mean()*100:.1f}%")

    # 枠順別（内/中/外）
    if "馬番" in many.columns and "出走頭数" in many.columns:
        many["相対枠"] = many["馬番"] / many["出走頭数"]
        many["枠位置"] = pd.cut(many["相対枠"], [0, 0.33, 0.66, 1.01],
                              labels=["内枠", "中枠", "外枠"])
        print("\n  [枠順別の複勝率]")
        for q, g in many.groupby("枠位置", observed=True):
            r, n = safe_rate(g["的中"] == 1, g["的中"] >= 0)
            if r is not None:
                mark = "🔴" if r < 50 else "🟡" if r < 58 else "🟢"
                print(f"    {mark} {q:<6} {n:>4}頭  複勝率{r:>5.1f}%")

    # 脚質別
    if "過去平均先行指数" in many.columns:
        print("\n  [脚質別の複勝率]")
        many["脚質帯"] = pd.qcut(many["過去平均先行指数"].rank(method="first"),
                               4, labels=["逃げ・先行", "好位", "中団", "後方"])
        for q, g in many.groupby("脚質帯", observed=True):
            r, n = safe_rate(g["的中"] == 1, g["的中"] >= 0)
            if r is not None:
                mark = "🔴" if r < 50 else "🟡" if r < 58 else "🟢"
                print(f"    {mark} {q:<10} {n:>4}頭  複勝率{r:>5.1f}%")

    # ── C. 飛んだ馬 vs 当たった馬 の特徴差 ──
    print("\n" + "=" * 58)
    print("C. 飛んだ馬 vs 当たった馬 の平均値比較（全体）")
    print("=" * 58)
    hit = top[top["的中"] == 1]
    miss = top[top["飛んだ"] == 1]
    compare_cols = ["MF複勝率", "前走着順", "同距離過去勝率", "過去出走数",
                    "体重増減", "年齢", "距離変化", "出走頭数"]
    print(f"  {'項目':<16} {'当たった馬':>10} {'飛んだ馬':>10} {'差':>8}")
    for c in compare_cols:
        if c in top.columns:
            h = hit[c].mean()
            m = miss[c].mean()
            if pd.notna(h) and pd.notna(m):
                print(f"  {c:<16} {h:>10.2f} {m:>10.2f} {h-m:>+8.2f}")
    print("  → 差が大きい項目 = 当たり外れを分ける要因 = 特徴量化の候補")

    print("\n分析完了")
    print("※ 🔴の条件が長距離・多頭数の中でも特に弱い部分。")
    print("  そこを狙った特徴量を作れば効率的に精度が上がる。")


if __name__ == "__main__":
    main()
