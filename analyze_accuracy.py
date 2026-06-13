# -*- coding: utf-8 -*-
"""
当日の予想(today_predictions.csv)と実際のレース結果を比較し、
AI評価の精度を多角的に分析する。

分析内容:
  1. ◎(印)馬の着順分布 → 印がどれだけ当たっているか
  2. 予測順位 vs 実際着順の相関 → モデルの順位付け精度
  3. AI予測1位馬の勝率・連対率・複勝率(実績)
  4. 純粋AI評価(MF) vs 通常モデルの精度比較
  5. 戦略該当馬の実績
  6. 人気帯別の精度(人気を当てているだけか、妙味を出せているか)
出力はすべてコンソール。結果はnetkeibaから取得(負荷軽減のため5秒間隔)。
"""
import os
import time
import pandas as pd
import numpy as np
from result_tracker import get_race_result, BASE_DIR

PRED_FILE = os.path.join(BASE_DIR, "today_predictions.csv")


def main():
    if not os.path.exists(PRED_FILE):
        print(f"予想ファイルがありません: {PRED_FILE}")
        return

    pred = pd.read_csv(PRED_FILE)
    pred["race_id"] = pred["race_id"].astype(str)
    race_ids = sorted(pred["race_id"].unique())
    print(f"対象レース数: {len(race_ids)}")

    # 各レースの結果を取得して予想とマージ
    merged_all = []
    for rid in race_ids:
        res = get_race_result(rid)
        if res is None:
            print(f"  結果取得失敗: {rid}")
            time.sleep(5)
            continue
        res = res[["馬名", "着順_num"]].copy()
        res["race_id"] = rid
        p = pred[pred["race_id"] == rid].copy()
        m = p.merge(res, on=["race_id", "馬名"], how="left")
        merged_all.append(m)
        print(f"  取得: {rid} ({len(m)}頭)")
        time.sleep(5)

    if not merged_all:
        print("結果が取得できませんでした")
        return

    df = pd.concat(merged_all, ignore_index=True)
    df = df.dropna(subset=["着順_num"])
    print(f"\n結果と照合できた頭数: {len(df)}")

    # ── 1. 印(推奨ランク)別の着順分布 ──
    print("\n" + "=" * 55)
    print("1. 印別の実際の着順")
    print("=" * 55)
    if "推奨ランク" in df.columns:
        for mark in ["◎", "○", "▲", "△", "×"]:
            sub = df[df["推奨ランク"] == mark]
            if len(sub) == 0:
                continue
            win = (sub["着順_num"] == 1).mean() * 100
            ren = (sub["着順_num"] <= 2).mean() * 100
            fuku = (sub["着順_num"] <= 3).mean() * 100
            avg = sub["着順_num"].mean()
            print(f"  {mark}: 頭数{len(sub):3d}  勝率{win:5.1f}%  "
                  f"連対率{ren:5.1f}%  複勝率{fuku:5.1f}%  平均着順{avg:.1f}")

    # ── 2. 予測順位 vs 実着順の相関 ──
    print("\n" + "=" * 55)
    print("2. 予測順位 vs 実着順の精度")
    print("=" * 55)
    if "予測順位" in df.columns:
        d2 = df.dropna(subset=["予測順位"])
        corr = d2["予測順位"].corr(d2["着順_num"])
        print(f"  順位相関(スピアマン的): {corr:.3f}  (1に近いほど良い)")
        # 予測1位が実際に何着だったか
        top1 = d2[d2["予測順位"] == 1]
        if len(top1) > 0:
            print(f"  予測1位馬({len(top1)}レース): "
                  f"勝率{(top1['着順_num']==1).mean()*100:.1f}%  "
                  f"連対率{(top1['着順_num']<=2).mean()*100:.1f}%  "
                  f"複勝率{(top1['着順_num']<=3).mean()*100:.1f}%")

    # ── 3. 通常モデル vs 純粋AI(MF)の予測1位精度比較 ──
    print("\n" + "=" * 55)
    print("3. 通常モデル vs 純粋AI評価(MF) の1位精度")
    print("=" * 55)
    for col, label in [("勝ち確率", "通常モデル"), ("MF勝ち確率", "純粋AI(MF)")]:
        if col not in df.columns:
            continue
        # レースごとに該当列の最大値の馬を取得
        idx = df.groupby("race_id")[col].idxmax().dropna()
        tops = df.loc[idx]
        if len(tops) > 0:
            print(f"  {label}1位: "
                  f"勝率{(tops['着順_num']==1).mean()*100:.1f}%  "
                  f"連対率{(tops['着順_num']<=2).mean()*100:.1f}%  "
                  f"複勝率{(tops['着順_num']<=3).mean()*100:.1f}%")

    # ── 4. 戦略該当馬の実績 ──
    print("\n" + "=" * 55)
    print("4. 戦略該当馬の実績")
    print("=" * 55)
    if "該当戦略" in df.columns:
        strat = df[df["該当戦略"].notna() & (df["該当戦略"].astype(str) != "")
                   & (df["該当戦略"].astype(str) != "nan")]
        if len(strat) > 0:
            for s, g in strat.groupby("該当戦略"):
                win = (g["着順_num"] == 1).mean() * 100
                fuku = (g["着順_num"] <= 3).mean() * 100
                print(f"  {s}: 頭数{len(g):2d}  勝率{win:.1f}%  複勝率{fuku:.1f}%")
        else:
            print("  戦略該当馬なし")

    # ── 5. 人気帯別の精度(モデルが人気を超える妙味を出せているか) ──
    print("\n" + "=" * 55)
    print("5. 人気帯別 AI予測1位の的中(人気馬偏重でないか)")
    print("=" * 55)
    if "予測順位" in df.columns and "人気" in df.columns:
        top1 = df[df["予測順位"] == 1].copy()
        top1["人気帯"] = pd.cut(top1["人気"], [0, 1, 3, 6, 100],
                              labels=["1人気", "2-3人気", "4-6人気", "7人気以下"])
        for band, g in top1.groupby("人気帯", observed=True):
            if len(g) == 0:
                continue
            win = (g["着順_num"] == 1).mean() * 100
            print(f"  AI1位が{band}: {len(g)}回  勝率{win:.1f}%")

    print("\n分析完了")


if __name__ == "__main__":
    main()
