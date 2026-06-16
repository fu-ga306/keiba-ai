# -*- coding: utf-8 -*-
"""
analyze_weakness.py
───────────────────
MFモデルが「どこで予想を外しているか」を多角的に分析する。
弱点を特定して、次に作るべき特徴量の方向性を決めるための診断ツール。

入力:
  model_mf_result.csv  (MFモデルの2025年予測結果)
  race_features.csv    (条件列: 距離/競馬場/クラス/馬場/芝ダート等を結合)

分析軸（MF複勝1位が実際に複勝圏3着内に入った率＝精度で評価）:
  1. 距離帯別の精度
  2. 競馬場別の精度
  3. クラス別の精度
  4. 芝/ダート別の精度
  5. 馬場状態別の精度
  6. 出走頭数別の精度
  7. 人気帯別の精度（MFが人気薄をどれだけ拾えているか）
  8. 「MFと市場の評価が割れたレース」での精度

使い方:
  python analyze_weakness.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MF_CSV   = os.path.join(BASE_DIR, "model_mf_result.csv")
FEAT_CSV = os.path.join(BASE_DIR, "race_features.csv")

JYO_MAP = {1:"札幌",2:"函館",3:"福島",4:"新潟",5:"東京",
           6:"中山",7:"中京",8:"京都",9:"阪神",10:"小倉"}
DIST_MAP = {1:"短距離(〜1400)", 2:"中距離(1600-2000)", 3:"長距離(2001〜)"}
CLASS_MAP = {1:"新馬",2:"未勝利",3:"1勝",4:"2勝",5:"3勝",6:"OP",7:"G3",8:"G2",9:"G1"}
BABA_MAP = {1:"良",2:"稍重",3:"重",4:"不良"}


def precision_by(df, col, label_map=None, min_n=20):
    """colの値ごとに、MF複勝1位の実複勝率（3着内率）と勝率1位の実勝率を出す。"""
    results = []
    for val, g in df.groupby(col):
        # そのグループのレースで MF複勝1位馬を集める
        fuku1 = g[g["MF複勝順位"] == 1]
        win1  = g[g["MF勝率順位"] == 1]
        if len(fuku1) < min_n:
            continue
        fuku_hit = (fuku1["着順_num"] <= 3).mean() * 100  # 複勝1位の実複勝率
        win_hit  = (win1["着順_num"] == 1).mean() * 100     # 勝率1位の実勝率
        name = label_map.get(val, str(val)) if label_map else str(val)
        results.append((name, len(fuku1), fuku_hit, win_hit))
    return results


def print_section(title, results, sort_idx=2):
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)
    if not results:
        print("  （データ不足）")
        return
    print(f"  {'区分':<18} {'N':>5} {'複勝1位の複勝率':>14} {'勝率1位の勝率':>12}")
    # 複勝率の低い順（弱点が上に来る）
    for name, n, fuku, win in sorted(results, key=lambda x: x[sort_idx]):
        mark = "🔴" if fuku < 50 else "🟡" if fuku < 58 else "🟢"
        print(f"  {mark} {name:<16} {n:>5} {fuku:>12.1f}% {win:>11.1f}%")


def main():
    if not os.path.exists(MF_CSV):
        print(f"MFモデルの結果CSVがありません: {MF_CSV}")
        print("先に market_free_model.py を実行してください。")
        return

    mf = pd.read_csv(MF_CSV)
    mf = mf.dropna(subset=["着順_num"])
    mf["race_id"] = mf["race_id"].astype(str)

    # race_features.csv から条件列を結合
    feat_cols = ["race_id", "馬名", "距離カテゴリ", "競馬場cd", "クラス_num",
                 "is_turf", "馬場状態_num", "出走頭数", "距離"]
    feat = pd.read_csv(FEAT_CSV, low_memory=False)
    feat["race_id"] = feat["race_id"].astype(str)
    use = [c for c in feat_cols if c in feat.columns]
    df = mf.merge(feat[use].drop_duplicates(subset=["race_id", "馬名"]),
                  on=["race_id", "馬名"], how="left")

    print("=" * 56)
    print("MFモデル 弱点分析（どこで外しているか）")
    print("=" * 56)
    print(f"対象: {df['race_id'].nunique()}レース")
    print("※ 複勝1位の複勝率が低い区分 = 弱点（赤🔴が改善対象）")
    print("※ 全体平均の複勝1位的中率は約56%が基準")

    # 1. 距離帯別
    if "距離カテゴリ" in df.columns:
        print_section("1. 距離帯別の精度", precision_by(df, "距離カテゴリ", DIST_MAP))

    # 2. 競馬場別
    if "競馬場cd" in df.columns:
        print_section("2. 競馬場別の精度", precision_by(df, "競馬場cd", JYO_MAP))

    # 3. クラス別
    if "クラス_num" in df.columns:
        print_section("3. クラス別の精度", precision_by(df, "クラス_num", CLASS_MAP))

    # 4. 芝/ダート別
    if "is_turf" in df.columns:
        print_section("4. 芝/ダート別の精度",
                      precision_by(df, "is_turf", {1:"芝", 0:"ダート"}))

    # 5. 馬場状態別
    if "馬場状態_num" in df.columns:
        print_section("5. 馬場状態別の精度", precision_by(df, "馬場状態_num", BABA_MAP))

    # 6. 出走頭数別
    if "出走頭数" in df.columns:
        df["頭数帯"] = pd.cut(df["出走頭数"], [0, 8, 12, 16, 30],
                            labels=["少頭数(〜8)", "中(9-12)", "多(13-16)", "フル(17〜)"])
        print_section("6. 出走頭数別の精度", precision_by(df, "頭数帯", min_n=20))

    # 7. 人気帯別（MFが人気薄を拾えているか）
    if "人気" in df.columns:
        fuku1 = df[df["MF複勝順位"] == 1].copy()
        fuku1["人気帯"] = pd.cut(fuku1["人気"], [0, 1, 3, 6, 100],
                               labels=["1番人気", "2-3番人気", "4-6番人気", "7番〜"])
        print("\n" + "=" * 56)
        print("7. MF複勝1位の人気帯別 実複勝率")
        print("=" * 56)
        for band, g in fuku1.groupby("人気帯", observed=True):
            if len(g) < 10:
                continue
            hit = (g["着順_num"] <= 3).mean() * 100
            mark = "🔴" if hit < 50 else "🟡" if hit < 58 else "🟢"
            print(f"  {mark} {band:<10} {len(g):>5}回  実複勝率{hit:>6.1f}%")
        print("  → 人気薄帯の複勝率が高ければ、妙味馬を実力で見抜けている")

    print("\n" + "=" * 56)
    print("【弱点の読み方】")
    print("=" * 56)
    print("  🔴の区分 = MFモデルが苦手な条件")
    print("  → そこに効く特徴量を作れば精度が上がる")
    print("  例: 長距離が🔴 → スタミナ・血統(長距離適性)の特徴量")
    print("      ダートが🔴 → ダート専用の実績特徴量")
    print("      多頭数が🔴 → 枠順・展開の特徴量")
    print("\n分析完了")


if __name__ == "__main__":
    main()
