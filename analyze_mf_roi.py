# -*- coding: utf-8 -*-
"""
analyze_mf_roi.py
─────────────────
市場フリー(MF)モデルの回収率を分析する。
「市場を見ない実力予想」が、実際に買って儲かるか（回収率）を検証する。

入力: model_mf_result.csv（market_free_model.py の3モデル学習で生成）
      列: race_id, 馬名, 着順_num, 単勝オッズ, 人気,
          MF勝率, MF連対率, MF複勝率,
          MF勝率順位, MF連対順位, MF複勝順位

評価軸: 的中率ではなく「回収率」（儲かるか）。
  ・MF軸◎（MF勝率1位）の単勝回収率
  ・人気帯別の回収率（人気薄で妙味があるか）
  ・通常モデル(model_result.csv)との回収率比較
  ・複勝回収率（payout_data.csv があれば実払戻、なければ概算）

使い方:
  python analyze_mf_roi.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MF_CSV     = os.path.join(BASE_DIR, "model_mf_result.csv")
NORMAL_CSV = os.path.join(BASE_DIR, "model_result.csv")
PAYOUT_CSV = os.path.join(BASE_DIR, "payout_data.csv")

BET = 100  # 1点100円


def tansho_roi(df_bets):
    """単勝回収率を計算（着順1着の馬のオッズ合計 / ベット総額）。"""
    n = len(df_bets)
    if n == 0:
        return 0, 0, 0
    wins = df_bets[df_bets["着順_num"] == 1]
    payout = (wins["単勝オッズ"] * BET).sum()
    invest = n * BET
    roi = payout / invest * 100 if invest > 0 else 0
    hit_rate = len(wins) / n * 100
    return roi, hit_rate, n


def main():
    if not os.path.exists(MF_CSV):
        print(f"MFモデルの結果CSVがありません: {MF_CSV}")
        print("先に market_free_model.py を実行してください。")
        return

    df = pd.read_csv(MF_CSV)
    df = df.dropna(subset=["着順_num"])
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    df = df.dropna(subset=["単勝オッズ"])

    print("=" * 56)
    print("MFモデル（市場フリー）回収率分析")
    print("=" * 56)
    print(f"対象: {df['race_id'].nunique()}レース / {len(df)}頭")

    # ── 1. MF軸◎（MF勝率1位）の単勝回収率 ──
    print("\n" + "=" * 56)
    print("1. MF軸◎（MF勝率1位）を単勝で買った場合")
    print("=" * 56)
    mf_axis = df[df["MF勝率順位"] == 1]
    roi, hit, n = tansho_roi(mf_axis)
    print(f"  ベット数: {n}回  的中率: {hit:.1f}%  回収率: {roi:.1f}%")
    if roi >= 100:
        print("  → 単勝で買って黒字（実力予想で妙味を出せている）")
    else:
        print("  → 単勝では赤字（人気馬中心 or 妙味不足）")

    # ── 2. MF軸◎を人気帯別に見た回収率（人気薄で妙味があるか）──
    print("\n" + "=" * 56)
    print("2. MF軸◎の人気帯別 回収率（妙味の所在）")
    print("=" * 56)
    if "人気" in mf_axis.columns:
        bands = [
            ("1番人気",     mf_axis["人気"] == 1),
            ("2-3番人気",   (mf_axis["人気"] >= 2) & (mf_axis["人気"] <= 3)),
            ("4-6番人気",   (mf_axis["人気"] >= 4) & (mf_axis["人気"] <= 6)),
            ("7番人気以下", mf_axis["人気"] >= 7),
        ]
        for label, mask in bands:
            sub = mf_axis[mask]
            roi, hit, n = tansho_roi(sub)
            if n == 0:
                continue
            mark = "🟢" if roi >= 100 else "🔴"
            print(f"  {mark} {label:10}: {n:4d}回  的中{hit:5.1f}%  回収率{roi:6.1f}%")
        print("  → 人気薄帯で回収率100%超なら、市場が見落とす妙味を捉えている")

    # ── 3. MF勝率1位 が 何番人気を選んでいるか（人気追従度）──
    print("\n" + "=" * 56)
    print("3. MF軸◎の人気分布（市場との独立性）")
    print("=" * 56)
    if "人気" in mf_axis.columns:
        fav = (mf_axis["人気"] == 1).mean() * 100
        non_fav = (mf_axis["人気"] >= 4).mean() * 100
        print(f"  1番人気を選んだ割合: {fav:.1f}%")
        print(f"  4番人気以下を選んだ割合: {non_fav:.1f}%")
        print("  → 通常モデル(85.6%が1番人気)より低ければ、独立した実力評価")

    # ── 4. 通常モデルとの回収率比較 ──
    print("\n" + "=" * 56)
    print("4. 通常モデル◎ vs MFモデル◎ 回収率比較")
    print("=" * 56)
    if os.path.exists(NORMAL_CSV):
        ndf = pd.read_csv(NORMAL_CSV)
        ndf = ndf.dropna(subset=["着順_num"])
        ndf["単勝オッズ"] = pd.to_numeric(ndf["単勝オッズ"], errors="coerce")
        ndf = ndf.dropna(subset=["単勝オッズ"])
        n_axis = ndf[ndf["予測順位"] == 1]
        roi_n, hit_n, cnt_n = tansho_roi(n_axis)
        roi_m, hit_m, cnt_m = tansho_roi(mf_axis)
        print(f"  通常モデル◎: {cnt_n}回  的中{hit_n:.1f}%  回収率{roi_n:.1f}%")
        print(f"  MFモデル◎:   {cnt_m}回  的中{hit_m:.1f}%  回収率{roi_m:.1f}%")
        if roi_m > roi_n:
            print("  → MFモデルの方が回収率が高い（妙味を捉えている）")
        else:
            print("  → 現状は通常モデルが上（MFは血統等の強化余地あり）")
    else:
        print("  model_result.csv がないため比較スキップ")

    # ── 5. 複勝回収率（払戻データがあれば実値、なければ概算）──
    print("\n" + "=" * 56)
    print("5. MF軸◎の複勝回収率")
    print("=" * 56)
    if os.path.exists(PAYOUT_CSV):
        pay = pd.read_csv(PAYOUT_CSV)
        fuku = pay[pay["券種"] == "複勝"].copy()
        # race_id + 馬番 で複勝払戻を引く必要があるが、
        # ここでは race_id 単位で MF◎が複勝圏(3着内)かどうかで概算
        mf_axis_fuku = mf_axis[mf_axis["着順_num"] <= 3]
        print(f"  MF◎の複勝的中: {len(mf_axis_fuku)}/{len(mf_axis)}回 "
              f"({len(mf_axis_fuku)/len(mf_axis)*100:.1f}%)")
        print("  ※ 実複勝オッズでの回収率は payout_data.csv 突合で別途計算")
    else:
        # 概算: 複勝率（3着内）だけ表示
        fuku_rate = (mf_axis["着順_num"] <= 3).mean() * 100
        print(f"  MF◎の複勝的中率: {fuku_rate:.1f}%（実払戻データ取得後に回収率計算）")
        print("  → payout_data.csv 取得後に複勝回収率を算出できる")

    print("\n分析完了")
    print("※ この結果は血統データ追加前のベースライン。")
    print("  血統追加後に再実行すれば改善度を測れる。")


if __name__ == "__main__":
    main()
