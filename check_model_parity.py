# -*- coding: utf-8 -*-
"""検証に使う bet_cache が、本番モデルと同じものかを確かめる（2026-08-17）

なぜ必要か
  2026-08-17に、検証の入力(bet_cache)と本番(train_mf_v2)で別のモデルを
  使っていたことが判明した。
    検証: LightGBM 1本・距離分割なし
    本番: LGB5シード + XGB2 + CatBoost2 + LambdaRank・1900mで距離分割
  この状態で測った「5年110.0%」は幻で、入力を揃えたら68.5%に落ちた。

  数字を何度検算しても、入力が違えば意味がない。だから買い方を測る前に、
  まず入力が本番と同じかをここで確かめる。

見るもの
  ① 荒れR率のような集計値が近いか（買う対象レースの量が合っているか）
  ② 馬ごとの順位が一致するか（実際に買う馬が同じか）
  ②が本命。集計が合っていても、選ぶ馬が違えば別の買い方になる。

  2026-08-17の実測では、集計は1.5pt差で近かったのに、
  MF勝率1位が同じ馬になるのは71.5%しかなかった。つまり28.5%のレースで
  軸が別の馬になる。集計だけ見ていると見逃す。

前提
  model_mf_result.csv が必要。本番と同じ作り方のOOS予測で、
    KEIBA_TEST_YEAR=2025 python train_mf_v2.py backtest
  で作れる（数時間かかる。model_mf_bt.pkl が2.6GB出るので終わったら消すこと）。

実行: python check_model_parity.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ここを下回ったら「別のモデルを見ている」とみなす
MIN_AXIS_AGREE = 0.90      # MF勝率1位が同じ馬になる割合
MIN_SPEARMAN = 0.97        # 複勝順位の順位相関
MAX_ARE_GAP = 1.0          # 荒れR率の差(pt)


def log(m):
    print(m, flush=True)


def main():
    if not os.path.exists("model_mf_result.csv"):
        log("model_mf_result.csv がありません。")
        log("  KEIBA_TEST_YEAR=2025 python train_mf_v2.py backtest  で作れます。")
        return
    E = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    E["race_id"] = E["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    E["年"] = E["race_id"].str[:4]
    years = sorted(E["年"].unique())
    log(f"本番モデルのOOS出力: {len(E):,}頭 / {E.race_id.nunique():,}レース  年={years}")

    ok_all = True
    for y in years:
        path = f"bet_cache_{y}.csv"
        if not os.path.exists(path):
            log(f"\n{y}: {path} が無いので照合できません")
            continue
        e = E[E["年"] == y].copy()
        C = pd.read_csv(path, dtype={"race_id": str, "bn": str})
        g = C.groupby("race_id")
        C["r1_c"] = g["p_win"].rank(ascending=False, method="first")

        # ① 集計: 荒れR率
        fe = e[pd.to_numeric(e["人気"], errors="coerce") == 1]
        fc = C[C.pr == 1]
        are_e = (pd.to_numeric(fe["MF複勝順位"], errors="coerce") >= 4).mean() * 100
        are_c = (fc.mr >= 4).mean() * 100

        # ② 馬ごと: 順位の一致
        m = e[["race_id", "馬名", "MF複勝順位", "MF勝率順位"]].rename(
            columns={"MF複勝順位": "mr_e", "MF勝率順位": "r1_e"})
        j = m.merge(C[["race_id", "馬名", "mr", "r1_c"]], on=["race_id", "馬名"], how="inner")
        if j.empty:
            log(f"\n{y}: 馬名で照合できませんでした")
            continue
        rho = j[["mr_e", "mr"]].corr(method="spearman").iloc[0, 1]

        def _axis(x):
            if x.r1_e.isna().all() or x.r1_c.isna().all():
                return np.nan
            return x.loc[x.r1_e.idxmin(), "馬名"] == x.loc[x.r1_c.idxmin(), "馬名"]
        agree = j.groupby("race_id").apply(_axis).mean()

        gap_ok = abs(are_e - are_c) <= MAX_ARE_GAP
        rho_ok = rho >= MIN_SPEARMAN
        ax_ok = agree >= MIN_AXIS_AGREE
        ok = gap_ok and rho_ok and ax_ok
        ok_all &= ok

        log(f"\n=== {y} ===")
        log(f"  荒れR率        本番{are_e:>5.1f}%  検証{are_c:>5.1f}%  差{abs(are_e-are_c):>4.1f}pt"
            f"   {'OK' if gap_ok else f'NG(>{MAX_ARE_GAP}pt)'}")
        log(f"  複勝順位の相関  {rho:.3f}"
            f"{'':>24}{'OK' if rho_ok else f'NG(<{MIN_SPEARMAN})'}")
        log(f"  MF勝率1位が同じ馬 {agree*100:>5.1f}%"
            f"{'':>20}{'OK' if ax_ok else f'NG(<{MIN_AXIS_AGREE*100:.0f}%)'}")
        log(f"  → {'✅ 同じモデルとみなせる' if ok else '⚠ 別のモデルを見ている。この bet_cache で買い方を決めてはいけない'}")

    log("\n" + ("=" * 60))
    if ok_all:
        log("✅ 検証の入力は本番と一致。買い方の検証に使ってよい。")
    else:
        log("⚠ 検証の入力が本番と一致しない。")
        log("  ここで測った回収率は本番の成績を表さない（2026-08-17に110.0%→68.5%）。")
        log("  prep_cache.py のモデル定義を train_mf_v2.py に揃えるか、")
        log("  train_mf_v2.py backtest を年ごとに回して本番と同じ入力を作ること。")


if __name__ == "__main__":
    main()
