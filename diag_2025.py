# -*- coding: utf-8 -*-
"""2025年だけ成績が悪い理由を切り分ける（2026-09-03）

年別（作り直し後の5年）
  2021:107%  2022:110%  2023:113%  2024:134%  2025:81%

⚠ 2025年はもう見てしまった。ここで分かったことをもとに構成を変えると
  事前登録の枠組みが無効になる。**理解までにとどめ、変更はしない。**

切り分ける観点
  ① 的中率が落ちたのか、払戻が落ちたのか
     的中率が同じで回収率だけ落ちたなら、当てた馬のオッズが安かった＝運
     的中率が落ちたなら、モデルの選別力そのものが効いていない
  ② シャッフル（中身ゼロ）の水準が年で違わないか
     2025のシャッフルも低いなら、市場側の性質が変わっている
  ③ データの質が年で違わないか（欠損率）
  ④ 買った馬の人気が年で違わないか
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "_diag_pred.pkl")
YEARS = [2021, 2022, 2023, 2024, 2025]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def get_pred():
    if os.path.exists(CACHE):
        log("キャッシュから読み込み")
        return pd.read_pickle(CACHE)
    sys.path.insert(0, BASE_DIR)
    import exp_model_202609 as E
    log("5年分の予測を作成（10分ほど）")
    D, BASE = E.load()
    P = E.fit_predict(D, BASE, YEARS, [42, 7, 123], 600)
    P.to_pickle(CACHE)
    return P


def with_payout(P):
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv = jv[jv.券種 == "単勝"].copy()
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    jv["bn"] = pd.to_numeric(jv["組み合わせ"], errors="coerce")
    P = P.copy()
    P["bn"] = pd.to_numeric(P["馬番"], errors="coerce")
    P = P.merge(jv[["race_id", "bn", "払戻金"]], on=["race_id", "bn"], how="left")
    P["払戻"] = P["払戻金"].fillna(0.0)
    return P


def main():
    P = with_payout(get_pred())
    ax = P.loc[P.groupby("race_id")["gap"].idxmax()]
    ax = ax[ax.gap >= 1.5]
    log(f"  軸 {len(ax):,}点（単勝のみ・gap>=1.5）")

    log("")
    log("  === ① 的中率が落ちたのか、払戻が落ちたのか ===")
    log("  %-6s %8s %9s %9s %12s %10s" %
        ("年", "点数", "的中率", "回収率", "的中時の平均払戻", "平均人気"))
    log("  " + "-" * 62)
    for y, g in ax.groupby("年"):
        hit = g["払戻"] > 0
        log("  %-6d %8d %8.1f%% %8.1f%% %11.0f円 %9.1f"
            % (y, len(g), hit.mean() * 100, g["払戻"].mean(),
               g.loc[hit, "払戻"].mean() if hit.any() else 0,
               pd.to_numeric(g["人気"], errors="coerce").mean()))

    log("")
    log("  === ② 中身をゼロにした水準（年ごと） ===")
    log("  市場側の性質が変わっていれば、シャッフルの水準も年で動く")
    rng = np.random.default_rng(903)
    log("  %-6s %14s %12s" % ("年", "シャッフル平均", "実測"))
    log("  " + "-" * 36)
    for y in YEARS:
        Py = P[P.年 == y]
        vals = []
        for _ in range(40):
            S = Py.assign(_r=rng.random(len(Py))).sort_values(["race_id", "_r"])
            S["g2"] = Py.sort_values("race_id")["gap"].values
            a = S.loc[S.groupby("race_id")["g2"].idxmax()]
            a = a[a["g2"] >= 1.5]
            if len(a):
                vals.append(a["払戻"].mean())
        obs = ax[ax.年 == y]["払戻"].mean()
        log("  %-6d %13.1f%% %11.1f%%" % (y, np.mean(vals), obs))

    log("")
    log("  === ③ 買った馬の人気帯（年ごと・単勝のみ） ===")
    bands = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 99, "10-")]
    log("  %-6s " % "年" + "  ".join("%8s" % b[2] for b in bands))
    log("  " + "-" * 46)
    for y, g in ax.groupby("年"):
        p = pd.to_numeric(g["人気"], errors="coerce")
        log("  %-6d " % y + "  ".join(
            "%7.1f%%" % (((p >= lo) & (p <= hi)).mean() * 100) for lo, hi, _ in bands))

    log("")
    log("  === ④ 人気帯ごとの回収率（年ごと） ===")
    for lo, hi, lab in bands:
        s = ax[(pd.to_numeric(ax["人気"], errors="coerce") >= lo)
               & (pd.to_numeric(ax["人気"], errors="coerce") <= hi)]
        row = []
        for y in YEARS:
            g = s[s.年 == y]
            row.append("%6.0f%%" % g["払戻"].mean() if len(g) >= 30 else "     -")
        log("    %-6s " % lab + " ".join(row) + "   ← " + " ".join(str(y) for y in YEARS))


if __name__ == "__main__":
    main()
