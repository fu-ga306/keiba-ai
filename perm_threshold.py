# -*- coding: utf-8 -*-
"""しきい値と回収率の単調な関係が、中身のあるものか確かめる（2026-09-03）

なぜ要るか
  事前登録した実験で、しきい値を上げるほど回収率が上がる形が出た。

    gap>=1.3  12,647点  121.3%
    gap>=1.5   8,322点  132.0%
    gap>=1.7   4,790点  155.6%
    gap>=2.0   2,229点  199.7%

  きれいすぎる。**「点数が減るほど回収率が上がる」だけの見かけ**かもしれない。
  中身をゼロにしても同じ形が出るなら、それは選び方の副作用でしかない。

やり方
  レース内で gap をシャッフルする。順位の情報だけを壊し、
  レースの顔ぶれ・オッズ・払戻はそのまま。
  シャッフル後も同じ単調性が出るなら、単調性に意味はない。

  単勝だけで測る（ワイドは組み合わせの探索が重く、本質でもない）。

実行
  python perm_threshold.py
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
CACHE = os.path.join(BASE_DIR, "_perm_pred_cache.pkl")
THS = (1.3, 1.5, 1.7, 2.0)
N_PERM = 200


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def get_pred():
    """予測は重いので一度だけ作って使い回す。"""
    if os.path.exists(CACHE):
        log("キャッシュから読み込み")
        return pd.read_pickle(CACHE)
    sys.path.insert(0, BASE_DIR)
    import exp_model_202609 as E
    log("予測を作成（8分ほど）")
    D, BASE = E.load()
    P = E.fit_predict(D, BASE, E.DEV_YEARS, [42, 7, 123], 600)
    P.to_pickle(CACHE)
    return P


def tansho_table(P):
    """単勝の払戻を各行に付ける。ここまでやれば以降はベクトル演算で済む。"""
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv = jv[jv.券種 == "単勝"].copy()
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    jv["bn"] = pd.to_numeric(jv["組み合わせ"], errors="coerce")
    P = P.copy()
    P["bn"] = pd.to_numeric(P["馬番"], errors="coerce")
    P = P.merge(jv[["race_id", "bn", "払戻金"]], on=["race_id", "bn"], how="left")
    P["払戻"] = P["払戻金"].fillna(0.0)
    return P


def roi_by_threshold(P, gap_col="gap"):
    """各レースで gap 最大の馬を軸にし、しきい値ごとの回収率を出す。"""
    idx = P.groupby("race_id")[gap_col].idxmax()
    ax = P.loc[idx, [gap_col, "払戻"]]
    out = {}
    for t in THS:
        s = ax[ax[gap_col] >= t]["払戻"]
        out[t] = (len(s), s.mean() if len(s) else np.nan)
    return out


def main():
    P = get_pred()
    P = tansho_table(P)
    log(f"  {len(P):,}頭 / {P.race_id.nunique():,}レース")

    obs = roi_by_threshold(P)
    log("")
    log("  === 実測（単勝のみ） ===")
    for t in THS:
        n, r = obs[t]
        log(f"    gap>={t}  {n:6,}点  {r:6.1f}%")

    log("")
    log(f"  === レース内でgapをシャッフル（{N_PERM}回） ===")
    rng = np.random.default_rng(903)
    g = P.groupby("race_id")[["gap"]]
    order = P.groupby("race_id").ngroup().values
    null = {t: [] for t in THS}
    S = P.copy()
    for i in range(N_PERM):
        # レース内だけで並べ替える
        S["gap"] = (P.assign(_r=rng.random(len(P)))
                    .sort_values(["race_id", "_r"])["gap"].values)
        S2 = S.sort_values("race_id")
        res = roi_by_threshold(S2)
        for t in THS:
            null[t].append(res[t][1])
        if (i + 1) % 50 == 0:
            log(f"    {i+1}/{N_PERM}")

    log("")
    log("  %-12s %-16s %-14s %s" % ("しきい値", "シャッフル平均", "実測", "実測以上が出る確率"))
    log("  " + "-" * 62)
    for t in THS:
        a = np.array([x for x in null[t] if np.isfinite(x)])
        n, r = obs[t]
        p = (a >= r).mean() if len(a) else np.nan
        mark = "○ 中身がある" if p < 0.05 else "⚠ 偶然で説明できる"
        log("  gap>=%-6.1f %10.1f%%      %8.1f%%     %5.1f%%  %s"
            % (t, a.mean(), r, p * 100, mark))
    log("")
    log("  シャッフルでも単調に上がるなら、単調性は選び方の副作用にすぎない。")
    for t in THS:
        a = np.array([x for x in null[t] if np.isfinite(x)])
        log(f"    シャッフル gap>={t}  {a.mean():.1f}%")


if __name__ == "__main__":
    main()
