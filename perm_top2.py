# -*- coding: utf-8 -*-
"""スリッページを通った構成に、探索込みの順列検定をかける（高速版・2026-08-16）

前版(perm_top.py)は「1頭ごとに rid_all == rr で相手を探す」書き方をしており
O(n^2)になって実測値の計算すら終わらなかった。レース単位に一度だけ展開して
以降は配列だけで回すよう作り直した（slippage_sim.py で使ったのと同じ方針）。

やること
  レース内でモデルの順位をシャッフルして「実力ゼロ」のデータを作り、
  同じ探索グリッドで最良ROIを記録する。それを何百回も繰り返して、
  実測の最良ROIがその分布のどこに位置するかを見る。

  実測が分布に埋もれていれば探索の産物。外に出ていれば本物の可能性がある。

⚠ 検定するのは「直近2年の馬単ROI」。採用判断に使った指標と揃える。

実行: python perm_top2.py → perm_top_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RECENT = [2024, 2025]
N_PERM = 400
MIN_PT = 100
rng = np.random.default_rng(20260816)


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in RECENT], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "馬場状態_num",
                              "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    UMA = {}
    for r in jv[jv.券種 == "馬単"].itertuples():
        UMA[(r.race_id, r.組み合わせ)] = r.払戻金
    log(f"検体（直近2年） {len(D):,}頭 / {D.race_id.nunique():,}レース")

    # ── レース単位に一度だけ展開する（ここが速度の肝）──
    races = []
    for rid, g in D.groupby("race_id", sort=False):
        n = len(g)
        bn = g.bn.values
        # 馬単の払戻を n×n 行列にしておく（i→j）
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = UMA.get((rid, f"{bn[i]}-{bn[j]}"), 0.0)
        races.append({
            "odds": g.odds.values.astype(float),
            "cw": g.c_win.values, "c2": g.c_top2.values, "c3": g.c_top3.values,
            "pr": g.pr.values.astype(float),
            "dist": float(g["距離"].iloc[0]) if pd.notna(g["距離"].iloc[0]) else 0.0,
            "baba": float(g["馬場状態_num"].iloc[0]) if pd.notna(g["馬場状態_num"].iloc[0]) else 0.0,
            "tosu": float(g["出走頭数"].iloc[0]) if pd.notna(g["出走頭数"].iloc[0]) else 0.0,
            "uma": M,
        })
    log(f"  展開完了 {len(races):,}レース\n")

    ODDS = [(1, 99), (1, 20), (5, 30)]
    AXN, MATEN = [1, 2], [3, 5]

    def rank_desc(v):
        # 大きいほど1位。同値は先着順（argsortを2回）
        o = np.argsort(-v, kind="stable")
        r = np.empty(len(v))
        r[o] = np.arange(1, len(v) + 1)
        return r

    def best_roi(shuffle):
        """全レースを1回走査して、グリッド内の最良ROIを返す。"""
        keys = [(c, b, o, a, m)
                for c in ("全体", "長距離", "道悪", "少頭数", "荒れR")
                for b in (0, 1, 2)
                for o in range(len(ODDS))
                for a in AXN for m in MATEN]
        cost = {k: 0.0 for k in keys}
        ret = {k: 0.0 for k in keys}
        for R in races:
            v = (R["cw"], R["c2"], R["c3"])
            if shuffle:
                p = rng.permutation(len(R["odds"]))
                rk = [rank_desc(x[p]) for x in v]
            else:
                rk = [rank_desc(x) for x in v]
            fav = np.nan
            m1 = R["pr"] == 1
            if m1.any():
                fav = rk[2][m1][0]
            flags = {"全体": True, "長距離": R["dist"] >= 1900,
                     "道悪": R["baba"] >= 3, "少頭数": R["tosu"] <= 12,
                     "荒れR": (not np.isnan(fav)) and fav >= 4}
            for oi, (olo, ohi) in enumerate(ODDS):
                om = (R["odds"] >= olo) & (R["odds"] < ohi)
                for bi in (0, 1, 2):
                    r = rk[bi]
                    for av in AXN:
                        ax = np.where((r <= av) & om)[0]
                        if not len(ax):
                            continue
                        i = ax[np.argmin(r[ax])]
                        for mn in MATEN:
                            mate = np.where((r <= mn) & (np.arange(len(r)) != i))[0]
                            if not len(mate):
                                continue
                            c = 100.0 * len(mate)
                            p_ = R["uma"][i, mate].sum()
                            for cn, ok in flags.items():
                                if ok:
                                    k = (cn, bi, oi, av, mn)
                                    cost[k] += c
                                    ret[k] += p_
        best = -1.0
        for k in keys:
            if cost[k] >= MIN_PT * 100:
                best = max(best, ret[k] / cost[k] * 100)
        return best

    log("実測の最良ROIを計算中...")
    obs = best_roi(False)
    log(f"  実測 最良ROI = {obs:.1f}%\n")

    log(f"シャッフル {N_PERM}回...")
    null = []
    for k in range(N_PERM):
        null.append(best_roi(True))
        if (k + 1) % 50 == 0:
            log(f"  {k+1}/{N_PERM}  帰無の中央 {np.median(null):.1f}%")
    null = np.array(null)
    p = float(np.mean(null >= obs))
    log("\n=== 結果 ===")
    log(f"  実測 最良ROI     {obs:.1f}%")
    log(f"  帰無の中央値     {np.median(null):.1f}%")
    log(f"  帰無の95%上限    {np.percentile(null, 95):.1f}%")
    log(f"  p値（探索込み）  {p:.4f}")
    log("  " + ("→ 有意。探索の産物では説明できない" if p < 0.05
                else "→ 有意でない。実力ゼロでも同じ数字が出る"))
    pd.DataFrame([{"実測最良ROI": round(obs, 1),
                   "帰無中央": round(float(np.median(null)), 1),
                   "帰無95%上限": round(float(np.percentile(null, 95)), 1),
                   "p値": round(p, 4), "試行回数": N_PERM}]) \
        .to_csv("perm_top_result.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
