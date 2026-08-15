# -*- coding: utf-8 -*-
"""選ぶ基準を券種に合わせる（2026-08-15・第5世代）

前回の反省（利用者の指摘）
  第4世代では軸も相手も全部「MF複勝順位（c_top3のランク）」で選んでいた。
  それでは複勝が有利になるのは当たり前で、券種ごとの優劣を比べたことにならない。

  実際、探索期間と検証期間の相関は
    複勝 +0.582 / ワイド +0.400 / 馬連 +0.243 / 単勝 -0.193 / 馬単裏 -0.205
  と、複勝系だけが再現した。これは「複勝順位で選んだから」かもしれない。

今回
  券種が問う着順に合わせて、選ぶ基準を変える。
    単勝・馬単の1着     → 勝率順位（c_win）
    馬連・馬単の2着以内 → 連対順位（c_top2）
    複勝・ワイド・3連複 → 複勝順位（c_top3）
  さらに全組み合わせ（3基準 × 全券種）も試して、
  「その券種にはどの基準が最適か」を実測する。

判定は第4世代と同じ作法。
  探索期間 2021-2023 で選び、一度も見ていない 2024-2025 で確認する。
  見るのは「探索ROIと検証ROIの順位相関」＝その買い方が再現するかどうか。

実行: python search_v5.py → search_v5_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN = [2021, 2022, 2023]
TEST = [2024, 2025]
MIN_HIT_TR = 15
MIN_PT_TE = 100


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in TRAIN + TEST], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "馬場状態_num",
                              "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    # 3つの順位を作る。これまでは r3 しか使っていなかった。
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False)     # 勝率順位
    D["r2"] = g["c_top2"].rank(ascending=False)    # 連対順位
    D["r3"] = g["c_top3"].rank(ascending=False)    # 複勝順位
    fav = D[D.pr == 1][["race_id", "r3"]].rename(columns={"r3": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    kinds = ("単勝", "複勝", "馬連", "馬単", "ワイド", "3連複")
    PAY = {k: {} for k in kinds}
    for r in jv[jv.券種.isin(kinds)].itertuples():
        PAY[r.券種][(r.race_id, r.組み合わせ)] = r.払戻金
    return D, PAY


CONDS = {
    "全体": lambda d: pd.Series(True, index=d.index),
    "堅いR": lambda d: d.fav_mr == 1,
    "荒れR": lambda d: d.fav_mr >= 4,
    "長距離1900+": lambda d: d["距離"] >= 1900,
    "芝": lambda d: d.is_turf == 1,
    "ダート": lambda d: d.is_turf == 0,
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
}
# 選ぶ基準。列名と表示名。
BASIS = {"勝率": "r1", "連対": "r2", "複勝": "r3"}
AXN = [1, 2]            # 軸の順位（〜N位）
MATEN = [3, 5]          # 相手の頭数（〜N位）
ODDS = [(1, 99), (1, 20), (5, 30)]


def blank():
    return {y: [0.0, 0.0, 0] for y in TRAIN + TEST}


def add(acc, y, cost, pay):
    acc[y][0] += cost
    acc[y][1] += pay
    acc[y][2] += 1 if pay > 0 else 0


def summarize(cond, kind, basis, acc):
    tc = sum(acc[y][0] for y in TRAIN); tr_ = sum(acc[y][1] for y in TRAIN)
    th = sum(acc[y][2] for y in TRAIN)
    ec = sum(acc[y][0] for y in TEST); er = sum(acc[y][1] for y in TEST)
    eh = sum(acc[y][2] for y in TEST)
    if th < MIN_HIT_TR or ec < MIN_PT_TE * 100:
        return None
    a, b = tr_ / tc * 100, er / ec * 100
    return {"買い方": kind, "選ぶ基準": basis, "条件": cond,
            "探索点数": int(tc / 100), "探索的中": th, "探索ROI": round(a, 1),
            "検証点数": int(ec / 100), "検証的中": eh, "検証ROI": round(b, 1),
            "差": round(b - a, 1)}


def s2(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def main():
    D, PAY = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    rows = []
    n = 0
    for cl, cf in CONDS.items():
        rid = list(D[cf(D)].race_id.unique())
        for bl, bcol in BASIS.items():
            for olo, ohi in ODDS:
                # 単系
                for av in AXN:
                    acc = {k: blank() for k in ("単勝", "複勝")}
                    for r in rid:
                        g = races[r]; y = int(r[:4])
                        sel = g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        for b in sel.bn:
                            add(acc["単勝"], y, 100, PAY["単勝"].get((r, b), 0.0))
                            add(acc["複勝"], y, 100, PAY["複勝"].get((r, b), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(f"{cl}/{bl}{av}位/{olo}-{ohi}倍", k, bl, a))
                # 軸-相手系（相手も同じ基準で選ぶ）
                for av, mn in itertools.product(AXN, MATEN):
                    acc = {k: blank() for k in ("馬単表", "馬単裏", "馬連", "ワイド")}
                    for r in rid:
                        g = races[r]; y = int(r[:4])
                        ax = g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        if ax.empty:
                            continue
                        a0 = ax.sort_values(bcol).bn.iloc[0]
                        mates = [b for b in g[g[bcol] <= mn].bn if b != a0]
                        for b in mates:
                            add(acc["馬単表"], y, 100, PAY["馬単"].get((r, f"{a0}-{b}"), 0.0))
                            add(acc["馬単裏"], y, 100, PAY["馬単"].get((r, f"{b}-{a0}"), 0.0))
                            add(acc["馬連"], y, 100, PAY["馬連"].get((r, s2(a0, b)), 0.0))
                            add(acc["ワイド"], y, 100, PAY["ワイド"].get((r, s2(a0, b)), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(
                            f"{cl}/{bl}{av}位軸x{bl}上位{mn}/{olo}-{ohi}倍", k, bl, a))
                # ボックス系
                for bn_ in (3, 4):
                    acc = {k: blank() for k in ("馬連BOX", "ワイドBOX", "3連複BOX")}
                    for r in rid:
                        g = races[r]; y = int(r[:4])
                        sel = g[(g[bcol] <= bn_) & (g.odds >= olo) & (g.odds < ohi)]
                        bs = sorted(sel.bn.tolist())
                        if len(bs) < 2:
                            continue
                        for a0, b in itertools.combinations(bs, 2):
                            add(acc["馬連BOX"], y, 100, PAY["馬連"].get((r, s2(a0, b)), 0.0))
                            add(acc["ワイドBOX"], y, 100, PAY["ワイド"].get((r, s2(a0, b)), 0.0))
                        for c3 in itertools.combinations(bs, 3):
                            add(acc["3連複BOX"], y, 100,
                                PAY["3連複"].get((r, "-".join(c3)), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(f"{cl}/{bl}BOX{bn_}/{olo}-{ohi}倍", k, bl, a))
        log(f"  {cl} 完了（累計 {n:,}構成）")

    R = pd.DataFrame([r for r in rows if r is not None])
    R.to_csv("search_v5_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索 {n:,}構成 / 評価できた {len(R):,}件\n")

    from scipy import stats
    log("=== 券種 × 選ぶ基準 ごとの再現性（探索ROIと検証ROIの順位相関）===")
    log("  正で有意なら『探索期間で良い構成は翌年も良い』＝選ぶ意味がある")
    log(f"{'買い方':<10}{'基準':<6}{'件数':>6}{'相関':>9}{'p値':>9}{'検証平均':>9}{'検証最高':>9}")
    for (k, b), g in R.groupby(["買い方", "選ぶ基準"]):
        if len(g) < 20:
            continue
        c = stats.spearmanr(g.探索ROI, g.検証ROI)
        mark = "  ← 再現" if (c.pvalue < 0.05 and c.correlation > 0) else ""
        log(f"{k:<10}{b:<6}{len(g):>6}{c.correlation:>+9.3f}{c.pvalue:>9.3f}"
            f"{g.検証ROI.mean():>8.1f}%{g.検証ROI.max():>8.1f}%{mark}")


if __name__ == "__main__":
    main()
