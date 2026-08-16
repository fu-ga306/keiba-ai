# -*- coding: utf-8 -*-
"""探索の網を広げる（2026-08-16・第7世代）

なぜ広げるか
  新基準（直近2年とも100%超・的中30本以上）で search_v6 の1,797構成を洗ったが、
  該当は14件で、前回スリッページ検証にかけたものと同じだった。
  新基準で採用できる構成を増やすには、探索の網自体を広げるしかない。

第6世代から増やしたもの
  レース条件  8 → 14（クラス・頭数帯・芝ダ×距離を追加）
  オッズ帯    3 → 6（狭い帯も入れる。ただしスリッページに弱いのは承知のうえ）
  軸の位置    1-2位 → 1-3位
  相手の数    3,5 → 2,3,4,5,6
  券種        従来どおり（単勝・複勝・馬連・ワイド・馬単表裏・各BOX）

判定は第6世代と同じく年ごとのROI。ここでは候補を出すだけで、
スリッページ検証は adopt_scan.py 側で行う。

⚠ 網を広げるほど偶然の当たりも増える。最後は必ず順列検定にかけること。
  探索数はこのスクリプトの出力に記録される。

実行: python search_v7.py → search_v7_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_HIT = 30
MIN_PT_Y = 15


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False)
    D["r2"] = g["c_top2"].rank(ascending=False)
    D["r3"] = g["c_top3"].rank(ascending=False)
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
    "中距離1600-1800": lambda d: (d["距離"] >= 1600) & (d["距離"] <= 1800),
    "短距離-1400": lambda d: d["距離"] <= 1400,
    "芝": lambda d: d.is_turf == 1,
    "ダート": lambda d: d.is_turf == 0,
    "芝長距離": lambda d: (d.is_turf == 1) & (d["距離"] >= 1900),
    "ダ短距離": lambda d: (d.is_turf == 0) & (d["距離"] <= 1400),
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
    "中頭数13-15": lambda d: (d["出走頭数"] >= 13) & (d["出走頭数"] <= 15),
    "上級クラス4+": lambda d: d["クラス_num"] >= 4,
}
BASIS = {"勝率": "r1", "連対": "r2", "複勝": "r3"}
AXN = [1, 2, 3]
MATEN = [2, 3, 4, 5, 6]
ODDS = [(1, 99), (1, 20), (5, 30), (3, 15), (10, 40), (1, 10)]


def blank():
    return {y: [0.0, 0.0, 0] for y in YEARS}


def add(acc, y, cost, pay):
    acc[y][0] += cost
    acc[y][1] += pay
    acc[y][2] += 1 if pay > 0 else 0


def summarize(cond, kind, basis, acc):
    hits = sum(acc[y][2] for y in YEARS)
    if hits < MIN_HIT:
        return None
    rois, ok = {}, 0
    for y in YEARS:
        c = acc[y][0]
        if c < MIN_PT_Y * 100:
            return None
        v = acc[y][1] / c * 100
        rois[y] = round(v, 1)
        ok += 1 if v >= 100 else 0
    tc = sum(acc[y][0] for y in YEARS)
    tr = sum(acc[y][1] for y in YEARS)
    return {"買い方": kind, "基準": basis, "条件": cond,
            "点数": int(tc / 100), "的中": hits, "通算ROI": round(tr / tc * 100, 1),
            **{f"y{y}": rois[y] for y in YEARS},
            "100超年数": ok, "最悪年": min(rois.values())}


def s2(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def main():
    D, PAY = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    rows, n = [], 0
    for cl, cf in CONDS.items():
        rid = list(D[cf(D)].race_id.unique())
        for bl, bcol in BASIS.items():
            for olo, ohi in ODDS:
                for av in AXN:
                    acc = {k: blank() for k in ("単勝", "複勝")}
                    for r in rid:
                        g = races[r]
                        y = int(r[:4])
                        for b in g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)].bn:
                            add(acc["単勝"], y, 100, PAY["単勝"].get((r, b), 0.0))
                            add(acc["複勝"], y, 100, PAY["複勝"].get((r, b), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(f"{cl}/{bl}{av}位/{olo}-{ohi}倍", k, bl, a))
                for av, mn in itertools.product(AXN, MATEN):
                    if mn <= av:
                        continue
                    acc = {k: blank() for k in ("馬単表", "馬単裏", "馬連", "ワイド")}
                    for r in rid:
                        g = races[r]
                        y = int(r[:4])
                        ax = g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        if ax.empty:
                            continue
                        a0 = ax.sort_values(bcol).bn.iloc[0]
                        for b in [x for x in g[g[bcol] <= mn].bn if x != a0]:
                            add(acc["馬単表"], y, 100, PAY["馬単"].get((r, f"{a0}-{b}"), 0.0))
                            add(acc["馬単裏"], y, 100, PAY["馬単"].get((r, f"{b}-{a0}"), 0.0))
                            add(acc["馬連"], y, 100, PAY["馬連"].get((r, s2(a0, b)), 0.0))
                            add(acc["ワイド"], y, 100, PAY["ワイド"].get((r, s2(a0, b)), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(
                            f"{cl}/{bl}{av}位軸x{bl}上位{mn}/{olo}-{ohi}倍", k, bl, a))
                for bn_ in (3, 4, 5):
                    acc = {k: blank() for k in ("馬連BOX", "ワイドBOX", "3連複BOX")}
                    for r in rid:
                        g = races[r]
                        y = int(r[:4])
                        bs = sorted(g[(g[bcol] <= bn_) & (g.odds >= olo)
                                      & (g.odds < ohi)].bn.tolist())
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
    R.to_csv("search_v7_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索 {n:,}構成 / 年単位で判定できた {len(R):,}件")
    ok = R[(R.y2024 >= 100) & (R.y2025 >= 100) & (R.的中 >= 30)]
    log(f"  直近2年とも100%超 ＆ 的中30本以上: {len(ok)}件（第6世代は14件）\n")
    log("=== 候補（直近2年の平均が高い順・上位25）===")
    if len(ok):
        o = ok.assign(直近2年=((ok.y2024 + ok.y2025) / 2).round(1))
        log(o.sort_values("直近2年", ascending=False).head(25)
            [["買い方", "基準", "条件", "点数", "的中", "y2024", "y2025",
              "直近2年", "通算ROI", "100超年数"]].to_string(index=False))


if __name__ == "__main__":
    main()
