# -*- coding: utf-8 -*-
"""年単位で判定する（2026-08-16・第6世代）

利用者の判断基準
  「年単位でプラスなら問題ない」。
  つまり信頼区間や再現性ではなく、各年の回収率が100%を超えているかで見る。

第5世代と同じグリッド（券種 × 選ぶ基準 × レース条件 × オッズ帯）を、
年ごとのROIで出し直す。判定は「5年のうち何年が100%超か」。

⚠ この基準には多重検定の問題がある。
  仮に実力ゼロでも、ある年に100%を超える確率が2割あるなら、
  5年連続する確率は 0.2^5 = 0.032%。1,872構成なら偶然でも 0.6件ほど出る。
  「5年連続プラス」が1〜2件見つかっても、それだけでは実力の証拠にならない。
  そこで実際の「年ごとに100%を超える確率」から期待件数を計算して併記する。

実行: python search_v6.py → search_v6_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_HIT = 25        # 5年合計の最低的中数
MIN_PT_Y = 20       # 各年の最低点数（少なすぎる年は判定に使えない）


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "馬場状態_num",
                              "出走頭数"]).drop_duplicates("race_id")
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
    "芝": lambda d: d.is_turf == 1,
    "ダート": lambda d: d.is_turf == 0,
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
}
BASIS = {"勝率": "r1", "連対": "r2", "複勝": "r3"}
AXN = [1, 2]
MATEN = [3, 5]
ODDS = [(1, 99), (1, 20), (5, 30)]


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
        c, r = acc[y][0], acc[y][1]
        if c < MIN_PT_Y * 100:
            return None            # 年の点数が少なすぎる構成は年単位で判定できない
        v = r / c * 100
        rois[y] = round(v, 1)
        ok += 1 if v >= 100 else 0
    tc = sum(acc[y][0] for y in YEARS); tr = sum(acc[y][1] for y in YEARS)
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
                        g = races[r]; y = int(r[:4])
                        sel = g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        for b in sel.bn:
                            add(acc["単勝"], y, 100, PAY["単勝"].get((r, b), 0.0))
                            add(acc["複勝"], y, 100, PAY["複勝"].get((r, b), 0.0))
                    for k, a in acc.items():
                        n += 1
                        rows.append(summarize(f"{cl}/{bl}{av}位/{olo}-{ohi}倍", k, bl, a))
                for av, mn in itertools.product(AXN, MATEN):
                    acc = {k: blank() for k in ("馬単表", "馬単裏", "馬連", "ワイド")}
                    for r in rid:
                        g = races[r]; y = int(r[:4])
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
                for bn_ in (3, 4):
                    acc = {k: blank() for k in ("馬連BOX", "ワイドBOX", "3連複BOX")}
                    for r in rid:
                        g = races[r]; y = int(r[:4])
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
    R.to_csv("search_v6_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索 {n:,}構成 / 年単位で判定できた {len(R):,}件\n")

    log("=== 100%を超えた年数の分布 ===")
    for k in range(6):
        c = int((R["100超年数"] == k).sum())
        log(f"  {k}/5年: {c:>5}件 ({c/len(R)*100:5.1f}%)")

    # 偶然でも何件出るかを、実際の「年が100%を超える率」から計算する
    p = R["100超年数"].sum() / (len(R) * 5)
    log(f"\n  1年が100%を超える確率（実測平均）: {p*100:.1f}%")
    for k in (4, 5):
        from math import comb
        exp = len(R) * comb(5, k) * p ** k * (1 - p) ** (5 - k)
        act = int((R["100超年数"] == k).sum())
        log(f"  {k}/5年: 実際 {act}件 / 偶然でも期待される件数 {exp:.1f}件")

    log("\n=== 5年すべて100%超の構成 ===")
    best = R[R["100超年数"] == 5].sort_values("通算ROI", ascending=False)
    log(best.to_string(index=False) if len(best) else "  なし")
    log("\n=== 4年以上100%超（上位15）===")
    log(R[R["100超年数"] >= 4].sort_values("通算ROI", ascending=False)
        .head(15).to_string(index=False))


if __name__ == "__main__":
    main()
