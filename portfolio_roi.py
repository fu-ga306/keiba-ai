# -*- coding: utf-8 -*-
"""採用した7構成を「実際に買う形」で合算した回収率を測る（2026-08-16）

なぜ必要か
  個別の構成は 106.8〜148.5%（7分前ベース）だが、それをそのまま足しても
  実際の回収率にはならない。理由は2つ。
    ① 6構成が同じ荒れRを対象にするので、同じ組み合わせが重複する。
       本番では1点にまとめて買うので、重複分は1回しか払わない。
    ② 構成ごとに点数が違うので、単純平均ではなく点数で重み付けされる。

  本番の _build_bet_rows と同じ「(券種,組み合わせ)で重複除去」を再現して、
  実際に買う買い目の回収率を出す。

  あわせて、重複を除去しない場合（各構成を別々に買う）も出して比較する。

実行: python portfolio_roi.py → portfolio_roi_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]
ARE_FAV_MR_MIN = 4
N_SIM = 40
rng = np.random.default_rng(20260816)

# keiba_predict.ARE_PLANS と同じ
ALL = {
    "A": ("荒れR勝率1位x2 馬単裏", "馬単裏", "r1", 1, 2, "are"),
    "B": ("クラス4+勝率1位x2 馬単裏", "馬単裏", "r1", 1, 2, "cls4"),
    "C": ("荒れR勝率1位x2 馬連", "馬連", "r1", 1, 2, "are"),
    "D": ("荒れR連対1位x4 馬単裏", "馬単裏", "r2", 1, 4, "are"),
    "E": ("荒れR勝率1位x3 馬単裏", "馬単裏", "r1", 1, 3, "are"),
    "F": ("荒れR連対1位x5 馬単裏", "馬単裏", "r2", 1, 5, "are"),
    "G": ("荒れR勝率1位x4 馬単表", "馬単表", "r1", 1, 4, "are"),
}
import os
PLANS = [ALL[k] for k in os.environ.get("PLANSET","ABCDEFG")]


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "クラス_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False, method="first")
    D["r2"] = g["c_top2"].rank(ascending=False, method="first")
    D["r3"] = g["c_top3"].rank(ascending=False, method="first")
    fav = D[D.pr == 1][["race_id", "r3"]].rename(columns={"r3": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    UMA, UREN = {}, {}
    for r in jv[jv.券種 == "馬単"].itertuples():
        UMA[(r.race_id, r.組み合わせ)] = r.払戻金
    for r in jv[jv.券種 == "馬連"].itertuples():
        UREN[(r.race_id, r.組み合わせ)] = r.払戻金
    return D, UMA, UREN


def build(g, rid, UMA, UREN, dedup=True):
    """1レースの買い目を作る。本番の _build_bet_rows と同じ形。"""
    fav = g.fav_mr.iloc[0] if pd.notna(g.fav_mr.iloc[0]) else np.nan
    cls = g["クラス_num"].iloc[0] if pd.notna(g["クラス_num"].iloc[0]) else np.nan
    is_are = pd.notna(fav) and fav >= ARE_FAV_MR_MIN
    is_c4 = pd.notna(cls) and cls >= 4
    seen = {}
    rows = []
    for nm, kind, bas, axr, mtr, cond in PLANS:
        if cond == "are" and not is_are:
            continue
        if cond == "cls4" and not is_c4:
            continue
        r = g[bas].values
        ax = g.bn.values[r == axr]
        if not len(ax):
            continue
        a = ax[0]
        mates = g.bn.values[(r <= mtr) & (g.bn.values != a)]
        for b in mates:
            if kind == "馬単裏":
                k, combo, pay = "馬単", f"{b}-{a}", UMA.get((rid, f"{b}-{a}"), 0.0)
            elif kind == "馬単表":
                k, combo, pay = "馬単", f"{a}-{b}", UMA.get((rid, f"{a}-{b}"), 0.0)
            else:
                c2 = f"{min(a,b)}-{max(a,b)}"
                k, combo, pay = "馬連", c2, UREN.get((rid, c2), 0.0)
            if dedup:
                seen[(k, combo)] = pay
            else:
                rows.append(pay)
    return list(seen.values()) if dedup else rows


def main():
    D, UMA, UREN = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")
    races = {r: g for r, g in D.groupby("race_id", sort=False)}

    for dedup, lab in ((True, "重複除去あり（本番と同じ）"), (False, "重複除去なし（各構成を別々に買う）")):
        acc = {y: [0.0, 0.0, 0] for y in YEARS}
        nrace = {y: 0 for y in YEARS}
        for rid, g in races.items():
            y = int(rid[:4])
            pays = build(g, rid, UMA, UREN, dedup)
            if not pays:
                continue
            nrace[y] += 1
            for p in pays:
                acc[y][0] += 100
                acc[y][1] += p
                acc[y][2] += 1 if p > 0 else 0
        tc = sum(acc[y][0] for y in YEARS)
        tr = sum(acc[y][1] for y in YEARS)
        th = sum(acc[y][2] for y in YEARS)
        rc = sum(acc[y][0] for y in RECENT)
        rr = sum(acc[y][1] for y in RECENT)
        rh = sum(acc[y][2] for y in RECENT)
        log(f"=== {lab} ===")
        log(f"  5年   {int(tc/100):>6,}点 的中{th:>4}  回収率 {tr/tc*100:6.1f}%"
            f"  買うレース {sum(nrace.values()):>5,}（1レース{tc/100/max(sum(nrace.values()),1):.1f}点）")
        log(f"  直近2年{int(rc/100):>6,}点 的中{rh:>4}  回収率 {rr/rc*100:6.1f}%")
        log("  年別: " + "  ".join(
            f"{y}:{acc[y][1]/acc[y][0]*100:.1f}%" if acc[y][0] else f"{y}:--"
            for y in YEARS))
        if dedup:
            v = []
            for rid, g in races.items():
                if int(rid[:4]) in RECENT:
                    v += build(g, rid, UMA, UREN, True)
            v = np.array(v)
            b = np.array([rng.choice(v, len(v)).mean() for _ in range(3000)])
            log(f"  直近2年の95%区間 [{np.percentile(b,2.5):.1f}, {np.percentile(b,97.5):.1f}]")
            log(f"  1レースあたりの投資 {tc/max(sum(nrace.values()),1):.0f}円"
                f"（1点500円換算で {tc/100/max(sum(nrace.values()),1)*500:.0f}円）")
        log("")


if __name__ == "__main__":
    main()
