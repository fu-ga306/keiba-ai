# -*- coding: utf-8 -*-
"""買い方を網羅する探索（2026-08-15・第4世代）

これまでの反省
  ・軸を1頭決めて相手に流す形しか試していなかった。
    ボックス・裏（相手→軸）・馬単の表裏両取りを試していない。
  ・「探索期間で最良の構成」を採用していたが、それは翌年に崩れると実証された
    （上位20構成の平均: 探索111.1% → 検証69.7%、-41.4pt）。

今回の作法（前回の holdout 実験で分かったこと）
  探索期間 2021-2023 で候補を作り、**一度も見ていない 2024-2025 で確認する**。
  採用の判断は探索期間のROIではなく **検証期間のROI**。
  単勝の高ROIは検証で-30〜-70pt崩れ、複勝は-2.6〜-11.7ptで済んだ。
  分散の違い（単勝の1点の標準偏差579% vs 複勝78%）が原因。

買い方の種類
  軸-相手系  馬単・表（軸→相手） / 馬単・裏（相手→軸） / 馬単・両
             馬連 / ワイド
  ボックス系 馬連BOX / ワイドBOX / 馬単BOX / 3連複BOX
  単系       単勝 / 複勝（比較の基準として）

実行: python search_v4.py → search_v4_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN = [2021, 2022, 2023]
TEST = [2024, 2025]
MIN_HIT_TR = 15      # 探索期間での最低的中数
MIN_PT_TE = 100      # 検証期間での最低点数（少ないと検証にならない）


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
    D["mr"] = D.groupby("race_id")["c_top3"].rank(ascending=False)
    fav = D[D.pr == 1][["race_id", "mr"]].rename(columns={"mr": "fav_mr"})
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
AX = {"MF1位": 1, "MF2位以内": 2, "MF3位以内": 3}
MATE = {"MF上位3": ("mr", 3), "MF上位5": ("mr", 5),
        "人気上位3": ("pr", 3), "人気上位5": ("pr", 5)}
BOX = {"MFBOX3": ("mr", 3), "MFBOX4": ("mr", 4),
       "人気BOX3": ("pr", 3), "人気BOX4": ("pr", 4)}
ODDS = [(1, 99), (1, 20), (5, 30)]


def blank():
    return {y: [0.0, 0.0, 0] for y in TRAIN + TEST}


def add(acc, y, cost, pay):
    acc[y][0] += cost
    acc[y][1] += pay
    acc[y][2] += 1 if pay > 0 else 0


def summarize(cond, kind, acc):
    tr_c = sum(acc[y][0] for y in TRAIN)
    tr_r = sum(acc[y][1] for y in TRAIN)
    tr_h = sum(acc[y][2] for y in TRAIN)
    te_c = sum(acc[y][0] for y in TEST)
    te_r = sum(acc[y][1] for y in TEST)
    te_h = sum(acc[y][2] for y in TEST)
    if tr_h < MIN_HIT_TR or te_c < MIN_PT_TE * 100:
        return None
    tr_roi = tr_r / tr_c * 100
    te_roi = te_r / te_c * 100
    return {"買い方": kind, "条件": cond,
            "探索点数": int(tr_c / 100), "探索的中": tr_h, "探索ROI": round(tr_roi, 1),
            "検証点数": int(te_c / 100), "検証的中": te_h, "検証ROI": round(te_roi, 1),
            "差": round(te_roi - tr_roi, 1)}


def s2(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def main():
    D, PAY = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    rows = []
    n_try = 0

    for cl, cf in CONDS.items():
        rid = list(D[cf(D)].race_id.unique())
        for olo, ohi in ODDS:
            # ── 単系（比較の基準）──
            for al, av in AX.items():
                acc = {k: blank() for k in ("単勝", "複勝")}
                for r in rid:
                    g = races[r]
                    y = int(r[:4])
                    sel = g[(g.mr <= av) & (g.odds >= olo) & (g.odds < ohi)]
                    for b in sel.bn:
                        add(acc["単勝"], y, 100, PAY["単勝"].get((r, b), 0.0))
                        add(acc["複勝"], y, 100, PAY["複勝"].get((r, b), 0.0))
                for k, a in acc.items():
                    n_try += 1
                    rows.append(summarize(f"{cl}/{al}/{olo}-{ohi}倍", k, a))

            # ── 軸-相手系 ──
            for al, av in AX.items():
                for ml, (mcol, mn) in MATE.items():
                    acc = {k: blank() for k in
                           ("馬単表", "馬単裏", "馬単両", "馬連", "ワイド")}
                    for r in rid:
                        g = races[r]
                        y = int(r[:4])
                        ax = g[(g.mr <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        if ax.empty:
                            continue
                        a = ax.sort_values("mr").bn.iloc[0]
                        mates = [b for b in g[g[mcol] <= mn].bn if b != a]
                        for b in mates:
                            add(acc["馬単表"], y, 100, PAY["馬単"].get((r, f"{a}-{b}"), 0.0))
                            add(acc["馬単裏"], y, 100, PAY["馬単"].get((r, f"{b}-{a}"), 0.0))
                            add(acc["馬連"], y, 100, PAY["馬連"].get((r, s2(a, b)), 0.0))
                            add(acc["ワイド"], y, 100, PAY["ワイド"].get((r, s2(a, b)), 0.0))
                            both = (PAY["馬単"].get((r, f"{a}-{b}"), 0.0)
                                    + PAY["馬単"].get((r, f"{b}-{a}"), 0.0))
                            add(acc["馬単両"], y, 200, both)
                    for k, a in acc.items():
                        n_try += 1
                        rows.append(summarize(f"{cl}/{al}軸x{ml}/{olo}-{ohi}倍", k, a))

            # ── ボックス系 ──
            for bl, (bcol, bn) in BOX.items():
                acc = {k: blank() for k in
                       ("馬連BOX", "ワイドBOX", "馬単BOX", "3連複BOX")}
                for r in rid:
                    g = races[r]
                    y = int(r[:4])
                    sel = g[(g[bcol] <= bn) & (g.odds >= olo) & (g.odds < ohi)]
                    bs = sorted(sel.bn.tolist())
                    if len(bs) < 2:
                        continue
                    for a, b in itertools.combinations(bs, 2):
                        add(acc["馬連BOX"], y, 100, PAY["馬連"].get((r, s2(a, b)), 0.0))
                        add(acc["ワイドBOX"], y, 100, PAY["ワイド"].get((r, s2(a, b)), 0.0))
                        add(acc["馬単BOX"], y, 100, PAY["馬単"].get((r, f"{a}-{b}"), 0.0))
                        add(acc["馬単BOX"], y, 100, PAY["馬単"].get((r, f"{b}-{a}"), 0.0))
                    for c3 in itertools.combinations(bs, 3):
                        add(acc["3連複BOX"], y, 100,
                            PAY["3連複"].get((r, "-".join(c3)), 0.0))
                for k, a in acc.items():
                    n_try += 1
                    rows.append(summarize(f"{cl}/{bl}/{olo}-{ohi}倍", k, a))
        log(f"  {cl} 完了（累計 {n_try:,}構成）")

    R = pd.DataFrame([r for r in rows if r is not None])
    R.to_csv("search_v4_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索 {n_try:,}構成 / 評価できた {len(R):,}件")
    ok = R[R.検証ROI >= 100]
    log(f"  検証期間で100%超: {len(ok)}件\n")
    log("=== 検証ROIが高い順 上位20 ===")
    log(R.sort_values("検証ROI", ascending=False).head(20).to_string(index=False))
    log("\n=== 買い方ごとの成績（検証期間の平均）===")
    log(R.groupby("買い方").agg(件数=("検証ROI", "size"),
                              探索平均=("探索ROI", "mean"),
                              検証平均=("検証ROI", "mean"),
                              検証最高=("検証ROI", "max")).round(1).to_string())


if __name__ == "__main__":
    main()
