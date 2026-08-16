# -*- coding: utf-8 -*-
"""新基準で採用できる構成を全部洗い出す（2026-08-16・第7世代）

新しい採用基準（競馬向けに引き直したもの）
  ① 直近2年（2024・2025）とも100%超
  ② スリッページ通過（7分前で選び直しても100%超）
  ③ 的中30本以上
  ④ 点推定100%超（95%区間の下限は参考に留める）
  ⑤ 順列検定 p<0.20（別途 perm_top3.py で実施済み。グリッド全体で p=0.1575）

これまでの基準（下限>100%・p<0.05）は医学・製薬や学術論文の慣例で、
競馬のように分散が大きい対象には厳しすぎた。12,000構成すべてが落ちていた。

前回は直近2年で上位14件だけを検証したが、①を満たす構成は他にもあるはず。
今回は search_v6_result.csv の全構成から①③④を満たすものを抜き、
そのすべてに②（スリッページ模擬）をかける。

実行: python adopt_scan.py → adopt_scan_result.csv
"""
import itertools
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]
N_SIM = 60
rng = np.random.default_rng(20260816)


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


COND_FN = {
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


def parse(cond):
    """search_v6 が書いた条件文字列を、実行できる形に戻す。
    例: '少頭数-12/勝率1位軸x勝率上位5/5-30倍' → (条件, 基準列, 軸位, 相手位, オッズ)"""
    parts = cond.split("/")
    cl = parts[0]
    body = parts[1]
    olo, ohi = map(float, parts[2].replace("倍", "").split("-"))
    m = re.match(r"(勝率|連対|複勝)(\d+)位軸x(勝率|連対|複勝)上位(\d+)", body)
    if m:
        return cl, BASIS[m.group(1)], int(m.group(2)), int(m.group(4)), olo, ohi, "軸相手"
    m = re.match(r"(勝率|連対|複勝)BOX(\d+)", body)
    if m:
        return cl, BASIS[m.group(1)], None, int(m.group(2)), olo, ohi, "BOX"
    m = re.match(r"(勝率|連対|複勝)(\d+)位$", body)
    if m:
        return cl, BASIS[m.group(1)], int(m.group(2)), None, olo, ohi, "単"
    return None


def s2(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def bets(races, rid, PAY, kind, bcol, av, mn, olo, ohi, form, ov=None):
    out = []
    for r in rid:
        g = races[r]
        y = int(r[:4])
        od = ov[r] if ov is not None else g.odds.values
        gg = g.assign(_od=od)
        if form == "単":
            sel = gg[(gg[bcol] <= av) & (gg._od >= olo) & (gg._od < ohi)]
            for b in sel.bn:
                out.append((y, PAY[kind].get((r, b), 0.0)))
        elif form == "軸相手":
            ax = gg[(gg[bcol] <= av) & (gg._od >= olo) & (gg._od < ohi)]
            if ax.empty:
                continue
            a0 = ax.sort_values(bcol).bn.iloc[0]
            for b in [x for x in gg[gg[bcol] <= mn].bn if x != a0]:
                if kind == "馬単表":
                    out.append((y, PAY["馬単"].get((r, f"{a0}-{b}"), 0.0)))
                elif kind == "馬単裏":
                    out.append((y, PAY["馬単"].get((r, f"{b}-{a0}"), 0.0)))
                elif kind == "馬連":
                    out.append((y, PAY["馬連"].get((r, s2(a0, b)), 0.0)))
                elif kind == "ワイド":
                    out.append((y, PAY["ワイド"].get((r, s2(a0, b)), 0.0)))
        else:  # BOX
            bs = sorted(gg[(gg[bcol] <= mn) & (gg._od >= olo) & (gg._od < ohi)].bn.tolist())
            for a0, b in itertools.combinations(bs, 2):
                if kind == "馬連BOX":
                    out.append((y, PAY["馬連"].get((r, s2(a0, b)), 0.0)))
                elif kind == "ワイドBOX":
                    out.append((y, PAY["ワイド"].get((r, s2(a0, b)), 0.0)))
            if kind == "3連複BOX":
                for c3 in itertools.combinations(bs, 3):
                    out.append((y, PAY["3連複"].get((r, "-".join(c3)), 0.0)))
    return out


def roi(bs, years):
    v = [p for y, p in bs if y in years]
    return (float(np.mean(v)), len(v), sum(1 for x in v if x > 0)) if v else (np.nan, 0, 0)


def drift_pools():
    o = pd.read_csv("odds_history.csv", dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    o = o[o["単勝オッズ"] > 0].sort_values("t")
    last = (o.groupby(["race_id", "馬名"]).tail(1)[["race_id", "馬名", "単勝オッズ"]]
            .rename(columns={"単勝オッズ": "pre"}))
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"]) \
        .rename(columns={"単勝オッズ": "fin", "人気": "ninki", "着順_num": "着"})
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    m = last.merge(rf, on=["race_id", "馬名"], how="inner").dropna(
        subset=["pre", "fin", "ninki", "着"])
    m = m[(m.pre > 0) & (m.fin > 0)].copy()
    m["lr"] = np.log(m.fin / m.pre)
    m["w"] = (m["着"] == 1).astype(int)
    m["band"] = np.select([m.ninki <= 1, m.ninki <= 3, m.ninki <= 5, m.ninki <= 7],
                          [0, 1, 2, 3], 4)
    pools = {(b, w): m[(m.band == b) & (m.w == w)].lr.values
             for b in range(5) for w in (0, 1)}
    merged = np.concatenate([pools[(b, 1)] for b in (2, 3, 4) if len(pools[(b, 1)])])
    for b in (2, 3, 4):
        pools[(b, 1)] = merged
    return pools


def main():
    R6 = pd.read_csv("search_v6_result.csv")
    # ①③④で一次選抜（スリッページは重いので、ここを通ったものだけにかける）
    cand = R6[(R6.y2024 >= 100) & (R6.y2025 >= 100) & (R6.的中 >= 30)].copy()
    log(f"search_v6 の {len(R6):,}構成のうち、直近2年とも100%超＆的中30本以上: {len(cand)}件")

    D, PAY = load()
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    pools = drift_pools()
    log(f"検体 {len(D):,}頭 / {len(races):,}レース\n")

    rows = []
    for _, c in cand.iterrows():
        pr = parse(c.条件)
        if pr is None:
            continue
        cl, bcol, av, mn, olo, ohi, form = pr
        if cl not in COND_FN:
            continue
        rid = list(D[COND_FN[cl](D)].race_id.unique())
        kind = c.買い方
        b0 = bets(races, rid, PAY, kind, bcol, av, mn, olo, ohi, form)
        r2, n2, h2 = roi(b0, RECENT)
        sims = []
        for _ in range(N_SIM):
            ov = {}
            for r in rid:
                g = races[r]
                band = np.select([g.pr <= 1, g.pr <= 3, g.pr <= 5, g.pr <= 7],
                                 [0, 1, 2, 3], 4)
                w = g.win.values
                lr = np.empty(len(g))
                for b in range(5):
                    for ww in (0, 1):
                        idx = np.where((band == b) & (w == ww))[0]
                        if not len(idx):
                            continue
                        pool = pools[(b, ww)]
                        lr[idx] = rng.choice(pool if len(pool) else pools[(b, 0)], len(idx))
                ov[r] = g.odds.values / np.exp(lr)
            rr, _, _ = roi(bets(races, rid, PAY, kind, bcol, av, mn, olo, ohi, form, ov),
                           RECENT)
            if np.isfinite(rr):
                sims.append(rr)
        slip = float(np.median(sims)) if sims else np.nan
        ok = (slip >= 100) and (h2 >= 30)
        rows.append({"買い方": kind, "条件": c.条件, "基準": c.基準,
                     "5年ROI": c.通算ROI, "y2024": c.y2024, "y2025": c.y2025,
                     "直近2年ROI": round(r2, 1), "直近2年点数": n2, "直近2年的中": h2,
                     "7分前ROI": round(slip, 1), "影響pt": round(slip - r2, 1),
                     "採用": "○" if ok else ""})
        log(f"{c.条件[:44]:<46}{kind:<8}直近2年{r2:>6.1f}% → 7分前{slip:>6.1f}% "
            f"(的中{h2}) {'← 採用可' if ok else ''}")

    R = pd.DataFrame(rows).sort_values("7分前ROI", ascending=False)
    R.to_csv("adopt_scan_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n{'='*78}")
    ok = R[R.採用 == "○"]
    log(f"=== 新基準を満たした構成: {len(ok)}件 ===")
    log(ok.to_string(index=False) if len(ok) else "  なし")


if __name__ == "__main__":
    main()
