# -*- coding: utf-8 -*-
"""第7世代の上位34件を検証する（2026-08-16）

対象
  search_v7（15,876構成）で
    ・直近2年（2024・2025）とも100%超
    ・的中30本以上
    ・5年通算も100%超
  を満たした34件。5年通算100%超は12,000構成の探索で一度も出ていなかった。

やること
  ① スリッページ模擬（7分前オッズで選び直す）… これまで全候補がここで落ちた
  ② ブートストラップ95%区間（直近2年）
  ③ 年ごとのROI（既に search_v7 が出しているが再掲）

順列検定は別途 perm_v7.py で実施する（探索数が15,876に増えたため、
第6世代の p=0.1575 は使えない。同じ網で検定し直す必要がある）。

実行: python verify34.py → verify34_result.csv
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


COND_FN = {
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


def parse(cond):
    parts = cond.split("/")
    cl, body = parts[0], parts[1]
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
            for b in gg[(gg[bcol] <= av) & (gg._od >= olo) & (gg._od < ohi)].bn:
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
        else:
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
    R7 = pd.read_csv("search_v7_result.csv")
    cand = R7[(R7.y2024 >= 100) & (R7.y2025 >= 100) & (R7.的中 >= 30)
              & (R7.通算ROI >= 100)].copy()
    log(f"対象 {len(cand)}件（直近2年とも100%超・的中30本以上・5年通算100%超）\n")

    D, PAY = load()
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    pools = drift_pools()

    rows = []
    for _, c in cand.iterrows():
        pr = parse(c.条件)
        if pr is None or pr[0] not in COND_FN:
            continue
        cl, bcol, av, mn, olo, ohi, form = pr
        rid = list(D[COND_FN[cl](D)].race_id.unique())
        b0 = bets(races, rid, PAY, c.買い方, bcol, av, mn, olo, ohi, form)
        r2, n2, h2 = roi(b0, RECENT)
        v2 = np.array([p for y, p in b0 if y in RECENT])
        bs = np.array([rng.choice(v2, len(v2)).mean() for _ in range(2000)]) \
            if len(v2) else np.array([np.nan])
        lo, hi = np.percentile(bs, [2.5, 97.5])
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
            rr, _, _ = roi(bets(races, rid, PAY, c.買い方, bcol, av, mn,
                                olo, ohi, form, ov), RECENT)
            if np.isfinite(rr):
                sims.append(rr)
        slip = float(np.median(sims)) if sims else np.nan
        ok = slip >= 100
        rows.append({"買い方": c.買い方, "条件": c.条件, "基準": c.基準,
                     "5年ROI": c.通算ROI, "100超年数": c["100超年数"],
                     "直近2年ROI": round(r2, 1), "直近2年的中": h2,
                     "CI下": round(lo, 1), "CI上": round(hi, 1),
                     "7分前ROI": round(slip, 1), "影響pt": round(slip - r2, 1),
                     "通過": "○" if ok else ""})
        log(f"{c.条件[:40]:<42}{c.買い方:<7}5年{c.通算ROI:>6.1f}% "
            f"直近{r2:>6.1f}% → 7分前{slip:>6.1f}% "
            f"[{lo:.0f},{hi:.0f}] {'← 通過' if ok else ''}")

    R = pd.DataFrame(rows).sort_values("7分前ROI", ascending=False)
    R.to_csv("verify34_result.csv", index=False, encoding="utf-8-sig")
    ok = R[R.通過 == "○"]
    log(f"\n{'='*80}")
    log(f"=== スリッページを通過: {len(ok)}/{len(R)}件 ===")
    log(ok.to_string(index=False) if len(ok) else "  なし")


if __name__ == "__main__":
    main()
