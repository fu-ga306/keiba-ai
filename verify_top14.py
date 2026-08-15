# -*- coding: utf-8 -*-
"""直近2年で好成績だった14構成を、確立した検証にかける（2026-08-16）

背景
  第6世代の探索（1,797構成）で、2024・2025の両方で100%を超えたものが14件あった。
  偶然でも5.6件は期待されるので、そのままでは採用できない。

  なお「初期年が低いのは各馬の過去データが少ないから」という仮説は否定された。
  平均過去出走数は 2021年6.84走 / 2025年7.63走 とほぼ同じで、欠損率も11-12%で横ばい。
  特徴量生成時に2019年より前の履歴も参照しているため、初期年も不利ではない。
  つまり2021-2023で負けていたのは、単にその期間は通用しなかったということ。

この検証で見るもの
  ① ブートストラップ95%区間（直近2年・5年通算の両方）
  ② 順列検定（1,797構成から選んだことを織り込む family-wise）
  ③ スリッページ模擬（7分前オッズで選び直す）

③が最重要。過去の全候補はここで落ちている（EV方式 109.8%→82.8%）。

実行: python verify_top14.py → verify_top14_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]
rng = np.random.default_rng(20260816)

# 第6世代で直近2年とも100%超だった14構成（買い方, 基準列, 条件の実装）
CANDS = [
    ("馬単表", "r1", "少頭数-12", 1, 5, 5, 30, "少頭数/勝率1位軸x勝率上位5/5-30倍"),
    ("単勝", "r1", "道悪", 2, None, 5, 30, "道悪/勝率2位/5-30倍"),
    ("馬連BOX", "r3", "道悪", None, 4, 5, 30, "道悪/複勝BOX4/5-30倍"),
    ("馬単表", "r1", "少頭数-12", 1, 3, 5, 30, "少頭数/勝率1位軸x勝率上位3/5-30倍"),
    ("単勝", "r1", "荒れR", 1, None, 5, 30, "荒れR/勝率1位/5-30倍"),
    ("馬単表", "r1", "道悪", 2, 5, 5, 30, "道悪/勝率2位軸x勝率上位5/5-30倍"),
    ("馬連BOX", "r2", "長距離1900+", None, 3, 5, 30, "長距離/連対BOX3/5-30倍"),
    ("馬単裏", "r2", "荒れR", 1, 5, 1, 20, "荒れR/連対1位軸x連対上位5/1-20倍"),
    ("馬単裏", "r1", "荒れR", 1, 3, 1, 20, "荒れR/勝率1位軸x勝率上位3/1-20倍"),
    ("単勝", "r2", "少頭数-12", 1, None, 5, 30, "少頭数/連対1位/5-30倍"),
    ("馬連BOX", "r3", "長距離1900+", None, 4, 5, 30, "長距離/複勝BOX4/5-30倍"),
    ("単勝", "r1", "長距離1900+", 1, None, 5, 30, "長距離/勝率1位/5-30倍"),
    ("馬単裏", "r1", "荒れR", 1, 3, 1, 99, "荒れR/勝率1位軸x勝率上位3/1-99倍"),
    ("単勝", "r1", "荒れR", 1, None, 1, 20, "荒れR/勝率1位/1-20倍"),
]
N_SEARCHED = 1797       # 探索した構成数（family-wise検定に使う）


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
    kinds = ("単勝", "馬連", "馬単")
    PAY = {k: {} for k in kinds}
    for r in jv[jv.券種.isin(kinds)].itertuples():
        PAY[r.券種][(r.race_id, r.組み合わせ)] = r.払戻金
    return D, PAY


COND_FN = {
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "荒れR": lambda d: d.fav_mr >= 4,
    "長距離1900+": lambda d: d["距離"] >= 1900,
}


def s2(a, b):
    return f"{min(a, b)}-{max(a, b)}"


def bets_for(races, rid, PAY, kind, bcol, av, mn, olo, ohi, odds_override=None):
    """1構成の全ベットを (年, 払戻) のリストで返す。1点100円。
    odds_override が渡された場合、そのオッズで軸/オッズ条件を判定する（7分前の模擬）。"""
    out = []
    for r in rid:
        g = races[r]
        y = int(r[:4])
        od = odds_override[r] if odds_override is not None else g.odds.values
        gg = g.assign(_od=od)
        if kind in ("単勝",):
            sel = gg[(gg[bcol] <= av) & (gg._od >= olo) & (gg._od < ohi)]
            for b in sel.bn:
                out.append((y, PAY["単勝"].get((r, b), 0.0)))
        elif kind in ("馬単表", "馬単裏"):
            ax = gg[(gg[bcol] <= av) & (gg._od >= olo) & (gg._od < ohi)]
            if ax.empty:
                continue
            a0 = ax.sort_values(bcol).bn.iloc[0]
            for b in [x for x in gg[gg[bcol] <= mn].bn if x != a0]:
                key = f"{a0}-{b}" if kind == "馬単表" else f"{b}-{a0}"
                out.append((y, PAY["馬単"].get((r, key), 0.0)))
        elif kind == "馬連BOX":
            bs = sorted(gg[(gg[bcol] <= mn) & (gg._od >= olo) & (gg._od < ohi)].bn.tolist())
            for a0, b in itertools.combinations(bs, 2):
                out.append((y, PAY["馬連"].get((r, s2(a0, b)), 0.0)))
    return out


def roi(bets, years=None):
    v = [p for y, p in bets if years is None or y in years]
    return (np.mean(v), len(v), sum(1 for x in v if x > 0)) if v else (np.nan, 0, 0)


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
    D, PAY = load()
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    pools = drift_pools()
    log(f"検体 {len(D):,}頭 / {len(races):,}レース\n")

    rows = []
    for kind, bcol, cl, av, mn, olo, ohi, name in CANDS:
        rid = list(D[COND_FN[cl](D)].race_id.unique())
        bets = bets_for(races, rid, PAY, kind, bcol, av, mn, olo, ohi)
        r5, n5, h5 = roi(bets, None)
        r2, n2, h2 = roi(bets, RECENT)
        # ブートストラップ（直近2年）
        v2 = np.array([p for y, p in bets if y in RECENT])
        b2 = np.array([rng.choice(v2, len(v2)).mean() for _ in range(3000)])
        lo2, hi2 = np.percentile(b2, [2.5, 97.5])
        # スリッページ模擬
        sims = []
        for _ in range(120):
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
            sb = bets_for(races, rid, PAY, kind, bcol, av, mn, olo, ohi, ov)
            rr, _, _ = roi(sb, RECENT)
            if np.isfinite(rr):
                sims.append(rr)
        sims = np.array(sims)
        rows.append({"買い方": kind, "構成": name,
                     "5年点数": n5, "5年的中": h5, "5年ROI": round(r5, 1),
                     "直近2年点数": n2, "直近2年的中": h2, "直近2年ROI": round(r2, 1),
                     "CI下": round(lo2, 1), "CI上": round(hi2, 1),
                     "7分前ROI": round(float(np.median(sims)), 1) if len(sims) else np.nan,
                     "影響pt": round(float(np.median(sims)) - r2, 1) if len(sims) else np.nan})
        x = rows[-1]
        log(f"{name[:42]:<44}5年{x['5年ROI']:>6.1f}% / 直近2年{x['直近2年ROI']:>6.1f}% "
            f"[{x['CI下']:.0f},{x['CI上']:.0f}] → 7分前{x['7分前ROI']:>6.1f}% ({x['影響pt']:+.1f}pt)")

    R = pd.DataFrame(rows)
    R.to_csv("verify_top14_result.csv", index=False, encoding="utf-8-sig")
    log("\n" + "=" * 78)
    log("=== 判定 ===")
    ok = R[(R["CI下"] >= 100) & (R["7分前ROI"] >= 100)]
    log(f"  直近2年のCI下限が100%超 かつ 7分前でも100%超: {len(ok)}件")
    if len(ok):
        log(ok.to_string(index=False))
    else:
        log("  なし")
        log(f"\n  CI下限が100%を超えたもの: {int((R['CI下'] >= 100).sum())}件")
        log(f"  7分前でも100%を超えたもの: {int((R['7分前ROI'] >= 100).sum())}件")


if __name__ == "__main__":
    main()
