# -*- coding: utf-8 -*-
"""レースごとに券種を選ぶ4案を試す（2026-09-04）

⚠ 事前登録_券種選択_202609.md に書いたものだけを試す。あとから足さない。
  しきい値は全券種で同じ値を使う。券種ごとに変えたらそれは探索。
  2025年はここでは触らない（評価専用）。
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
DEV_YEARS = [2021, 2022, 2023, 2024]
MONTHS = 48.0
AX_GAP = 1.5
MATE_GAP = 1.3
MAX_MATE = 3
EV_MIN = 1.0           # 期待値のしきい値。全券種で同じ値
RNG = np.random.default_rng(20260904)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred.csv"),
                    dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["馬番"] = pd.to_numeric(d["bn"], errors="coerce")
    d["年"] = d["race_id"].str[:4].astype(int)
    parts = []
    for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                          usecols=["race_id", "is_turf"], dtype={"race_id": str},
                          chunksize=200000):
        parts.append(ch.drop_duplicates("race_id"))
    rf = pd.concat(parts).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    return d[d.年.isin(DEV_YEARS)]


def payouts():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    keep = ("単勝", "複勝", "ワイド", "馬連")
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金
            for r in jv[jv.券種.isin(keep)].itertuples()}


def race_options(g):
    """そのレースで買える候補を (券種, 組み合わせ, 期待値) で返す。

    オッズは単勝オッズからHarvilleで推定する（実オッズを持っていないため）。
    既存 resid_kinds.py と同じ流儀。
    """
    g = g.sort_values("gap", ascending=False)
    ax = g.iloc[0]
    if ax["gap"] < AX_GAP:
        return None, []
    bn = str(int(ax["馬番"])).zfill(2)
    opts = []
    # 単勝: P(1着) × 単勝オッズ
    opts.append(("単勝", bn, float(ax.p1 * ax.odds)))
    # 複勝: P(3着以内) × 推定複勝オッズ（市場の3着以内確率の逆数×控除後）
    if ax.q3 > 0:
        fuku_odds = 0.8 / float(ax.q3)
        opts.append(("複勝", bn, float(ax.p3 * fuku_odds)))
    mates = g.iloc[1:]
    mates = mates[mates["gap"] >= MATE_GAP].head(MAX_MATE)
    for _, mt in mates.iterrows():
        mb = str(int(mt["馬番"])).zfill(2)
        combo = "-".join(sorted([bn, mb]))
        # ワイド: 2頭とも3着以内。Harvilleの近似
        pw = float(ax.p3 * mt.p3)
        qw = float(ax.q3 * mt.q3)
        if qw > 0:
            opts.append(("ワイド", combo, pw * (0.775 / qw)))
        # 馬連: 2頭で1-2着（順不同）。同じくHarville近似
        pu = float(ax.p2 * mt.p2)
        qu = float(ax.q2 * mt.q2)
        if qu > 0:
            opts.append(("馬連", combo, pu * (0.775 / qu)))
    return ax, opts


def stat(ret):
    n = len(ret)
    if n < 100:
        return None
    s = RNG.choice(ret, size=(3000, n)).mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])

    def need(hi_n=60000):
        if ret.mean() <= 100:
            return None
        def bl(k):
            return np.percentile(RNG.choice(ret, size=(1200, k)).mean(axis=1), 2.5)
        if bl(hi_n) <= 100:
            return None
        a, b = 100, hi_n
        while a < b:
            mid = (a + b) // 2
            if bl(mid) > 100:
                b = mid
            else:
                a = mid + 1
        return a

    nd = need()
    return dict(n=n, roi=ret.mean(), hit=(ret > 0).mean() * 100, lo=lo, hi=hi,
                need=nd, months=(nd / (n / MONTHS) if nd else None))


def main():
    d = load()
    PAY = payouts()
    log(f"  開発データ {len(d):,}頭 / {d.race_id.nunique():,}レース（{DEV_YEARS}）")

    # Dの境目は開発データの中央値で機械的に決める（数字を見て動かさない）
    ax_gaps = d.loc[d.groupby("race_id")["gap"].idxmax(), "gap"]
    gap_mid = float(ax_gaps[ax_gaps >= AX_GAP].median())
    log(f"  Dの境目（軸gapの中央値）= {gap_mid:.3f}  ※機械的に決定")

    acc = {k: [] for k in ("A 現行", "B 期待値最大の1点", "C 期待値がしきい値以上を全部",
                           "D gapで切替")}
    for rid, g in d.groupby("race_id", sort=False):
        ax, opts = race_options(g)
        if ax is None:
            continue
        bn = str(int(ax["馬番"])).zfill(2)
        turf = bool(ax["is_turf"])

        # A 現行
        r = [PAY.get((rid, "単勝", bn), 0.0)]
        if not turf:
            for k, cb, _ in opts:
                if k == "ワイド":
                    r.append(PAY.get((rid, "ワイド", cb), 0.0))
        acc["A 現行"].extend(r)

        # B 期待値が最大の1点
        best = max(opts, key=lambda x: x[2])
        acc["B 期待値最大の1点"].append(PAY.get((rid, best[0], best[1]), 0.0))

        # C しきい値以上を全部
        for k, cb, ev in opts:
            if ev >= EV_MIN:
                acc["C 期待値がしきい値以上を全部"].append(PAY.get((rid, k, cb), 0.0))

        # D gapで切替
        if float(ax["gap"]) >= gap_mid:
            acc["D gapで切替"].append(PAY.get((rid, "単勝", bn), 0.0))
        else:
            acc["D gapで切替"].append(PAY.get((rid, "複勝", bn), 0.0))

    log("")
    log("  === 全結果（2021-2024。2025は見ていない） ===")
    log("  %-30s %7s %7s %8s %17s %10s" %
        ("案", "点数", "的中率", "回収率", "95%区間", "証明まで"))
    log("  " + "-" * 86)
    rows = []
    for k, v in acc.items():
        st = stat(np.array(v))
        if st is None:
            log(f"  {k:<30} 点数が少なすぎます")
            continue
        mo = f"{st['months']:.0f}か月" if st["months"] else "届かない"
        log("  %-30s %7d %6.1f%% %7.1f%% [%5.1f, %6.1f] %10s %s"
            % (k, st["n"], st["hit"], st["roi"], st["lo"], st["hi"], mo,
               "○" if st["lo"] > 100 else ""))
        rows.append({"案": k, **st})

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(BASE_DIR, "exp_kindselect_result.csv"),
             index=False, encoding="utf-8-sig")
    ok = R[R["lo"] > 100]
    log("")
    log(f"  95%区間が100%を上回った案: {len(ok)} / {len(R)}")
    if len(ok):
        cand = ok.dropna(subset=["months"]).sort_values("months")
        log("  規則2: 証明までが最短の順")
        for r in cand.itertuples():
            log(f"    {r.案:<30} {r.months:.0f}か月")
        if "A 現行" in set(ok["案"]):
            log("  規則3: 現行も候補に残っているので**現行を優先**")
        else:
            log(f"  → 現行は候補外。候補は {cand.iloc[0]['案']}")
    else:
        log("    無し。規則3により、何も変えない。")


if __name__ == "__main__":
    main()
