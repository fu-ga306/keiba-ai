# -*- coding: utf-8 -*-
"""事前登録した6つの券種を全部試す（2026-09-04）

⚠ 事前登録_券種_202609.md に書いたものだけを試す。あとから足さない。
  結果は**全部**出す。良いものだけ載せない。
  2025年はここでは触らない（評価専用）。

実行
  python exp_bettype.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_YEARS = [2021, 2022, 2023, 2024]
MONTHS = 48.0
MATE_GAP = 1.3
MAX_MATE = 3
RNG = np.random.default_rng(20260904)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def payouts():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    keep = ("単勝", "複勝", "ワイド", "馬連")
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金
            for r in jv[jv.券種.isin(keep)].itertuples()}


def bets_for(g, PAY, kind):
    """1レース分の払戻の並びを返す。軸は gap 最大かつ1.5以上の1頭。"""
    g = g.sort_values("gap", ascending=False)
    ax = g.iloc[0]
    if ax["gap"] < 1.5:
        return []
    rid = ax["race_id"]
    bn = str(int(ax["馬番"])).zfill(2)
    turf = bool(ax["is_turf"])
    out = []
    if kind in ("単勝", "単勝+ワイド(ダのみ)", "単勝+ワイド(全部)",
                "単勝+複勝", "単勝+馬連"):
        out.append(PAY.get((rid, "単勝", bn), 0.0))
    if kind == "複勝":
        out.append(PAY.get((rid, "複勝", bn), 0.0))
    if kind == "単勝+複勝":
        out.append(PAY.get((rid, "複勝", bn), 0.0))
    if kind in ("単勝+ワイド(ダのみ)", "単勝+ワイド(全部)", "単勝+馬連"):
        if kind == "単勝+ワイド(ダのみ)" and turf:
            return out
        mates = g.iloc[1:]
        mates = mates[mates["gap"] >= MATE_GAP].head(MAX_MATE)
        for _, mt in mates.iterrows():
            mb = str(int(mt["馬番"])).zfill(2)
            combo = "-".join(sorted([bn, mb]))
            tk = "馬連" if kind == "単勝+馬連" else "ワイド"
            out.append(PAY.get((rid, tk, combo), 0.0))
    return out


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
    pm = n / MONTHS
    return dict(n=n, roi=ret.mean(), hit=(ret > 0).mean() * 100,
                lo=lo, hi=hi, need=nd, months=(nd / pm if nd else None))


def main():
    sys.path.insert(0, BASE_DIR)
    import resid_io

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
    d = d[d.年.isin(DEV_YEARS)]
    log(f"  開発データ {len(d):,}頭 / {d.race_id.nunique():,}レース（{DEV_YEARS}）")

    PAY = payouts()
    KINDS = ["単勝", "単勝+ワイド(ダのみ)", "単勝+ワイド(全部)",
             "単勝+複勝", "単勝+馬連", "複勝"]
    acc = {k: [] for k in KINDS}
    for rid, g in d.groupby("race_id", sort=False):
        for k in KINDS:
            acc[k].extend(bets_for(g, PAY, k))

    log("")
    log("  === 全結果（2021-2024。2025は見ていない） ===")
    log("  %-22s %7s %7s %8s %16s %10s" %
        ("券種", "点数", "的中率", "回収率", "95%区間", "証明まで"))
    log("  " + "-" * 78)
    rows = []
    for k in KINDS:
        st = stat(np.array(acc[k]))
        if st is None:
            log(f"  {k:<22} 点数が少なすぎます")
            continue
        mark = "○" if st["lo"] > 100 else ""
        mo = f"{st['months']:.0f}か月" if st["months"] else "届かない"
        log("  %-22s %7d %6.1f%% %7.1f%% [%5.1f, %6.1f] %10s %s"
            % (k, st["n"], st["hit"], st["roi"], st["lo"], st["hi"], mo, mark))
        rows.append({"券種": k, **st})

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(BASE_DIR, "exp_bettype_result.csv"),
             index=False, encoding="utf-8-sig")
    log("")
    ok = R[R["lo"] > 100]
    log(f"  95%区間が100%を上回った券種: {len(ok)} / {len(R)}")
    if len(ok):
        log("    " + " / ".join(ok["券種"]))
        cand = ok.dropna(subset=["months"]).sort_values("months")
        if len(cand):
            log("")
            log("  規則2: 証明までの月数が最短のもの")
            for r in cand.itertuples():
                log(f"    {r.券種:<22} {r.months:.0f}か月")
            log(f"  → 候補は {cand.iloc[0]['券種']}")
            cur = "単勝+ワイド(ダのみ)"
            if cur in set(ok["券種"]):
                log(f"  規則3: 現行（{cur}）も候補に残っているので**現行を優先**")
    else:
        log("    無し。規則3により、何も変えない。")


if __name__ == "__main__":
    main()
