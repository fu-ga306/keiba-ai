# -*- coding: utf-8 -*-
"""軸馬を複勝で買ったらどうなるかを測る（2026-08-31）

きっかけ
  同種の開発者が「複勝 的中率26% 回収率142.8%」と公表していた。
  的中率26%は人気薄を買っている証拠なので、我々の軸（人気薄に寄る）と
  相性がある可能性がある。以前の複勝テスト(86.6%)と条件が違うかもしれない。

⚠ これは探索なので、数字を見る前に条件を固定する
  ・買うのは resid_io.pick_bets が選んだ軸だけ（選び方は一切変えない）
  ・出す表は**全部出す**。良いものだけ見せない
  ・順列検定を通す（払戻をシャッフルして同じ数字が出ないこと）
  ・年別を必ず出す（通算だけで判断しない）
"""
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import resid_io

def log(m):
    print(m, flush=True)

d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
d["gap"] = d.p1 / d.q
d["馬番"] = pd.to_numeric(d["bn"], errors="coerce")
_p = []
for ch in pd.read_csv("race_features.csv", usecols=["race_id", "is_turf"],
                      dtype={"race_id": str}, chunksize=200000):
    _p.append(ch.drop_duplicates("race_id"))
rf = pd.concat(_p).drop_duplicates("race_id")
rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
d = d.merge(rf, on="race_id", how="left"); del _p

jv = pd.read_csv("jv_payouts.csv", dtype=str)
jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
       for r in jv[jv.券種.isin(("単勝", "複勝", "ワイド"))].itertuples()}

rows = []
for rid, g in d.groupby("race_id", sort=False):
    bets = resid_io.pick_bets(g, model={"gap_min": resid_io.AX_GAP})
    ax = next((b for b in bets if b["券種"] == "単勝"), None)
    if ax is None:
        continue
    bn = str(ax["組み合わせ"]).zfill(2)
    row = g[pd.to_numeric(g["bn"], errors="coerce") == int(bn)]
    if row.empty:
        continue
    rows.append({
        "race_id": rid, "年": int(rid[:4]),
        "人気": float(row["人気"].iat[0]), "gap": float(row["gap"].iat[0]),
        "odds": float(row["odds"].iat[0]),
        "turf": bool(g["is_turf"].iat[0]),
        "単勝": PAY.get((rid, "単勝", bn), 0.0),
        "複勝": PAY.get((rid, "複勝", bn), 0.0),
    })
R = pd.DataFrame(rows)
log(f"  軸 {len(R):,}点（5年）\n")

rng = np.random.default_rng(831)

def show(x, lab):
    if len(x) < 50:
        log(f"    {lab:<22} {len(x):>5}点  （少なすぎる）"); return
    for kind in ("単勝", "複勝"):
        v = x[kind].values
        s = rng.choice(v, size=(4000, len(v))).mean(axis=1)
        lo, hi = np.percentile(s, [2.5, 97.5])
        yr = x.groupby("年")[kind].mean()
        log("    %-8s %-10s %5d点 的中%5.1f%% 回収%6.1f%% 95%%[%3.0f,%4.0f] 年別 %s"
            % (lab, kind, len(v), (v > 0).mean() * 100, v.mean(), lo, hi,
               " ".join("%.0f" % z for z in yr)))

log("  === 全体 ===")
show(R, "すべて")
log("")
log("  === 人気帯ごと（事前に決めた区切り） ===")
for lo, hi, lab in [(1,3,"1-3番人気"), (4,6,"4-6"), (7,9,"7-9"),
                    (10,12,"10-12"), (13,99,"13番人気以下")]:
    show(R[(R.人気 >= lo) & (R.人気 <= hi)], lab)
log("")
log("  === gap帯ごと ===")
for lo, hi, lab in [(1.5,2.0,"gap1.5-2.0"), (2.0,3.0,"gap2.0-3.0"), (3.0,99,"gap3.0以上")]:
    show(R[(R.gap >= lo) & (R.gap < hi)], lab)
log("")
log("  === 芝ダート ===")
for t, lab in ((True, "芝"), (False, "ダート")):
    show(R[R.turf == t], lab)

# 順列検定：払戻をシャッフルして、複勝の通算が同じくらい出るか
log("")
log("  === 順列検定（複勝・全体） ===")
obs = R["複勝"].mean()
null = []
for _ in range(2000):
    null.append(rng.permutation(R["複勝"].values)[:len(R)].mean())
log(f"    実測 {obs:.1f}%  シャッフル平均 {np.mean(null):.1f}%")
log("    ※ 選び方を変えていないので、この検定は「点数の切り出し」しか見ない。")
log("      本当に見るべきは年別のばらつき。")
