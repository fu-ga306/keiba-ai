# -*- coding: utf-8 -*-
"""1着/2着以内/3着以内の確率が正しいかを検算する（2026-09-04）

券種を期待値で選ぶ前に、その材料が正しいかを確かめる。
材料が狂っていれば、その上に何を積んでも意味がない。

確かめること
  ① レース内の合計が理屈どおりか
       p1 の合計 ≈ 1（1着は1頭）
       p2 の合計 ≈ 2（2着以内は2頭）
       p3 の合計 ≈ 3（3着以内は3頭）
  ② 較正されているか
       「p1が0.2の馬」は本当に20%勝つか。帯ごとに実測と比べる
  ③ 市場側（q, q2, q3）も同じか
  ④ 2頭の同時確率を独立の積で近似してよいか
       ワイド = 2頭とも3着以内。独立を仮定すると p3(A)×p3(B)。
       実際は排他性があるので**過大になる**はず。実測で確かめる。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from itertools import combinations

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()]
    log(f"  {len(d):,}頭 / {d.race_id.nunique():,}レース\n")

    log("  === ① レース内の合計 ===")
    log("  %-8s %10s %10s %10s" % ("", "理屈", "実際の中央値", "実際の平均"))
    log("  " + "-" * 44)
    for c, want in (("p1", 1), ("p2", 2), ("p3", 3),
                    ("q", 1), ("q2", 2), ("q3", 3)):
        s = d.groupby("race_id")[c].sum()
        log("  %-8s %10d %10.3f %10.3f" % (c, want, s.median(), s.mean()))

    log("")
    log("  === ② 較正（予測した確率どおりに来ているか） ===")
    for col, cond, lab in (("p1", d["着"] == 1, "1着"),
                           ("p2", d["着"] <= 2, "2着以内"),
                           ("p3", d["着"] <= 3, "3着以内")):
        log(f"  【{col} → {lab}】")
        v = d[col]
        for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
            m = (v >= lo) & (v < hi)
            if m.sum() < 200:
                continue
            log("    予測 %4.0f-%4.0f%%  %6d頭  予測平均%5.1f%%  実際%5.1f%%  差%+5.1f"
                % (lo * 100, hi * 100, m.sum(), v[m].mean() * 100,
                   cond[m].mean() * 100, cond[m].mean() * 100 - v[m].mean() * 100))
        log("")

    log("  === ④ 2頭の同時確率：独立の積でよいか ===")
    log("  ワイド（2頭とも3着以内）を、上位2頭の組で確かめる")
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        g = g.sort_values("p1", ascending=False)
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        rows.append({"独立の積": float(a.p3 * b.p3),
                     "実際": int(a["着"] <= 3 and b["着"] <= 3)})
    R = pd.DataFrame(rows)
    log(f"    {len(R):,}レース")
    log(f"    独立の積の平均  {R['独立の積'].mean()*100:.1f}%")
    log(f"    実際の的中率    {R['実際'].mean()*100:.1f}%")
    r = R["実際"].mean() / R["独立の積"].mean()
    log(f"    実際 ÷ 独立の積 = {r:.3f}")
    if r < 0.9:
        log("    → 独立の積は**過大**。そのまま期待値に使うと買いすぎる")
    elif r > 1.1:
        log("    → 独立の積は**過小**")
    else:
        log("    → 近似として使える範囲")

    log("")
    log("  帯ごとに見る")
    R["帯"] = pd.cut(R["独立の積"], [0, .05, .1, .2, .35, 1.01])
    for k, g in R.groupby("帯", observed=True):
        if len(g) < 200:
            continue
        log("    積 %-14s %6d件  積の平均%5.1f%%  実際%5.1f%%  比 %.2f"
            % (str(k), len(g), g["独立の積"].mean() * 100, g["実際"].mean() * 100,
               g["実際"].mean() / max(g["独立の積"].mean(), 1e-9)))


if __name__ == "__main__":
    main()
