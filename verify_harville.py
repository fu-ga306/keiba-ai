# -*- coding: utf-8 -*-
"""Harvilleで組んだ確率が実測と合うかを検算する（2026-09-04）

独立の積は 実際÷推定 = 0.650 で1.5倍に見積もっていた。
Harville なら 0.95〜1.05 に入るはず。**入らなければ使わない。**

確かめること
  ① 3着以内の確率：モデルのp3 と Harville由来 のどちらが実測に近いか
  ② ワイド（2頭とも3着以内）：Harville は実測と合うか
  ③ 馬連（2頭で1-2着）：同じく
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
N_RACES = 4000          # 全部やると重いので抽出


def log(m):
    print(m, flush=True)


def band_report(pred, actual, label, edges=(0, .05, .1, .2, .35, .6, 1.01)):
    log(f"  【{label}】")
    log("    %-16s %8s %10s %10s %8s" % ("予測の帯", "件数", "予測平均", "実際", "比"))
    log("    " + "-" * 58)
    tot_p = tot_a = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() < 100:
            continue
        p, a = pred[m].mean(), actual[m].mean()
        tot_p += p * m.sum()
        tot_a += a * m.sum()
        log("    %4.0f-%4.0f%%      %8d %9.1f%% %9.1f%% %8.2f"
            % (lo * 100, hi * 100, m.sum(), p * 100, a * 100, a / max(p, 1e-9)))
    log("    全体             %8d %9.1f%% %9.1f%% %8.2f"
        % (len(pred), pred.mean() * 100, actual.mean() * 100,
           actual.mean() / max(pred.mean(), 1e-9)))
    log("")


def main():
    sys.path.insert(0, BASE_DIR)
    import harville as H

    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()]
    rids = d.race_id.unique()
    rng = np.random.default_rng(904)
    pick = set(rng.choice(rids, size=min(N_RACES, len(rids)), replace=False))
    d = d[d.race_id.isin(pick)]
    log(f"  抽出 {d.race_id.nunique():,}レース / {len(d):,}頭\n")

    m_p3, h_p3, act3 = [], [], []
    w_ind, w_har, w_act = [], [], []
    u_har, u_act = [], []
    t0 = datetime.now()
    for rid, g in d.groupby("race_id", sort=False):
        g = g.sort_values("p1", ascending=False).reset_index(drop=True)
        pi = g["p1"].to_numpy(dtype=float)
        if len(pi) < 4:
            continue
        h3 = H.top3_prob(pi)
        m_p3.extend(g["p3"].to_numpy())
        h_p3.extend(h3)
        act3.extend((g["着"] <= 3).astype(float).to_numpy())
        # 上位2頭の組で2頭ぶんの確率を見る
        a, b = 0, 1
        w_ind.append(float(g.p3.iat[a] * g.p3.iat[b]))
        w_har.append(H.pair_top3_prob(pi, a, b))
        w_act.append(int(g["着"].iat[a] <= 3 and g["着"].iat[b] <= 3))
        u_har.append(H.pair_top2_prob(pi, a, b))
        u_act.append(int(g["着"].iat[a] <= 2 and g["着"].iat[b] <= 2))
    log(f"  計算 {(datetime.now()-t0).total_seconds():.0f}秒\n")

    log("  === ① 3着以内の確率 ===")
    band_report(np.array(m_p3), np.array(act3), "モデルの p3（いま使っているもの）")
    band_report(np.array(h_p3), np.array(act3), "Harville（p1から組み立て）")

    log("  === ② ワイド（2頭とも3着以内） ===")
    band_report(np.array(w_ind), np.array(w_act, dtype=float), "独立の積（いまの近似）",
                edges=(0, .1, .2, .35, .5, 1.01))
    band_report(np.array(w_har), np.array(w_act, dtype=float), "Harville",
                edges=(0, .1, .2, .35, .5, 1.01))

    log("  === ③ 馬連（2頭で1-2着） ===")
    band_report(np.array(u_har), np.array(u_act, dtype=float), "Harville",
                edges=(0, .02, .05, .1, .2, 1.01))

    log("  === 判定 ===")
    for lab, p, a in (("ワイド 独立の積", np.array(w_ind), np.array(w_act, dtype=float)),
                      ("ワイド Harville", np.array(w_har), np.array(w_act, dtype=float)),
                      ("馬連 Harville", np.array(u_har), np.array(u_act, dtype=float))):
        r = a.mean() / max(p.mean(), 1e-9)
        ok = "○ 使える" if 0.95 <= r <= 1.05 else "✗ 使えない"
        log(f"    {lab:<20} 実際÷推定 = {r:.3f}  {ok}")


if __name__ == "__main__":
    main()
