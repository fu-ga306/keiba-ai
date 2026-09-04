# -*- coding: utf-8 -*-
"""2頭の同時確率を較正する（2026-09-04）

ここまでの経過
  独立の積（較正前）    実際÷推定 = 0.660
  Harville（較正後）    0.772
  独立の積（較正後）    0.920   ← 一番良い

  Harvilleが劣るのは、p1_cal から組み立て直す過程で
  p1側の誤差（最大3.4pt）が2乗で効くため。
  較正済みの p3 は既に実測に合わせてあるので、そのまま使うほうが近い。

  残る8%は「2頭が同じ3枠を奪い合う」ぶん。独立ではないので当然出る。
  **これも同じやり方で較正する。**

やり方
  x = p3_cal(A) × p3_cal(B) を入力、実際に2頭とも3着以内だったかを目標にして
  等調回帰をかける。**その年より前のデータだけで作る。**

  馬連（2頭で1-2着）も同様に p2_cal の積から作る。

出力
  pair_calib.pkl
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
from sklearn.isotonic import IsotonicRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "pair_calib.pkl")
AX_GAP = 1.5
MATE_GAP = 1.3
MAX_MATE = 3


def log(m):
    print(m, flush=True)


def build_pairs(d):
    """軸とその相手の組を作る。実際に買う組だけを対象にする。"""
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        g = g.sort_values("gap", ascending=False)
        ax = g.iloc[0]
        if ax["gap"] < AX_GAP:
            continue
        mates = g.iloc[1:]
        mates = mates[mates["gap"] >= MATE_GAP].head(MAX_MATE)
        for _, mt in mates.iterrows():
            rows.append({
                "race_id": rid, "年": int(ax["年"]),
                "w_x": float(ax["p3_cal"] * mt["p3_cal"]),
                "u_x": float(ax["p2_cal"] * mt["p2_cal"]),
                "w_y": int(ax["着"] <= 3 and mt["着"] <= 3),
                "u_y": int(ax["着"] <= 2 and mt["着"] <= 2),
            })
    return pd.DataFrame(rows)


def report(pred, act, label):
    r = act.mean() / max(pred.mean(), 1e-9)
    log(f"  {label:<26} 推定{pred.mean()*100:5.1f}%  実際{act.mean()*100:5.1f}%  "
        f"比 {r:.3f}  {'○ 使える' if 0.95 <= r <= 1.05 else '✗'}")
    return r


def main():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred_cal.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()].copy()
    d["gap"] = d["p1"] / d["q"]
    P = build_pairs(d)
    years = sorted(P["年"].unique())
    log(f"  組 {len(P):,}件 / 年 {years}\n")

    models = {}
    parts = []
    for y in years:
        tr, te = P[P["年"] < y], P[P["年"] == y].copy()
        if len(tr) < 3000:
            log(f"  {y}年: 過去データ不足（{len(tr)}件）→ そのまま")
            te["w_cal"], te["u_cal"] = te["w_x"], te["u_x"]
            parts.append(te)
            continue
        models[y] = {}
        for k, x, yy in (("w", "w_x", "w_y"), ("u", "u_x", "u_y")):
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(tr[x].to_numpy(), tr[yy].to_numpy(dtype=float))
            te[k + "_cal"] = ir.predict(te[x].to_numpy())
            models[y][k] = ir
        parts.append(te)
        log(f"  {y}年: 過去{len(tr):,}件で較正器を作成")

    D = pd.concat(parts, ignore_index=True)
    ev = D[D["年"] > years[0]]
    log(f"\n  評価 {len(ev):,}件（{sorted(ev['年'].unique())}）\n")

    log("  === ワイド（2頭とも3着以内） ===")
    report(ev["w_x"].to_numpy(), ev["w_y"].to_numpy(dtype=float), "較正前（独立の積）")
    rw = report(ev["w_cal"].to_numpy(), ev["w_y"].to_numpy(dtype=float), "較正後")
    log("")
    log("  === 馬連（2頭で1-2着） ===")
    report(ev["u_x"].to_numpy(), ev["u_y"].to_numpy(dtype=float), "較正前（独立の積）")
    ru = report(ev["u_cal"].to_numpy(), ev["u_y"].to_numpy(dtype=float), "較正後")

    log("")
    log("  === 帯ごと（ワイド・較正後） ===")
    for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, 1.01)):
        m = (ev["w_cal"] >= lo) & (ev["w_cal"] < hi)
        if m.sum() < 100:
            continue
        log("    %4.0f-%4.0f%%  %7d件  推定%5.1f%%  実際%5.1f%%  比%.2f"
            % (lo * 100, hi * 100, m.sum(), ev.loc[m, "w_cal"].mean() * 100,
               ev.loc[m, "w_y"].mean() * 100,
               ev.loc[m, "w_y"].mean() / max(ev.loc[m, "w_cal"].mean(), 1e-9)))

    with open(OUT, "wb") as f:
        pickle.dump({"models": models, "作成": datetime.now().strftime("%Y-%m-%d %H:%M")}, f)
    log(f"\n  pair_calib.pkl に保存")
    return 0 if (0.95 <= rw <= 1.05 and 0.95 <= ru <= 1.05) else 1


if __name__ == "__main__":
    sys.exit(main())
