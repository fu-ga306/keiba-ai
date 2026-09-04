# -*- coding: utf-8 -*-
"""全ての2頭の組で較正器を作り、軸の組に当てはまるか確かめる（2026-09-04）

前回の行き詰まり
  軸＋相手の組だけで較正しようとしたが、年1,500件しか作れず、
  低確率帯（0-10%）で比0.83と合わなかった。

  「実際に買う組と性質が違うから全組は使えない」と切り捨てたが、
  **それ自体を検証していなかった。**

今回
  全ての2頭の組で較正器を作る（年30万件・200倍）。
  そのうえで**軸の組だけで評価**する。当てはまるなら材料は足りる。

  ⚠ 較正器を作るのは「その年より前」のデータだけ。
  ⚠ 評価は軸の組だけ。全組で評価すると当然合う。

さらに、レース頭数で違いが出るかも見る。
3枠を何頭で奪い合うかで、独立からのズレ方が変わるはずなので。
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
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "pair_calib.pkl")
AX_GAP = 1.5
MATE_GAP = 1.3
MAX_MATE = 3
MAX_PAIRS_PER_RACE = 60      # 全組だと重いので上位から抽出
# 頭数の区切り。3枠を奪い合う頭数で、独立からのズレ方が変わる
FIELD_BINS = [(0, 12), (12, 14), (14, 16), (16, 99)]


def log(m):
    print(m, flush=True)


def make_pairs(d):
    """全組（上位から抽出）と、軸の組かどうかの印を返す。"""
    allp, axp = [], []
    for rid, g in d.groupby("race_id", sort=False):
        g = g.sort_values("gap", ascending=False).reset_index(drop=True)
        n = len(g)
        if n < 4:
            continue
        yy = int(g["年"].iat[0])
        p3 = g["p3_cal"].to_numpy(float)
        p2 = g["p2_cal"].to_numpy(float)
        ch = g["着"].to_numpy(float)
        # 軸の組
        ax_ok = g["gap"].iat[0] >= AX_GAP
        mates = set()
        if ax_ok:
            for j in range(1, n):
                if g["gap"].iat[j] >= MATE_GAP and len(mates) < MAX_MATE:
                    mates.add(j)
        idx = list(range(min(n, 12)))
        cnt = 0
        for i, j in combinations(idx, 2):
            if cnt >= MAX_PAIRS_PER_RACE:
                break
            cnt += 1
            rec = {"年": yy, "頭数": n,
                   "w_x": p3[i] * p3[j], "u_x": p2[i] * p2[j],
                   "w_y": int(ch[i] <= 3 and ch[j] <= 3),
                   "u_y": int(ch[i] <= 2 and ch[j] <= 2),
                   "軸の組": int(ax_ok and i == 0 and j in mates)}
            allp.append(rec)
            if rec["軸の組"]:
                axp.append(rec)
    return pd.DataFrame(allp), pd.DataFrame(axp)


def report(pred, act, label):
    r = act.mean() / max(pred.mean(), 1e-9)
    ok = 0.95 <= r <= 1.05
    log(f"  {label:<30} 推定{pred.mean()*100:5.1f}%  実際{act.mean()*100:5.1f}%  "
        f"比 {r:.3f}  {'○' if ok else '✗'}")
    return r


def main():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred_cal.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()].copy()
    d["gap"] = d["p1"] / d["q"]
    t0 = datetime.now()
    A, X = make_pairs(d)
    log(f"  全組 {len(A):,}件 / うち軸の組 {len(X):,}件  "
        f"（{(datetime.now()-t0).total_seconds():.0f}秒）")
    years = sorted(A["年"].unique())
    log(f"  年 {years}\n")

    models, parts = {}, []
    for y in years:
        tr = A[A["年"] < y]
        te = A[A["年"] == y].copy()
        if len(tr) < 20000:
            log(f"  {y}年: 過去{len(tr):,}件で不足 → そのまま")
            te["w_cal"], te["u_cal"] = te["w_x"], te["u_x"]
            parts.append(te)
            continue
        # ⚠ 頭数ごとに較正器を分ける（2026-09-04）
        #   3枠を何頭で奪い合うかで、独立からのズレ方が変わる。
        #   実測: 0-11頭 比1.17 / 11-14頭 比1.03 / 14-20頭 比0.78
        #   1本の較正器では、この違いを吸収できない。
        models[y] = {}
        te["w_cal"] = np.nan
        te["u_cal"] = np.nan
        for lo, hi in FIELD_BINS:
            mtr = (tr["頭数"] >= lo) & (tr["頭数"] < hi)
            mte = (te["頭数"] >= lo) & (te["頭数"] < hi)
            if mtr.sum() < 5000 or mte.sum() == 0:
                continue
            for k, x, yy in (("w", "w_x", "w_y"), ("u", "u_x", "u_y")):
                ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                ir.fit(tr.loc[mtr, x].to_numpy(), tr.loc[mtr, yy].to_numpy(dtype=float))
                te.loc[mte, k + "_cal"] = ir.predict(te.loc[mte, x].to_numpy())
                models[y][(k, lo, hi)] = ir
        # 該当する帯が無かった行は素の値で埋める
        te["w_cal"] = te["w_cal"].fillna(te["w_x"])
        te["u_cal"] = te["u_cal"].fillna(te["u_x"])
        parts.append(te)
        log(f"  {y}年: 過去{len(tr):,}件で較正器を作成")

    D = pd.concat(parts, ignore_index=True)
    ev = D[(D["年"] > years[0]) & (D["軸の組"] == 1)]
    log(f"\n  === 評価：軸の組だけ {len(ev):,}件 ===")
    log("  （較正器は全組で作り、評価は軸の組だけで行う）\n")

    log("  ワイド（2頭とも3着以内）")
    report(ev["w_x"].to_numpy(), ev["w_y"].to_numpy(float), "較正前（独立の積）")
    rw = report(ev["w_cal"].to_numpy(), ev["w_y"].to_numpy(float), "較正後（全組で作った較正器）")
    log("")
    log("  馬連（2頭で1-2着）")
    report(ev["u_x"].to_numpy(), ev["u_y"].to_numpy(float), "較正前（独立の積）")
    ru = report(ev["u_cal"].to_numpy(), ev["u_y"].to_numpy(float), "較正後（全組で作った較正器）")

    log("")
    log("  === 帯ごと（ワイド・軸の組） ===")
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .35), (.35, 1.01)):
        m = (ev["w_cal"] >= lo) & (ev["w_cal"] < hi)
        if m.sum() < 80:
            continue
        log("    %4.0f-%4.0f%%  %6d件  推定%5.1f%%  実際%5.1f%%  比%.2f"
            % (lo * 100, hi * 100, m.sum(), ev.loc[m, "w_cal"].mean() * 100,
               ev.loc[m, "w_y"].mean() * 100,
               ev.loc[m, "w_y"].mean() / max(ev.loc[m, "w_cal"].mean(), 1e-9)))

    log("")
    log("  === 頭数で違いが出るか（ワイド・軸の組・較正後） ===")
    for lo, hi in ((0, 11), (11, 14), (14, 20)):
        m = (ev["頭数"] >= lo) & (ev["頭数"] < hi)
        if m.sum() < 80:
            continue
        log("    %2d-%2d頭  %6d件  推定%5.1f%%  実際%5.1f%%  比%.2f"
            % (lo, hi, m.sum(), ev.loc[m, "w_cal"].mean() * 100,
               ev.loc[m, "w_y"].mean() * 100,
               ev.loc[m, "w_y"].mean() / max(ev.loc[m, "w_cal"].mean(), 1e-9)))

    with open(OUT, "wb") as f:
        pickle.dump({"models": models, "方式": "全組で学習・頭数別", "field_bins": FIELD_BINS,
                     "作成": datetime.now().strftime("%Y-%m-%d %H:%M")}, f)
    log(f"\n  pair_calib.pkl に保存")
    return 0 if (0.95 <= rw <= 1.05 and 0.95 <= ru <= 1.05) else 1


if __name__ == "__main__":
    sys.exit(main())
