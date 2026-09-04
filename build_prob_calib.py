# -*- coding: utf-8 -*-
"""1着/2着以内/3着以内の確率を較正する（2026-09-04）

なぜ要るか
  p1 が高い側で過大、低い側で過小になっていた。

      予測60%超  予測平均81.8%  実際64.0%  比0.78
      予測0-5%   予測平均 2.0%  実際 4.0%  比2.04

  2頭を組むと歪みが掛け算になり、ワイドの確率が実測の1.5倍になっていた
  （0.78の2乗 ≈ 0.61 ≈ 観測された0.64）。
  Harvilleで組み直しても同じ0.64だったので、**組み方ではなく材料の問題**。

  現行の買い方は gap=p1/q の**順位**しか使わないので影響を受けない。
  期待値で券種を選ぼうとして初めて表面化した。

やり方
  等調回帰（isotonic）で較正する。単調性を保つので順位は変わらない。
  ⚠ その年より前のデータだけで較正器を作る。同じ年で作れば当然合う。
  較正後、レース内で合計が1（p1）／2（p2）／3（p3）になるよう正規化する。

出力
  prob_calib.pkl   年ごとの較正器
  検算結果を標準出力に出す

実行
  python build_prob_calib.py
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
OUT = os.path.join(BASE_DIR, "prob_calib.pkl")
TARGETS = {"p1": ("着", 1, 1.0), "p2": ("着", 2, 2.0), "p3": ("着", 3, 3.0)}


def log(m):
    print(m, flush=True)


def renorm(df, col, total):
    """レース内で合計が total になるよう正規化する。"""
    s = df.groupby("race_id")[col].transform("sum")
    return (df[col] / s.replace(0, np.nan) * total).clip(0, 1)


def band_report(pred, actual, label):
    log(f"  【{label}】")
    log("    %-14s %8s %10s %10s %8s" % ("予測の帯", "件数", "予測平均", "実際", "比"))
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .35), (.35, .6), (.6, 1.01)):
        m = (pred >= lo) & (pred < hi)
        if m.sum() < 100:
            continue
        p, a = pred[m].mean(), actual[m].mean()
        log("    %4.0f-%4.0f%%    %8d %9.1f%% %9.1f%% %8.2f"
            % (lo * 100, hi * 100, m.sum(), p * 100, a * 100, a / max(p, 1e-9)))
    worst = 0.0
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .35), (.35, .6), (.6, 1.01)):
        m = (pred >= lo) & (pred < hi)
        if m.sum() >= 100:
            worst = max(worst, abs(actual[m].mean() - pred[m].mean()) * 100)
    log(f"    最大のズレ {worst:.1f}ポイント")
    log("")
    return worst


def main():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()].copy()
    d["年"] = d["race_id"].str[:4].astype(int)
    years = sorted(d["年"].unique())
    log(f"  {len(d):,}頭 / {d.race_id.nunique():,}レース  年 {years}\n")

    models = {}
    out_parts = []
    for y in years:
        tr = d[d["年"] < y]
        te = d[d["年"] == y].copy()
        if len(tr) < 20000:
            log(f"  {y}年: 較正に使える過去データが足りません（{len(tr)}頭）→ そのまま")
            for c in TARGETS:
                te[c + "_cal"] = te[c]
            out_parts.append(te)
            continue
        models[y] = {}
        for c, (col, k, tot) in TARGETS.items():
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(tr[c].to_numpy(), (tr[col] <= k).astype(float).to_numpy())
            te[c + "_cal"] = ir.predict(te[c].to_numpy())
            te[c + "_cal"] = renorm(te, c + "_cal", tot)
            models[y][c] = ir
        out_parts.append(te)
        log(f"  {y}年: 過去{len(tr):,}頭で較正器を作成")

    D = pd.concat(out_parts, ignore_index=True)
    ev = D[D["年"] > years[0]]          # 較正できた年だけで評価
    log(f"\n  評価に使う年 {sorted(ev['年'].unique())}（{len(ev):,}頭）\n")

    log("  === 較正の前後 ===")
    res = {}
    for c, (col, k, _) in TARGETS.items():
        act = (ev[col] <= k).astype(float).to_numpy()
        b = band_report(ev[c].to_numpy(), act, f"{c} 較正前")
        a = band_report(ev[c + "_cal"].to_numpy(), act, f"{c} 較正後")
        res[c] = (b, a)

    log("  === 判定 ===")
    ok = True
    for c, (b, a) in res.items():
        mark = "○" if a <= 3.0 else "✗"
        if a > 3.0:
            ok = False
        log(f"    {c}  最大のズレ {b:5.1f}pt → {a:5.1f}pt   {mark}（3pt以内が目標）")

    with open(OUT, "wb") as f:
        pickle.dump({"models": models, "years": years,
                     "作成": datetime.now().strftime("%Y-%m-%d %H:%M")}, f)
    log(f"\n  prob_calib.pkl に保存（{len(models)}年ぶん）")
    D.to_csv(os.path.join(BASE_DIR, "resid_kinds_pred_cal.csv"),
             index=False, encoding="utf-8-sig")
    log("  resid_kinds_pred_cal.csv に較正後の確率を保存")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
