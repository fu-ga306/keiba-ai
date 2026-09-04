# -*- coding: utf-8 -*-
"""BTだけが持つ「同じ日の先行レースの結果」がどれだけ効くかを測る（2026-09-04）

なぜ測るか
  「2021-2024のBTと同じ仕様で本番を動かす」ことを目標にしたい。
  だが、消せない差が1つある。

    BT    12Rの特徴量を作るとき、同じ日の1R〜11Rの結果を使える
    本番  race_data_clean.csv は週次更新。当日の結果は持てない

  これが小さければ「ほぼ同じ仕様」と言えるし、大きければ言えない。
  **実装に入る前に、どちらかを確かめる。**

やり方
  同じレースの特徴量を2通りで作る。
    ① 履歴に同じ日の先行レースを**含める**（BTの条件）
    ② 履歴から同じ日のレースを**全部除く**（本番の条件）
  何列が変わるか、どれだけ変わるかを見る。

  1日の後半のレース（先行レースが多い）で測る。差が最大になる場所。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK = ["着順", "着順_num", "タイム", "上り", "通過", "着差", "賞金"]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)

    ids = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))
    last = ids[-1]
    meet = last[:10]                       # 同じ開催日のレース
    same_day = sorted([r for r in ids if r[:10] == meet])
    log(f"  対象 {last}（{same_day.index(last)+1}番目 / 同日{len(same_day)}レース）")

    g = clean[clean.race_id == last].copy()
    race_df = g.drop(columns=[c for c in MASK if c in g.columns])

    # ① BTの条件：同じ日の先行レースを含む
    hist_bt = clean[clean.race_id != last].copy()
    # ② 本番の条件：同じ日のレースを全部除く
    hist_live = clean[~clean.race_id.isin(same_day)].copy()
    log(f"  BT条件の履歴 {len(hist_bt):,}行 / 本番条件 {len(hist_live):,}行"
        f"（差 {len(hist_bt)-len(hist_live)}行）")

    out = {}
    for lab, h in (("BT条件", hist_bt), ("本番条件", hist_live)):
        t0 = time.time()
        out[lab] = F.build_features_for_prediction(race_df.copy(), h)
        log(f"    {lab}: {time.time()-t0:.0f}秒")

    a, b = out["BT条件"], out["本番条件"]
    diffs = []
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x = pd.to_numeric(a[c], errors="coerce")
        y = pd.to_numeric(b[c], errors="coerce")
        v = x.notna() & y.notna()
        if v.sum() == 0:
            if x.isna().mean() != y.isna().mean():
                diffs.append((c, np.nan, "欠損の有無が違う"))
            continue
        d = (x[v] - y[v]).abs()
        if (d > 1e-9).any():
            rel = (d / x[v].abs().replace(0, np.nan)).mean()
            diffs.append((c, d.mean(), f"相対差 {rel*100:.1f}%" if rel == rel else ""))

    log("")
    log(f"  === 同じ日の結果を使えるかどうかで変わる列: {len(diffs)} / {len(cols)} ===")
    for c, dv, note in sorted(diffs, key=lambda x: -(x[1] if x[1] == x[1] else 0))[:25]:
        log("    %-32s 平均差 %10s  %s"
            % (c[:32], f"{dv:.4f}" if dv == dv else "-", note))
    log("")
    log(f"  変わらない列 {len(cols)-len(diffs)} / {len(cols)}"
        f"（{(len(cols)-len(diffs))/len(cols)*100:.1f}%）")
    log("")
    log("  この差は実装では消せない。BTが本番より多くの情報を使っている分。")


if __name__ == "__main__":
    main()
