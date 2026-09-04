# -*- coding: utf-8 -*-
"""朝の計算を使い回し、変わる列だけ差し替えて足りるかを確かめる（2026-09-03）

やりたいこと（現状の作りは変えない）
  07:00   いままで通り全部計算する
  40分前  馬体重など**変わるものだけ**取り直して予想
  それ以降 オッズなど**変わるものだけ**取り直して予想

確かめること
  ① 馬体重が入るかどうかで、実際にどの列が変わるか
  ② 馬場状態が変わると、どの列が変わるか
  ③ その列だけ差し替えれば、全部計算し直したのと同じになるか

  ③が成り立つなら、40分前と直前の計算量を大幅に減らせる。
  成り立たない列があれば、それを名指しする。

⚠ 過去のレースを使う。スクレイピングはしない。
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


def build(F, race_df, hist, tag):
    t0 = time.time()
    pdf = F.build_features_for_prediction(race_df.copy(), hist)
    log(f"    {tag}: {time.time()-t0:.0f}秒")
    return pdf


def diff_cols(a, b, cols):
    """2つの結果で値が違う列を返す。"""
    out = []
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x = pd.to_numeric(a[c], errors="coerce")
        y = pd.to_numeric(b[c], errors="coerce")
        both_na = x.isna() & y.isna()
        if both_na.all():
            continue
        v = x.notna() & y.notna()
        if v.sum() == 0:
            out.append((c, "片方だけ欠損", np.nan))
            continue
        d = (x[v] - y[v]).abs()
        if (d > 1e-9).any():
            out.append((c, "値が違う", d.mean()))
    return out


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    model_cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    rid = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))[-1]
    log(f"  対象 {rid}")

    g = clean[clean.race_id == rid].copy()
    hist = clean[clean.race_id != rid].copy()
    base = g.drop(columns=[c for c in MASK if c in g.columns])

    # ── ① 馬体重の有無で何が変わるか ──────────────────────────
    log("")
    log("  === ① 馬体重の有無 ===")
    with_w = build(F, base, hist, "体重あり（40分前の状態）")
    no_w = base.copy()
    for c in ("馬体重", "体重増減"):
        if c in no_w.columns:
            no_w[c] = np.nan
    without_w = build(F, no_w, hist, "体重なし（朝の状態）")

    d1 = diff_cols(with_w, without_w, model_cols)
    log(f"    体重の有無で変わる列: {len(d1)} / {len(model_cols)}")
    for c, why, dv in sorted(d1, key=lambda x: -(x[2] if x[2] == x[2] else 0))[:25]:
        log("      %-30s %-12s 平均差 %s" % (c[:30], why, f"{dv:.4f}" if dv == dv else "-"))

    # ── ② 馬場状態が変わると何が変わるか ────────────────────────
    log("")
    log("  === ② 馬場状態（良→稍重）で変わる列 ===")
    wet = base.copy()
    if "馬場状態" in wet.columns:
        wet["馬場状態"] = "稍重"
    if "馬場状態_num" in wet.columns:
        wet["馬場状態_num"] = 2
    wet_pdf = build(F, wet, hist, "稍重")
    d2 = diff_cols(with_w, wet_pdf, model_cols)
    log(f"    馬場状態で変わる列: {len(d2)} / {len(model_cols)}")
    for c, why, dv in sorted(d2, key=lambda x: -(x[2] if x[2] == x[2] else 0))[:20]:
        log("      %-30s %-12s 平均差 %s" % (c[:30], why, f"{dv:.4f}" if dv == dv else "-"))

    # ── ③ まとめ ────────────────────────────────────────
    s1 = {c for c, *_ in d1}
    s2 = {c for c, *_ in d2}
    log("")
    log("  === ③ まとめ ===")
    log(f"    体重で変わる      {len(s1)}列")
    log(f"    馬場状態で変わる   {len(s2)}列")
    log(f"    合計（重複除く）   {len(s1 | s2)}列 / {len(model_cols)}")
    log("")
    log(f"    → 40分前に更新すべきは {len(s1 | s2)}列。"
        f"残り {len(model_cols)-len(s1|s2)}列は朝のまま使える")
    pd.DataFrame(sorted(s1 | s2), columns=["更新が要る列"]).to_csv(
        "incremental_cols.csv", index=False, encoding="utf-8-sig")
    log("    incremental_cols.csv に保存")


if __name__ == "__main__":
    main()
