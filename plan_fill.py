# -*- coding: utf-8 -*-
"""どの列を何で埋めるかを、実測から決める（2026-09-03）

原則
  ○ 埋める   学習時（race_features.csv）に値があるのに本番で欠損している列
            モデルは値がある前提で学習しており、NaNは未知の入力になる
  ✕ 埋めない  学習時もNaNの列（初出走の馬の過去成績など）
            モデルはNaNとして学習している。埋めると別物になる

判定
  学習時の欠損率が低い（<5%）のに、本番の欠損率がそれより10ポイント以上高い列を
  「埋めるべき」とする。それ以外は触らない。

埋め方は列の性質で決める
  騎手・調教師・馬主の成績  事前集計から引ける（正確）
  コース脚質バイアス       既存CSVから
  その他                学習時の中央値（最後の手段。印を残す）

実行
  python plan_fill.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle
import re

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_NA_OK = 5.0        # 学習時の欠損率がこれ未満なら「値があるのが普通」
GAP_MIN = 10.0           # 本番がこれ以上多く欠けていたら埋める対象


def log(m):
    print(m, flush=True)


def source_of(col):
    """何で埋めるかを列名から決める。"""
    if re.search(r"騎手", col):
        return "騎手の事前集計"
    if re.search(r"調教師", col):
        return "調教師の事前集計"
    if re.search(r"馬主", col):
        return "馬主の事前集計"
    if re.search(r"コース脚質バイアス", col):
        return "course_style_bias.csv"
    if re.search(r"回り", col):
        return "course_turn.csv"
    if re.search(r"賞金", col):
        return "(クラス,着順)の定額表"
    return "学習時の中央値"


def main():
    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]

    s = pd.read_csv(os.path.join(BASE_DIR, "pred_features.csv"),
                    dtype={"race_id": str}, low_memory=False)
    s = s[s["記録時刻"].astype(str) >= "2026/08/30 11:20"]     # 名寄せ導入後だけ
    live = {c: pd.to_numeric(s[c], errors="coerce").isna().mean() * 100
            for c in cols if c in s.columns}
    log(f"  本番の記録 {len(s)}頭 / {s.race_id.nunique()}レース")

    # 学習側の欠損率（列を小分けにして読む）
    tr = {}
    BATCH = 40
    tgt = [c for c in cols if c in live]
    for i in range(0, len(tgt), BATCH):
        part = tgt[i:i + BATCH]
        acc = np.zeros(len(part))
        n = 0
        for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                              usecols=part, chunksize=100000, low_memory=True):
            acc += ch.reindex(columns=part).isna().sum().values
            n += len(ch)
        for c, v in zip(part, acc / n * 100):
            tr[c] = v
    log(f"  学習側 {n:,}行\n")

    rows = []
    for c in tgt:
        rows.append({"列": c, "本番": live[c], "学習": tr[c],
                     "差": live[c] - tr[c], "埋め方": source_of(c)})
    R = pd.DataFrame(rows)
    fill = R[(R["学習"] < TRAIN_NA_OK) & (R["差"] >= GAP_MIN)].sort_values("差", ascending=False)
    keep = R[~R.index.isin(fill.index)]

    log(f"  === 埋めるべき列 {len(fill)} / {len(tgt)} ===")
    log("  %-30s %8s %8s %8s  %s" % ("列", "本番欠損", "学習欠損", "差", "埋め方"))
    log("  " + "-" * 82)
    for r in fill.itertuples():
        log("  %-30s %7.1f%% %7.1f%% %7.1f  %s" % (r.列[:30], r.本番, r.学習, r.差, r.埋め方))

    log("")
    log("  === 埋め方ごとの内訳 ===")
    for k, v in fill["埋め方"].value_counts().items():
        log(f"    {k:<24} {v:>3}列")

    log("")
    log(f"  === 埋めない列 {len(keep)} ===")
    hi = keep[keep["本番"] > 30].sort_values("本番", ascending=False).head(10)
    log("  本番の欠損が多いが、学習時も欠けているので触らないもの（上位10）")
    for r in hi.itertuples():
        log("    %-30s 本番%6.1f%%  学習%6.1f%%" % (r.列[:30], r.本番, r.学習))

    fill.to_csv(os.path.join(BASE_DIR, "fill_plan.csv"), index=False,
                encoding="utf-8-sig")
    log("\n  fill_plan.csv に保存")


if __name__ == "__main__":
    main()
