# -*- coding: utf-8 -*-
"""まとめ計算を、正しい条件と正しい物差しで測り直す（2026-09-04）

前回の測り方の誤り
  ① 完全一致（差<1e-6）だけを一致と数えた。
     実際には平均差が7分の1〜29分の1に縮んでいたのに「悪化」と報告した。
  ② 履歴から36レース分を除いた。本番が知り得ないのは**同じ日の分だけ**。
     同日レースの影響は3列しかないことが後で分かった。

今回の条件
  ・同じ開催日のレースだけを履歴から除く（本番の実際の条件）
  ・上限なし（まとめれば絞り込みが実質不要）
  ・一致の物差しを3段階で出す
      完全一致    差 < 1e-6
      実用一致    相対差 < 0.1%
      おおむね一致 相対差 < 1%
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
    meet = ids[-1][:10]
    targets = sorted([r for r in ids if r[:10] == meet])
    log(f"  同じ開催日の {len(targets)}レースをまとめて計算する")

    race_df = clean[clean.race_id.isin(targets)].copy()
    race_df = race_df.drop(columns=[c for c in MASK if c in race_df.columns])
    hist = clean[~clean.race_id.isin(targets)].copy()
    log(f"  出馬表 {len(race_df)}頭 / 履歴 {len(hist):,}行")

    F._MAX_PRED_HISTORY_OVERRIDE = None
    t0 = time.time()
    pdf = F.build_features_for_prediction(race_df, hist)
    el = time.time() - t0
    log(f"  まとめて計算 {el:.0f}秒（{len(pdf)}頭・{len(targets)}レース）")
    log(f"    1レースあたり {el/len(targets):.1f}秒 相当")

    need = list(set(cols) | {"race_id", "馬番"})
    parts = []
    for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                          usecols=lambda c: c in need, dtype={"race_id": str},
                          chunksize=200000, low_memory=False):
        ch["race_id"] = ch["race_id"].str.replace(r"\.0$", "", regex=True)
        x = ch[ch.race_id.isin(targets)]
        if len(x):
            parts.append(x)
    BT = pd.concat(parts)
    BT["bn"] = pd.to_numeric(BT["馬番"], errors="coerce")
    pdf = pdf.copy()
    pdf["bn"] = pd.to_numeric(pdf["馬番"], errors="coerce")
    mg = pdf.merge(BT, on=["race_id", "bn"], suffixes=("_本番", "_BT"))
    log(f"  突き合わせ {len(mg)}頭")

    exact = ok01 = ok1 = tot = 0
    worst = []
    for c in cols:
        a = pd.to_numeric(mg.get(c + "_本番"), errors="coerce")
        b = pd.to_numeric(mg.get(c + "_BT"), errors="coerce")
        if a is None or b is None:
            continue
        v = a.notna() & b.notna()
        if v.sum() < 10:
            continue
        tot += 1
        d = (a[v] - b[v]).abs()
        rel = (d / b[v].abs().clip(lower=1e-9))
        if (d < 1e-6).mean() > 0.999:
            exact += 1
        if (rel < 0.001).mean() > 0.999:
            ok01 += 1
        if (rel < 0.01).mean() > 0.999:
            ok1 += 1
        else:
            worst.append((c, d.mean(), rel.mean() * 100))

    log("")
    log("  === 一致の度合い（%d列で比較） ===" % tot)
    log(f"    完全一致（差<1e-6）      {exact:3d}列  {exact/tot*100:5.1f}%")
    log(f"    実用一致（相対差<0.1%）   {ok01:3d}列  {ok01/tot*100:5.1f}%")
    log(f"    おおむね一致（相対差<1%）  {ok1:3d}列  {ok1/tot*100:5.1f}%")
    log("")
    log(f"  1%以上ずれる列 {len(worst)}")
    for c, dv, rl in sorted(worst, key=lambda x: -x[2])[:20]:
        log("    %-32s 平均差 %8.4f  相対差 %6.1f%%" % (c[:32], dv, rl))


if __name__ == "__main__":
    main()
