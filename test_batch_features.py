# -*- coding: utf-8 -*-
"""全レースまとめて特徴量を作れば、速くて正確になるかを確かめる（2026-09-03）

いまの作り
  1レースごとに build_features_for_prediction を呼ぶ。
  そのたびに履歴を「そのレースの馬・騎手・調教師」に絞り、
  20,000行を超えたら古い行を捨てる。
    → 1レース99秒。36レースで約59分。しかもBTと値が合わない（79列）。

試すこと
  当日の全レースを**まとめて1回**で計算する。
  行数は変わらないので時間はほぼ同じ（1回ぶん）。
  絞り込みが実質不要になるので、上限で削られることもない。

確かめること
  ① まとめて計算した値が、BT(race_features.csv)と一致するか
  ② 1回あたりの時間

⚠ 過去のレースを「まだ結果が出ていないもの」に見立てて測る。
  スクレイピングはしない。
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
    model_cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)

    # 8/30の全レースを「当日ぶん」に見立てる
    ids = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))
    day = [r for r in ids if r[:10] == ids[-1][:10]]
    targets = sorted(set(clean.loc[clean.race_id.isin(ids[-36:]), "race_id"]))
    log(f"  当日ぶんに見立てるレース {len(targets)}件")

    race_df = clean[clean.race_id.isin(targets)].copy()
    race_df = race_df.drop(columns=[c for c in MASK if c in race_df.columns])
    hist = clean[~clean.race_id.isin(targets)].copy()
    log(f"  出馬表 {len(race_df)}頭 / 履歴 {len(hist):,}行")

    F._MAX_PRED_HISTORY_OVERRIDE = None      # 絞り込みの上限を外す
    t0 = time.time()
    pdf = F.build_features_for_prediction(race_df, hist)
    el = time.time() - t0
    log(f"  まとめて計算 完了 {el:.0f}秒（{len(pdf)}頭）")

    # BT側と突き合わせ
    need = list(set(model_cols) | {"race_id", "馬番"})
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

    ok = tot = 0
    bad = []
    for c in model_cols:
        a = pd.to_numeric(mg.get(c + "_本番"), errors="coerce")
        d = pd.to_numeric(mg.get(c + "_BT"), errors="coerce")
        if a is None or d is None:
            continue
        v = a.notna() & d.notna()
        if v.sum() < 10:
            continue
        tot += 1
        agree = ((a[v] - d[v]).abs() < 1e-6).mean()
        if agree > 0.999:
            ok += 1
        else:
            bad.append((c, agree * 100, (a[v] - d[v]).abs().mean()))

    log("")
    log("  === 結果 ===")
    log(f"    一致した列  {ok} / {tot}  （{ok/tot*100:.1f}%）")
    log(f"    計算時間    {el:.0f}秒（{len(targets)}レースぶんを1回で）")
    log(f"    いまの作り  99秒 × {len(targets)}レース = {99*len(targets)/60:.0f}分")
    log("")
    if bad:
        log(f"  まだ一致しない列 {len(bad)}")
        for c, ag, df_ in sorted(bad, key=lambda x: x[1])[:15]:
            log("    %-30s 一致%5.1f%%  平均差 %.4f" % (c[:30], ag, df_))


if __name__ == "__main__":
    main()
