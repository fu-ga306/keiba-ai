# -*- coding: utf-8 -*-
"""履歴の上限が特徴量をどれだけ歪めているか測る（2026-09-03）

背景
  本番は build_features_for_prediction で特徴量をその場で計算する。
  そのとき履歴を「今日の馬・騎手・調教師」に絞ってから使うが、
  絞り込み後が20,000行を超えると**古い行を捨てる**安全弁がある。

    _MAX_PRED_HISTORY = 20000

  この値は「絞り込みが空振りして5〜11k行しか残らない」前提で決められた。
  ところが2026-08-30に名寄せを直した結果、騎手・調教師が正しく一致するようになり、
  38,000〜68,000行が残るようになった。**その6〜7割が捨てられている。**

  実際に27レース全部で上限に当たっていた。
  BTは全履歴で計算するので、本番とBTで値が違う。突き合わせで79列がズレていた。

測ること
  上限を変えたときの ① 値の一致率 ② 計算時間
  スクレイピングはしない。過去のレースを出馬表に見立てて使う。

実行
  python test_history_cap.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPS = [20000, 60000, 200000, None]        # None = 制限なし
N_RACES = 3                                 # 測るレース数（1レースあたり時間がかかる）


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F
    import pickle

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    model_cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)

    # 8/30のレースを出馬表に見立てる（本番が実際に予想した日）
    targets = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))[-N_RACES:]
    log(f"  対象レース {targets}")

    # BT側の正解（race_features.csv）
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
    log(f"  BT側 {len(BT)}頭")

    orig_cap = F.__dict__.get("_MAX_PRED_HISTORY_OVERRIDE")
    rows = []
    for cap in CAPS:
        lab = "制限なし" if cap is None else f"{cap:,}行"
        agree_all, t_all = [], []
        for rid in targets:
            g = clean[clean.race_id == rid].copy()
            hist = clean[clean.race_id != rid].copy()
            race_df = g.drop(columns=[c for c in ("着順", "着順_num", "タイム", "上り",
                                                  "通過", "着差", "賞金")
                                      if c in g.columns])
            F._MAX_PRED_HISTORY_OVERRIDE = cap
            t0 = time.time()
            try:
                pdf = F.build_features_for_prediction(race_df, hist)
            except Exception as e:
                log(f"    {lab} {rid} 失敗: {type(e).__name__}: {e}")
                continue
            t_all.append(time.time() - t0)
            pdf = pdf.copy()
            pdf["bn"] = pd.to_numeric(pdf["馬番"], errors="coerce")
            b = BT[BT.race_id == rid]
            mg = pdf.merge(b, on="bn", suffixes=("_本番", "_BT"))
            ok = 0
            tot = 0
            for c in model_cols:
                a = pd.to_numeric(mg.get(c + "_本番"), errors="coerce")
                d = pd.to_numeric(mg.get(c + "_BT"), errors="coerce")
                if a is None or d is None:
                    continue
                v = a.notna() & d.notna()
                if v.sum() < 3:
                    continue
                tot += 1
                if ((a[v] - d[v]).abs() < 1e-6).mean() > 0.999:
                    ok += 1
            if tot:
                agree_all.append(ok / tot * 100)
        if agree_all:
            rows.append((lab, np.mean(agree_all), np.mean(t_all)))
            log(f"  {lab:<10} 一致した列 {np.mean(agree_all):5.1f}%   "
                f"1レースあたり {np.mean(t_all):5.1f}秒")
    F._MAX_PRED_HISTORY_OVERRIDE = orig_cap

    log("")
    log("  === まとめ ===")
    log("  %-12s %14s %16s" % ("上限", "一致した列", "1レースの計算時間"))
    log("  " + "-" * 46)
    for lab, ag, t in rows:
        log("  %-12s %12.1f%% %14.1f秒" % (lab, ag, t))
    log("")
    log("  本番は7分前に予想するので、1レース60秒を超えると間に合わない。")


if __name__ == "__main__":
    main()
