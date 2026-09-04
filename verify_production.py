# -*- coding: utf-8 -*-
"""明日の本番で予想が動くかを、スクレイピングなしで確かめる（2026-09-04）

なぜ要るか
  features.py は予想の中核。今日そこを4か所直した。
  壊れていれば明日の予想が全部止まる。**開催前に確かめる。**

確かめること
  ① 必要なファイルが揃っているか
  ② 本番と同じ条件（上限20000・絞り込みあり）で特徴量が作れるか
  ③ そこから gap を計算し、買い目まで作れるか
  ④ 欠損率が異常でないか
  ⑤ 1レースあたりの時間が間に合うか（予想の間隔は最小5分）

過去のレースを出馬表に見立てて通す。スクレイピングはしない。
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
NEED_FILES = ["race_data_clean.csv", "model_resid.pkl", "speed_baseline.csv",
              "agari_baseline.csv", "course_style_bias.csv",
              "course_style_bias_dated.csv", "sire_stats_father_train.csv",
              "course_turn.csv", "name_master.csv", "owner_master.csv"]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    ng = 0

    log("=== ① 必要なファイル ===")
    for f in NEED_FILES:
        p = os.path.join(BASE_DIR, f)
        if os.path.exists(p):
            log(f"  ○ {f:<34} {os.path.getsize(p)/1024:>9,.0f} KB")
        else:
            log(f"  ✗ {f:<34} ありません")
            ng += 1

    import features as F
    import resid_io
    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    ids = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))

    log("")
    log("=== ②③ 本番と同じ条件で3レース通す（上限20000・絞り込みあり） ===")
    # 上限は既定のまま（本番と同じ）
    if hasattr(F, "_MAX_PRED_HISTORY_OVERRIDE"):
        F._MAX_PRED_HISTORY_OVERRIDE = 20000
    times, results = [], []
    for rid in ids[-3:]:
        g = clean[clean.race_id == rid].copy()
        hist = clean[clean.race_id != rid].copy()
        race_df = g.drop(columns=[c for c in MASK if c in g.columns])
        t0 = time.time()
        try:
            pdf = F.build_features_for_prediction(race_df, hist)
        except Exception as e:
            log(f"  ✗ {rid} 特徴量で失敗: {type(e).__name__}: {e}")
            ng += 1
            continue
        el = time.time() - t0
        times.append(el)
        try:
            gg = resid_io.predict_gap(m, pdf)
            if gg is None:
                log(f"  ✗ {rid} gapを計算できません")
                ng += 1
                continue
            bets = resid_io.pick_bets(gg, model={"gap_min": resid_io.AX_GAP})
        except Exception as e:
            log(f"  ✗ {rid} 買い目で失敗: {type(e).__name__}: {e}")
            ng += 1
            continue
        miss = pd.DataFrame({c: pd.to_numeric(pdf[c], errors="coerce")
                             for c in m["use_cols"] if c in pdf.columns}).isna().mean().mean()
        results.append((rid, len(pdf), el, gg["gap"].max(), len(bets), miss * 100))
        log(f"  ○ {rid}  {len(pdf):2d}頭  {el:5.1f}秒  "
            f"最大gap {gg['gap'].max():.2f}  買い目 {len(bets)}点  欠損 {miss*100:.1f}%")

    log("")
    log("=== ④ 欠損率 ===")
    if results:
        mi = np.mean([r[5] for r in results])
        log(f"  平均 {mi:.1f}%")
        if mi > 25:
            log("  ⚠ 25%を超えています。以前の実測は17〜19%程度")
            ng += 1
        else:
            log("  ○ 想定の範囲")

    log("")
    log("=== ⑤ 時間 ===")
    if times:
        log(f"  1レース {np.mean(times):.0f}秒（最大 {max(times):.0f}秒）")
        log("  予想の間隔は最小5分（300秒）")
        if max(times) > 240:
            log("  ⚠ 240秒を超えています。次のレースに食い込む恐れ")
            ng += 1
        else:
            log("  ○ 間に合います")

    log("")
    log("=" * 50)
    if ng == 0:
        log("  ✅ 明日の本番で動きます")
    else:
        log(f"  ⚠ 問題 {ng}件。直してから開催を迎えること")
    return ng


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
