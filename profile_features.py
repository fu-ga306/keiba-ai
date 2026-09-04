# -*- coding: utf-8 -*-
"""特徴量計算のどこに時間がかかっているかを測る（2026-09-04）

なぜ要るか
  1レース99秒（上限20,000）／242秒（上限なし）。
  上限を外せば本番とBTが一致するが、時間が足りない。
  **どこが遅いのかを知らないと、正しい場所を直せない。**

  コードのコメントには add_horse_history_features が遅いと書いてあるが、
  騎手・調教師の集計も df.iterrows() で全行を回している。
  推測ではなく実測で決める。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import cProfile
import pstats
import io as _io
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK = ["着順", "着順_num", "タイム", "上り", "通過", "着差", "賞金"]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    rid = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))[-1]
    g = clean[clean.race_id == rid].copy()
    hist = clean[clean.race_id != rid].copy()
    race_df = g.drop(columns=[c for c in MASK if c in g.columns])
    log(f"  対象 {rid}  履歴 {len(hist):,}行")

    F._MAX_PRED_HISTORY_OVERRIDE = None       # 上限なし＝BTと同じ条件
    log("  上限なしで実行して内訳を測る")

    pr = cProfile.Profile()
    pr.enable()
    F.build_features_for_prediction(race_df, hist)
    pr.disable()

    s = _io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    st.print_stats(40)
    txt = s.getvalue()

    log("")
    log("  === 累計時間の上位（features.py の関数のみ） ===")
    for line in txt.splitlines():
        if "features.py" in line or "iterrows" in line or "{built-in" in line:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    cum = float(parts[3])
                except ValueError:
                    continue
                if cum >= 3.0:
                    log("    " + line.strip()[:150])

    log("")
    log("  === 全体の上位20（参考） ===")
    for line in txt.splitlines()[4:30]:
        log("    " + line.rstrip()[:150])


if __name__ == "__main__":
    main()
