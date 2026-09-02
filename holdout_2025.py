# -*- coding: utf-8 -*-
"""2025年で1回だけ評価する（2026-09-03）

⚠ これは**一度きり**の評価。ここで下がっても構成を選び直さない。
  選び直したら、この枠組み自体が無効になる。

事前登録（事前登録_202609.md）にしたがって決めた構成
  A 特徴量  いまの全部（外すとどれも悪化した）
  B 券種    単勝＋ワイド（基準が最良）
  C しきい値 gap>=1.5（証明までの月数が最短。回収率の高さでは選ばない）
  D シード   3（7にしても区間が狭くならなかった）
  E 学習量  600 → **300**（証明まで24か月→16か月）

  変えるのは E だけ。

比較のため、いまの構成（rounds=600）も同じ2025年で測る。
これは候補探しではなく、**採用した1つの変更と現行の1つを並べる**だけ。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import exp_model_202609 as E

    D, BASE = E.load()
    PAY = E.payouts()
    log(f"  {len(D):,}頭  特徴量{len(BASE)}列")
    log("  2025年を、2024年までのデータで学習したモデルで予測する")

    rng = np.random.default_rng(20260903)
    rows = []
    for nr, lab in ((300, "採用案 rounds=300"), (600, "現行 rounds=600")):
        t0 = datetime.now()
        P = E.fit_predict(D, BASE, [2025], [42, 7, 123], nr)
        ret = E.evaluate(P, PAY, 1.5, True)
        s = rng.choice(ret, size=(4000, len(ret))).mean(axis=1)
        lo, hi = np.percentile(s, [2.5, 97.5])
        rows.append((lab, len(ret), ret.mean(), lo, hi))
        log(f"  {lab:<20} {len(ret):5d}点  {ret.mean():6.1f}%  "
            f"95%[{lo:5.1f}, {hi:5.1f}]   {(datetime.now()-t0).total_seconds()/60:.0f}分")

    log("")
    log("  === 2025年（一度きりの評価） ===")
    for lab, n, roi, lo, hi in rows:
        mark = "○ 100%超" if lo > 100 else ""
        log(f"    {lab:<20} {n:5d}点  {roi:6.1f}%  95%[{lo:5.1f}, {hi:5.1f}]  {mark}")
    log("")
    log("  開発（2021-2024）では 300が141.7% / 600が132.0% だった。")
    log("  2025年でも同じ向きなら、変更に意味がある。")
    log("  逆なら、開発での差は偶然だったということ。")

    pd.DataFrame(rows, columns=["構成", "点数", "回収率", "下限", "上限"]).to_csv(
        "holdout_2025_result.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
