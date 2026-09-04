# -*- coding: utf-8 -*-
"""モデルが古びたのか、2025年が特殊なのかを分ける（2026-09-04）

いまの状態
  軸の実際の勝率  2021:16.0% 2022:13.7% 2023:15.2% 2024:17.3% 2025:11.3%
  予測勝率はどの年も18〜23%で同水準。**2025だけ当たらない。**

  年ごとに学習し直しているので、2025年のモデルは2019-2024で学習している。
  データは一番多い。なのに一番当たらない。

切り分け方
  **同じモデルを2024年と2025年の両方に当てる。**
    2019-2023 で学習 → 2024 で評価（out-of-sample）
    2019-2023 で学習 → 2025 で評価（out-of-sample）
  同じモデルなので、差が出れば「年の違い」。
  2024が良く2025が悪ければ、モデルではなく年が原因。

さらに
  ・学習年を変えたモデルの重要度を比べ、モデル自体が変わっているか見る
  ・2025年を除いて学習したモデルと、含めたモデルで違いが出るか
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import exp_model_202609 as E

    D, BASE = E.load()
    PAY = E.payouts()
    log(f"  {len(D):,}頭  特徴量{len(BASE)}列")

    # ── ① 同じモデルを2024と2025に当てる ──────────────────────
    log("")
    log("  === ① 2019-2023で学習した1つのモデルを、2024と2025に当てる ===")
    tr = D[D.年 <= 2023]
    log(f"    学習 {len(tr):,}頭（2019-2023）")
    import lightgbm as lgb
    preds = []
    for sd in (42, 7, 123):
        p = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                 num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, seed=sd, verbose=-1,
                 num_threads=4)
        ds = lgb.Dataset(tr[BASE], tr["win"].values,
                         init_score=tr["lq"].values, free_raw_data=False)
        preds.append(lgb.train(p, ds, num_boost_round=600))
    log("    学習完了")

    log("")
    log("    %-6s %8s %10s %12s %10s %8s" %
        ("年", "軸の数", "予測勝率", "実際の勝率", "比", "回収率"))
    log("    " + "-" * 60)
    for y in (2024, 2025):
        te = D[D.年 == y]
        f = np.mean([m.predict(te[BASE], raw_score=True) for m in preds], axis=0)
        t = te[["race_id", "馬番", "odds", "is_turf"]].copy()
        sc = f + te["lq"].values
        e = np.exp(sc - pd.Series(sc, index=te.index).groupby(te.race_id).transform("max"))
        t["p"] = (e / e.groupby(te.race_id.values).transform("sum")).values
        t["gap"] = t["p"] / te["q"].values
        t["着"] = te["着"].values
        ax = t.loc[t.groupby("race_id")["gap"].idxmax()]
        ax = ax[ax["gap"] >= 1.5]
        ret = E.evaluate(t, PAY, 1.5, False)
        log("    %-6d %8d %9.1f%% %11.1f%% %10.2f %7.1f%%"
            % (y, len(ax), ax["p"].mean() * 100, (ax["着"] == 1).mean() * 100,
               (ax["着"] == 1).mean() / max(ax["p"].mean(), 1e-9), ret.mean()))

    # ── ② 学習年を変えて重要度を比べる ────────────────────────
    log("")
    log("  === ② 学習年を変えると、モデルは変わるか（重要度の上位10） ===")
    imps = {}
    for last in (2022, 2023, 2024):
        t2 = D[D.年 <= last]
        ds = lgb.Dataset(t2[BASE], t2["win"].values,
                         init_score=t2["lq"].values, free_raw_data=False)
        p = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                 num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1,
                 num_threads=4)
        mm = lgb.train(p, ds, num_boost_round=600)
        v = mm.feature_importance(importance_type="gain")
        imps[last] = pd.Series(v / v.sum() * 100, index=BASE)
        log(f"    〜{last}年で学習: 完了")

    top = imps[2024].sort_values(ascending=False).head(10).index
    log("")
    log("    %-30s %8s %8s %8s" % ("列", "〜2022", "〜2023", "〜2024"))
    log("    " + "-" * 58)
    for c in top:
        log("    %-30s %7.2f%% %7.2f%% %7.2f%%"
            % (c[:30], imps[2022][c], imps[2023][c], imps[2024][c]))

    # 順位の相関
    log("")
    for a, b in ((2022, 2023), (2023, 2024)):
        r = imps[a].rank().corr(imps[b].rank(), method="spearman")
        log(f"    重要度の順位相関 {a}年まで vs {b}年まで: {r:.3f}")
    log("    → 1.0に近ければモデルの性格は変わっていない")


if __name__ == "__main__":
    main()
