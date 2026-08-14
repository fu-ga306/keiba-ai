# -*- coding: utf-8 -*-
"""条件別の特徴量除外を、実装可能な単一ルールに落として検証する（2026-08-14）

グループ除去実験で分かったこと（クリーンデータ）
  長距離2100+  基準+0.00198 / 全特徴が最良（血統を抜くと1/5に激減）
  短距離-1400  基準+0.00025 / −騎手厩舎 +0.00040（1.6倍）
  道悪         基準+0.00027 / −騎手厩舎 +0.00096（3.6倍）
  多頭数16+    基準+0.00025 / −騎手厩舎 +0.00032（1.3倍）

  → 「騎手厩舎は長距離以外では害」という形に見える。

⚠ そのまま実装できない。条件が重なる（道悪かつ短距離など）。
  実装するなら「距離で切って、それ以外は騎手厩舎を抜く」のような単一規則にする。
  その規則が本当に効くかを、条件を切る前の全体で検証する。

比較する3案
  A 現行        : 全特徴（323列）
  B 全体で除外   : 騎手厩舎31列を常に抜く
  C 距離で切替  : 距離>=閾値は全特徴 / 未満は騎手厩舎を抜く

walk-forward（学習=検証年より前）。全体と条件別の両方でΔR²を出す。

実行: python verify_rule.py → verify_rule_result.csv
"""
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from market_free_model import FEATURE_COLS_MF

EPS = 1e-6
YEARS = [2022, 2023, 2024, 2025]
SWITCH = 1900          # この距離以上は全特徴を使う
JOCKEY = r"騎手|調教師|馬主"
P = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
         num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def dr2(y, mkt, mdl):
    if y.sum() < 30:
        return np.nan
    out = []
    for X in (np.column_stack([logit(mkt)]),
              np.column_stack([logit(mkt), logit(mdl)])):
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        p = np.clip(lr.predict_proba(X)[:, 1], EPS, 1 - EPS)
        ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        b = y.mean()
        out.append(1 - ll / ((y * np.log(b) + (1 - y) * np.log(1 - b)).sum()))
    return out[1] - out[0]


def main():
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    NOJ = [c for c in BASE if not re.search(JOCKEY, c)]
    log(f"全特徴 {len(BASE)}列 / 騎手厩舎を抜くと {len(NOJ)}列（-{len(BASE)-len(NOJ)}）")
    use = list(dict.fromkeys(["race_id", "馬名", "着順_num", "単勝オッズ",
                              "距離", "馬場状態_num", "出走頭数"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[D["着"].notna() & D["odds"].notna() & (D["odds"] > 0)].copy()
    D["win"] = (D["着"] == 1).astype(int)
    D["m"] = D.groupby("race_id")["odds"].transform(lambda s: (1 / s) / (1 / s).sum())
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")

    def oos(cols, mask=None):
        """walk-forwardで予測を作る。maskがあればその行だけ学習・予測。"""
        out = pd.Series(np.nan, index=D.index)
        for ty in YEARS:
            tr = D[(D["年"] < ty) & (mask if mask is not None else True)]
            te = (D["年"] == ty) & (mask if mask is not None else True)
            if len(tr) < 3000 or te.sum() < 300:
                continue
            m = lgb.train(P, lgb.Dataset(tr[cols], tr["win"]), num_boost_round=400)
            out[te] = m.predict(D.loc[te, cols])
        return out

    log("\nA 現行（全特徴）を学習中...")
    pA = oos(BASE)
    log("B 全体で騎手厩舎を除外を学習中...")
    pB = oos(NOJ)
    log(f"C 距離{SWITCH}mで切替を学習中...")
    lo_m = D["距離"] < SWITCH
    hi_m = ~lo_m
    pC = pd.Series(np.nan, index=D.index)
    pC[lo_m] = oos(NOJ, lo_m)[lo_m]
    pC[hi_m] = oos(BASE, hi_m)[hi_m]

    CONDS = {"全体": pd.Series(True, index=D.index),
             f"長距離{SWITCH}+": D["距離"] >= SWITCH,
             "短距離-1400": D["距離"] <= 1400,
             "道悪": D["馬場状態_num"] >= 3,
             "多頭数16+": D["出走頭数"] >= 16}
    rows = []
    log(f"\n{'条件':<14}{'A 現行':>10}{'B 全除外':>10}{'C 距離切替':>11}{'最良':>10}")
    for cl, sel in CONDS.items():
        r = {"条件": cl}
        for name, p in (("A 現行", pA), ("B 全除外", pB), ("C 距離切替", pC)):
            t = D[sel & p.notna()]
            r[name] = round(dr2(t["win"].values, t["m"].values, p[t.index].values), 5)
        r["頭数"] = int((sel & pA.notna()).sum())
        best = max(("A 現行", "B 全除外", "C 距離切替"), key=lambda k: r[k])
        r["最良"] = best
        rows.append(r)
        log(f"{cl:<12}{r['A 現行']:>+10.5f}{r['B 全除外']:>+10.5f}"
            f"{r['C 距離切替']:>+11.5f}{best:>12}")
    pd.DataFrame(rows).to_csv("verify_rule_result.csv", index=False,
                              encoding="utf-8-sig")
    log("\n保存 → verify_rule_result.csv")


if __name__ == "__main__":
    main()
