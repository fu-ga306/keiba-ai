# -*- coding: utf-8 -*-
"""条件ごとに使う特徴量を変えると精度は上がるか（2026-08-14・クリーンデータ）

問い（利用者）
  構成を細かくして、構成に応じて予想に使う情報を操作できないか。
  例: 長距離では血統・スタミナを重く、短距離ではスピードを重く。

⚠ GBDTは木なので、距離・馬場・競馬場で内部的に既に条件分岐している。
  「条件ごとに特徴を変える」ことの利得は、すでにある程度モデルに入っている。
  それでも明示的に絞れば上がるのかを、特徴量グループの除去実験で測る。

やり方
  条件（長距離/短距離/道悪/多頭数）ごとに、特徴量グループを1つずつ「抜いて」
  学習し直し、その条件でのΔR²（市場を条件に入れたうえでの追加情報量）を比べる。
    抜いて悪化する → その条件でそのグループは効いている
    抜いて改善する → その条件ではノイズ。条件別に落とす価値がある
  walk-forwardは維持する（学習=検証年より前）。

実行: python feat_by_cond.py → feat_by_cond_result.csv
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
P = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
         num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)

GROUPS = {
    "血統": r"父系|母父系|血統|sire",
    "騎手厩舎": r"騎手|調教師|馬主",
    "スピード": r"速度|上り|タイム|上指",
    "脚質ペース": r"脚質|先行|差し|逃げ|位置|ペース",
    "距離適性": r"距離",
    "調教": r"chk_|調教",
}


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
    use = list(dict.fromkeys(["race_id", "馬名", "着順_num", "単勝オッズ", "人気",
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
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}個")

    CONDS = {
        "長距離2100+": D["距離"] >= 2100,
        "短距離-1400": D["距離"] <= 1400,
        "道悪": D["馬場状態_num"] >= 3,
        "多頭数16+": D["出走頭数"] >= 16,
    }
    for g, pat in GROUPS.items():
        log(f"  グループ {g}: {len([c for c in BASE if re.search(pat, c)])}列")

    rows = []
    for cl, sel in CONDS.items():
        S = D[sel].copy()
        log(f"\n=== {cl}  {len(S):,}頭 ===")
        variants = {"全特徴（基準）": BASE}
        for g, pat in GROUPS.items():
            drop = [c for c in BASE if re.search(pat, c)]
            if 5 <= len(drop) < len(BASE) - 20:
                variants[f"−{g}"] = [c for c in BASE if c not in drop]
        for name, cols in variants.items():
            S["_p"] = np.nan
            for ty in YEARS:
                tr, te = S[S["年"] < ty], S["年"] == ty
                if len(tr) < 3000 or te.sum() < 300:
                    continue
                m = lgb.train(P, lgb.Dataset(tr[cols], tr["win"]), num_boost_round=400)
                S.loc[te, "_p"] = m.predict(S.loc[te, cols])
            T = S[S["_p"].notna()]
            if len(T) < 1000:
                continue
            d = dr2(T["win"].values, T["m"].values, T["_p"].values)
            rows.append({"条件": cl, "構成": name, "頭数": len(T),
                         "特徴量数": len(cols), "ΔR²": round(d, 5)})
            log(f"  {name:<16}{len(cols):>4}列  ΔR² {d:+.5f}")
            pd.DataFrame(rows).to_csv("feat_by_cond_result.csv", index=False,
                                      encoding="utf-8-sig")

    R = pd.DataFrame(rows)
    log("\n" + "=" * 64)
    for cl in CONDS:
        t = R[R.条件 == cl]
        if t.empty:
            continue
        base = t[t.構成 == "全特徴（基準）"]["ΔR²"].iloc[0]
        best = t.sort_values("ΔR²", ascending=False).iloc[0]
        log(f"{cl:<14} 基準 {base:+.5f}  最良 {best['構成']:<12}{best['ΔR²']:+.5f}"
            + ("  ← 改善" if best["ΔR²"] > base else "  改善なし"))


if __name__ == "__main__":
    main()
