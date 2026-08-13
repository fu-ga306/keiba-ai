# -*- coding: utf-8 -*-
"""条件別の専門モデルは、汎用モデルより強いのか（2026-08-13）

背景
  条件別にΔR²を測ったら、優位が最大11倍まで変わった。5年すべて正。
    クラス6 +0.0057 / 長距離2100-2400 +0.0025 / 馬場状態3 +0.0018 / 全体 +0.0005
  ならば条件ごとにモデルを分ければもっと強くなるのでは、という提案の検証。

⚠ 自明ではない。検体が1/20になるので、分散が増えて悪化する可能性が高い。
  GBDTは木なので、競馬場cd・距離で内部的に既に条件分岐している。
  「分ける」ことの利得は、既にある程度モデルに入っている。

やり方
  汎用モデルの予測は bet_cache_*.csv の p_win（walk-forward OOS）をそのまま使う。
  専門モデルだけを新たに訓練する。学習は同じ walk-forward（年<TY）で、
  ただし学習データを当該条件に限定する。
  同じテスト行の上で、市場を条件に入れたΔR²を比べる。

実行: python specialist.py → specialist_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from market_free_model import FEATURE_COLS_MF

EPS = 1e-6
YEARS = [2021, 2022, 2023, 2024, 2025]
P = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def dr2(y, market, model):
    """市場を条件に入れたうえでモデルが足す情報量。"""
    if y.sum() < 15:
        return np.nan
    out = []
    for X in (np.column_stack([logit(market)]),
              np.column_stack([logit(market), logit(model)])):
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        p = np.clip(lr.predict_proba(X)[:, 1], EPS, 1 - EPS)
        ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        b = y.mean()
        out.append(1 - ll / ((y * np.log(b) + (1 - y) * np.log(1 - b)).sum()))
    return out[1] - out[0]


def main():
    log("読み込み中...")
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(["race_id", "馬名", "着順_num", "人気", "単勝オッズ",
                              "距離", "クラス_num", "馬場状態_num", "出走頭数"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[D["着"].notna() & D["odds"].notna() & (D["odds"] > 0)].copy()
    D["win"] = (D["着"] == 1).astype(int)
    log(f"  {len(D):,}頭 / {D.race_id.nunique():,}レース")

    # 汎用モデルのOOS予測（既存の bet_cache をそのまま使う）
    G = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   [["race_id", "馬名", "p_win"]] for y in YEARS], ignore_index=True)
    D = D.merge(G, on=["race_id", "馬名"], how="left")

    SEGS = [("長距離 2100-2400m", (D["距離"] >= 2100) & (D["距離"] <= 2400)),
            ("長距離 2500m〜", D["距離"] >= 2500),
            ("重賞級 クラス6+", D["クラス_num"] >= 6),
            ("道悪 馬場状態3+", D["馬場状態_num"] >= 3),
            ("多頭数 17頭〜", D["出走頭数"] >= 17)]

    rows = []
    for lbl, sel in SEGS:
        S = D[sel].copy()
        log(f"\n=== {lbl} ===  全体{len(S):,}頭")
        S["spec"] = np.nan
        for ty in YEARS:
            tr = S[(S["年"] < ty) & (S["年"] >= 2019)]
            te = S["年"] == ty
            if len(tr) < 2000 or te.sum() < 200:
                continue
            m = lgb.train(P, lgb.Dataset(tr[BASE], tr["win"]), num_boost_round=600)
            S.loc[te, "spec"] = m.predict(S.loc[te, BASE])
            log(f"  {ty} 専門モデル学習 {len(tr):,}頭 → 検証 {int(te.sum()):,}頭")
        T = S[S["spec"].notna() & S["p_win"].notna()].copy()
        if len(T) < 500:
            log("  検体不足でスキップ")
            continue
        T["m"] = T.groupby("race_id")["odds"].transform(lambda s: (1 / s) / (1 / s).sum())
        y = T["win"].values
        g = dr2(y, T["m"].values, T["p_win"].values)
        s = dr2(y, T["m"].values, T["spec"].values)
        # 両方を入れたら（合議）どうなるか
        both = np.sqrt(np.clip(T["p_win"].values, EPS, 1) * np.clip(T["spec"].values, EPS, 1))
        b = dr2(y, T["m"].values, both)
        log(f"  → 汎用 {g:+.4f} / 専門 {s:+.4f} / 併用 {b:+.4f}   検証{len(T):,}頭")
        rows.append({"条件": lbl, "検証頭数": len(T), "汎用ΔR²": round(g, 4),
                     "専門ΔR²": round(s, 4), "併用ΔR²": round(b, 4),
                     "専門の優位": round(s - g, 4)})
        pd.DataFrame(rows).to_csv("specialist_result.csv", index=False,
                                  encoding="utf-8-sig")

    log("\n" + "=" * 66)
    r = pd.DataFrame(rows)
    log(r.to_string(index=False))
    log("\n保存 → specialist_result.csv")


if __name__ == "__main__":
    main()
