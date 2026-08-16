# -*- coding: utf-8 -*-
"""残差モデルを学習して本番用に保存する（2026-08-17）

このモデルは何か
  市場のオッズを出発点（LightGBM の init_score）に固定し、そこからのズレだけを
  学習する。つまり「市場が何を見落としているか」を直接覚える。
  従来のMFモデル（市場を見ずにゼロから勝率を当てる）とは役割が違う。

  検証値（walk-forward 5年・resid_robust.py）:
    ΔR² 0.0046（Benter基準の25.7%・従来の11.5倍）
    gap>=2.0 の1頭を単勝: 2,752点 的中260 ROI 135.9% 95%区間[100.9, 180.1]
    順列検定 p=0.0000（400回中0回）

使い方
  検証用（walk-forward・過去の成績を測る）:
      python train_resid.py backtest
      → resid_pred.csv（各年を、その年より前だけで学習して予測）
  本番用（全期間で学習してデプロイ）:
      python train_resid.py
      → model_resid.pkl

⚠ 本番用は全期間で学習するので、過去の成績を測るのには使えない（in-sample）。
  成績を語るときは必ず backtest 側の数字を使うこと。
⚠ 予測にはオッズが要る。オッズが出るまで予測できない。
⚠ 毎年（できれば毎月）学習し直すこと。2023年までで固定して2025年を当てると
  70%まで落ちた。効果が古くなる可能性がある。

実行: python train_resid.py [backtest]
"""
import pickle
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from market_free_model import FEATURE_COLS_MF

EPS = 1e-9
GAP_MIN = 2.0            # 事前登録した買い判断のしきい値
N_ROUNDS = 600
SEEDS = [42, 7, 123]     # 複数シードの平均を使う（1シードだと12.3pt振れる）
BT_YEARS = [2021, 2022, 2023, 2024, 2025]


def log(m):
    print(m, flush=True)


def params(seed):
    return dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed)


def market_prob(df):
    """オッズからレース内の市場確率を作る。控除率は正規化で消える。"""
    inv = 1.0 / pd.to_numeric(df["単勝オッズ"], errors="coerce").clip(lower=1.01)
    return inv / inv.groupby(df["race_id"]).transform("sum")


def load(train_only=True):
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(["race_id", "馬名", "馬番", "着順_num", "人気",
                              "単勝オッズ"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[(D.odds > 0)].copy()
    if train_only:
        D = D[D["着"].notna()].copy()
    D["win"] = (D["着"] == 1).astype(float)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy().reset_index(drop=True)
    D["q"] = market_prob(D)
    D["lq"] = np.log(D.q.clip(EPS))
    return D, BASE


def fit(tr, BASE, rounds=N_ROUNDS):
    """複数シードで学習して平均する。1シードだと結果が12.3pt振れるため。"""
    ms = []
    for sd in SEEDS:
        ms.append(lgb.train(params(sd),
                            lgb.Dataset(tr[BASE], tr.win, init_score=tr.lq.values),
                            num_boost_round=rounds))
    return ms


def raw_score(models, X):
    return np.mean([m.predict(X, raw_score=True) for m in models], axis=0)


def add_gap(df, f):
    """特徴量ぶんのスコア f から、レース内の予測確率と gap を作る。

    p ∝ exp(f) × 市場確率。gap = p / 市場確率。
    レース内の順位は f だけで決まるのでオッズに依存しない（スリッページ耐性）。
    """
    d = df.copy()
    d["f"] = f
    sc = d.f + d.lq
    e = np.exp(sc - sc.groupby(d.race_id).transform("max"))
    d["p"] = e / e.groupby(d.race_id).transform("sum")
    d["gap"] = d.p / d.q
    d["EV"] = d.p * d.odds
    return d


def backtest():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}列")
    log(f"シード {SEEDS} の平均・{N_ROUNDS}回\n")
    out = []
    for y in BT_YEARS:
        tr, te = D[D.年 < y], D[D.年 == y]
        if len(tr) < 5000 or te.empty:
            continue
        ms = fit(tr, BASE)
        out.append(add_gap(te, raw_score(ms, te[BASE])))
        log(f"  {y} 完了（学習{len(tr):,}頭 → 検証{len(te):,}頭）")
    d = pd.concat(out)
    d.to_csv("resid_pred.csv", index=False, encoding="utf-8-sig",
             columns=["race_id", "馬名", "馬番", "年", "人気", "odds", "着", "win",
                      "q", "p", "gap", "EV"])
    log(f"\n→ resid_pred.csv（{len(d):,}頭）")

    sel = d.loc[d.groupby("race_id")["gap"].idxmax()]
    buy = sel[sel.gap >= GAP_MIN]
    rng = np.random.default_rng(20260817)
    v = (buy.win * buy.odds * 100).values
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(4000)])
    log(f"\n=== 事前登録の買い方（gap>={GAP_MIN} の1頭を単勝1点）===")
    log(f"  {len(buy):,}点  的中{int(buy.win.sum())}  ROI {v.mean():.1f}%"
        f"  95%区間 [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]")
    log("  年別: " + "  ".join(
        f"{y}:{(g.win*g.odds).sum()/len(g)*100:.0f}%" for y, g in buy.groupby("年")))
    log(f"  1年あたり {len(buy)/len(BT_YEARS):.0f}点（1日36レースなら"
        f" {len(buy)/d.race_id.nunique()*36:.1f}点）")


def deploy():
    D, BASE = load()
    log(f"全期間で学習: {len(D):,}頭 / {D.race_id.nunique():,}レース")
    log(f"  年: {D.年.min()}〜{D.年.max()}  シード{SEEDS}・{N_ROUNDS}回")
    ms = fit(D, BASE)
    with open("model_resid.pkl", "wb") as fh:
        pickle.dump({"models": ms, "use_cols": BASE, "seeds": SEEDS,
                     "rounds": N_ROUNDS, "gap_min": GAP_MIN,
                     "years": [int(D.年.min()), int(D.年.max())], "n": len(D),
                     "format": "resid_v1"}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log("→ model_resid.pkl")
    log("\n⚠ このpklは全期間で学習しているので、過去の成績を測るのには使えない。")
    log("  成績は `python train_resid.py backtest` の数字を使うこと。")


if __name__ == "__main__":
    (backtest if "backtest" in sys.argv else deploy)()
