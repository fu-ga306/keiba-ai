# -*- coding: utf-8 -*-
"""市場を出発点にして「市場が外している分」だけを学習させる（2026-08-17）

なぜこれを試すか
  診断の結果、モデルが市場に足している情報はほぼ無かった。
    単一LGB      ΔR² 0.0005（Benter基準の 3.0%）
    本番アンサンブル ΔR² 0.0001（同 0.6%・2025年）
  10モデル束ねても変わらない。作り方を変えないと動かない。

  今のやり方は「市場を見ずにゼロから勝率を当てる」。その結果、モデルが学ぶのは
  ほとんど市場も知っていること（人気馬は強い）になり、足し算にならない。

  Benterのやり方は違う。市場のオッズを出発点（オフセット）に固定し、
  そこからのズレだけを学習させる。つまり「市場が何を外しているか」を直接学ぶ。
  最初から市場が知っていることを学び直さないので、上乗せだけが残る。

やること
  LightGBM の init_score に log(市場確率) を入れる。
  これで木は「市場の予測に対する補正項」だけを覚える。
  目的関数もレース単位にする（lambdarank）版も比べる。

比べるもの
  ① 現行      : 市場を見ずに binary で勝率を学習（bet_cache と同じ）
  ② 残差学習   : 市場を init_score に固定して binary
  ③ 残差＋順位 : 同上を lambdarank で

  どれも walk-forward（その年より前だけで学習）。
  評価は条件付きロジットの ΔR²（Benterと同じ物差し）。

実行: python resid_model.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import model_diag as M
from market_free_model import FEATURE_COLS_MF

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
P = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
PR = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3],
          learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
          verbose=-1, seed=42, lambdarank_truncation_level=20)


def log(m):
    print(m, flush=True)


def load():
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(["race_id", "馬名", "着順_num", "人気", "単勝オッズ"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[D["着"].notna() & (D.odds > 0) & D["年"].between(2019, 2025)].copy()
    D["win"] = (D["着"] == 1).astype(int)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy()
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    D["lq"] = np.log(D.q.clip(EPS))
    return D.reset_index(drop=True), BASE


def grouped(df):
    """lambdarank 用のグループ長。race_id が連続している前提で並べ替える。"""
    d = df.sort_values("race_id")
    return d, d.groupby("race_id", sort=False).size().values


def fit_predict(tr, te, cols, mode):
    if mode == "plain":
        m = lgb.train(P, lgb.Dataset(tr[cols], tr.win), num_boost_round=600)
        return m.predict(te[cols])
    if mode == "resid":
        ds = lgb.Dataset(tr[cols], tr.win, init_score=tr.lq.values)
        m = lgb.train(P, ds, num_boost_round=600)
        return m.predict(te[cols], raw_score=True) + te.lq.values
    if mode == "rank_resid":
        trs, gp = grouped(tr)
        ds = lgb.Dataset(trs[cols], trs.win, group=gp, init_score=trs.lq.values)
        m = lgb.train(PR, ds, num_boost_round=600)
        return m.predict(te[cols], raw_score=True) + te.lq.values
    raise ValueError(mode)


def dr2(te, score_col):
    """条件付きロジットで ΔR² を出す。score は log スケールの相対値でよい。"""
    d = te.copy()
    d["_rc"] = pd.factorize(d["race_id"])[0]
    l0 = M.null_ll(d)
    _, lm = M.clogit(d, ["lq"])
    _, lb = M.clogit(d, ["lq", score_col])
    return (1 - lb / l0) - (1 - lm / l0)


def main():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}列")
    log("評価はBenterと同じΔR²（市場だけのときからの改善）。基準 0.0178\n")

    res = {}
    for mode, lab in (("plain", "① 現行（市場を見ずに学習）"),
                      ("resid", "② 残差学習（市場を出発点に）"),
                      ("rank_resid", "③ 残差＋レース単位の順位学習")):
        vals = []
        for y in YEARS:
            tr, te = D[D.年 < y], D[D.年 == y].copy()
            if len(tr) < 5000 or te.empty:
                continue
            s = fit_predict(tr, te, BASE, mode)
            # レース内で相対化（定数ずれは条件付きロジットに影響しないが念のため）
            te["sc"] = s
            te["sc"] = te["sc"] - te.groupby("race_id")["sc"].transform("mean")
            vals.append((y, dr2(te, "sc")))
            log(f"  {lab}  {y}: ΔR²={vals[-1][1]:+.4f}")
        res[lab] = vals
        log("")

    log("=== まとめ ===")
    log(f"  {'作り方':<30}{'2021':>8}{'2022':>8}{'2023':>8}{'2024':>8}{'2025':>8}{'平均':>9}{'Benter比':>10}")
    for lab, vals in res.items():
        d = dict(vals)
        avg = np.mean([v for _, v in vals])
        log(f"  {lab:<30}" + "".join(f"{d.get(y,np.nan):>8.4f}" for y in YEARS)
            + f"{avg:>9.4f}{avg/0.0178*100:>9.1f}%")


if __name__ == "__main__":
    main()
