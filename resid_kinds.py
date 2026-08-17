# -*- coding: utf-8 -*-
"""残差モデルで券種を広げ、レースごとに最適な買い目を選ぶ（2026-08-17）

いまの状態
  単勝のみ。gap>=2.0 の1頭を1点。5年通算157.1%（的中203本）だが、
  年別は 178/92/227/266/86 で2年が100%割れ。点数が少なく標本が足りない。

  券種を増やせば点数が増え、標本も増える。ただし控除率は券種で違う。
    単勝・複勝 20% ／ 馬連・ワイド 22.5% ／ 馬単・三連複 25% ／ 三連単 27.5%
  控除率が高いほど不利なので、増やせば良いというものではない。

事前登録（ROIを見る前に固定。あとから券種も条件も足さない）
  ① 残差モデルを3つ学習する（1着 / 2着以内 / 3着以内）。作り方は勝率と同じで、
     市場確率を出発点にしてズレだけを学ぶ。市場側もそれぞれの目標に合わせる
     （2着以内・3着以内の市場確率はHarvilleでオッズから作る）。
  ② 各券種の当たる確率を、モデルの確率から組み立てる。
     単勝  P(1着)
     複勝  P(3着以内)
     馬連  P(2頭が1-2着・順不同)
     ワイド P(2頭が3着以内)
     馬単  P(A→Bの順で1-2着)
     いずれもHarville（1着を決めたら残りで2着を決める）で組む。
  ③ 期待値 = 当たる確率 × オッズ。オッズは単勝オッズからHarvilleで推定する
     （馬連などの実オッズは事前に持っていないため）。
  ④ 買うのは「期待値がしきい値を超えた券種のうち、期待値が最大のもの1点」。
     しきい値は単勝で決めた gap>=2.0 に相当する水準として、
     **期待値 >= 2.0** に固定する（券種によらず同じ数字を使う）。

  ⚠ 券種ごとにしきい値を変えると、それは探索になる。同じ数字を使う。
  ⚠ 三連複・三連単は入れない。控除率25-27.5%で、過去の検証でも優位ゼロだった。

判定
  5年通算・年別・順列検定・95%区間を出す。単勝のみの現行を上回らなければ
  広げない。

実行: python resid_kinds.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from market_free_model import FEATURE_COLS_MF

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
SEEDS = [42, 7, 123]
N_ROUNDS = 600
EV_MIN = 2.0                 # 事前登録。券種によらず同じ
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def params(seed):
    return dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed)


def harville_top(q, k):
    """市場の勝率 q から「k着以内に入る確率」をHarvilleで作る。"""
    q = np.clip(q, 1e-9, 1 - 1e-9)
    r = q / (1 - q)
    p2 = q * (r.sum() - r)
    if k == 2:
        return np.clip(q + p2, 1e-6, 1.0)
    den = 1 - q[:, None] - q[None, :]
    M = (q[:, None] * q[None, :]) / ((1 - q)[:, None] * np.clip(den, 1e-9, None))
    np.fill_diagonal(M, 0.0)
    p3 = q * (M.sum() - M.sum(1) - M.sum(0))
    return np.clip(q + p2 + p3, 1e-6, 1.0)


def load():
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
    D = D[D["着"].notna() & (D.odds > 0)].copy()
    D["win"] = (D["着"] == 1).astype(float)
    D["t2"] = (D["着"] <= 2).astype(float)
    D["t3"] = (D["着"] <= 3).astype(float)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy().reset_index(drop=True)
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    # 2着以内・3着以内の市場確率（Harville）
    for k, col in ((2, "q2"), (3, "q3")):
        vals = np.empty(len(D))
        for _, idx in D.groupby("race_id", sort=False).indices.items():
            vals[idx] = harville_top(D.q.values[idx], k)
        D[col] = vals
    for a, b in (("q", "lq"), ("q2", "lq2"), ("q3", "lq3")):
        D[b] = np.log(D[a].clip(EPS))
    D["bn"] = pd.to_numeric(D["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    return D, BASE


def fit_predict(D, BASE, tgt, off):
    """walk-forward で残差モデルを学習し、レース内で正規化した確率を返す。"""
    out = pd.Series(np.nan, index=D.index)
    for y in YEARS:
        tr, te = D[D.年 < y], D.年 == y
        if len(tr) < 5000 or not te.any():
            continue
        f = np.mean([lgb.train(params(sd),
                               lgb.Dataset(tr[BASE], tr[tgt], init_score=tr[off].values),
                               num_boost_round=N_ROUNDS).predict(D.loc[te, BASE],
                                                                 raw_score=True)
                     for sd in SEEDS], axis=0)
        out[te] = f + D.loc[te, off].values
    return out


def main():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}列")
    log(f"事前登録: 期待値 >= {EV_MIN} の券種のうち最大のもの1点。券種共通のしきい値。")
    log("券種: 単勝 / 複勝 / 馬連 / ワイド / 馬単（三連系は控除率が高いので入れない）\n")

    for tgt, off, col in (("win", "lq", "s1"), ("t2", "lq2", "s2"), ("t3", "lq3", "s3")):
        D[col] = fit_predict(D, BASE, tgt, off)
        log(f"  {tgt} 学習完了")
    d = D[D.s1.notna()].copy()
    # レース内で正規化（1着確率は合計1、2着以内は合計2、3着以内は合計3）
    for col, pc, tot in (("s1", "p1", 1.0), ("s2", "p2", 2.0), ("s3", "p3", 3.0)):
        e = np.exp(d[col] - d.groupby("race_id")[col].transform("max"))
        d[pc] = (e / e.groupby(d.race_id).transform("sum") * tot).clip(1e-6, 0.999)
    d.to_csv("resid_kinds_pred.csv", index=False, encoding="utf-8-sig",
             columns=["race_id", "馬名", "bn", "年", "人気", "odds", "着", "win",
                      "q", "q2", "q3", "p1", "p2", "p3"])
    log(f"\n→ resid_kinds_pred.csv（{len(d):,}頭）")
    log("\n※ 各券種の期待値計算と回収率は resid_kinds_eval.py で行う")


if __name__ == "__main__":
    main()
