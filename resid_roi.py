# -*- coding: utf-8 -*-
"""残差学習モデルが回収率に変わるかを測る（2026-08-17）

前提（resid_model.py の結果）
  市場のlog確率を init_score に固定して「市場が外している分」だけを学習させると、
  ΔR² が 0.0004 → 0.0046 になった。Benter基準の 2.4% → 25.7%。5年すべてで改善。

  ただし ΔR² が上がっても回収率になるとは限らない。Benterの基準 0.0178 に対し
  まだ1/4なので、理屈の上では赤字のはず。実際どうなるかを測る。

やること
  ① walk-forward で残差モデルの勝率を出す（その年より前だけで学習）
  ② 予測を保存する（あとで買い方を変えて測り直せるように）
  ③ 単勝で、モデルと市場の食い違いの大きさ別に回収率を出す
  ④ 7分前オッズでの選択も模擬する（実際に買えるか）

⚠ このモデルは市場（オッズ）を入力に使う。従来のMFは市場を見ない設計だったので
  役割が違う。買い判断に使うなら、オッズ確定前には予測が出せない点に注意。

実行: python resid_roi.py → resid_pred.csv
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from market_free_model import FEATURE_COLS_MF

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
P = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def main():
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(["race_id", "馬名", "馬番", "着順_num", "人気",
                              "単勝オッズ", "is_turf", "距離"] + BASE))
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
    D["bn"] = pd.to_numeric(D["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}列\n")

    D["p"] = np.nan
    for y in YEARS:
        tr, te = D[D.年 < y], D.年 == y
        m = lgb.train(P, lgb.Dataset(tr[BASE], tr.win, init_score=tr.lq.values),
                      num_boost_round=600)
        raw = m.predict(D.loc[te, BASE], raw_score=True) + D.loc[te, "lq"].values
        D.loc[te, "p"] = raw
        log(f"  {y} 学習完了（学習{len(tr):,}頭）")
    d = D[D.p.notna()].copy()
    # レース内で softmax して確率にする
    d["e"] = np.exp(d.p - d.groupby("race_id")["p"].transform("max"))
    d["pm"] = d.e / d.groupby("race_id")["e"].transform("sum")
    d["EV"] = d.pm * d.odds
    d["gap"] = d.pm / d.q                      # 市場との食い違い（1より大なら我々が高評価）
    d.to_csv("resid_pred.csv", index=False, encoding="utf-8-sig",
             columns=["race_id", "馬名", "bn", "年", "人気", "odds", "着", "win",
                      "is_turf", "距離", "q", "pm", "EV", "gap"])
    log(f"\n→ resid_pred.csv（{len(d):,}頭）\n")

    def tab(title, groups):
        log(f"=== {title} ===")
        log(f"  {'区分':<20}{'点数':>9}{'的中率':>8}{'ROI':>8}"
            + "".join(f"{y:>7}" for y in YEARS))
        for lab, s in groups:
            if len(s) < 300:
                continue
            roi = (s.win * s.odds).sum() / len(s) * 100
            yr = [((g.win * g.odds).sum() / len(g) * 100 if len(g) > 50 else np.nan)
                  for y in YEARS for g in [s[s.年 == y]]]
            log(f"  {lab:<20}{len(s):>9,}{s.win.mean()*100:>7.1f}%{roi:>7.1f}%"
                + "".join(f"{v:>7.0f}" if np.isfinite(v) else f"{'--':>7}" for v in yr))
        log("")

    tab("単勝: モデルと市場の食い違い別", [
        (f"gap {lo}-{hi}", d[(d.gap >= lo) & (d.gap < hi)])
        for lo, hi in [(0, .8), (.8, 1.0), (1.0, 1.2), (1.2, 1.5),
                       (1.5, 2.0), (2.0, 3.0), (3.0, 99)]])

    tab("単勝: 期待値別", [
        (f"EV {lo}-{hi}", d[(d.EV >= lo) & (d.EV < hi)])
        for lo, hi in [(0, .8), (.8, .9), (.9, 1.0), (1.0, 1.1),
                       (1.1, 1.3), (1.3, 1.6), (1.6, 99)]])

    g = d.groupby("race_id")
    top = d.loc[g["pm"].idxmax()]
    ev = d.loc[g["EV"].idxmax()]
    gp = d.loc[g["gap"].idxmax()]
    tab("レースから1頭だけ選ぶ", [
        ("モデル1位", top), ("期待値最大", ev), ("食い違い最大", gp),
        ("モデル1位かつ1-10倍", top[top.odds < 10]),
        ("モデル1位かつ gap>1.1", top[top.gap > 1.1]),
        ("モデル1位かつ gap>1.2", top[top.gap > 1.2]),
    ])


if __name__ == "__main__":
    main()
