# -*- coding: utf-8 -*-
"""本番用「市場の歪み補正モデル」を学習して保存する。

能力データは一切使わず、オッズとレース文脈だけから勝率を学習する。
これは予測モデルではなく“市場の歪みの地図”であり、用途は
「そのレースで買ってはいけない（過剰人気の）馬を動的に除外する」こと。

2025検証(<=2024学習)での効果:
  全馬の単勝ROI 72.25% → 補正係数の下位50%を除外して 82.34%（複勝 71.56%→81.40%）
  歪みは大穴に集中（補正係数: 30倍以下=0.99〜1.03 / 100-300倍=0.61 / 300倍超=0.20）
  ※逆方向（過小評価馬を買う）は期間を変えると再現しないため採用しない。

出力: debias_model.pkl  （keiba_predict.py が読み込んで使う）
使い方: python train_debias.py
"""
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["log_odds", "q", "人気", "人気率", "odds比", "q_top", "q_std",
         "出走頭数", "クラス_num", "距離", "is_turf", "枠番", "場"]


def build_market_features(df):
    """オッズとレース文脈だけの特徴を作る。予測時も学習時も同じ関数を使う。"""
    d = df.copy()
    for c in ["単勝オッズ", "人気", "出走頭数", "クラス_num", "距離", "is_turf", "枠番"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan
    if "場" not in d.columns:
        d["場"] = d["race_id"].astype(str).str[4:6].astype(int)
    if d["出走頭数"].isna().all():
        d["出走頭数"] = d.groupby("race_id")["単勝オッズ"].transform("size")
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["log_odds"] = np.log(d["単勝オッズ"].clip(lower=1.0))
    d["fav_odds"] = d.groupby("race_id")["単勝オッズ"].transform("min")
    d["odds比"] = d["単勝オッズ"] / d["fav_odds"]
    d["q_top"] = d.groupby("race_id")["q"].transform("max")
    d["q_std"] = d.groupby("race_id")["q"].transform("std")
    d["人気率"] = d["人気"] / d["出走頭数"]
    return d


def apply_debias(model, pdf):
    """予測用: pdfに 補正確率p_adj と 補正係数bias を付けて返す。
    bias < 1 は市場が過剰に買っている（＝買ってはいけない）馬。"""
    d = build_market_features(pdf)
    ok = d["単勝オッズ"].notna() & (d["単勝オッズ"] > 0)
    d["p_adj"] = np.nan
    if ok.sum() >= 2:
        p = model.predict_proba(d.loc[ok, FEATS])[:, 1]
        d.loc[ok, "p_adj"] = p
        s = d.groupby("race_id")["p_adj"].transform("sum")
        d["p_adj"] = d["p_adj"] / s.replace(0, np.nan)
    d["bias"] = d["p_adj"] / d["q"].clip(lower=1e-9)
    return d[["p_adj", "bias"]]


def main():
    d = pd.read_csv(os.path.join(BASE, "race_features.csv"), dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着", "単勝オッズ", "人気"])
    d = build_market_features(d)
    d["win"] = (d["着"] == 1).astype(float)
    d = d.dropna(subset=["q", "log_odds"])
    # 当年を学習に含めると、その年での評価が自己採点になり閾値もズレる
    # （全期間学習だと bias<0.90 が6倍の馬まで切ってしまった）。前年までで学習する。
    import datetime
    this_year = datetime.date.today().year
    d = d[d["race_id"].str[:4].astype(int) < this_year]
    print(f"学習データ: {len(d):,}頭 / {d['race_id'].nunique():,}レース "
          f"({d['race_id'].str[:4].min()}〜{d['race_id'].str[:4].max()})", flush=True)
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                           n_estimators=700, min_child_samples=100,
                           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                           verbose=-1, seed=42)
    m.fit(d[FEATS], d["win"])
    path = os.path.join(BASE, "debias_model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": m, "feats": FEATS, "format": "debias_v1"}, f)
    print(f"保存 → {path} ({os.path.getsize(path)/1024:.0f}KB)", flush=True)

    # 学習データ上での確認（参考値）
    d["p_adj"] = m.predict_proba(d[FEATS])[:, 1]
    d["p_adj"] = d["p_adj"] / d.groupby("race_id")["p_adj"].transform("sum")
    d["bias"] = d["p_adj"] / d["q"].clip(lower=1e-9)
    print("\nオッズ帯別の補正係数（学習データ・参考）", flush=True)
    d["帯"] = pd.cut(d["単勝オッズ"], [0, 5, 10, 30, 60, 100, 300, 99999],
                    labels=["〜5", "5-10", "10-30", "30-60", "60-100", "100-300", "300+"])
    for k, g in d.groupby("帯", observed=True):
        print(f"  {str(k):<10} n={len(g):6d}  補正係数 {g['bias'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
