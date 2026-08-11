# -*- coding: utf-8 -*-
"""市場確率を説明変数に入れた2次元較正器を作る（購入判定専用）。

なぜ必要か（2026-08-12）
  5年 walk-forward OOF 207,518頭で、EVと実払戻の順位相関が5年とも負だった。
  EVが高い馬ほど損をする（オプティマイザの呪い）。
  EV = p × オッズ なので、オッズが大きい側でpが過大だとEVは二重に膨らみ、
  そこだけを選ぶと誤差の上側の裾を集めることになる。
  本番ルール該当馬は 予測18.0% / 実勝率6.8%、実効EVは0.95で赤字が確定していた。

  1次元Isotonic（mf_calibrator.pkl）は全馬まとめた較正なので、これを直せない。
  全体のECEは0.0052と良いのに、選んだ馬だけ外れる、という構造だった。

やり方
  バケット分割は大穴帯でサンプルが薄く階段状になるので使わない。
  ロジスティック回帰で連続的に較正する。

    入力: logit(MF勝率), logit(市場確率), その交互作用
    出力: 実勝率

  市場確率はレース内で正規化した 1/オッズ（控除率を抜く）。
  市場と割れるほどモデル側を割り引く、という関数を学習する。

⚠ この較正器を印・★・順位・S〜D評価に使ってはいけない。
  市場に約9倍の重みが付くため、MF順位が人気順のコピーになる（順位相関0.9924）。
  5年で★が4,507頭→19頭に消える。市場フリーという設計そのものが壊れる。
  用途は「購入判定と実効EVの表示」だけ。

⚠ bet_cache の c_win_n で学習してはいけない。あれは prep_cache.py が独自に
  訓練した別モデルの出力で、本番MFとは分布が違う（中央値 0.0409 vs 0.0788）。
  logit を入力に取る以上、必ず本番MFのOOS出力で学習すること。

入力: model_mf_result.csv（train_mf_v2 backtestモードの正直なOOS出力）
出力: mf_calib2d.pkl
実行: python build_calib2d.py
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "model_mf_result.csv")
OUT = os.path.join(BASE_DIR, "mf_calib2d.pkl")
EPS = 1e-6


def log(m):
    print(m, flush=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def design(model_p, market_p):
    lp, lm = logit(model_p), logit(market_p)
    return np.column_stack([lp, lm, lp * lm])


def main():
    if not os.path.exists(SRC):
        log(f"{os.path.basename(SRC)} がありません。"
            "train_mf_v2.py を backtest モードで実行してください。")
        return 1
    d = pd.read_csv(SRC, dtype={"race_id": str}, low_memory=False)
    d = d.dropna(subset=["着順_num", "MF勝率", "単勝オッズ"])
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d = d[d["odds"] > 0].copy()
    d["win"] = (pd.to_numeric(d["着順_num"], errors="coerce") == 1).astype(int)
    # 市場確率: 1/オッズをレース内で正規化して控除率を抜く
    d["m"] = d.groupby("race_id")["odds"].transform(lambda s: (1 / s) / (1 / s).sum())
    # MF勝率もレース内で正規化してから較正する（本番の扱いと揃える）
    d["p"] = d.groupby("race_id")["MF勝率"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0)
    d = d[(d.p > 0) & (d.m > 0)]
    yr = sorted(d["race_id"].str[:4].unique())
    log(f"学習データ {len(d):,}頭 / {d.race_id.nunique():,}レース  年 {yr[0]}〜{yr[-1]}")

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(design(d.p.values, d.m.values), d.win.values)
    c = lr.coef_[0]
    log(f"\n  係数  モデル {c[0]:+.4f}   市場 {c[1]:+.4f}   交互作用 {c[2]:+.4f}"
        f"   切片 {lr.intercept_[0]:+.4f}")
    log(f"  → 市場にモデルの約 {abs(c[1]/c[0]):.1f} 倍の重み")

    # 効果の確認（学習データ上。OOSの厳密評価は calib2d.py が5年で済ませている）
    d["p2"] = lr.predict_proba(design(d.p.values, d.m.values))[:, 1]
    d["p2n"] = d.groupby("race_id")["p2"].transform(lambda s: s / s.sum())
    d["ev1"], d["ev2"] = d.p * d.odds, d.p2n * d.odds
    from scipy import stats
    log(f"\n  EVと実払戻の順位相関  較正前 {stats.spearmanr(d.ev1, d.win*d.odds).correlation:+.4f}"
        f"  → 2次元 {stats.spearmanr(d.ev2, d.win*d.odds).correlation:+.4f}")
    log(f"\n  {'EV帯':>10} {'頭数':>7} {'予測':>7} {'実勝率':>7} {'比':>6}")
    for lo, hi in [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 99)]:
        s = d[(d.ev2 >= lo) & (d.ev2 < hi)]
        if len(s) < 100:
            continue
        log(f"  {lo:.1f}-{hi:<5.1f} {len(s):>7,} {s.p2n.mean():>7.4f} "
            f"{s.win.mean():>7.4f} {s.win.mean()/s.p2n.mean():>6.2f}")
    for th in (1.0, 1.7, 2.2):
        n = int((d.ev2 >= th).sum())
        log(f"    実効EV>={th}: {n:,}頭 ({n/len(d)*100:.2f}%)")

    with open(OUT, "wb") as fh:
        pickle.dump({"coef": c, "intercept": float(lr.intercept_[0]),
                     "src": os.path.basename(SRC), "years": [yr[0], yr[-1]],
                     "n": int(len(d))}, fh)
    log(f"\n保存 → {os.path.basename(OUT)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
