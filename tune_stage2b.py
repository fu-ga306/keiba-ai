# -*- coding: utf-8 -*-
"""非1番人気専用モデルの改善案を検証（2024検証の前に、伸ばせる軸を洗う）。

現状の最良: 全特徴(能力+市場+血統+残差)・目的=勝つか・上位1%購入
            → 単勝100.0%(前96.4/後102.6) 勝率33.9% n=401

まだ試していない改善軸:
  E. 期待値で買う   … 予測確率×オッズ が閾値超のものを買う（確率だけで選ぶより
                      配当を考慮でき、同じ勝率でも高配当を拾える）
  F. シード平均     … 複数シードの平均で予測を安定させる（1シードは偶然に弱い）
  G. オッズ帯の限定 … 極端な人気薄を除く/含める
  H. 券種          … 単勝と複勝のどちらで運用するのが良いか
  I. レース単位     … 1レース1点に制限（同一レースの重複購入を避ける）
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb

from tune_stage2 import load, ABILITY, MARKET, BLOOD, RESID
import tune_stage2 as T
from market_free_model import FEATURE_COLS_MF

P = dict(objective="binary", learning_rate=0.03, num_leaves=31, n_estimators=600,
         min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, verbose=-1)


def log(m):
    print(m, flush=True)


def score(s, tan, fuk, mid, label):
    if len(s) < 80:
        log(f"  {label:<38} サンプル不足({len(s)})")
        return

    def roi(x, tbl):
        return sum(tbl.get((r.race_id, r.bn), 0) for r in x.itertuples()) / len(x)

    a, f = roi(s, tan), roi(s, fuk)
    h1, h2 = roi(s[s["dt"] <= mid], tan), roi(s[s["dt"] > mid], tan)
    mk = "◎" if min(h1, h2) >= 100 else ("○" if min(h1, h2) >= 95 else "")
    log(f"  {label:<38}{len(s):5,} 勝率{s['win'].mean()*100:5.1f}% "
        f"平均{s['単勝オッズ'].mean():5.1f}倍 単勝{a:6.1f}%(前{h1:6.1f}/後{h2:6.1f}) "
        f"複勝{f:6.1f}%{mk}")


def main():
    d, tan, fuk = load()
    ability = [c for c in FEATURE_COLS_MF if c in d.columns]
    feats = [c for c in dict.fromkeys(ability + MARKET + BLOOD + RESID)
             if c in d.columns]
    tr = d[(d["年"] <= 2024) & (d["人気"] >= 2)]
    te = d[(d["年"] == 2025) & (d["人気"] >= 2)].copy()
    mid = te["dt"].quantile(0.5)
    log(f"学習 {len(tr):,} / 検証 {len(te):,} / 特徴 {len(feats)}")

    log("\n【F】シード平均で安定化（1シードは偶然に弱い）")
    preds = []
    for sd in (42, 7, 123, 2024, 99):
        m = lgb.LGBMClassifier(**P, seed=sd).fit(tr[feats], tr["win"])
        preds.append(m.predict_proba(te[feats])[:, 1])
    te["p1"] = preds[0]
    te["p"] = np.mean(preds, axis=0)
    for col, nm in [("p1", "1シード"), ("p", "5シード平均")]:
        for frac in (0.01, 0.02, 0.03):
            th = te[col].quantile(1 - frac)
            score(te[te[col] >= th], tan, fuk, mid, f"{nm} 上位{frac*100:.0f}%")

    log("\n【E】期待値で買う（予測確率×オッズ）")
    te["EV"] = te["p"] * te["単勝オッズ"]
    for lo in (0.9, 1.0, 1.1, 1.2, 1.4):
        score(te[te["EV"] >= lo], tan, fuk, mid, f"期待値 ≥ {lo:.1f}")
    for frac in (0.01, 0.02, 0.03):
        th = te["EV"].quantile(1 - frac)
        score(te[te["EV"] >= th], tan, fuk, mid, f"期待値 上位{frac*100:.0f}%")

    log("\n【G】オッズ帯の限定（上位2%の中で）")
    th = te["p"].quantile(0.98)
    top = te[te["p"] >= th]
    for lo, hi, nm in [(0, 5, "〜5倍"), (5, 10, "5-10倍"), (10, 20, "10-20倍"),
                       (0, 10, "〜10倍"), (3, 15, "3-15倍")]:
        score(top[(top["単勝オッズ"] >= lo) & (top["単勝オッズ"] < hi)],
              tan, fuk, mid, f"上位2% × {nm}")

    log("\n【I】1レース1点に制限（同一レースの重複を避ける）")
    for frac in (0.02, 0.03, 0.05):
        th = te["p"].quantile(1 - frac)
        s = te[te["p"] >= th]
        one = s.sort_values("p", ascending=False).groupby("race_id").head(1)
        score(one, tan, fuk, mid, f"上位{frac*100:.0f}% → 各レース最上位のみ")

    log("\n【H】券種の比較（上位1-3%）")
    for frac in (0.01, 0.02, 0.03):
        th = te["p"].quantile(1 - frac)
        s = te[te["p"] >= th]
        log(f"  上位{frac*100:.0f}%: 単勝と複勝の比較は各行の右2列を参照"
            f"（複勝は的中率{s['fuku'].mean()*100:.1f}%）")


if __name__ == "__main__":
    main()
