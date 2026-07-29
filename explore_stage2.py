# -*- coding: utf-8 -*-
"""二段構え: 市場の人気順を入力に、1番人気が負ける7割を拾いに行く専用モデル。

ユーザー着想:
  ①通常通り予想する
  ②その予想と市場の人気順、および実結果との乖離を使って、別の機械学習をさせる
  1番人気の勝率は約32%＝7割は外れている。その7割を拾いに行きたい。

現行モデルの問題: 全馬を等しく扱うため、予測能力の大半が「1番人気が勝つか」に
使われる。勝率8%の非1番人気は「その他大勢」として粗く扱われている。

3つの形で検証（すべて<=2024学習/2025検証・2025は前後半でも確認）:
  A. 非1番人気だけで学習した専用モデル（対象を絞って капацитеを集中）
  B. 「1番人気に先着するか」を目的変数にする（正例率が約50%で学習しやすい）
  C. 二段構え: レース単位で「1番人気が飛ぶ確率」を出し、飛ぶと読んだレースで
     Aのモデルの推奨を買う
入力には市場情報(人気・オッズ・市場想定確率)と主モデル予測、その乖離を含める。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from market_free_model import FEATURE_COLS_MF

LGBP = dict(objective="binary", learning_rate=0.03, num_leaves=31, n_estimators=600,
            min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着"])
    for c in ["単勝オッズ", "人気", "馬番", "出走頭数"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["単勝オッズ", "人気", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["年"] = d["race_id"].str[:4].astype(int)
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    # 市場情報
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["fav_odds"] = d.groupby("race_id")["単勝オッズ"].transform("min")
    d["odds比"] = d["単勝オッズ"] / d["fav_odds"]
    d["q_top"] = d.groupby("race_id")["q"].transform("max")
    d["人気率"] = d["人気"] / d["出走頭数"]
    # 1番人気の着順（レース単位）
    fav = d[d["人気"] == 1].groupby("race_id")["着"].min()
    d["fav着"] = d["race_id"].map(fav)
    d["先着"] = (d["着"] < d["fav着"]).astype(float)      # 1番人気に先着したか
    d["fav負け"] = (d["fav着"] > 1).astype(float)          # そのレースで1番人気が負けたか
    # 主モデルの予測（2025のみ存在）を乖離特徴に使う
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測スコア"]].rename(columns={"予測スコア": "p3"})
    d = d.merge(p3, on=["race_id", "馬名"], how="left")
    d["p3"] = pd.to_numeric(d["p3"], errors="coerce")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    return d, tan, fuk


def roi(s, t):
    if not len(s):
        return float("nan")
    return sum(t.get((r.race_id, r.bn), 0) for r in s.itertuples()) / len(s)


def report(s, tan, fuk, label, mid):
    if len(s) < 80:
        return
    a = roi(s, tan)
    h1 = roi(s[s["dt"] <= mid], tan)
    h2 = roi(s[s["dt"] > mid], tan)
    mk = "◎" if min(h1, h2) >= 100 else ("○" if min(h1, h2) >= 90 else "")
    log(f"  {label:<34}{len(s):6,}  勝率{s['win'].mean()*100:5.1f}%  "
        f"平均{s['単勝オッズ'].mean():5.1f}倍  単勝{a:6.1f}% (前{h1:5.1f}/後{h2:5.1f}) "
        f"複勝{roi(s, fuk):6.1f}%{mk}")


def main():
    d, tan, fuk = load()
    base = [c for c in FEATURE_COLS_MF if c in d.columns]
    mkt = ["人気", "単勝オッズ", "q", "odds比", "q_top", "人気率", "出走頭数"]
    # FEATURE_COLS_MFに既に含まれる列(出走頭数など)との重複を除く
    feats = list(dict.fromkeys(base + mkt))
    tr = d[d["年"] <= 2024]
    te = d[d["年"] == 2025].copy()
    mid = te["dt"].quantile(0.5)
    log(f"学習 {len(tr):,} / 検証 {len(te):,} / 特徴 {len(feats)}（能力＋市場）")
    log(f"1番人気の勝率 {te[te['人気']==1]['win'].mean()*100:.1f}% "
        f"→ {100-te[te['人気']==1]['win'].mean()*100:.1f}%が『拾いに行く』対象\n")

    log("=" * 108)
    log("【A】非1番人気だけで学習した専用モデル（対象を絞って能力を集中）")
    log("=" * 108)
    trA = tr[tr["人気"] >= 2]
    teA = te[te["人気"] >= 2].copy()
    mA = lgb.LGBMClassifier(**LGBP).fit(trA[feats], trA["win"])
    teA["pA"] = mA.predict_proba(teA[feats])[:, 1]
    teA["rkA"] = teA.groupby("race_id")["pA"].rank(ascending=False, method="min")
    log(f"  AUC(非1番人気内で勝ち馬を当てる) {roc_auc_score(teA['win'], teA['pA']):.4f}")
    report(teA, tan, fuk, "非1番人気すべて(基準)", mid)
    report(teA[teA["rkA"] == 1], tan, fuk, "専用モデルの1位", mid)
    for q in (0.98, 0.95, 0.90):
        th = teA["pA"].quantile(q)
        report(teA[teA["pA"] >= th], tan, fuk, f"専用モデル 上位{(1-q)*100:.0f}%", mid)

    log("\n" + "=" * 108)
    log("【B】『1番人気に先着するか』を目的にする（正例率が高く学習しやすい）")
    log("=" * 108)
    trB = tr[tr["人気"] >= 2]
    log(f"  正例率(1番人気に先着) {trB['先着'].mean()*100:.1f}%  ※勝率8%より学習しやすい")
    mB = lgb.LGBMClassifier(**LGBP).fit(trB[feats], trB["先着"])
    teA["pB"] = mB.predict_proba(teA[feats])[:, 1]
    teA["rkB"] = teA.groupby("race_id")["pB"].rank(ascending=False, method="min")
    log(f"  AUC(先着予測) {roc_auc_score(teA['先着'], teA['pB']):.4f}")
    report(teA[teA["rkB"] == 1], tan, fuk, "先着モデルの1位", mid)
    for q in (0.98, 0.95, 0.90):
        th = teA["pB"].quantile(q)
        report(teA[teA["pB"] >= th], tan, fuk, f"先着モデル 上位{(1-q)*100:.0f}%", mid)

    log("\n" + "=" * 108)
    log("【C】二段構え: 1番人気が飛ぶレースを選び、そこで専用モデルの推奨を買う")
    log("=" * 108)
    trF = tr[tr["人気"] == 1]
    teF = te[te["人気"] == 1].copy()
    mF = lgb.LGBMClassifier(**LGBP).fit(trF[feats], trF["fav負け"])
    teF["pF"] = mF.predict_proba(teF[feats])[:, 1]
    log(f"  AUC(1番人気が負けるか) {roc_auc_score(teF['fav負け'], teF['pF']):.4f}"
        f"  実際に負けた率 {teF['fav負け'].mean()*100:.1f}%")
    for q in (0.9, 0.8, 0.7, 0.5):
        th = teF["pF"].quantile(q)
        rids = set(teF[teF["pF"] >= th]["race_id"])
        sub = teA[teA["race_id"].isin(rids)]
        act = teF[teF["pF"] >= th]["fav負け"].mean() * 100
        log(f"  ― 飛ぶ確率 上位{(1-q)*100:.0f}%のレース({len(rids)}R・実際に飛んだ率{act:.1f}%) ―")
        report(sub[sub["rkA"] == 1], tan, fuk, "  そこで専用モデル1位", mid)
        report(sub[sub["rkB"] == 1], tan, fuk, "  そこで先着モデル1位", mid)


if __name__ == "__main__":
    main()
