# -*- coding: utf-8 -*-
"""競馬場ごとにモデルを作り（＝場別の特徴量重み付け）、市場との乖離を狙えるか検証。

仮説: 全国一律モデルは場ごとの効き方（中山の内枠/東京の長い直線/阪神の坂など）を
      平均化して潰している。場別に学習すれば市場に上乗せできるのでは。

判定は「市場に情報を足せているか」で行う:
  ・LL増分 … log(市場確率)だけのロジスティック回帰に、場別モデル確率を足した時の
              対数尤度の伸び。+0.001未満なら実質ゼロ。
  ・乖離ベットROI … (モデル確率 - 市場確率)の上位を買った時の単勝/複勝ROI。
  ・再現性 … 2025を実開催日で前半/後半に割り、両方で100%超かを確認（偶然対策）。
全国モデル（同じ特徴量・同じ設定で全場まとめて学習）と必ず比較する。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

TY = 2025
JYO = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
       6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}
LGB = dict(objective="binary", learning_rate=0.05, num_leaves=31, n_estimators=500,
           min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def load():
    from market_free_model import FEATURE_COLS_MF
    # 単勝オッズは race_features に元から入っている（model_result.csvは2025のみで
    # 全期間の学習データが消えるため使わない）
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["odds"] = d["単勝オッズ"]
    d["年"] = d["race_id"].str[:4].astype(int)
    d["場"] = d["race_id"].str[4:6].astype(int)
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d = d.dropna(subset=["着", "odds"])
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["raw"] = 1 / d["odds"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    use = [c for c in FEATURE_COLS_MF if c in d.columns]
    return d, use


def ll_gain(sub, pcol):
    """市場のみ→市場＋モデル の対数尤度・AUCの増分。"""
    s = sub.dropna(subset=["q", pcol])
    if len(s) < 500 or s["win"].nunique() < 2:
        return None
    eps = 1e-6
    X0 = np.log(np.clip(s[["q"]], eps, 1)).values
    y = s["win"].values
    r0 = LogisticRegression(C=1e6).fit(X0, y)
    p0 = r0.predict_proba(X0)[:, 1]
    X1 = np.hstack([X0, np.log(np.clip(s[[pcol]], eps, 1)).values])
    r1 = LogisticRegression(C=1e6).fit(X1, y)
    p1 = r1.predict_proba(X1)[:, 1]
    return (-log_loss(y, p1)) - (-log_loss(y, p0)), roc_auc_score(y, p1) - roc_auc_score(y, p0)


def edge_roi(sub, pcol, frac=1 / 6):
    """乖離(p-q)上位を買った時の単勝ROIと、期間前半/後半の再現性。"""
    s = sub.dropna(subset=[pcol, "q"]).copy()
    if len(s) < 300:
        return None
    s["edge"] = s[pcol] - s["q"]
    th = s["edge"].quantile(1 - frac)
    p = s[s["edge"] >= th]
    if len(p) < 80:
        return None
    roi = (p["win"] * p["odds"]).sum() / len(p) * 100
    mid = s["dt"].median()
    h1 = p[p["dt"] <= mid]
    h2 = p[p["dt"] > mid]
    r1 = (h1["win"] * h1["odds"]).sum() / len(h1) * 100 if len(h1) > 30 else np.nan
    r2 = (h2["win"] * h2["odds"]).sum() / len(h2) * 100 if len(h2) > 30 else np.nan
    return roi, r1, r2, len(p)


def main():
    d, use = load()
    log(f"データ {len(d):,}行 / 特徴量{len(use)}列 / 検証年{TY}")
    tr_all = d[d["年"] <= TY - 1]
    te_all = d[d["年"] == TY].copy()

    log("\n【全国モデル】(比較の基準・全場まとめて学習)")
    g = lgb.LGBMClassifier(**LGB).fit(tr_all[use], tr_all["win"])
    te_all["p_all"] = g.predict_proba(te_all[use])[:, 1]
    te_all["p_all"] = te_all["p_all"] / te_all.groupby("race_id")["p_all"].transform("sum")
    r = ll_gain(te_all, "p_all")
    log(f"  市場への上乗せ: LL{r[0]:+.5f}  AUC{r[1]:+.4f}   ※+0.001未満は実質ゼロ")
    er = edge_roi(te_all, "p_all")
    log(f"  乖離ベット単勝ROI: {er[0]:.1f}%  (前半{er[1]:.1f}% / 後半{er[2]:.1f}% / n={er[3]})")

    log("\n【場別モデル】各場のデータだけで学習＝場ごとの特徴量重み付け")
    log(f"  {'場':<6}{'学習':>7}{'検証':>6}{'LL増分':>10}{'AUC増分':>9}"
        f"{'乖離ROI':>9}{'前半':>8}{'後半':>8}{'判定':>5}")
    rows = []
    for code, name in JYO.items():
        tr = tr_all[tr_all["場"] == code]
        te = te_all[te_all["場"] == code].copy()
        if len(tr) < 5000 or len(te) < 800:
            continue
        mdl = lgb.LGBMClassifier(**LGB).fit(tr[use], tr["win"])
        te["p_v"] = mdl.predict_proba(te[use])[:, 1]
        te["p_v"] = te["p_v"] / te.groupby("race_id")["p_v"].transform("sum")
        g1 = ll_gain(te, "p_v")
        e1 = edge_roi(te, "p_v")
        if g1 is None or e1 is None:
            continue
        ok = "○" if (e1[0] > 100 and e1[1] > 100 and e1[2] > 100) else (
            "△" if e1[0] > 100 else "")
        log(f"  {name:<6}{len(tr):7d}{len(te):6d}{g1[0]:+10.5f}{g1[1]:+9.4f}"
            f"{e1[0]:8.1f}%{e1[1]:7.1f}%{e1[2]:7.1f}%{ok:>5}")
        rows.append((name, g1[0], e1[0], e1[1], e1[2]))

    log("\n【まとめ】")
    if rows:
        best = max(rows, key=lambda x: x[2])
        log(f"  乖離ROI最良: {best[0]} {best[2]:.1f}% (前半{best[3]:.1f}/後半{best[4]:.1f})")
        pos = [r for r in rows if r[2] > 100 and r[3] > 100 and r[4] > 100]
        log(f"  前後半とも100%超の場: {[r[0] for r in pos] if pos else 'なし'}")
        log(f"  LL増分の最大: {max(r[1] for r in rows):+.5f}"
            f"（+0.001未満なら場別にしても市場に勝てていない）")


if __name__ == "__main__":
    main()
