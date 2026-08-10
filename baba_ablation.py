# -*- coding: utf-8 -*-
"""クッション値・含水率を足すと精度と回収率が上がるかを検証する。

背景（2026-08-10）
  モデルが持つ馬場情報は「良/稍重/重/不良」の4段階だけだった。同じ「良」でも
  クッション値7.5と10.5では走破時計が別物になる。JRAが2020年から公表しており
  fetch_baba.py で取得済みだったのに、race_features.csv に結合されておらず
  モデルは一度も見ていなかった。

  これまで失敗した9つの試み（速度指数・血統・Elo・調教など）は全て
  「馬個体の過去成績の加工」だった。クッション値はレース当日の環境そのもので、
  質的に異なる情報。ただし過去の教訓どおり、2年で効いて5年で消える可能性がある。

やり方
  同じ学習設定で「あり」と「なし」を作り、2024年と2025年で比べる。
  クッション値が揃うのは2021年以降なので学習は2021年から。
    学習 2021〜(test-2) / 較正 (test-1) / 検証 test

⚠ 学習設定は比較用に軽くしてある（LGB2シード・win/place3のみ）。
   あり/なしの差だけを見る。絶対値を本番と比べないこと。

実行: python baba_ablation.py → baba_ablation_result.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from market_free_model import FEATURE_COLS_MF, add_race_rank_features
from features import BABA_COLS, add_baba_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7]
TIME_WEIGHT_MAX = 2.0
POS_W = {"win": 2.0, "place3": 1.5}
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05,
                  lambda_l2=0.05, verbose=-1, min_gain_to_split=0.05)
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
CASES = [(2024, 2021), (2025, 2021)]


def _mk(chaku, t):
    return (chaku == 1).astype(int) if t == "win" else (chaku <= 3).astype(int)


def _fit(tr, cols, target):
    X, y = tr[cols], _mk(tr["着順_num"], target)
    ymax, ymin = tr["年"].max(), tr["年"].min()
    tw = 1.0 + (tr["年"] - ymin) / max(ymax - ymin, 1) * (TIME_WEIGHT_MAX - 1.0)
    Xt, Xc, yt, yc, wt, _ = train_test_split(X, y, tw, test_size=0.2, random_state=42)
    w = np.where(yt == 1, POS_W[target], 1.0) * wt.values
    out = []
    for sd in SEEDS:
        p = dict(LGB_PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        b = lgb.LGBMClassifier(**p, n_estimators=5000)
        b.fit(Xt, yt, sample_weight=w,
              callbacks=[lgb.early_stopping(100, verbose=False)], eval_set=[(Xc, yc)])
        c = CalibratedClassifierCV(estimator=b, method="isotonic", cv=None)
        c.fit(Xc, yc)
        out.append(c)
    return out


def _pred(models, df, cols):
    return np.mean([m.predict_proba(df[cols])[:, 1] for m in models], axis=0)


def evaluate(te, win_p, p3_p):
    """カバー率（3着内を上位で拾えた割合）と、本番の買い方での回収率。"""
    d = te[["race_id", "着順_num", "単勝オッズ", "人気"]].copy()
    d["w"], d["f"] = win_p, p3_p
    d["p"] = d.groupby("race_id")["w"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0)
    d["mfr"] = d.groupby("race_id")["f"].rank(ascending=False)
    d["ev"] = d["p"] * pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["gap"] = pd.to_numeric(d["人気"], errors="coerce") - d["mfr"]
    # カバー率: 3着以内の馬をモデル上位3頭で何頭拾えたか
    top3 = d[d.mfr <= 3]
    cover = (top3["着順_num"] <= 3).sum() / max((d["着順_num"] <= 3).sum(), 1) * 100
    od = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    hit = ((d.gap >= GAP_MIN) & (od <= ODDS_MAX) &
           (((d.mfr == 1) & (d.ev >= EV_TOP)) |
            (d.mfr.between(2, 5) & (d.ev >= EV_SUB))))
    cand = d[hit.fillna(False)]
    if cand.empty:
        return cover, 0, 0.0
    pick = cand.sort_values("ev", ascending=False).groupby("race_id").head(1)
    ret = np.where(pick["着順_num"] == 1,
                   pd.to_numeric(pick["単勝オッズ"], errors="coerce") * 100, 0).sum()
    return cover, len(pick), ret / (len(pick) * 100) * 100


def main():
    print("特徴量を読み込み中...", flush=True)
    df = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False)
    df = add_race_rank_features(df)
    df = add_baba_features(df)          # 既存ファイルに後付けする
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    base = [c for c in FEATURE_COLS_MF if c in df.columns]
    with_baba = base + [c for c in BABA_COLS if c in df.columns]
    print(f"  {len(df)}行 / なし{len(base)}列 / あり{len(with_baba)}列", flush=True)

    rows = []
    for test_y, from_y in CASES:
        tr = df[(df["年"] >= from_y) & (df["年"] <= test_y - 2)]
        ca = df[df["年"] == test_y - 1]
        te = df[df["年"] == test_y]
        print(f"\n=== test{test_y}: 学習{from_y}〜{test_y-2}({len(tr)}行) ===", flush=True)
        for lbl, cols in (("なし", base), ("あり", with_baba)):
            t0 = time.time()
            preds = {}
            for tgt in ("win", "place3"):
                m = _fit(tr, cols, tgt)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(_pred(m, ca, cols), _mk(ca["着順_num"], tgt))
                preds[tgt] = iso.predict(_pred(m, te, cols))
            cover, n, roi = evaluate(te, preds["win"], preds["place3"])
            rows.append({"テスト年": test_y, "馬場データ": lbl,
                         "カバー率": round(cover, 1), "買い点数": n,
                         "回収率": round(roi, 1), "秒": int(time.time() - t0)})
            print(f"  {lbl}: カバー率 {cover:.1f}%  回収率 {roi:.1f}%  "
                  f"({n}点 / {int(time.time()-t0)}秒)", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(BASE_DIR, "baba_ablation_result.csv"),
                index=False, encoding="utf-8-sig")

    r = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(r.to_string(index=False))
    print()
    for y in sorted(r["テスト年"].unique()):
        a = r[(r["テスト年"] == y) & (r["馬場データ"] == "あり")].iloc[0]
        b = r[(r["テスト年"] == y) & (r["馬場データ"] == "なし")].iloc[0]
        print(f"  {y}: カバー率 {a.カバー率 - b.カバー率:+.1f}pt  "
              f"回収率 {a.回収率 - b.回収率:+.1f}pt")
    print("\n※ 過去の教訓: 2年で効いて5年で消えることがある。両年で同じ向きに")
    print("   出ていなければ採用しないこと。")


if __name__ == "__main__":
    sys.exit(main())
