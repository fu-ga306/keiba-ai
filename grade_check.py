# -*- coding: utf-8 -*-
"""評価グレード（S〜D）の較正精度と、買い目への足切り効果を2年で検証する。

背景（2026-08-10）
  2025年のOOS 43,189頭では、評価で足切りすると回収率が上がって見えた
  （現行83.4% → A以上94.3% → S以上99.2%）。しかし
    ・しきい値を決めた年と評価した年が同じ（過学習の疑い）
    ・開催前半/後半で分けると再現しない（S以上: 前半49.5% / 後半154.4%）
    ・最良でも100%を割る
  ため採用しなかった。独立した年で確かめる必要がある。

やること
  test年ごとに
    学習   : from年 〜 (test-2)年
    較正   : (test-1)年  ← 等張回帰。検証年は較正に使わない
    検証   : test年
  を回し、win/place2/place3 の3つを出して合成スコアを作る。
  2024と2025の両方を同じ設定で回すので、年をまたいで再現するかが見える。

⚠ 学習設定は本番より軽い（LGB2シード。本番はLGB5＋XGB2＋Cat2）。
   2年間の比較にのみ使う。絶対値を本番の数字と並べないこと。
⚠ 本番の model_mf_result.csv は上書きしない（別ファイルに出す）。

実行: python grade_check.py       → grade_check_result.csv / grade_check_oos_<年>.csv
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7]
TIME_WEIGHT_MAX = 2.0
POS_W = {"win": 2.0, "place2": 1.7, "place3": 1.5}
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05,
                  lambda_l2=0.05, verbose=-1, min_gain_to_split=0.05)

GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
GRADE_TH = [(1.05, "S"), (0.80, "A"), (0.58, "B"), (0.36, "C")]

CASES = [(2024, 2019), (2025, 2019)]      # (テスト年, 学習開始年)


def _mk(chaku, t):
    if t == "win":
        return (chaku == 1).astype(int)
    return (chaku <= (2 if t == "place2" else 3)).astype(int)


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


def _roi(p):
    """単勝1点・1,000円想定。戻り値は (点数, 回収率, 的中率)。"""
    if not len(p):
        return 0, 0.0, 0.0
    ret = np.where(p["着順_num"] == 1,
                   pd.to_numeric(p["単勝オッズ"], errors="coerce") * 100, 0).sum()
    return len(p), ret / (len(p) * 100) * 100, (p["着順_num"] == 1).mean() * 100


def main():
    print("特徴量を読み込み中...", flush=True)
    df = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False)
    df = add_race_rank_features(df)
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    cols = [c for c in FEATURE_COLS_MF if c in df.columns]
    print(f"  {len(df)}行 / 特徴量{len(cols)}列", flush=True)

    rows = []
    for test_y, from_y in CASES:
        t0 = time.time()
        tr = df[(df["年"] >= from_y) & (df["年"] <= test_y - 2)]
        ca = df[df["年"] == test_y - 1]
        te = df[df["年"] == test_y].copy()
        print(f"\n=== test{test_y}: 学習{from_y}〜{test_y-2}({len(tr)}行) / "
              f"較正{test_y-1} / 検証{len(te)}行 ===", flush=True)

        for tgt in ("win", "place2", "place3"):
            m = _fit(tr, cols, tgt)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(_pred(m, ca, cols), _mk(ca["着順_num"], tgt))
            te[tgt] = iso.predict(_pred(m, te, cols))
            print(f"  {tgt} 完了 {time.time()-t0:.0f}秒", flush=True)

        # 本番と同じく勝率だけレース内で正規化してから期待値にする
        te["p"] = te.groupby("race_id")["win"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 0)
        te["score"] = te["p"] + te["place2"] + te["place3"]
        te["pt"] = np.select([te["着順_num"] == 1, te["着順_num"] <= 2,
                              te["着順_num"] <= 3], [3, 2, 1], 0)
        te["grade"] = np.select([te["score"] >= t for t, _ in GRADE_TH],
                                [g for _, g in GRADE_TH], "D")
        keep = ["race_id", "馬名", "着順_num", "単勝オッズ", "人気",
                "win", "place2", "place3", "p", "score", "grade", "pt"]
        te[[c for c in keep if c in te.columns]].to_csv(
            os.path.join(BASE_DIR, f"grade_check_oos_{test_y}.csv"),
            index=False, encoding="utf-8-sig")

        print(f"\n  [評価グレードの較正] 合成スコア平均 {te.score.mean():.3f} / "
              f"実測の平均獲得点 {te.pt.mean():.3f} / ズレ {te.score.mean()-te.pt.mean():+.3f}")
        for g in "SABCD":
            s = te[te.grade == g]
            if not len(s):
                continue
            rows.append({"テスト年": test_y, "種別": "grade", "条件": g,
                         "頭数": len(s),
                         "勝率": round((s["着順_num"] == 1).mean() * 100, 1),
                         "連対": round((s["着順_num"] <= 2).mean() * 100, 1),
                         "複勝": round((s["着順_num"] <= 3).mean() * 100, 1),
                         "平均点": round(s.pt.mean(), 2)})
            print(f"    {g}: {len(s):>6d}頭  勝率{(s['着順_num']==1).mean()*100:5.1f}%  "
                  f"複勝{(s['着順_num']<=3).mean()*100:5.1f}%  平均点{s.pt.mean():.2f}")

        # 買い目に足切りを入れるとどうなるか
        te["ev"] = te["p"] * pd.to_numeric(te["単勝オッズ"], errors="coerce")
        mr = te.groupby("race_id")["place3"].rank(ascending=False)
        gap = pd.to_numeric(te["人気"], errors="coerce") - mr
        od = pd.to_numeric(te["単勝オッズ"], errors="coerce")
        base = ((gap >= GAP_MIN) & (od <= ODDS_MAX) &
                (((mr == 1) & (te.ev >= EV_TOP)) |
                 (mr.between(2, 5) & (te.ev >= EV_SUB))))
        cand = te[base.fillna(False)]
        print(f"\n  [買い目への足切り]")
        for th, lbl in [(0.0, "現行"), (0.36, "C以上"), (0.58, "B以上"),
                        (0.80, "A以上"), (1.05, "S以上")]:
            pick = cand[cand.score >= th].sort_values(
                "ev", ascending=False).groupby("race_id").head(1)
            n, roi, hit = _roi(pick)
            rows.append({"テスト年": test_y, "種別": "filter", "条件": lbl,
                         "頭数": n, "回収率": round(roi, 1), "的中率": round(hit, 1)})
            print(f"    {lbl:6s} {n:>4d}点  回収率{roi:>7.1f}%  的中{hit:>5.1f}%")
        pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "grade_check_result.csv"),
                                  index=False, encoding="utf-8-sig")
        print(f"\n  所要 {int(time.time()-t0)}秒", flush=True)

    print("\n保存 → grade_check_result.csv / grade_check_oos_*.csv")


if __name__ == "__main__":
    sys.exit(main())
