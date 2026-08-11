# -*- coding: utf-8 -*-
"""「芝のみ」の優位が、独立した年（2026）でも再現するかを検証する。

見つかったこと（2026-08-11）
  5年（2021-2025）で買い方を探索し直したところ、**芝のみに絞ると
  5年すべて100%超・ブートストラップ下限110.8** という構成が出た。
  これまで130通り以上試して一度も達成できなかった水準。

      芝    101.9 / 170.0 / 147.4 / 122.8 / 189.8  → 通算148.9%
      ダート  80.7 /  57.9 /  86.1 / 108.8 /  69.2  → 通算 80.3%

  機構も説明がつく。モデルの当てる力は芝もダートも同じ（カバー率48.2/48.3%）
  なのに、選ばれた馬の勝率は同人気帯の 芝2.19倍 / ダート1.13倍。
  違うのはモデルではなく**市場側**で、芝では市場が見落としているが
  ダートでは市場が正しく評価している。「市場のズレで稼ぐ」戦略と整合する。

  ただし探索の中から見つけたものなので、**独立した年での再現確認が要る**。

やり方
  本番モデル(model_mf.pkl)は2026年まで全データで学習しているので、
  2026年に当てるとin-sampleになり使えない。学習をやり直して正直な予測を作る。

    対照(A): 学習≤2023 / 較正2024 / 検証2025 … bet_cacheの結果を再現できるか
    本番(B): 学習≤2024 / 較正2025 / 検証2026 … 独立検証

  Aで芝>>ダートが再現すれば、この軽い学習設定でも判定に使えると分かる。
  そのうえでBを見る。

⚠ 学習設定は本番より軽い（LGB2シード）。芝とダートの差を見るのが目的で、
   絶対値を本番の数字と並べないこと。

実行: python validate2026.py → validate2026_result.csv
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
POS_W = {"win": 2.0, "place3": 1.5}
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05,
                  lambda_l2=0.05, verbose=-1, min_gain_to_split=0.05)
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
CASES = [(2025, 2020, "対照"), (2026, 2021, "独立検証")]
rng = np.random.default_rng(12345)


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


def load_pay():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}


def evaluate(te, pay, turf=None):
    """本番の買い方（単勝1,000円＋馬単500円）で回収率を出す。"""
    d = te.copy()
    if turf is not None:
        d = d[d["is_turf"] == turf]
    if d.empty:
        return None
    pts = []
    for rid, g in d.groupby("race_id", sort=False):
        c = g[(g["gap"] >= GAP_MIN) & (g["odds"] <= ODDS_MAX) &
              (((g["mr"] == 1) & (g["ev"] >= EV_TOP)) |
               (g["mr"].between(2, 5) & (g["ev"] >= EV_SUB)))]
        if not len(c):
            continue
        ax = c.sort_values("ev", ascending=False).bn.iloc[0]
        pr = g["人気"].rank(method="first")
        cost, ret = 1000.0, pay.get((rid, "単勝", ax), 0.0) * 10
        for m in g[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= 3)].bn:
            if m == ax:
                continue
            cost += 500
            ret += pay.get((rid, "馬単", f"{ax}-{m}"), 0.0) * 5
        pts.append((cost, ret))
    if len(pts) < 20:
        return None
    a = np.array(pts, float)
    idx = rng.integers(0, len(a), (5000, len(a)))
    bs = a[:, 1][idx].sum(1) / a[:, 0][idx].sum(1) * 100
    return {"R数": len(a), "回収率": round(a[:, 1].sum() / a[:, 0].sum() * 100, 1),
            "CI下": round(float(np.percentile(bs, 2.5)), 1),
            "的中": int((a[:, 1] > 0).sum()),
            "P100": round(float(np.mean(bs > 100) * 100), 1)}


def main():
    print("特徴量を読み込み中...", flush=True)
    df = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False)
    df = add_race_rank_features(df)
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    cols = [c for c in FEATURE_COLS_MF if c in df.columns]
    pay = load_pay()
    print(f"  {len(df)}行 / 特徴量{len(cols)}列", flush=True)

    rows = []
    for test_y, from_y, tag in CASES:
        t0 = time.time()
        tr = df[(df["年"] >= from_y) & (df["年"] <= test_y - 2)]
        ca = df[df["年"] == test_y - 1]
        te = df[df["年"] == test_y].copy()
        print(f"\n=== {tag} test{test_y}: 学習{from_y}〜{test_y-2}({len(tr)}行) / "
              f"較正{test_y-1} / 検証{len(te)}行 ===", flush=True)
        preds = {}
        for tgt in ("win", "place3"):
            m = _fit(tr, cols, tgt)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(_pred(m, ca, cols), _mk(ca["着順_num"], tgt))
            preds[tgt] = iso.predict(_pred(m, te, cols))
            print(f"  {tgt} 完了 {time.time()-t0:.0f}秒", flush=True)

        # bet_cache と同じ定義に揃える
        te["bn"] = pd.to_numeric(te["馬番"], errors="coerce").astype("Int64") \
            .astype(str).str.zfill(2)
        te["odds"] = pd.to_numeric(te["単勝オッズ"], errors="coerce")
        te["p"] = preds["win"]
        te["p"] = te.groupby("race_id")["p"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 0)
        te["ev"] = te["p"] * te["odds"]
        te["mr"] = preds["place3"]
        te["mr"] = te.groupby("race_id")["mr"].rank(ascending=False)
        te["gap"] = pd.to_numeric(te["人気"], errors="coerce") - te["mr"]

        for turf, lbl in ((None, "全体"), (1, "芝のみ"), (0, "ダートのみ")):
            r = evaluate(te, pay, turf)
            if r:
                rows.append({"区分": tag, "テスト年": test_y, "コース": lbl, **r})
                print(f"  {lbl:8s} {r['R数']:>4d}レース 回収率{r['回収率']:7.1f}% "
                      f"CI下{r['CI下']:6.1f} 的中{r['的中']:>3d} P100 {r['P100']:.0f}%",
                      flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "validate2026_result.csv"),
                                  index=False, encoding="utf-8-sig")

    print("\n" + "=" * 76)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n判定のしかた:")
    print("  対照(2025)で 芝>>ダート が再現 → この軽い学習設定でも判定に使える")
    print("  独立検証(2026)でも 芝>>ダート → 芝限定は本物の可能性が高い")
    print("  2026で崩れる → 5年で見つけた芝の優位は過剰適合だった")


if __name__ == "__main__":
    sys.exit(main())
