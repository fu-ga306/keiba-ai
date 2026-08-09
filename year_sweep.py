# -*- coding: utf-8 -*-
"""学習に使う年数を変えて、本番の買い方での回収率を比べる。

問い（2026-08-09）
  MFモデルは何年分で学習するのが一番回収率が高いのか。
  これまで「学習期間の拡大は+0.4pt＝誤差」という精度の話しか測っていない。
  回収率で年数を振った実験は無かった。

やり方
  train  … from年 〜 (test-2)年で学習
  calib  … (test-1)年で確率を較正（本番と同じく、検証年を較正に使わない）
  test   … test年で本番の買い方を適用して回収率を出す

  本番の買い方（keiba_predict.py の EV方式）
    乖離(人気順位 − MF複勝順位) >= 3 かつ 単勝オッズ <= 20 で
    MF複勝順位1位は EV>=1.7 / 2〜5位は EV>=2.2 を満たす馬のうち
    期待値が最大の1頭を単勝1点。

⚠ 学習の設定を本番より軽くしている（LGB2シード・win/place3のみ）。
   年数どうしの比較が目的なので、全条件で同じ軽さに揃えてある。
   ここで決まった年数で、最後に本番設定の学習を一度だけ回すこと。

実行: python year_sweep.py        （結果は year_sweep_result.csv）
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
OUT = os.path.join(BASE_DIR, "year_sweep_result.csv")

SEEDS = [42, 7]                 # 本番は5シード＋XGB/Cat。比較用に軽くする
TIME_WEIGHT_MAX = 2.0
POS_W = {"win": 2.0, "place3": 1.5}
LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05,
                  lambda_l2=0.05, verbose=-1, min_gain_to_split=0.05)

# 本番の買い方（keiba_predict.py と同じ値）
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2

CASES = [           # (テスト年, 学習開始年)
    (2025, 2022), (2025, 2020), (2025, 2019),
    (2024, 2021), (2024, 2019),
]


def _mk(chaku, t):
    return (chaku == 1).astype(int) if t == "win" else (chaku <= 3).astype(int)


def _fit(train_df, cols, target):
    X = train_df[cols]
    y = _mk(train_df["着順_num"], target)
    ymax, ymin = train_df["年"].max(), train_df["年"].min()
    rng = max(ymax - ymin, 1)
    tw = 1.0 + (train_df["年"] - ymin) / rng * (TIME_WEIGHT_MAX - 1.0)
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


def roi_of(te, win_p, p3_p):
    """本番の買い方を当てはめて回収率を出す。"""
    d = te[["race_id", "馬番", "着順_num", "単勝オッズ", "人気"]].copy()
    d["mf_win"] = win_p
    d["mf_p3"] = p3_p
    # レース内で確率を正規化してから期待値にする（本番と同じ）
    d["p"] = d.groupby("race_id")["mf_win"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0)
    d["ev"] = d["p"] * pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["mfr"] = d.groupby("race_id")["mf_p3"].rank(ascending=False)
    d["pop"] = pd.to_numeric(d["人気"], errors="coerce")
    d["gap"] = d["pop"] - d["mfr"]
    od = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    hit = ((d["gap"] >= GAP_MIN) & (od <= ODDS_MAX) &
           (((d["mfr"] == 1) & (d["ev"] >= EV_TOP)) |
            (d["mfr"].between(2, 5) & (d["ev"] >= EV_SUB))))
    cand = d[hit.fillna(False)]
    if cand.empty:
        return 0, 0.0, 0.0
    pick = cand.sort_values("ev", ascending=False).groupby("race_id").head(1)
    ret = np.where(pick["着順_num"] == 1,
                   pd.to_numeric(pick["単勝オッズ"], errors="coerce") * 100, 0).sum()
    n = len(pick)
    return n, ret / (n * 100) * 100, (pick["着順_num"] == 1).mean() * 100


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
        te = df[df["年"] == test_y]
        nyears = test_y - 1 - from_y            # 学習に使った年数（較正年を除く）
        print(f"\n=== test{test_y} / 学習{from_y}〜{test_y-2}（{nyears}年・{len(tr)}行）===",
              flush=True)
        res = {"テスト年": test_y, "学習開始": from_y, "学習年数": nyears,
               "学習行数": len(tr)}
        preds = {}
        for tgt in ("win", "place3"):
            m = _fit(tr, cols, tgt)
            # 較正年で等張回帰をかける（本番の mf_calibrator と同じ考え方）
            pc = _pred(m, ca, cols)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pc, _mk(ca["着順_num"], tgt))
            preds[tgt] = iso.predict(_pred(m, te, cols))
            print(f"  {tgt} 完了 {time.time()-t0:.0f}秒", flush=True)
        n, roi, hitr = roi_of(te, preds["win"], preds["place3"])
        res.update({"買い点数": n, "回収率": round(roi, 1), "的中率": round(hitr, 1),
                    "所要秒": int(time.time() - t0)})
        print(f"  → 回収率 {roi:.1f}%  的中 {hitr:.1f}%  {n}点", flush=True)
        rows.append(res)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print(f"\n保存 → {OUT}")
    print("\n⚠ 学習設定は本番より軽い（LGB2シード・win/place3のみ）。")
    print("   年数どうしの比較にのみ使うこと。")


if __name__ == "__main__":
    sys.exit(main())
