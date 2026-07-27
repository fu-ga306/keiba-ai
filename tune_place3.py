# -*- coding: utf-8 -*-
"""place3(複勝モデル)専用チューニング実験。

複勝順位は「相手プール」の選定に使われ、収支への影響が最も大きい
(2026-07-27の検証: place3が僅かに劣化しただけで馬単251.9→233.6%)。
現状は win/place2/place3 が同一設計のため、place3 専用の最適化余地を探る。

評価指標(2025・<=2024学習):
  複妙複勝率  : MF複勝1位が3着内に来る率（軸の質）
  捕捉@5      : MF複勝上位5に実際の3着内が何頭入るか（相手プールの質＝本丸）
  2頭以上率   : 上位5に2頭以上入ったレースの割合
  AUC         : place3の識別力

使い方: python tune_place3.py [variant ...]   （省略時は全variant）
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from market_free_model import FEATURE_COLS_MF, add_race_rank_features
from model import LambdaRankWrapper

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

TEST_YEAR = 2025
SEEDS_LGB = [42, 7, 123, 2024, 99]
SEEDS_XC = [42, 7]

BASE_LGB = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05, lambda_l2=0.05,
                verbose=-1, min_gain_to_split=0.05)

# variant名 -> 設定差分。base=現行と同一(対照群/再現性確認用)
VARIANTS = {
    "base":      dict(),
    "posw1.0":   dict(pos_w=1.0),                      # 複勝は陽性が多い→過剰な重み付けを外す
    "tw3.0":     dict(time_weight=3.0),                # 直近重視を強める(drift対策)
    "leaves127": dict(lgb=dict(num_leaves=127, min_child_samples=30)),  # 表現力↑+過学習抑制
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_data():
    df = pd.read_csv("race_features.csv")
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]
    df = add_race_rank_features(df)
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    tr = df[df["年"] <= TEST_YEAR - 1].copy()
    te = df[df["年"] == TEST_YEAR].copy()
    use = [c for c in FEATURE_COLS_MF if c in tr.columns]
    return tr, te, use


def train_place3(tr, te, use, pos_w=1.5, time_weight=2.0, lgb_over=None):
    """place3だけを学習し、モデル毎のテスト予測行列(n_test, n_models)を返す。"""
    y = (tr["着順_num"] <= 3).astype(int)
    X = tr[use]
    ymin, ymax = tr["年"].min(), tr["年"].max()
    tw = 1.0 + (tr["年"] - ymin) / max(ymax - ymin, 1) * (time_weight - 1.0)
    X_tr, X_cal, y_tr, y_cal, w_tr, w_cal = train_test_split(
        X, y, tw, test_size=0.2, random_state=42)
    w_main = np.where(y_tr == 1, pos_w, 1.0) * w_tr.values
    Xt = te[use]
    preds, names = [], []

    p_lgb = dict(BASE_LGB, **(lgb_over or {}))
    for sd in SEEDS_LGB:
        p = dict(p_lgb, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        b = lgb.LGBMClassifier(**p, n_estimators=5000)
        b.fit(X_tr, y_tr, sample_weight=w_main,
              callbacks=[lgb.early_stopping(100, verbose=False)], eval_set=[(X_cal, y_cal)])
        c = CalibratedClassifierCV(estimator=b, method="isotonic", cv=None)
        c.fit(X_cal, y_cal)
        preds.append(c.predict_proba(Xt)[:, 1]); names.append(f"lgb{sd}")
    log(f"    LGB x{len(SEEDS_LGB)} done")

    if HAS_XGB:
        for sd in SEEDS_XC:
            xt = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.03, max_depth=5,
                                   n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=pos_w, eval_metric="logloss",
                                   early_stopping_rounds=100, verbosity=0, random_state=sd)
            xt.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], sample_weight=w_tr.values, verbose=False)
            xm = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.03, max_depth=5,
                                   n_estimators=max(xt.best_iteration, 10), subsample=0.8,
                                   colsample_bytree=0.8, scale_pos_weight=pos_w,
                                   verbosity=0, random_state=sd)
            xm.fit(X_tr, y_tr, sample_weight=w_tr.values, verbose=False)
            c = CalibratedClassifierCV(estimator=xm, method="isotonic", cv=None)
            c.fit(X_cal, y_cal)
            preds.append(c.predict_proba(Xt)[:, 1]); names.append(f"xgb{sd}")
        log(f"    XGB x{len(SEEDS_XC)} done")

    if HAS_CAT:
        for sd in SEEDS_XC:
            ct = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5,
                                    loss_function="Logloss", eval_metric="Logloss",
                                    early_stopping_rounds=100, verbose=False, random_seed=sd)
            ct.fit(X_tr, y_tr, eval_set=(X_cal, y_cal), sample_weight=w_main)
            cm = CatBoostClassifier(iterations=max(ct.best_iteration_, 10), learning_rate=0.03,
                                    depth=5, loss_function="Logloss", verbose=False, random_seed=sd)
            cm.fit(X_tr, y_tr, sample_weight=w_main)
            c = CalibratedClassifierCV(estimator=cm, method="isotonic", cv=None)
            c.fit(X_cal, y_cal)
            preds.append(c.predict_proba(Xt)[:, 1]); names.append(f"cat{sd}")
        log(f"    Cat x{len(SEEDS_XC)} done")

    try:
        idx = X_tr.index
        rid = tr.loc[idx, "race_id"].values
        srt = np.argsort(rid, kind="stable")
        Xr = X_tr.values[srt]
        ch = tr.loc[idx, "着順_num"].values[srt]
        sh = tr.loc[idx, "出走頭数"].fillna(18).astype(int).values[srt]
        yr = np.clip(sh - ch, 0, None).astype(int)
        _, grp = np.unique(rid[srt], return_counts=True)
        rp = dict(p_lgb)
        rp.update(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3, 5])
        rp.pop("min_child_samples", None)
        bst = lgb.train(rp, lgb.Dataset(Xr, label=yr, group=grp), num_boost_round=2000)
        preds.append(LambdaRankWrapper(bst).predict_proba(Xt)[:, 1]); names.append("rank")
        log("    LambdaRank done")
    except Exception as e:
        log(f"    LambdaRank skip: {e}")

    return np.column_stack(preds), names


def evaluate(te, score, label):
    """score(1次元)から複勝順位を作り、軸の質と相手プールの質を測る。"""
    d = te[["race_id", "着順_num"]].copy()
    d["s"] = score
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["rank"] = d.groupby("race_id")["s"].rank(ascending=False, method="min")
    axis_rate = d[d["rank"] == 1]["fuku"].mean() * 100
    cap = d[d["rank"] <= 5].groupby("race_id")["fuku"].sum()
    auc = roc_auc_score(d["fuku"], d["s"])
    return {"label": label, "複妙複勝率": axis_rate, "捕捉@5": cap.mean(),
            "2頭以上": (cap >= 2).mean() * 100, "3頭": (cap >= 3).mean() * 100, "AUC": auc}


def aggregations(P, names):
    """1回の学習から複数の集約方式を評価（追加学習コストゼロ）。"""
    out = {"確率平均(現行)": P.mean(axis=1)}
    r = pd.DataFrame(P).rank(ascending=True).values      # 大きいほど良い向きに揃える
    out["順位平均"] = r.mean(axis=1)
    out["中央値"] = np.median(P, axis=1)
    if "rank" in names:
        i = names.index("rank")
        w = np.ones(P.shape[1]); w[i] = 3.0              # LambdaRankを重めに
        out["LambdaRank重視"] = (P * w).sum(axis=1) / w.sum()
        out["LambdaRank単体"] = P[:, i]
    return out


def main():
    want = sys.argv[1:] or list(VARIANTS)
    log("データ読み込み...")
    tr, te, use = load_data()
    log(f"train {len(tr)} / test {len(te)} / feat {len(use)}")
    rows = []
    for v in want:
        cfg = VARIANTS[v]
        log(f"=== variant: {v} {cfg or '(現行と同一)'} ===")
        t0 = time.time()
        P, names = train_place3(tr, te, use,
                                pos_w=cfg.get("pos_w", 1.5),
                                time_weight=cfg.get("time_weight", 2.0),
                                lgb_over=cfg.get("lgb"))
        log(f"  学習{(time.time()-t0)/60:.0f}分")
        for agg_name, s in aggregations(P, names).items():
            m = evaluate(te, s, f"{v} / {agg_name}")
            rows.append(m)
            log(f"  {agg_name:<16} 複妙{m['複妙複勝率']:.2f}% 捕捉{m['捕捉@5']:.3f} "
                f"2頭以上{m['2頭以上']:.1f}% AUC{m['AUC']:.4f}")
        pd.DataFrame(rows).to_csv("tune_place3_result.csv", index=False, encoding="utf-8-sig")
    log("=" * 70)
    r = pd.DataFrame(rows).sort_values("捕捉@5", ascending=False)
    log("【捕捉@5の高い順】ベースライン: 複妙57.86% 捕捉1.988 2頭以上74.8%")
    for _, x in r.head(12).iterrows():
        log(f"  {x['label']:<28} 複妙{x['複妙複勝率']:6.2f}% 捕捉{x['捕捉@5']:.3f} "
            f"2頭以上{x['2頭以上']:5.1f}% AUC{x['AUC']:.4f}")
    log("→ tune_place3_result.csv に保存")


if __name__ == "__main__":
    main()
