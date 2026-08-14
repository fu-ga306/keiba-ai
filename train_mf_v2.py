# -*- coding: utf-8 -*-
"""
train_mf_v2.py  ── 市場フリーモデル(MF) 改善版（デプロイ用）
現行MFに無い要素を追加:
  1. シードバギング（LGB×5 / XGB×2 / Cat×2 シード平均→分散低減=安定化）
  2. LambdaRank 追加（レース内順位学習→穴馬の順位精度↑）
  3. 直近重み強化（時系列重み 2.0→2.5・2026へのdrift対策）
学習≤2024 / 検証2025。特徴量は全てレース前情報（当日の着順/上がり/タイムは目的変数のみで不使用）。
出力: model_mf.pkl（デプロイ・現行は model_mf_backup.pkl に退避） + model_mf_result_v2.csv（検証）。
"""
import warnings, os, sys, shutil, pickle, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from market_free_model import FEATURE_COLS_MF, add_race_rank_features
from model import LambdaRankWrapper

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError: HAS_XGB = False
try:
    from catboost import CatBoostClassifier; HAS_CAT = True
except ImportError: HAS_CAT = False

LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                  num_leaves=63, min_child_samples=15, feature_fraction=0.75,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.05, lambda_l2=0.05,
                  verbose=-1, min_gain_to_split=0.05)
LGB_RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3, 5],
                       learning_rate=0.03, num_leaves=63, min_child_samples=15,
                       feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
                       lambda_l1=0.05, lambda_l2=0.05, verbose=-1, min_gain_to_split=0.05)
POS_W = {"win": 2.0, "place2": 1.7, "place3": 1.5}
SEEDS_LGB = [42, 7, 123, 2024, 99]   # LGB 5シード
SEEDS_XC  = [42, 7]                   # XGB/Cat 2シード
TIME_WEIGHT_MAX = 2.0                 # 検証で2.5より2.0が良かったため2.0を採用

def _mk(chaku, t):
    return (chaku == 1).astype(int) if t == "win" else (chaku <= (2 if t == "place2" else 3)).astype(int)

def _train_one(train_df, test_df, use_cols, target):
    print(f"\n### MF-v2 target={target} ###", flush=True)
    X_all = train_df[use_cols].copy(); y_all = _mk(train_df["着順_num"], target)
    ymax = train_df["年"].max(); ymin = train_df["年"].min(); rng = max(ymax - ymin, 1)
    tw = 1.0 + (train_df["年"] - ymin) / rng * (TIME_WEIGHT_MAX - 1.0)
    X_tr, X_cal, y_tr, y_cal, w_tr, w_cal = train_test_split(X_all, y_all, tw, test_size=0.2, random_state=42)
    pos = POS_W.get(target, 2.0); w_main = np.where(y_tr == 1, pos, 1.0) * w_tr.values
    Xt = test_df[use_cols].copy(); models = []

    # 1) LightGBM シードバギング(5)
    for sd in SEEDS_LGB:
        p = dict(LGB_PARAMS); p["seed"] = sd; p["bagging_seed"] = sd; p["feature_fraction_seed"] = sd
        b = lgb.LGBMClassifier(**p, n_estimators=5000)
        b.fit(X_tr, y_tr, sample_weight=w_main,
              callbacks=[lgb.early_stopping(100, verbose=False)], eval_set=[(X_cal, y_cal)])
        c = CalibratedClassifierCV(estimator=b, method="isotonic", cv=None); c.fit(X_cal, y_cal)
        models.append(c)
    print(f"  LGB x{len(SEEDS_LGB)}seed done", flush=True)

    # 2) XGBoost シードバギング(2)
    if HAS_XGB:
        for sd in SEEDS_XC:
            xt = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.03, max_depth=5,
                  n_estimators=1000, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos,
                  eval_metric="logloss", early_stopping_rounds=100, verbosity=0, random_state=sd)
            xt.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], sample_weight=w_tr.values, verbose=False)
            xm = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.03, max_depth=5,
                  n_estimators=xt.best_iteration, subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=pos, verbosity=0, random_state=sd)
            xm.fit(X_tr, y_tr, sample_weight=w_tr.values, verbose=False)
            c = CalibratedClassifierCV(estimator=xm, method="isotonic", cv=None); c.fit(X_cal, y_cal)
            models.append(c)
        print(f"  XGB x{len(SEEDS_XC)}seed done", flush=True)

    # 3) CatBoost シードバギング(2)
    if HAS_CAT:
        for sd in SEEDS_XC:
            ct = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, loss_function="Logloss",
                  eval_metric="Logloss", early_stopping_rounds=100, verbose=False, random_seed=sd)
            ct.fit(X_tr, y_tr, eval_set=(X_cal, y_cal), sample_weight=w_main)
            cm = CatBoostClassifier(iterations=ct.best_iteration_, learning_rate=0.03, depth=5,
                  loss_function="Logloss", verbose=False, random_seed=sd)
            cm.fit(X_tr, y_tr, sample_weight=w_main)
            c = CalibratedClassifierCV(estimator=cm, method="isotonic", cv=None); c.fit(X_cal, y_cal)
            models.append(c)
        print(f"  Cat x{len(SEEDS_XC)}seed done", flush=True)

    # 4) LambdaRank（順位学習）
    try:
        idx_tr = X_tr.index; rid = train_df.loc[idx_tr, "race_id"].values
        srt = np.argsort(rid, kind="stable"); Xr = X_tr.values[srt]
        ch = train_df.loc[idx_tr, "着順_num"].values[srt]
        sh = train_df.loc[idx_tr, "出走頭数"].fillna(18).astype(int).values[srt]
        yr = np.clip(sh - ch, 0, None).astype(int)
        _, grp = np.unique(rid[srt], return_counts=True)
        dtr = lgb.Dataset(Xr, label=yr, group=grp)
        rp = dict(LGB_RANK_PARAMS); rp["ndcg_eval_at"] = ([2, 3] if target == "place2" else [3, 5])
        bst = lgb.train(rp, dtr, num_boost_round=2000, valid_sets=[dtr],
                        callbacks=[lgb.log_evaluation(period=-1)])
        models.append(LambdaRankWrapper(bst)); print("  LambdaRank done", flush=True)
    except Exception as e:
        print(f"  LambdaRank skip: {e}", flush=True)

    score = np.mean([m.predict_proba(Xt)[:, 1] for m in models], axis=0)
    return models, score

def main():
    # ── 学習モード（2026-07-16 二本立て化）─────────────────────────────
    #   本番（デフォルト）    : 全データ学習 → model_mf.pkl（model_mf_result.csvは触らない）
    #   backtest（引数指定時）: 〜2024学習/2025検証 → model_mf_bt.pkl + model_mf_result.csv
    bt_mode = "backtest" in sys.argv
    df = pd.read_csv("race_features.csv"); df = df.dropna(subset=["着順_num"]); df = df[df["着順_num"] >= 1]
    df = add_race_rank_features(df); df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    if bt_mode:
        _ty = int(os.environ.get("KEIBA_TEST_YEAR", "2025"))   # 多年度検証用: テスト年を可変に
        tr = df[df["年"] <= _ty - 1].copy(); te = df[df["年"] == _ty].copy()
        print(f"[backtestモード] train<={_ty-1} / test {_ty}", flush=True)
    else:
        tr = df.copy(); te = df[df["年"] == df["年"].max()].copy()
        print(f"[本番モード] 全期間〜{df['年'].max()}で学習（テスト表示はin-sample参考値）", flush=True)
    # ── 距離で2モデルに分ける（2026-08-14）──────────────────────────
    #   長距離(>=MF_DIST_SPLIT): 全特徴 / 短距離: 騎手厩舎31列を除外
    #   クリーンデータのwalk-forward検証で5条件すべて改善（全体1.6倍）。
    #   根拠と数値は market_free_model.MF_DIST_SPLIT のコメント参照。
    from market_free_model import MF_DIST_SPLIT, mf_cols_for
    use = [c for c in FEATURE_COLS_MF if c in tr.columns]
    use_short = [c for c in mf_cols_for(0) if c in tr.columns]
    print(f"train {len(tr)} / test {len(te)}  LGBseeds={SEEDS_LGB} tw={TIME_WEIGHT_MAX}", flush=True)
    print(f"  長距離({MF_DIST_SPLIT}m以上) {len(use)}列 / 短距離 {len(use_short)}列"
          f"（騎手厩舎 -{len(use)-len(use_short)}）", flush=True)
    if "距離" not in tr.columns:
        raise RuntimeError("距離列が無いため距離分割ができません。race_features.csv を確認してください。")
    tr_lo, tr_hi = tr[tr["距離"] < MF_DIST_SPLIT], tr[tr["距離"] >= MF_DIST_SPLIT]
    te_lo, te_hi = te[te["距離"] < MF_DIST_SPLIT], te[te["距離"] >= MF_DIST_SPLIT]
    print(f"  学習 短{len(tr_lo):,} / 長{len(tr_hi):,}   検証 短{len(te_lo):,} / 長{len(te_hi):,}",
          flush=True)

    result = {}; out = te[["race_id", "馬名", "着順_num"]].copy()
    if "単勝オッズ" in te.columns: out["単勝オッズ"] = te["単勝オッズ"].values
    if "人気" in te.columns: out["人気"] = te["人気"].values
    for t in ["win", "place2", "place3"]:
        m_hi, s_hi = _train_one(tr_hi, te_hi, use, t)
        m_lo, s_lo = _train_one(tr_lo, te_lo, use_short, t)
        result[t] = {"models": m_hi, "use_cols": use,
                     "models_short": m_lo, "use_cols_short": use_short,
                     "dist_split": MF_DIST_SPLIT}
        # 検証行のスコアを距離ごとに埋め戻す
        score = pd.Series(np.nan, index=te.index)
        score.loc[te_hi.index] = s_hi
        score.loc[te_lo.index] = s_lo
        out[{"win": "MF勝率", "place2": "MF連対率", "place3": "MF複勝率"}[t]] = score.values
    for c, r in [("MF勝率", "MF勝率順位"), ("MF連対率", "MF連対順位"), ("MF複勝率", "MF複勝順位")]:
        out[r] = out.groupby("race_id")[c].rank(ascending=False)
    if bt_mode:
        # honest backtest資産はbacktestモードでのみ更新（本番モードのin-sample値で汚さない）
        out.to_csv("model_mf_result.csv", index=False, encoding="utf-8-sig")
        print("saved -> model_mf_result.csv", flush=True)

    save = {"win": result["win"], "place2": result["place2"], "place3": result["place3"],
            "models": result["win"]["models"], "use_cols": use,
            "dist_split": MF_DIST_SPLIT, "use_cols_short": use_short,
            "format": "multi_v1"}
    if bt_mode:
        with open("model_mf_bt.pkl", "wb") as f:
            pickle.dump(save, f)
        print("saved -> model_mf_bt.pkl (backtest用・本番には使われない)", flush=True)
    else:
        # ── デプロイ: 現行 model_mf.pkl をバックアップして上書き ──
        if os.path.exists("model_mf.pkl"):
            shutil.copy("model_mf.pkl", "model_mf_backup.pkl")
            print("backup -> model_mf_backup.pkl", flush=True)
        with open("model_mf.pkl", "wb") as f:
            pickle.dump(save, f)
        print("deployed -> model_mf.pkl (全データ学習・本番)", flush=True)
        # ── 分割保存(2026-07-27〜): 一括pickle.loadのピークRAMクラッシュ対策。
        #    読込側(keiba_auto/keiba_predict)は model_mf_parts/ を優先し逐次読込する。
        from mf_model_io import save_mf_split
        d = save_mf_split(save, ".")
        print(f"deployed -> {d}/ (分割保存・逐次読込用)", flush=True)

if __name__ == "__main__":
    main()
