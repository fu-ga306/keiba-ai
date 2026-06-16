"""
market_free_model.py
────────────────────
市場情報（人気・オッズ）を一切使わない純粋能力ベースモデル。
通常モデル（model.pkl）と並行運用して「市場乖離スコア」を生成する。

使い方:
    python market_free_model.py
"""

import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ── 市場依存列を除いた特徴量（人気・単勝オッズを除外）──────────────────
FEATURE_COLS_MF = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    # 「人気」「単勝オッズ」を除外 ← ここが通常モデルとの違い
    "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "体重増減_過去標準偏差", "体重増減_異常度",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順",
    "良馬場勝率", "重馬場勝率",
    "過去最速上り", "上り偏差", "距離別過去平均上り",
    # 追加特徴量（存在すれば使用）
    "斤量変化", "乗り替わり", "連闘", "休み明け", "負担率",
    # レース内ランク（能力ベースなのでOK）
    "レース内_過去勝率ランク",
    "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク",
    "レース内_騎手勝率ランク",
    # 競馬場×距離 適性
    "競馬場距離過去勝率", "競馬場距離過去平均着順",
    "競馬場過去勝率", "競馬場過去平均着順",
    # 脚質
    "過去平均先行指数", "先行馬フラグ",
    "想定先行馬数", "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
    # 開催時期
    "開催月", "開催季節",
    # 前走情報
    "前走着順", "前走上り", "前走距離", "距離変化",
    # 連続好走・成長
    "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
    # タイム差
    "平均タイム差",
    # 騎手×競馬場
    "騎手競馬場勝率",
    # 交互作用特徴量（市場情報を含まないものだけ）
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率",
    "芝ダート×先行_過去勝率",
    # 「前走好走×人気薄」「前走着順×人気_乖離」は人気を含むため除外（純粋実力モデル）
    "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
    # 距離変化系（長距離・距離延長の弱点対策・市場情報なし）
    "距離変化比率", "大幅延長フラグ", "大幅短縮フラグ",
    "距離延長幅", "長距離フラグ", "大幅延長×長距離", "距離延長×先行",
    "経験最長距離", "経験範囲超過", "延長×距離経験不足",
    "長距離複勝率", "前走余力", "延長×前走余力",
]

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.03,
    "num_leaves":        31,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "verbose":           -1,
    "min_gain_to_split": 0.1,
}


def add_race_rank_features(df):
    """レース内ランク特徴量を生成（市場情報不使用）"""
    df["レース内_過去勝率ランク"] = df.groupby("race_id")["過去勝率"].rank(
        ascending=False, method="min"
    )
    df["レース内_直近3走平均着順ランク"] = df.groupby("race_id")[
        "直近3走平均着順"
    ].rank(ascending=True, method="min")
    df["レース内_過去平均上りランク"] = df.groupby("race_id")["過去平均上り"].rank(
        ascending=True, method="min"
    )
    df["レース内_騎手勝率ランク"] = df.groupby("race_id")["騎手勝率"].rank(
        ascending=False, method="min"
    )
    return df


def train_market_free_model(csv_path="race_features.csv"):
    print("=" * 50)
    print("🤖 市場フリーモデル 学習開始")
    print("=" * 50)

    print("特徴量データ読み込み中...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]

    print("レース内ランク特徴量を生成中...")
    df = add_race_rank_features(df)

    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    train_df = df[df["年"] <= 2024].copy()
    test_df  = df[df["年"] == 2025].copy()

    print(f"学習データ: {len(train_df)}行（〜2024）")
    print(f"検証データ: {len(test_df)}行（2025）")

    # 存在する列のみ使用
    use_cols = [c for c in FEATURE_COLS_MF if c in train_df.columns]
    print(f"使用特徴量: {len(use_cols)}列（人気・オッズ除外）")
    print(f"  除外確認 - 人気: {'人気' not in use_cols} / 単勝オッズ: {'単勝オッズ' not in use_cols}")

    X_train_all = train_df[use_cols].copy()
    y_train_all = (train_df["着順_num"] == 1).astype(int)
    X_test      = test_df[use_cols].copy()

    # ── 時系列重み（直近年ほど重みを大きくする） ──────────────────────
    TIME_WEIGHT_MAX = 1.0
    year_min = train_df["年"].min()
    year_max = train_df["年"].max()
    year_range = max(year_max - year_min, 1)
    train_df["時系列重み"] = 1.0 + (train_df["年"] - year_min) / year_range * (TIME_WEIGHT_MAX - 1.0)

    # prefit方式でキャリブレーション
    X_train_main, X_cal, y_train_main, y_cal, w_time_main, w_time_cal = train_test_split(
        X_train_all, y_train_all, train_df["時系列重み"], test_size=0.2, random_state=42
    )
    w_main = np.where(y_train_main == 1, 2.0, 1.0) * w_time_main.values
    print(f"  時系列重み: {year_min}年=1.0倍 〜 {year_max}年={TIME_WEIGHT_MAX}倍")

    print("ベースLightGBMモデルを学習中（アーリーストッピング付き）...")
    base_model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=5000)
    base_model.fit(
        X_train_main, y_train_main,
        sample_weight=w_main,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        eval_set=[(X_cal, y_cal)],
    )
    print(f"  アーリーストッピング: {base_model.best_iteration_}本で停止")

    print("確率キャリブレーション中...")
    # sklearn 1.8+ では cv="prefit" が廃止、cv=None が同等動作
    calibrated = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=None
    )
    calibrated.fit(X_cal, y_cal)

    models = [calibrated]

    # ── テストデータで評価 ────────────────────────────────────────────
    test_df = test_df.copy()
    test_df["mf_score"] = calibrated.predict_proba(X_test)[:, 1]
    test_df["mf_rank"]  = test_df.groupby("race_id")["mf_score"].rank(ascending=False)

    def normalize(group):
        probs = group["mf_score"].values
        total = probs.sum()
        return pd.Series(probs / total if total > 0 else probs, index=group.index)

    test_df["mf_win_prob"] = (
        test_df.groupby("race_id").apply(normalize)
        .reset_index(level=0, drop=True)
    )

    # ── バックテスト ─────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("🔥 市場フリーモデル バックテスト（2025年）")
    print(f"{'='*50}")

    # 単勝期待値は通常モデルと異なり「純粋AI勝率×オッズ」
    if "単勝オッズ" in test_df.columns:
        test_df["mf_ev"] = test_df["mf_win_prob"] * test_df["単勝オッズ"] - 1

        for label, ev_thresh, rank_cond, odds_min, odds_max in [
            ("純粋AI予測1位 × 期待値≥0.3",   0.3, [1],    1.5, 20.0),
            ("純粋AI予測1位 × 期待値≥0.3 × 人気3番手以下（市場乖離）", 0.3, [1], 1.5, 20.0),
        ]:
            bets = test_df[
                (test_df["mf_rank"].isin(rank_cond)) &
                (test_df["mf_ev"] >= ev_thresh) &
                (test_df["単勝オッズ"] >= odds_min) &
                (test_df["単勝オッズ"] <= odds_max)
            ]
            if "市場乖離" in label and "人気" in test_df.columns:
                bets = bets[bets["人気"] >= 3]

            if len(bets) > 0:
                wins = bets[bets["着順_num"] == 1]
                print(f"\n── {label} ──")
                print(f"ベット数: {len(bets)}回")
                print(f"的中数:   {len(wins)}回")
                print(f"的中率:   {len(wins)/len(bets)*100:.1f}%")
                print(f"回収率:   {wins['単勝オッズ'].sum() / len(bets) * 100:.1f}%")

    # ── 通常モデルとの比較分析 ─────────────────────────────────────
    print(f"\n{'='*50}")
    print("📊 通常モデル vs 市場フリーモデル 比較")
    print(f"{'='*50}")

    try:
        with open("model.pkl", "rb") as f:
            saved = pickle.load(f)
        normal_models = saved["models"]
        normal_cols   = saved["use_cols"]

        # 通常モデルの列に合わせてNaNで補完
        normal_cols_exist = [c for c in normal_cols if c in test_df.columns]
        X_test_normal = test_df[normal_cols_exist].reindex(columns=normal_cols, fill_value=np.nan).copy()
        test_df["normal_score"] = np.mean(
            [m.predict_proba(X_test_normal)[:, 1] for m in normal_models], axis=0
        )
        test_df["normal_rank"] = test_df.groupby("race_id")["normal_score"].rank(ascending=False)

        # 乖離スコア（プラスほど「AIは高評価・市場は低評価」）
        test_df["乖離スコア"] = test_df["normal_rank"] - test_df["mf_rank"]

        # 乖離が大きい馬の的中率
        print("\n── 乖離スコア別成績（市場フリー順位 - 通常順位）──")
        print("  プラス = AIは高評価だが市場は低評価（穴馬候補）")
        for thresh in [3, 5, 7]:
            high_gap = test_df[
                (test_df["乖離スコア"] >= thresh) &
                (test_df["mf_rank"] == 1)
            ]
            if len(high_gap) > 0:
                wins = high_gap[high_gap["着順_num"] == 1]
                if "単勝オッズ" in high_gap.columns:
                    roi = wins["単勝オッズ"].sum() / len(high_gap) * 100
                    print(f"  乖離≥{thresh} × MF予測1位: {len(high_gap)}回 的中率{len(wins)/len(high_gap)*100:.1f}% 回収率{roi:.1f}%")

    except FileNotFoundError:
        print("  model.pkl が見つかりません（通常モデルとの比較をスキップ）")
    except Exception as e:
        print(f"  比較エラー: {e}")

    # ── 保存 ──────────────────────────────────────────────────────────
    out = test_df[["race_id", "馬名", "着順_num", "mf_score", "mf_rank", "mf_win_prob"]].copy()
    if "単勝オッズ" in test_df.columns:
        out["mf_ev"] = test_df["mf_ev"]
    if "乖離スコア" in test_df.columns:
        out["乖離スコア"] = test_df["乖離スコア"]
    out.to_csv("model_mf_result.csv", index=False, encoding="utf-8-sig")

    with open("model_mf.pkl", "wb") as f:
        pickle.dump({"models": models, "use_cols": use_cols}, f)
    print("\nモデル保存完了 → model_mf.pkl")

    return models, test_df, use_cols


# ════════════════════════════════════════════════════════════════════
# MF版 3モデル化（win/place2/place3）— 市場を見ない実力ベースの3予想
# ════════════════════════════════════════════════════════════════════

def _mf_make_target(chaku, target):
    if target == "win":
        return (chaku == 1).astype(int)
    elif target == "place2":
        return (chaku <= 2).astype(int)
    elif target == "place3":
        return (chaku <= 3).astype(int)
    raise ValueError(target)


_MF_POS_WEIGHT = {"win": 2.0, "place2": 1.7, "place3": 1.5}


def _train_mf_one(train_df, test_df, use_cols, target):
    """MFモデルを1ターゲット分学習してアンサンブル(LGB+XGB+CatBoost)を返す。"""
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

    print(f"\n{'#'*46}\n# MF ターゲット: {target}\n{'#'*46}")
    X_train_all = train_df[use_cols].copy()
    y_train_all = _mf_make_target(train_df["着順_num"], target)

    TIME_WEIGHT_MAX = 1.0
    year_min = train_df["年"].min()
    year_max = train_df["年"].max()
    year_range = max(year_max - year_min, 1)
    train_df = train_df.copy()
    train_df["時系列重み"] = 1.0 + (train_df["年"] - year_min) / year_range * (TIME_WEIGHT_MAX - 1.0)

    X_tr, X_cal, y_tr, y_cal, w_tr, w_cal = train_test_split(
        X_train_all, y_train_all, train_df["時系列重み"], test_size=0.2, random_state=42
    )
    pos_w = _MF_POS_WEIGHT.get(target, 2.0)
    w_main = np.where(y_tr == 1, pos_w, 1.0) * w_tr.values
    print(f"  正例重み({target}): {pos_w}倍")

    models = []
    # LightGBM
    base = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=5000)
    base.fit(X_tr, y_tr, sample_weight=w_main,
             callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=200)],
             eval_set=[(X_cal, y_cal)])
    cal_lgb = CalibratedClassifierCV(estimator=base, method="isotonic", cv=None)
    cal_lgb.fit(X_cal, y_cal)
    models.append(cal_lgb)
    print(f"  LightGBM完了（{base.best_iteration_}本）")

    # XGBoost
    if HAS_XGB:
        xtmp = xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.03, max_depth=5,
            n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, eval_metric="logloss",
            early_stopping_rounds=100, verbosity=0, random_state=42)
        xtmp.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                 sample_weight=w_tr.values, verbose=False)
        best_it = xtmp.best_iteration
        xgbm = xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.03, max_depth=5,
            n_estimators=best_it, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, verbosity=0, random_state=42)
        xgbm.fit(X_tr, y_tr, sample_weight=w_tr.values, verbose=False)
        cal_xgb = CalibratedClassifierCV(estimator=xgbm, method="isotonic", cv=None)
        cal_xgb.fit(X_cal, y_cal)
        models.append(cal_xgb)
        print(f"  XGBoost完了（{best_it}本）")

    # CatBoost
    if HAS_CAT:
        ctmp = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=5,
            loss_function="Logloss", eval_metric="Logloss",
            early_stopping_rounds=100, verbose=False, random_seed=42)
        ctmp.fit(X_tr, y_tr, eval_set=(X_cal, y_cal), sample_weight=w_main)
        best_it = ctmp.best_iteration_
        cbm = CatBoostClassifier(
            iterations=best_it, learning_rate=0.03, depth=5,
            loss_function="Logloss", verbose=False, random_seed=42)
        cbm.fit(X_tr, y_tr, sample_weight=w_main)
        cal_cat = CalibratedClassifierCV(estimator=cbm, method="isotonic", cv=None)
        cal_cat.fit(X_cal, y_cal)
        models.append(cal_cat)
        print(f"  CatBoost完了（{best_it}本）")

    # 評価
    X_test = test_df[use_cols].copy()
    score = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    tdf = test_df.copy()
    tdf["_score"] = score
    tdf["_rank"]  = tdf.groupby("race_id")["_score"].rank(ascending=False)
    top1 = tdf[tdf["_rank"] == 1]
    if len(top1) > 0:
        hit = _mf_make_target(top1["着順_num"], target).mean() * 100
        print(f"  [MF {target}] 予測1位の的中率: {hit:.1f}% ({len(top1)}レース)")

    return models


def train_mf_all_targets(csv_path="race_features.csv"):
    """MF版の win/place2/place3 を学習し model_mf.pkl に3セット保存する。"""
    print("=" * 50)
    print("🤖 市場フリーモデル 3モデル化 学習開始")
    print("=" * 50)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]
    df = add_race_rank_features(df)
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    train_df = df[df["年"] <= 2024].copy()
    test_df  = df[df["年"] == 2025].copy()
    use_cols = [c for c in FEATURE_COLS_MF if c in train_df.columns]
    print(f"使用特徴量: {len(use_cols)}列（人気・オッズ・人気交互作用すべて除外）")
    print(f"  人気混入チェック: {[c for c in use_cols if '人気' in c or 'オッズ' in c]}（空ならOK）")

    result = {}
    test_scores = {}
    for target in ["win", "place2", "place3"]:
        models = _train_mf_one(train_df, test_df, use_cols, target)
        result[target] = {"models": models, "use_cols": use_cols}
        # テストデータの予測スコアも保存（回収率分析用）
        Xt = test_df[use_cols].copy()
        test_scores[target] = np.mean([m.predict_proba(Xt)[:, 1] for m in models], axis=0)

    # ── 回収率分析用の結果CSVを出力 ──
    out = test_df[["race_id", "馬名", "着順_num"]].copy()
    if "単勝オッズ" in test_df.columns:
        out["単勝オッズ"] = test_df["単勝オッズ"].values
    if "人気" in test_df.columns:
        out["人気"] = test_df["人気"].values
    out["MF勝率"]   = test_scores["win"]
    out["MF連対率"] = test_scores["place2"]
    out["MF複勝率"] = test_scores["place3"]
    # レース内順位
    out["MF勝率順位"]   = out.groupby("race_id")["MF勝率"].rank(ascending=False)
    out["MF連対順位"]   = out.groupby("race_id")["MF連対率"].rank(ascending=False)
    out["MF複勝順位"]   = out.groupby("race_id")["MF複勝率"].rank(ascending=False)
    out.to_csv("model_mf_result.csv", index=False, encoding="utf-8-sig")
    print("  回収率分析用CSV出力 → model_mf_result.csv")

    save_dict = {
        "win":    result["win"],
        "place2": result["place2"],
        "place3": result["place3"],
        # 旧形式互換（既存コードが model_mf["models"] を読めるように）
        "models":   result["win"]["models"],
        "use_cols": result["win"]["use_cols"],
        "format":   "multi_v1",
    }
    with open("model_mf.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print(f"\n✅ MF 3モデル保存完了 → model_mf.pkl")
    print(f"   win/place2/place3 各{len(result['win']['models'])}モデル（市場フリー）")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # 旧来の単一モデル学習
        train_market_free_model()
    else:
        # デフォルト: 3モデル化（推奨）
        train_mf_all_targets()