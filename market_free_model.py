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
    # 交互作用特徴量
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率",
    "芝ダート×先行_過去勝率",
    "前走好走×人気薄", "前走着順×人気_乖離",
    "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
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


if __name__ == "__main__":
    models, df, use_cols = train_market_free_model()