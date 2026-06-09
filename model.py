import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import sklearn

FEATURE_COLS = [
    "枠番",
    "馬番",
    "斤量",
    "斤量_相対",
    "年齢",
    "is_male",
    "is_female",
    "is_castrated",
    "馬体重",
    "体重増減",
    "馬体重_相対",
    "人気",
    "出走頭数",
    "競馬場cd",
    "レース番号",
    "過去出走数",
    "過去平均着順",
    "過去勝率",
    "過去複勝率",
    "過去平均上り",
    "直近3走平均着順",
    "過去平均タイム秒",
    "直近3走平均タイム秒",
    "過去最速タイム秒",
    "直近3走平均上り",
    "過去平均体重増減",
    "距離カテゴリ",
    "距離別過去平均着順",
    "騎手勝率",
    "騎手複勝率",
    "調教師勝率",
    "調教師複勝率",
    "距離",
    "馬場状態_num",
    "is_turf",
    "クラス_num",
    "前走間隔",
    "同距離過去勝率",
    "同距離過去平均着順",
    "良馬場勝率",
    "重馬場勝率",
    # 上がり関連特徴量（過去データのみ・当日混入なし）
    "過去最速上り",
    "上り偏差",
    "距離別過去平均上り",
    # 回収率向上：追加特徴量
    "斤量変化",
    "乗り替わり",
    "連闘",
    "休み明け",
    "負担率",
    # レース内ランク特徴量
    "レース内_過去勝率ランク",
    "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク",
    "レース内_騎手勝率ランク",
    # 競馬場×距離 適性
    "競馬場距離過去勝率", "競馬場距離過去平均着順",
    "競馬場過去勝率", "競馬場過去平均着順",
    # 脚質
    "過去平均先行指数", "先行馬フラグ",
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
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "min_gain_to_split": 0.1,
}


def add_extra_features(df):
    """
    回収率向上のための追加特徴量を生成する。
    feature.py の build_features 後に呼び出す想定。
    """
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)

    # ① 斤量変化（前走との差）
    df["斤量変化"] = df.groupby("馬名")["斤量"].diff()

    # ② 騎手乗り替わりフラグ
    df["乗り替わり"] = (
        df.groupby("馬名")["騎手"].shift(1) != df["騎手"]
    ).astype(int)

    # ③ 連闘フラグ（前走間隔が1週以内）
    df["連闘"] = (df["前走間隔"] <= 1).astype(int)

    # ④ 休み明けフラグ（前走間隔が12週以上）
    df["休み明け"] = (df["前走間隔"] >= 12).astype(int)

    # ⑤ 負担率（斤量 / 馬体重）
    df["負担率"] = df["斤量"] / df["馬体重"].replace(0, np.nan)

    # ⑥ 過去最速上り（過去レースの最高上がり）
    df["過去最速上り"] = df.groupby("馬名")["過去平均上り"].transform(
        lambda x: x.expanding().min().shift(1)
    )

    # ⑦ 上り偏差（過去上がりのばらつき＝安定性指標）
    df["上り偏差"] = df.groupby("馬名")["過去平均上り"].transform(
        lambda x: x.expanding().std().shift(1)
    )

    # ⑧ 距離別過去平均上り
    df["距離別過去平均上り"] = df.groupby(["馬名", "距離カテゴリ"])["過去平均上り"].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    return df


def train_model(csv_path="race_features.csv"):
    print("特徴量データ読み込み中...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]

    # レース内ランク特徴量
    print("レース内相対特徴量を生成中...")
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

    # 追加特徴量（騎手列が存在する場合のみ）
    print("追加特徴量を生成中...")
    if "騎手" in df.columns:
        df = add_extra_features(df)
    else:
        for col in ["斤量変化", "乗り替わり", "連闘", "休み明け", "負担率"]:
            df[col] = np.nan

    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    train_df = df[df["年"] <= 2024].copy()
    test_df  = df[df["年"] == 2025].copy()

    print(f"学習データ: {len(train_df)}行（〜2024）")
    print(f"検証データ: {len(test_df)}行（2025）")

    # FEATURE_COLS のうち実際に存在する列のみ使用
    use_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"使用特徴量: {len(use_cols)}列")

    X_train_all = train_df[use_cols].copy()
    y_train_all = (train_df["着順_num"] == 1).astype(int)
    X_test      = test_df[use_cols].copy()
    y_test      = test_df["着順_num"]

    # ── prefit方式：学習用80% + キャリブレーション用20% に分割 ──
    # sample_weight を確実に分割後のデータに対応させるため index で管理
    X_train_main, X_cal, y_train_main, y_cal = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42
    )

    # 正例（1着）を2倍に重み付け
    w_main = np.where(y_train_main == 1, 2.0, 1.0)

    print("ベースLightGBMモデルを学習中（アーリーストッピング付き）...")
    base_model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=5000)
    base_model.fit(
        X_train_main,
        y_train_main,
        sample_weight=w_main,
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(period=100)],
        eval_set=[(X_cal, y_cal)],
    )
    best_iter = base_model.best_iteration_
    print(f"  アーリーストッピング: {best_iter}本で停止")

    # ── isotonic回帰でキャリブレーション ──
    # sklearn 1.8+ では cv="prefit" が廃止され cv=None が同等動作
    # cv=None = 渡した estimator がすでに fit 済みであることを前提にキャリブレーションのみ実行
    print("確率キャリブレーション中（isotonic / cv=None）...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=None
    )
    calibrated_model.fit(X_cal, y_cal)

    # ── テストデータで評価 ──
    test_df = test_df.copy()
    test_df["予測勝率スコア"] = calibrated_model.predict_proba(X_test)[:, 1]

    models = [calibrated_model]

    test_df["予測順位"] = test_df.groupby("race_id")["予測勝率スコア"].rank(
        ascending=False
    )

    def calc_win_prob_normalized(group):
        probs = group["予測勝率スコア"].values
        total = probs.sum()
        return pd.Series(
            probs / total if total > 0 else probs, index=group.index
        )

    test_df["勝ち確率"] = (
        test_df.groupby("race_id")
        .apply(calc_win_prob_normalized)
        .reset_index(level=0, drop=True)
    )

    # 単勝期待値（単一定義）
    test_df["単勝期待値"] = test_df["単勝オッズ"] * test_df["勝ち確率"] - 1

    out = test_df[
        [
            "race_id",
            "馬名",
            "着順_num",
            "予測勝率スコア",
            "予測順位",
            "単勝オッズ",
            "人気",
            "勝ち確率",
            "単勝期待値",
        ]
    ].copy()
    out = out.sort_values(["race_id", "予測順位"])
    out.to_csv("model_result.csv", index=False, encoding="utf-8-sig")

    with open("model.pkl", "wb") as f:
        pickle.dump({"models": models, "use_cols": use_cols}, f)
    print("モデル保存完了 → model.pkl")

    return models, test_df, use_cols


if __name__ == "__main__":
    models, df, use_cols = train_model()

    print(f"\n{'='*40}\n🔥 バックテスト\n{'='*40}")

    TARGET_EV = 0.3

    # 戦略A: 予測1位 × 期待値>=0.3 × オッズ(1.5〜20倍)
    bets_a = df[
        (df["予測順位"] == 1)
        & (df["単勝期待値"] >= TARGET_EV)
        & (df["単勝オッズ"] >= 1.5)
        & (df["単勝オッズ"] <= 20.0)
    ]
    if len(bets_a) > 0:
        wins_a = bets_a[bets_a["着順_num"] == 1]
        print(f"\n── 戦略A: 予測1位 × 期待値>={TARGET_EV} × オッズ(1.5〜20倍) ──")
        print(f"ベット数: {len(bets_a)}回")
        print(f"的中数:   {len(wins_a)}回")
        print(f"的中率:   {len(wins_a)/len(bets_a)*100:.1f}%")
        print(f"回収率:   {wins_a['単勝オッズ'].sum() / len(bets_a) * 100:.1f}%")

    # 戦略A-2: 戦略Aから「1番人気」を除外
    bets_a2 = bets_a[bets_a["人気"] >= 2]
    if len(bets_a2) > 0:
        wins_a2 = bets_a2[bets_a2["着順_num"] == 1]
        print(f"\n── 戦略A-2: 戦略Aから「1番人気」を除外 ──")
        print(f"ベット数: {len(bets_a2)}回")
        print(f"的中数:   {len(wins_a2)}回")
        print(f"的中率:   {len(wins_a2)/len(bets_a2)*100:.1f}%")
        print(f"回収率:   {wins_a2['単勝オッズ'].sum() / len(bets_a2) * 100:.1f}%")

    # 戦略B: 回収率92.8%（赤字）のため廃止

    # 戦略C: 人気3番手以下 × 予測1位（市場乖離）× 期待値>=0.3（緩和）
    bets_c = df[
        (df["人気"] >= 3)
        & (df["予測順位"] == 1)
        & (df["単勝期待値"] >= 0.3)
    ]
    if len(bets_c) > 0:
        wins_c = bets_c[bets_c["着順_num"] == 1]
        print(f"\n── 戦略C: 人気3番手以下 × 予測1位（市場乖離） × 期待値>=0.3 ──")
        print(f"ベット数: {len(bets_c)}回")
        print(f"的中数:   {len(wins_c)}回")
        print(f"的中率:   {len(wins_c)/len(bets_c)*100:.1f}%")
        print(f"回収率:   {wins_c['単勝オッズ'].sum() / len(bets_c) * 100:.1f}%")

    # 戦略D: 前走間隔2〜4週 × 予測1位 × 期待値>=0.2（緩和）
    if "前走間隔" in df.columns:
        bets_d = df[
            (df["前走間隔"] >= 2)
            & (df["前走間隔"] <= 4)
            & (df["予測順位"] == 1)
            & (df["単勝期待値"] >= 0.2)
        ]
        if len(bets_d) > 0:
            wins_d = bets_d[bets_d["着順_num"] == 1]
            print(f"\n── 戦略D: 前走間隔2〜4週 × 予測1位 × 期待値>=0.2 ──")
            print(f"ベット数: {len(bets_d)}回")
            print(f"的中数:   {len(wins_d)}回")
            print(f"的中率:   {len(wins_d)/len(bets_d)*100:.1f}%")
            print(f"回収率:   {wins_d['単勝オッズ'].sum() / len(bets_d) * 100:.1f}%")