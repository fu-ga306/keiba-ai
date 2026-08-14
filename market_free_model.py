"""
market_free_model.py
────────────────────
市場情報（人気・オッズ）を一切使わない純粋能力ベースモデル。
通常モデル（model.pkl）と並行運用して「市場乖離スコア」を生成する。

使い方:
    python market_free_model.py
"""

import os
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
    "枠番", "馬番", "斤量",
    "斤量_相対", "年齢", "is_male",
    "is_female", "is_castrated", "馬体重",
    "体重増減", "馬体重_相対", "出走頭数",
    "競馬場cd", "レース番号", "日",
    "回", "過去出走数", "過去平均着順",
    "過去勝率", "過去複勝率", "過去平均上り",
    "直近3走平均着順", "過去平均タイム秒", "直近3走平均タイム秒",
    "過去最速タイム秒", "直近3走平均上り", "過去平均体重増減",
    "体重増減_過去標準偏差", "体重増減_異常度", "距離カテゴリ",
    "距離別過去平均着順", "騎手勝率", "騎手複勝率",
    "調教師勝率", "調教師複勝率", "距離",
    "馬場状態_num", "is_turf", "クラス_num",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順",
    "良馬場勝率", "重馬場勝率", "過去最速上り",
    "上り偏差", "距離別過去平均上り", "斤量変化",
    "乗り替わり", "連闘", "休み明け",
    "負担率", "レース内_過去勝率ランク", "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク", "レース内_騎手勝率ランク", "競馬場距離過去勝率",
    "競馬場距離過去平均着順", "競馬場過去勝率", "競馬場過去平均着順",
    "過去平均先行指数", "先行馬フラグ", "想定先行馬数",
    "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
    "開催月", "開催季節", "前走着順",
    "前走上り", "前走距離", "距離変化",
    "クラス変化", "前走着差_秒", "過去平均着差_秒",
    "近5走平均着差_秒", "近5走着差_std", "前走4角位置",
    "過去平均4角位置", "前走脚質指数", "芝ダート変更",
    "重賞出走フラグ", "過去重賞出走数", "連続複勝フラグ",
    "連続勝利フラグ", "近走改善度", "平均タイム差",
    "騎手競馬場勝率", "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率", "芝ダート×先行_過去勝率", "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走", "休養明け×距離延長",
    "連闘×距離短縮", "距離変化比率", "大幅延長フラグ",
    "大幅短縮フラグ", "距離延長幅", "長距離フラグ",
    "大幅延長×長距離", "距離延長×先行", "経験最長距離",
    "経験範囲超過", "延長×距離経験不足", "長距離複勝率",
    "前走余力", "延長×前走余力", "騎手直近勝率",
    "騎手直近複勝率", "騎手調子トレンド", "直近5走勝利数",
    "直近5走複勝数", "直近5走平均着順", "過去獲得賞金累計",
    "過去平均獲得賞金", "回り_num", "回り別_過去勝率",
    "回り別_過去複勝率", "直線長_m", "坂あり",
    "クラス_num_R偏差", "クラス_num_R順", "クラス変化_R偏差",
    "クラス変化_R順", "コース先行勝率_R偏差", "コース先行勝率_R順",
    "コース好走相対4角_R偏差", "コース好走相対4角_R順", "コース差し複勝率_R偏差",
    "コース差し複勝率_R順", "コース脚質バイアス_R偏差", "コース脚質バイアス_R順",
    "メンバークラス平均_R偏差", "メンバークラス平均_R順", "メンバー賞金平均_R偏差",
    "メンバー賞金平均_R順", "メンバー過去勝率平均_R偏差", "メンバー過去勝率平均_R順",
    "メンバー過去勝率最大_R偏差", "メンバー過去勝率最大_R順", "レース番号_R偏差",
    "レース番号_R順", "他馬想定先行馬数_R偏差", "他馬想定先行馬数_R順",
    "体重増減_R偏差", "体重増減_R順", "体重増減_異常度_R偏差",
    "体重増減_異常度_R順", "先行圧_R偏差", "先行圧_R順",
    "先行有利コース×先行馬_R偏差", "先行有利コース×先行馬_R順", "出走頭数_R偏差",
    "出走頭数_R順", "前走4角位置_R偏差", "前走4角位置_R順",
    "前走上り_R偏差", "前走上り_R順", "前走余力_R偏差",
    "前走余力_R順", "前走着差_秒_R偏差", "前走着差_秒_R順",
    "前走着順_R偏差", "前走着順_R順", "前走脚質指数_R偏差",
    "前走脚質指数_R順", "前走距離_R偏差", "前走距離_R順",
    "前走間隔_R偏差", "前走間隔_R順", "同距離過去勝率_R偏差",
    "同距離過去勝率_R順", "同距離過去平均着順_R偏差", "同距離過去平均着順_R順",
    "回り_num_R偏差", "回り_num_R順", "回り別_過去勝率_R偏差",
    "回り別_過去勝率_R順", "回り別_過去複勝率_R偏差", "回り別_過去複勝率_R順",
    "差し×先行圧_R偏差", "差し×先行圧_R順", "差し有利コース×差し馬_R偏差",
    "差し有利コース×差し馬_R順", "差し馬×ハイペース想定_R偏差", "差し馬×ハイペース想定_R順",
    "年齢_R偏差", "年齢_R順", "延長×前走余力_R偏差",
    "延長×前走余力_R順", "延長×距離経験不足_R偏差", "延長×距離経験不足_R順",
    "想定先行馬数_R偏差", "想定先行馬数_R順", "想定先行馬率_R偏差",
    "想定先行馬率_R順", "想定逃げ馬数_R偏差", "想定逃げ馬数_R順",
    "斤量_R偏差", "斤量_R順", "斤量_相対_R偏差",
    "斤量_相対_R順", "斤量×年齢_負担_R偏差", "斤量×年齢_負担_R順",
    "斤量変化_R偏差", "斤量変化_R順", "日_R偏差",
    "日_R順", "枠番_R偏差", "枠番_R順",
    "父系_今回距離適性_R偏差", "父系_今回距離適性_R順", "父系_芝ダ適性_R偏差",
    "父系_芝ダ適性_R順", "父系_複勝率_R偏差", "父系_複勝率_R順",
    "父系_長距離勝率_R偏差", "父系_長距離勝率_R順", "直線長_m_R偏差",
    "直線長_m_R順", "直線長変化_R偏差", "直線長変化_R順",
    "直近3走平均タイム秒_R偏差", "直近3走平均タイム秒_R順", "直近3走平均上り_R偏差",
    "直近3走平均上り_R順", "直近3走平均着順_R偏差", "直近3走平均着順_R順",
    "直近5走勝利数_R偏差", "直近5走勝利数_R順", "直近5走平均着順_R偏差",
    "直近5走平均着順_R順", "直近5走着外率_R偏差", "直近5走着外率_R順",
    "直近5走着順_std_R偏差", "直近5走着順_std_R順", "直近5走複勝数_R偏差",
    "直近5走複勝数_R順", "競馬場cd_R偏差", "競馬場cd_R順",
    "競馬場過去勝率_R偏差", "競馬場過去勝率_R順", "競馬場過去平均着順_R偏差",
    "競馬場過去平均着順_R順", "経験最長距離_R偏差", "経験最長距離_R順",
    "経験範囲超過_R偏差", "経験範囲超過_R順", "脚質コース適合_R偏差",
    "脚質コース適合_R順", "脚質スコア_R偏差", "脚質スコア_R順",
    "芝ダート×先行_過去勝率_R偏差", "芝ダート×先行_過去勝率_R順", "調教師勝率_R偏差",
    "調教師勝率_R順", "調教師勝率_sm_R偏差", "調教師勝率_sm_R順",
    "調教師複勝率_R偏差", "調教師複勝率_R順", "負担率_R偏差",
    "負担率_R順", "距離_R偏差", "距離_R順",
    "距離×クラス_過去勝率_R偏差", "距離×クラス_過去勝率_R順", "距離×馬場_過去平均着順_R偏差",
    "距離×馬場_過去平均着順_R順", "距離カテゴリ_R偏差", "距離カテゴリ_R順",
    "距離別過去平均上り_R偏差", "距離別過去平均上り_R順", "距離別過去平均着順_R偏差",
    "距離別過去平均着順_R順", "距離変化_R偏差", "距離変化_R順",
    "距離変化比率_R偏差", "距離変化比率_R順", "距離延長×先行_R偏差",
    "距離延長×先行_R順", "距離延長幅_R偏差", "距離延長幅_R順",
    "近5走平均着差_秒_R偏差", "近5走平均着差_秒_R順", "近5走着差_std_R偏差",
    "近5走着差_std_R順", "近走改善度_R偏差", "近走改善度_R順",
    "逃げ争い_R偏差", "逃げ争い_R順", "過去出走数_R偏差",
    "過去出走数_R順", "過去勝率_R偏差", "過去勝率_R順",
    "過去平均4角位置_R偏差", "過去平均4角位置_R順", "過去平均タイム秒_R偏差",
    "過去平均タイム秒_R順", "過去平均上り_R偏差", "過去平均上り_R順",
    "過去平均体重増減_R偏差", "過去平均体重増減_R順", "過去平均先行指数_R偏差",
    "過去平均先行指数_R順", "過去平均獲得賞金_R偏差", "過去平均獲得賞金_R順",
    "過去平均着差_秒_R偏差", "過去平均着差_秒_R順", "過去平均着順_R偏差",
    "過去平均着順_R順", "過去最速タイム秒_R偏差", "過去最速タイム秒_R順",
    "過去最速上り_R偏差", "過去最速上り_R順", "過去獲得賞金累計_R偏差",
    "過去獲得賞金累計_R順", "過去着順_std_R偏差", "過去着順_std_R順",
    "過去複勝率_R偏差", "過去複勝率_R順", "過去重賞出走数_R偏差",
    "過去重賞出走数_R順", "開催季節_R偏差", "開催季節_R順",
    "開催月_R偏差", "開催月_R順", "馬主勝率_R偏差",
    "馬主勝率_R順", "馬主勝率_sm_R偏差", "馬主勝率_sm_R順",
    "馬主複勝率_R偏差", "馬主複勝率_R順", "馬体重_R偏差",
    "馬体重_R順", "馬体重_相対_R偏差", "馬体重_相対_R順",
    "騎手勝率_R偏差", "騎手勝率_R順", "騎手勝率_sm_R偏差",
    "騎手勝率_sm_R順", "騎手直近複勝率_R偏差", "騎手直近複勝率_R順",
    "騎手競馬場勝率_R偏差", "騎手競馬場勝率_R順", "騎手複勝率_R偏差",
    "騎手複勝率_R順",
]

# ── 差し替え用の新しい特徴量リスト（2026-08-03準備・まだ未使用）─────────────
# 2026-08-03の調査で判明した3点を反映したもの:
#   ①レース内定数(距離・競馬場cd・出走頭数・開催月など22列)の相対化を除去。
#     _R偏差は全行NaN、_R順は意味のない連番で、44列が中身なく枠を占めていた。
#   ②コース系の生列を追加（+0.7pt / raw_ablation.py）。
#     レース内定数なので相対化では消えるが、生の値には情報がある。
#   ③速度・上り指数の履歴14列を追加（+0.4pt / speed_ablation.py）。
# 見送ったもの: メンバー系(±0)・展開系(-0.4)・血統/馬主(-1.2)・相対化の素材拡大(-0.6)。
#
# ⚠️ FEATURE_COLS_MF はデプロイ済みモデルの use_cols と一致していないと予測が落ちる。
#    切り替えは「特徴量の再生成 → MF再学習 → 検証」を一続きで行うこと。
#    手順は apply_v2.py にまとめてある。単独で FEATURE_COLS_MF = FEATURE_COLS_MF_V2
#    と書き換えるだけでは本番が停止する。
def _build_v2():
    try:
        from features import RACE_CONST_COLS, RAW_ADOPT_COLS, SPEED_HIST_COLS
    except Exception:
        return list(FEATURE_COLS_MF)
    dead = {f"{c}{s}" for c in RACE_CONST_COLS for s in ("_R偏差", "_R順")}
    keep = [c for c in FEATURE_COLS_MF if c not in dead]
    add = [c for c in list(RAW_ADOPT_COLS) + list(SPEED_HIST_COLS) if c not in keep]
    return keep + add


FEATURE_COLS_MF_V2 = _build_v2()

# features.py と同じフラグファイルで切り替える。両方が同時に切り替わらないと
# 「CSVの列」と「モデルが要求する列」が食い違って予測が落ちる。
# フラグがあるのに切り替わらない状態が一番危険なので、その場合は黙って
# 続行せず落とす（握り潰すと学習と予測がズレたまま本番が動いてしまう）。
_V2_FLAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "features_v2.flag")
if os.path.exists(_V2_FLAG):
    if len(FEATURE_COLS_MF_V2) == len(FEATURE_COLS_MF):
        raise RuntimeError(
            "features_v2.flag があるのに V2 の特徴量リストを組み立てられていません。"
            "features.py の RACE_CONST_COLS / RAW_ADOPT_COLS / SPEED_HIST_COLS を確認してください。")
    FEATURE_COLS_MF = FEATURE_COLS_MF_V2

# ── 距離で2モデルに分ける（2026-08-14）────────────────────────────────
#   なぜ分けるか
#     騎手・調教師・馬主の特徴31列を抜くと、市場を条件に入れたΔR²が上がる。
#     さらに距離で切って別モデルにすると、もう一段上がる。
#     walk-forward（学習=検証年より前）でクリーンデータ 322,205頭を検証:
#
#       条件          A 現行(全特徴)  B 全体で除外  C 距離切替
#       全体            +0.00039     +0.00054    +0.00062  ← C
#       長距離1900+     +0.00041     +0.00076    +0.00079  ← C
#       短距離-1400     +0.00033     +0.00051    +0.00062  ← C
#       道悪            +0.00123     +0.00139    +0.00158  ← C
#       多頭数16+       +0.00028     +0.00049    +0.00051  ← C
#
#     5条件すべてでCが最良。全体で 1.6倍 になる。
#
#   なぜ騎手厩舎が害になるか（解釈）
#     騎手の勝率は「どの馬に乗るか」で決まる面が大きく、その情報は既にオッズに
#     入っている。市場を条件に入れると、残るのはノイズだけになる。
#
#   ⚠ 回収率が上がる保証はない。必要量(Benter 0.0178)の1/29が1/19になるだけ。
#   ⚠ この定数を変えたら必ずMFの再学習が要る。use_cols が合わないと予測が落ちる。
MF_DIST_SPLIT = 1900          # この距離以上は全特徴、未満は騎手厩舎を除外
_JOCKEY_PAT = r"騎手|調教師|馬主"


def mf_cols_for(distance=None):
    """距離に応じた特徴量リストを返す。距離不明なら安全側で全特徴。"""
    import re as _re
    if distance is None:
        return list(FEATURE_COLS_MF)
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return list(FEATURE_COLS_MF)
    if d >= MF_DIST_SPLIT:
        return list(FEATURE_COLS_MF)
    return [c for c in FEATURE_COLS_MF if not _re.search(_JOCKEY_PAT, c)]


FEATURE_COLS_MF_SHORT = mf_cols_for(0)      # 短距離側（騎手厩舎を除外）

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "learning_rate":     0.03,
    "num_leaves":        63,
    "min_child_samples": 15,
    "feature_fraction":  0.75,
    "bagging_fraction":  0.8,
    "bagging_freq":      1,
    "lambda_l1":         0.05,
    "lambda_l2":         0.05,
    "verbose":           -1,
    "min_gain_to_split": 0.05,
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
    print("[MF] 市場フリーモデル 学習開始")
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
    TIME_WEIGHT_MAX = 2.0  # 最古年=1.0倍、最新年=2.0倍（model.py と同値）
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
    print("[!!] 市場フリーモデル バックテスト（2025年）")
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
    print("[分析] 通常モデル vs 市場フリーモデル 比較")
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

    TIME_WEIGHT_MAX = 2.0  # 最古年=1.0倍、最新年=2.0倍
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
    print("[MF] 市場フリーモデル 3モデル化 学習開始")
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
    print(f"\n[完了] MF 3モデル保存完了 → model_mf.pkl")
    print(f"   win/place2/place3 各{len(result['win']['models'])}モデル（市場フリー）")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # 旧来の単一モデル学習
        train_market_free_model()
    else:
        # デフォルト: 3モデル化（推奨）
        train_mf_all_targets()