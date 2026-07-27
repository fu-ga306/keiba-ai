import os
import sys
import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 学習モード（2026-07-16 二本立て化）──────────────────────────────────
#   本番（デフォルト）      : 全データで学習 → model.pkl（直近レースまで取り込みdrift低減）
#   backtest（引数指定時）  : 〜2024学習/2025検証 → model_bt.pkl + model_result*.csv
#   ※ honest backtest資産(model_result*.csv)は backtestモードでのみ更新される。
#     本番モードは絶対に上書きしない（2025がin-sampleになり数字が嘘になるため）。
BT_MODE = "backtest" in sys.argv

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

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

FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気", "出走頭数", "競馬場cd", "レース番号", "日", "回",
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
    "斤量変化", "乗り替わり", "連闘", "休み明け", "負担率",
    "レース内_過去勝率ランク", "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク", "レース内_騎手勝率ランク",
    "競馬場距離過去勝率", "競馬場距離過去平均着順",
    "競馬場過去勝率", "競馬場過去平均着順",
    "過去平均先行指数", "先行馬フラグ",
    "想定先行馬数", "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
    # 精密展開特徴（2026-07-27）は検証で悪化したためモデル未使用。
    #   同条件BT: 全体198.5%→191.8%(収支-17.7万)。勝率側は改善(単勝192.8→197.9%)したが
    #   複勝側が悪化し相手プールが劣化→馬単251.9→233.6%で相殺。features.pyでは生成のみ継続。
    "開催月", "開催季節",
    "前走着順", "前走上り", "前走距離", "距離変化", "クラス変化",
    "前走着差_秒", "過去平均着差_秒", "近5走平均着差_秒", "近5走着差_std",
    "前走4角位置", "過去平均4角位置", "前走脚質指数",
    "芝ダート変更",
    "重賞出走フラグ", "過去重賞出走数",
    "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
    "平均タイム差", "騎手競馬場勝率",
    # 交互作用特徴量
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率",
    "芝ダート×先行_過去勝率",
    "前走好走×人気薄", "前走着順×人気_乖離",
    "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
    "休養明け×距離延長", "連闘×距離短縮",
    "接戦×人気薄",
    # 距離変化系（長距離・距離延長の弱点対策）
    "距離変化比率", "大幅延長フラグ", "大幅短縮フラグ",
    "距離延長幅", "長距離フラグ", "大幅延長×長距離", "距離延長×先行",
    "経験最長距離", "経験範囲超過", "延長×距離経験不足",
    "長距離複勝率", "前走余力", "延長×前走余力",
    "騎手直近勝率", "騎手直近複勝率", "騎手調子トレンド",
    "直近5走勝利数", "直近5走複勝数", "直近5走平均着順",
    # 過去獲得賞金（B-4: 実力指標）
    "過去獲得賞金累計", "過去平均獲得賞金",
    # コース形態（B-2: データ整備完了）
    "回り_num", "回り別_過去勝率", "回り別_過去複勝率",
    "直線長_m", "坂あり",
    # 血統特徴量（C-3: horse_id回復後に効く）
    "父系_今回距離適性", "母父系_今回距離適性",
    "父系_長距離勝率", "母父系_長距離勝率",
    "父系_芝ダ適性", "父系_複勝率",
    # 着順安定性（複勝精度向け）
    "過去着順_std", "直近5走着順_std", "直近5走着外率",
    # ターゲットエンコーディング（B1: 馬主・スムージング版）
    "騎手勝率_sm", "調教師勝率_sm",
    "馬主勝率", "馬主複勝率", "馬主勝率_sm",
    # コース脚質バイアス（実装1）
    "コース好走相対4角", "コース先行勝率", "コース差し複勝率",
    "脚質コース適合", "先行有利コース×先行馬", "差し有利コース×差し馬",
    # コース替わり（実装2）
    "回り替わりフラグ", "直線長変化", "坂替わりフラグ", "初コースフラグ",
    # レース内偏差値化（相対競争の強調）
    "過去勝率_レース内偏差", "過去複勝率_レース内偏差", "直近3走平均着順_レース内偏差",
    "過去平均上り_レース内偏差", "過去獲得賞金累計_レース内偏差", "騎手勝率_レース内偏差",
    "父系_今回距離適性_レース内偏差", "直近5走平均着順_レース内偏差",
    # メンバーレベル（相手の強さ）
    "メンバー過去勝率平均", "メンバー過去勝率最大", "過去勝率_対相手",
    "メンバークラス平均", "メンバー賞金平均", "賞金_対相手",
]

# SHAP特徴量選択（実装3）で寄与ほぼ0と判定された列。
# select_features_shap.py の出力を貼り付けて再学習すると、これらを除外する。
# 空リストなら全特徴量を使用（従来通り）。
# 2026/07/02 SHAP再算出結果（170列・平均|SHAP| < 0.0005・寄与ほぼ0）:
SHAP_DROP_COLS = [
    "大幅短縮フラグ",
    "馬場状態_num",
    "前走好走×人気薄",
    "連続複勝フラグ",
    "距離延長×前走好走",
    "距離短縮×前走好走",
    "大幅延長フラグ",
    "is_castrated",
    "大幅延長×長距離",
    "連続勝利フラグ",
    "先行馬フラグ",
    "長距離フラグ",
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 63,          # 31→63: 特徴量増加に合わせモデルを少し複雑に
    "min_child_samples": 15,   # 20→15: 少し細かい分岐を許可
    "feature_fraction": 0.75,  # 0.8→0.75: 特徴量増加後は少し絞る
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.05,         # L1正則化追加（過学習防止）
    "lambda_l2": 0.05,         # L2正則化追加
    "verbose": -1,
    "min_gain_to_split": 0.05, # 0.1→0.05: 新特徴量の弱いシグナルも拾う
}



def tune_lgb_params(X_train, y_train, X_val, y_val, w_train, n_trials=60):
    """Optuna で LightGBM のハイパーパラメータを最適化する。
    Optuna 未インストール時はデフォルトパラメータを返す。"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna未インストール → デフォルトパラメータを使用")
        return LGB_PARAMS.copy()

    from sklearn.metrics import log_loss

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "bagging_freq": 1,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 0.3),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 0.3),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.1),
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        return log_loss(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update({"objective": "binary", "metric": "binary_logloss", "verbose": -1, "bagging_freq": 1})
    print(f"  Optuna完了: {n_trials}試行 best_loss={study.best_value:.4f} "
          f"leaves={best.get('num_leaves')} lr={best.get('learning_rate'):.4f}")
    return best


def optimize_ensemble_weights(models, X_val, y_val):
    """検証データでアンサンブル各モデルの重みをlog-loss最小化で最適化する。
    非負・合計1に正規化した重みを返す。最適化失敗時は均等重み。"""
    from sklearn.metrics import log_loss
    preds = np.column_stack([m.predict_proba(X_val)[:, 1] for m in models])
    n = preds.shape[1]
    equal = np.ones(n) / n
    if n <= 1:
        return equal
    try:
        from scipy.optimize import minimize

        def loss(raw):
            w = np.abs(raw)
            s = w.sum()
            if s <= 0:
                return 1e9
            w = w / s
            p = np.clip(preds @ w, 1e-6, 1 - 1e-6)
            return log_loss(y_val, p)

        # 複数初期値から最良を採用（Nelder-Meadは初期値依存があるため）
        best_w, best_l = equal, loss(equal)
        for init in [equal, np.eye(n)[0], np.eye(n)[1] if n > 1 else equal]:
            res = minimize(loss, init, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
            w = np.abs(res.x); w = w / w.sum()
            if res.fun < best_l:
                best_l, best_w = res.fun, w
        l_equal = loss(equal)
        print(f"  アンサンブル重み最適化: 均等log_loss={l_equal:.4f} → 最適={best_l:.4f}")
        print(f"    重み: {[f'{x:.2f}' for x in best_w]}")
        return best_w
    except Exception as e:
        print(f"  重み最適化スキップ（均等重み使用）: {e}")
        return equal


def weighted_proba(models, X, weights=None):
    """重み付きアンサンブル予測。weightsがNoneなら均等平均。"""
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    if weights is None:
        return preds.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return preds @ w


class LambdaRankWrapper:
    """LambdaRankモデルをsklearn風にラップ（pickle対応）"""
    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        import numpy as _np
        if hasattr(X, "values"):
            X = X.values
        scores = self.booster.predict(X)
        std    = scores.std()
        probs  = 1 / (1 + _np.exp(-scores / max(std, 1e-9)))
        return _np.column_stack([1 - probs, probs])


LGB_RANK_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5],
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 15,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 0.05,
    "verbose": -1,
    "min_gain_to_split": 0.05,
}

def add_extra_features(df):
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
    df["斤量変化"]  = df.groupby("馬名")["斤量"].diff()
    df["乗り替わり"] = (df.groupby("馬名")["騎手"].shift(1) != df["騎手"]).astype(int)
    df["連闘"]      = (df["前走間隔"] <= 1).astype(int)
    df["休み明け"]  = (df["前走間隔"] >= 12).astype(int)
    df["負担率"]    = df["斤量"] / df["馬体重"].replace(0, np.nan)
    df["過去最速上り"] = df.groupby("馬名")["過去平均上り"].transform(lambda x: x.expanding().min().shift(1))
    df["上り偏差"]     = df.groupby("馬名")["過去平均上り"].transform(lambda x: x.expanding().std().shift(1))
    df["距離別過去平均上り"] = df.groupby(["馬名", "距離カテゴリ"])["過去平均上り"].transform(lambda x: x.expanding().mean().shift(1))
    return df


def make_target(chaku_series, target):
    """目的変数を作る。target に応じて正例の定義を変える。
      win    : 1着       (着順==1)
      place2 : 連対(2着以内, 着順<=2)
      place3 : 複勝(3着以内, 着順<=3)
    """
    if target == "win":
        return (chaku_series == 1).astype(int)
    elif target == "place2":
        return (chaku_series <= 2).astype(int)
    elif target == "place3":
        return (chaku_series <= 3).astype(int)
    else:
        raise ValueError(f"未知のtarget: {target}")


# target ごとの「正例を重視する重み」（正例にこの倍率をかける）
TARGET_POS_WEIGHT = {"win": 2.0, "place2": 1.7, "place3": 1.5}


def train_model(csv_path="race_features.csv", target="win", df=None):
    """1つの目的変数(target)についてアンサンブルモデルを学習して返す。
    target: "win" / "place2" / "place3"
    df: 既に読み込み済みのDataFrameがあれば渡す（3回読み込まないため）
    """
    if df is None:
        print("特徴量データ読み込み中...")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["着順_num"])
        df = df[df["着順_num"] >= 1]
    print(f"\n{'#'*50}\n# ターゲット: {target} を学習\n{'#'*50}")

    print("レース内相対特徴量を生成中...")
    df["レース内_過去勝率ランク"]         = df.groupby("race_id")["過去勝率"].rank(ascending=False, method="min")
    df["レース内_直近3走平均着順ランク"]  = df.groupby("race_id")["直近3走平均着順"].rank(ascending=True, method="min")
    df["レース内_過去平均上りランク"]     = df.groupby("race_id")["過去平均上り"].rank(ascending=True, method="min")
    df["レース内_騎手勝率ランク"]         = df.groupby("race_id")["騎手勝率"].rank(ascending=False, method="min")

    print("追加特徴量を生成中...")
    if "騎手" in df.columns:
        df = add_extra_features(df)
    else:
        # features.py で計算済みの場合は上書きしない（NaN率が高い場合のみ NaN でセット）
        for col in ["斤量変化", "連闘", "休み明け", "負担率", "乗り替わり"]:
            if col not in df.columns or df[col].isna().all():
                df[col] = np.nan

    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    if BT_MODE:
        # バックテストモード: 〜(TEST_YEAR-1)学習 / TEST_YEAR検証（honest評価）
        _ty = int(os.environ.get("KEIBA_TEST_YEAR", "2025"))   # 多年度検証用: テスト年を可変に
        train_df = df[df["年"] <= _ty - 1].copy()
        test_df  = df[df["年"] == _ty].copy()
        print(f"学習データ: {len(train_df)}行（〜{_ty-1}）[backtestモード]")
        print(f"検証データ: {len(test_df)}行（{_ty}）")
    else:
        # 本番モード: 全データで学習（直近年まで）。test_dfは最新年のin-sample参考値のみ。
        train_df = df.copy()
        test_df  = df[df["年"] == df["年"].max()].copy()
        print(f"学習データ: {len(train_df)}行（全期間〜{df['年'].max()}）[本番モード]")
        print(f"参考表示: {len(test_df)}行（{df['年'].max()}・in-sampleのため精度指標は参考値）")

    use_cols = [c for c in FEATURE_COLS if c in train_df.columns and c not in SHAP_DROP_COLS]
    if SHAP_DROP_COLS:
        print(f"  SHAP除外: {len(SHAP_DROP_COLS)}列を除外")
    print(f"使用特徴量: {len(use_cols)}列")

    X_train_all = train_df[use_cols].copy()
    y_train_all = make_target(train_df["着順_num"], target)
    X_test      = test_df[use_cols].copy()
    y_test      = test_df["着順_num"]

    # ── 時系列重み（直近年ほど重みを大きくする） ──────────────────────
    # 最古年=1.0倍、最新年=TIME_WEIGHT_MAX倍 で線形増加。
    # 1.0だと全年同重みで無効。2.0が実用的な設定。
    TIME_WEIGHT_MAX = 2.0
    year_min = train_df["年"].min()
    year_max = train_df["年"].max()
    year_range = max(year_max - year_min, 1)
    train_df["時系列重み"] = 1.0 + (train_df["年"] - year_min) / year_range * (TIME_WEIGHT_MAX - 1.0)

    X_train_main, X_cal_full, y_train_main, y_cal_full, w_time_main, w_time_cal = train_test_split(
        X_train_all, y_train_all, train_df["時系列重み"], test_size=0.2, random_state=42
    )
    # X_cal_full をさらに分割: calibration/early-stopping用(X_cal) と アンサンブル重み最適化専用(X_calw)。
    # 重み最適化を calibration と同じデータで行うと、isotonic過学習したモデルを過大評価する
    # リークが生じる（重みがLGB単独に偏る）。独立ホールドアウトで防ぐ。
    X_cal, X_calw, y_cal, y_calw = train_test_split(
        X_cal_full, y_cal_full, test_size=0.4, random_state=42
    )
    # 着順重み（正例を重視）× 時系列重み（直近年を重視）
    pos_w = TARGET_POS_WEIGHT.get(target, 2.0)
    w_main = np.where(y_train_main == 1, pos_w, 1.0) * w_time_main.values
    print(f"  時系列重み: {year_min}年=1.0倍 〜 {year_max}年={TIME_WEIGHT_MAX}倍")
    print(f"  正例重み({target}): {pos_w}倍")

    # ── LightGBM（Optunaでハイパーパラメータ最適化）──
    print("Optunaでハイパーパラメータ最適化中（60試行）...")
    best_lgb_params = tune_lgb_params(
        X_train_main, y_train_main, X_cal, y_cal, w_main, n_trials=60
    )
    print("ベースLightGBMモデルを学習中（アーリーストッピング付き）...")
    base_model = lgb.LGBMClassifier(**best_lgb_params, n_estimators=5000)
    base_model.fit(
        X_train_main, y_train_main,
        sample_weight=w_main,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=100)],
        eval_set=[(X_cal, y_cal)],
    )
    print(f"  アーリーストッピング: {base_model.best_iteration_}本で停止")
    print("確率キャリブレーション中（LightGBM）...")
    calibrated_lgb = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=None)
    calibrated_lgb.fit(X_cal, y_cal)
    models = [calibrated_lgb]

    # ── XGBoost ──
    if HAS_XGB:
        print("XGBoostモデルを学習中...")
        # step1: early_stoppingでベストイテレーション取得
        xgb_tmp = xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.03, max_depth=5,
            n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, eval_metric="logloss",
            early_stopping_rounds=100, verbosity=0, random_state=42,
        )
        xgb_tmp.fit(X_train_main, y_train_main, eval_set=[(X_cal, y_cal)],
                    sample_weight=w_time_main.values, verbose=False)
        best_iter_xgb = xgb_tmp.best_iteration
        print(f"  XGBoost ベストイテレーション: {best_iter_xgb}本")
        # step2: fixed iterationsで再学習（early_stopping不要）
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic", learning_rate=0.03, max_depth=5,
            n_estimators=best_iter_xgb, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, verbosity=0, random_state=42,
        )
        xgb_model.fit(X_train_main, y_train_main, sample_weight=w_time_main.values, verbose=False)
        calibrated_xgb = CalibratedClassifierCV(estimator=xgb_model, method="isotonic", cv=None)
        calibrated_xgb.fit(X_cal, y_cal)
        models.append(calibrated_xgb)
        print(f"  XGBoost完了: {best_iter_xgb}本")
    else:
        print("  XGBoostスキップ（pip install xgboost）")

    # ── CatBoost ──
    if HAS_CAT:
        print("CatBoostモデルを学習中...")
        # step1: early_stoppingでベストイテレーション取得
        cat_tmp = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=5,
            loss_function="Logloss", eval_metric="Logloss",
            early_stopping_rounds=100, verbose=False, random_seed=42,
        )
        cat_tmp.fit(X_train_main, y_train_main, eval_set=(X_cal, y_cal), sample_weight=w_main)
        best_iter_cat = cat_tmp.best_iteration_
        print(f"  CatBoost ベストイテレーション: {best_iter_cat}本")
        # step2: fixed iterationsで再学習（early_stopping不要）
        cat_model = CatBoostClassifier(
            iterations=best_iter_cat, learning_rate=0.03, depth=5,
            loss_function="Logloss", verbose=False, random_seed=42,
        )
        cat_model.fit(X_train_main, y_train_main, sample_weight=w_main)
        calibrated_cat = CalibratedClassifierCV(estimator=cat_model, method="isotonic", cv=None)
        calibrated_cat.fit(X_cal, y_cal)
        models.append(calibrated_cat)
        print(f"  CatBoost完了: {best_iter_cat}本")
    else:
        print("  CatBoostスキップ（pip install catboost）")

    # ── LightGBM LambdaRank（着順ランキング学習） ── 全ターゲット共通
    # ndcg_eval_at はターゲットに応じて調整: win=top3&5, place2=top2&3, place3=top3&5
    _ndcg_at = [2, 3] if target == "place2" else [3, 5]
    lambda_wrapper = None
    print(f"LambdaRankモデルを学習中（ndcg@{_ndcg_at}）...")
    try:
        idx_train = X_train_main.index
        idx_cal   = X_cal.index

        race_id_train = train_df.loc[idx_train, "race_id"].values
        race_id_cal   = train_df.loc[idx_cal,   "race_id"].values

        sort_train = np.argsort(race_id_train, kind="stable")
        sort_cal   = np.argsort(race_id_cal,   kind="stable")

        X_rank_train = X_train_main.values[sort_train]
        X_rank_cal   = X_cal.values[sort_cal]

        chakujun_train = train_df.loc[idx_train, "着順_num"].values[sort_train]
        chakujun_cal   = train_df.loc[idx_cal,   "着順_num"].values[sort_cal]
        shutsu_train   = train_df.loc[idx_train, "出走頭数"].fillna(18).astype(int).values[sort_train]
        shutsu_cal     = train_df.loc[idx_cal,   "出走頭数"].fillna(18).astype(int).values[sort_cal]

        y_rank_train = np.clip(shutsu_train - chakujun_train, 0, None).astype(int)
        y_rank_cal   = np.clip(shutsu_cal   - chakujun_cal,   0, None).astype(int)

        _, group_train = np.unique(race_id_train[sort_train], return_counts=True)
        _, group_cal   = np.unique(race_id_cal[sort_cal],     return_counts=True)

        train_data_rank = lgb.Dataset(X_rank_train, label=y_rank_train, group=group_train)
        val_data_rank   = lgb.Dataset(X_rank_cal,   label=y_rank_cal,   group=group_cal,
                                      reference=train_data_rank)

        rank_params = dict(LGB_RANK_PARAMS)
        rank_params["ndcg_eval_at"] = _ndcg_at
        rank_booster = lgb.train(
            rank_params,
            train_data_rank,
            num_boost_round=3000,
            valid_sets=[val_data_rank],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=200)],
        )

        lambda_wrapper = LambdaRankWrapper(rank_booster)
        models.append(lambda_wrapper)
        print(f"  LambdaRank完了（{rank_booster.best_iteration}本）→ アンサンブルに追加")
    except Exception as e:
        lambda_wrapper = None
        print(f"  LambdaRankスキップ: {e}")
        import traceback; traceback.print_exc()

    n_base = len(models) - (1 if lambda_wrapper else 0)
    n_total = len(models)
    print(f"\nアンサンブル: {n_total}モデルで予測（LGB+XGB+CatBoost"
          + ("+LambdaRank" if lambda_wrapper else "") + "）")

    # ── アンサンブル重み最適化（calibration未使用の独立ホールドアウトで探索）──
    print("アンサンブル重みを最適化中（独立ホールドアウト使用）...")
    ens_weights = optimize_ensemble_weights(models, X_calw, y_calw)

        # ── テストデータで評価 ──
    test_df = test_df.copy()
    # target汎用のスコア列（その目的の確率）※最適重みで加重平均
    test_df["予測スコア"] = weighted_proba(models, X_test, ens_weights)
    test_df["予測順位"] = test_df.groupby("race_id")["予測スコア"].rank(ascending=False)

    # 正例実績（この target における的中）
    y_test_bin = make_target(test_df["着順_num"], target)
    top1 = test_df[test_df["予測順位"] == 1]
    if len(top1) > 0:
        hit_rate = make_target(top1["着順_num"], target).mean() * 100
        print(f"  [{target}] 予測1位の的中率(この目的での): {hit_rate:.1f}%  ({len(top1)}レース)")

    if target == "win":
        # 案B: 勝ち確率は「生確率」をそのまま使う（正規化しない）。
        # 生確率 = winモデルが出す「この馬が1着になる絶対確率」。
        # これにより 勝率 ≤ 連対率 ≤ 複勝率 の包含関係が自然に保たれる。
        test_df["勝ち確率"] = test_df["予測スコア"]
        test_df["単勝期待値"] = test_df["単勝オッズ"] * test_df["勝ち確率"] - 1
        if BT_MODE:
            out = test_df[["race_id","馬名","着順_num","予測スコア","予測順位","単勝オッズ","人気","勝ち確率","単勝期待値"]].copy()
            out = out.sort_values(["race_id", "予測順位"])
            out.to_csv("model_result.csv", index=False, encoding="utf-8-sig")

    elif target in ("place2", "place3") and BT_MODE:
        # ○▲△×バックテスト用: 連対/複勝モデルのテスト予測を出力する（backtestモードのみ）。
        # 本番モードはin-sampleなので出力しない＝honest資産を上書きしない。
        _cols = [c for c in ["race_id","馬名","着順_num","予測スコア","予測順位","単勝オッズ","人気"]
                 if c in test_df.columns]
        out = test_df[_cols].copy().sort_values(["race_id", "予測順位"])
        out.to_csv(f"model_result_{target}.csv", index=False, encoding="utf-8-sig")

    return models, test_df, use_cols, lambda_wrapper, ens_weights


def train_all_targets(csv_path="race_features.csv"):
    """win / place2 / place3 の3モデルを学習し、model.pkl に3セット保存する。"""
    print("特徴量データ読み込み中（共通）...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]

    result = {}
    win_test_df = None
    for target in ["win", "place2", "place3"]:
        models, test_df, use_cols, lambda_wrapper, ens_weights = train_model(
            csv_path=csv_path, target=target, df=df.copy()
        )
        entry = {"models": models, "use_cols": use_cols, "weights": ens_weights}
        if lambda_wrapper is not None:
            entry["lambda_booster"] = lambda_wrapper.booster
        result[target] = entry
        if target == "win":
            win_test_df = test_df

    # 後方互換: 旧コードが model["models"] を読めるよう、winを最上位にも置く
    save_dict = {
        "win":    result["win"],
        "place2": result["place2"],
        "place3": result["place3"],
        # ↓旧形式互換（keiba_predict等が未対応でも動くように）
        "models":   result["win"]["models"],
        "use_cols": result["win"]["use_cols"],
        "weights":  result["win"]["weights"],
        "format":   "multi_v1",
    }
    if "lambda_booster" in result["win"]:
        save_dict["lambda_booster"] = result["win"]["lambda_booster"]

    _pkl = "model_bt.pkl" if BT_MODE else "model.pkl"
    with open(_pkl, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"\n[完了] 3モデル保存完了 → {_pkl}" + ("（backtest用・本番には使われない）" if BT_MODE else ""))
    print(f"   win: {len(result['win']['models'])}モデル / "
          f"place2: {len(result['place2']['models'])}モデル / "
          f"place3: {len(result['place3']['models'])}モデル")
    print(f"   ※旧形式互換: model['models']=winモデル も保持")

    return win_test_df


if __name__ == "__main__":
    df = train_all_targets()

    print(f"\n{'='*40}\n[!!] バックテスト（winモデル・生確率ベース）\n{'='*40}")

    # ── 期待値閾値スイープ（生確率での最適閾値を探す） ──
    print(f"\n{'─'*40}\n[分析] 戦略A 期待値閾値スイープ（最適値を探索）\n{'─'*40}")
    print("  閾値   ベット数  的中率   回収率")
    best_ev, best_roi = 0.3, 0
    for ev_th in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        bets = df[(df["予測順位"]==1)&(df["単勝期待値"]>=ev_th)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
        if len(bets) < 10:
            print(f"  {ev_th:+.1f}    {len(bets):4d}回    （サンプル不足）")
            continue
        wins = bets[bets["着順_num"]==1]
        roi = wins["単勝オッズ"].sum()/len(bets)*100
        rate = len(wins)/len(bets)*100
        mark = ""
        if roi > best_roi and len(bets) >= 30:
            best_roi, best_ev = roi, ev_th
            mark = " ←最高"
        print(f"  {ev_th:+.1f}    {len(bets):4d}回   {rate:5.1f}%  {roi:6.1f}%{mark}")
    print(f"\n  → 最適閾値: 期待値>={best_ev:+.1f}（回収率{best_roi:.1f}%, ベット30回以上で最高）")

    TARGET_EV = best_ev  # スイープで見つけた最適閾値を以降の戦略でも使う
    print(f"\n  以降の戦略A系は TARGET_EV={TARGET_EV:+.1f} を使用")

    bets_a = df[(df["予測順位"]==1)&(df["単勝期待値"]>=TARGET_EV)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
    if len(bets_a) > 0:
        wins_a = bets_a[bets_a["着順_num"]==1]
        print(f"\n── 戦略A: 予測1位 × 期待値>={TARGET_EV:+.1f} × オッズ(1.5〜20倍) ──")
        print(f"ベット数: {len(bets_a)}回\n的中数:   {len(wins_a)}回\n的中率:   {len(wins_a)/len(bets_a)*100:.1f}%\n回収率:   {wins_a['単勝オッズ'].sum()/len(bets_a)*100:.1f}%")

    bets_a2 = bets_a[bets_a["人気"]>=2]
    if len(bets_a2) > 0:
        wins_a2 = bets_a2[bets_a2["着順_num"]==1]
        print(f"\n── 戦略A-2: 戦略Aから1番人気を除外 ──")
        print(f"ベット数: {len(bets_a2)}回\n的中数:   {len(wins_a2)}回\n的中率:   {len(wins_a2)/len(bets_a2)*100:.1f}%\n回収率:   {wins_a2['単勝オッズ'].sum()/len(bets_a2)*100:.1f}%")

    bets_c = df[(df["人気"]>=3)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)]
    if len(bets_c) > 0:
        wins_c = bets_c[bets_c["着順_num"]==1]
        print(f"\n── 戦略C: 人気3番手以下 × 予測1位 × 期待値>=0.3 ──")
        print(f"ベット数: {len(bets_c)}回\n的中数:   {len(wins_c)}回\n的中率:   {len(wins_c)/len(bets_c)*100:.1f}%\n回収率:   {wins_c['単勝オッズ'].sum()/len(bets_c)*100:.1f}%")

    if "前走間隔" in df.columns:
        bets_d = df[(df["前走間隔"]>=2)&(df["前走間隔"]<=4)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.2)]
        if len(bets_d) > 0:
            wins_d = bets_d[bets_d["着順_num"]==1]
            print(f"\n── 戦略D: 前走間隔2〜4週 × 予測1位 × 期待値>=0.2 ──")
            print(f"ベット数: {len(bets_d)}回\n的中数:   {len(wins_d)}回\n的中率:   {len(wins_d)/len(bets_d)*100:.1f}%\n回収率:   {wins_d['単勝オッズ'].sum()/len(bets_d)*100:.1f}%")

    print(f"\n{'='*40}\n📍 競馬場別バックテスト（戦略A）\n{'='*40}")
    JYO_MAP = {1:"札幌",2:"函館",3:"福島",4:"新潟",5:"東京",6:"中山",7:"中京",8:"京都",9:"阪神",10:"小倉"}
    if "競馬場cd" in df.columns:
        jyo_results = []
        for jyo_cd, jyo_df in df.groupby("競馬場cd"):
            bets = jyo_df[(jyo_df["予測順位"]==1)&(jyo_df["単勝期待値"]>=0.3)&(jyo_df["単勝オッズ"]>=1.5)&(jyo_df["単勝オッズ"]<=20.0)]
            if len(bets) < 5: continue
            wins = bets[bets["着順_num"]==1]
            roi  = wins["単勝オッズ"].sum()/len(bets)*100
            jyo_results.append((JYO_MAP.get(jyo_cd, str(jyo_cd)), len(bets), len(wins)/len(bets)*100, roi))
        for name, n, rate, roi in sorted(jyo_results, key=lambda x: x[3], reverse=True):
            mark = "🟢" if roi>=120 else "🟡" if roi>=100 else "🔴"
            print(f"  {mark} {name:4} {n:3}回 {rate:.1f}% 回収率{roi:.1f}%")

    print(f"\n{'='*40}\n📏 距離別バックテスト（戦略A）\n{'='*40}")
    DIST_MAP = {1:"短距離(〜1400m)", 2:"中距離(1600〜2000m)", 3:"長距離(2001m〜)"}
    if "距離カテゴリ" in df.columns:
        for cat, label in DIST_MAP.items():
            bets = df[(df["距離カテゴリ"]==cat)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
            if len(bets) < 5: continue
            wins = bets[bets["着順_num"]==1]
            roi  = wins["単勝オッズ"].sum()/len(bets)*100
            mark = "🟢" if roi>=120 else "🟡" if roi>=100 else "🔴"
            print(f"  {mark} {label}: {len(bets)}回 {len(wins)/len(bets)*100:.1f}% 回収率{roi:.1f}%")

    print(f"\n{'='*40}\n🌿 芝・ダート別バックテスト（戦略A）\n{'='*40}")
    if "is_turf" in df.columns:
        for turf_val, label in [(1,"芝"),(0,"ダート")]:
            bets = df[(df["is_turf"]==turf_val)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
            if len(bets) < 5: continue
            wins = bets[bets["着順_num"]==1]
            roi  = wins["単勝オッズ"].sum()/len(bets)*100
            mark = "🟢" if roi>=120 else "🟡" if roi>=100 else "🔴"
            print(f"  {mark} {label}: {len(bets)}回 {len(wins)/len(bets)*100:.1f}% 回収率{roi:.1f}%")

    print(f"\n{'='*40}\n[結果] クラス別バックテスト（戦略A）\n{'='*40}")
    CLASS_MAP = {1:"新馬",2:"未勝利",3:"1勝クラス",4:"2勝クラス",5:"3勝クラス",6:"OP",7:"G3",8:"G2",9:"G1"}
    if "クラス_num" in df.columns:
        cls_results = []
        for cls_val, cls_label in CLASS_MAP.items():
            bets = df[(df["クラス_num"]==cls_val)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
            if len(bets) < 5: continue
            wins = bets[bets["着順_num"]==1]
            roi  = wins["単勝オッズ"].sum()/len(bets)*100
            cls_results.append((cls_label, len(bets), len(wins)/len(bets)*100, roi))
        for label, n, rate, roi in sorted(cls_results, key=lambda x: x[3], reverse=True):
            mark = "🟢" if roi>=120 else "🟡" if roi>=100 else "🔴"
            print(f"  {mark} {label:8}: {n:3}回 {rate:.1f}% 回収率{roi:.1f}%")

    print(f"\n{'='*40}\n🆕 新戦略バックテスト\n{'='*40}")

    if "競馬場cd" in df.columns:
        bets_f = df[(df["競馬場cd"].isin([5,7]))&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
        if len(bets_f) > 0:
            wins_f = bets_f[bets_f["着順_num"]==1]
            print(f"\n── 戦略F: 中京・東京 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_f)}回\n的中数:   {len(wins_f)}回\n的中率:   {len(wins_f)/len(bets_f)*100:.1f}%\n回収率:   {wins_f['単勝オッズ'].sum()/len(bets_f)*100:.1f}%")

    if "距離" in df.columns:
        bets_g = df[(df["距離"]<=1400)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
        if len(bets_g) > 0:
            wins_g = bets_g[bets_g["着順_num"]==1]
            print(f"\n── 戦略G: 短距離(〜1400m) × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_g)}回\n的中数:   {len(wins_g)}回\n的中率:   {len(wins_g)/len(bets_g)*100:.1f}%\n回収率:   {wins_g['単勝オッズ'].sum()/len(bets_g)*100:.1f}%")

    if "競馬場cd" in df.columns and "距離" in df.columns:
        bets_fg = df[(df["競馬場cd"].isin([5,7]))&(df["距離"]<=1400)&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
        if len(bets_fg) > 0:
            wins_fg = bets_fg[bets_fg["着順_num"]==1]
            print(f"\n── 戦略FG: 中京・東京 × 短距離 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_fg)}回\n的中数:   {len(wins_fg)}回\n的中率:   {len(wins_fg)/len(bets_fg)*100:.1f}%\n回収率:   {wins_fg['単勝オッズ'].sum()/len(bets_fg)*100:.1f}%")

    if "競馬場cd" in df.columns:
        bets_h = df[(df["競馬場cd"].isin([6,10]))&(df["予測順位"]==1)&(df["単勝期待値"]>=0.3)&(df["単勝オッズ"]>=1.5)&(df["単勝オッズ"]<=20.0)]
        if len(bets_h) > 0:
            wins_h = bets_h[bets_h["着順_num"]==1]
            print(f"\n── 戦略H: 中山・小倉 × 予測1位 × 期待値>=0.3 ──")
            print(f"ベット数: {len(bets_h)}回\n的中数:   {len(wins_h)}回\n的中率:   {len(wins_h)/len(bets_h)*100:.1f}%\n回収率:   {wins_h['単勝オッズ'].sum()/len(bets_h)*100:.1f}%")