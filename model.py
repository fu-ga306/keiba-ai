import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

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
    "人気", "出走頭数", "競馬場cd", "レース番号",
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
    "開催月", "開催季節",
    "前走着順", "前走上り", "前走距離", "距離変化",
    "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
    "平均タイム差", "騎手競馬場勝率",
    # 交互作用特徴量
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率",
    "芝ダート×先行_過去勝率",
    "前走好走×人気薄", "前走着順×人気_乖離",
    "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
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
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "min_gain_to_split": 0.1,
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
        for col in ["斤量変化", "乗り替わり", "連闘", "休み明け", "負担率"]:
            df[col] = np.nan

    df["年"] = df["race_id"].astype(str).str[:4].astype(int)
    train_df = df[df["年"] <= 2024].copy()
    test_df  = df[df["年"] == 2025].copy()

    print(f"学習データ: {len(train_df)}行（〜2024）")
    print(f"検証データ: {len(test_df)}行（2025）")

    use_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"使用特徴量: {len(use_cols)}列")

    X_train_all = train_df[use_cols].copy()
    y_train_all = make_target(train_df["着順_num"], target)
    X_test      = test_df[use_cols].copy()
    y_test      = test_df["着順_num"]

    # ── 時系列重み（直近年ほど重みを大きくする） ──────────────────────
    # 2019年=基準1.0、2024年=最大重み
    # 線形に増加：年差0→1.0, 年差5→TIME_WEIGHT_MAX
    TIME_WEIGHT_MAX = 1.0
    year_min = train_df["年"].min()
    year_max = train_df["年"].max()
    year_range = max(year_max - year_min, 1)
    train_df["時系列重み"] = 1.0 + (train_df["年"] - year_min) / year_range * (TIME_WEIGHT_MAX - 1.0)

    X_train_main, X_cal, y_train_main, y_cal, w_time_main, w_time_cal = train_test_split(
        X_train_all, y_train_all, train_df["時系列重み"], test_size=0.2, random_state=42
    )
    # 着順重み（正例を重視）× 時系列重み（直近年を重視）
    pos_w = TARGET_POS_WEIGHT.get(target, 2.0)
    w_main = np.where(y_train_main == 1, pos_w, 1.0) * w_time_main.values
    print(f"  時系列重み: {year_min}年=1.0倍 〜 {year_max}年={TIME_WEIGHT_MAX}倍")
    print(f"  正例重み({target}): {pos_w}倍")

    # ── LightGBM ──
    print("ベースLightGBMモデルを学習中（アーリーストッピング付き）...")
    base_model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=5000)
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

    # ── LightGBM LambdaRank（着順ランキング学習） ── ※winターゲットのみ
    lambda_wrapper = None
    if target == "win":
        print("LambdaRankモデルを学習中...")
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

            rank_booster = lgb.train(
                LGB_RANK_PARAMS,
                train_data_rank,
                num_boost_round=3000,
                valid_sets=[val_data_rank],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=200)],
            )

            lambda_wrapper = LambdaRankWrapper(rank_booster)
            print(f"  LambdaRank完了（{rank_booster.best_iteration}本）")
        except Exception as e:
            lambda_wrapper = None
            print(f"  LambdaRankスキップ: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"  LambdaRankは{target}では学習しません（winのみ）")

    print(f"\nアンサンブル: {len(models)}モデルの平均で予測（LGB+XGB+CatBoost）")
    if lambda_wrapper:
        print(f"  ※LambdaRankは参考スコアとして別途計算")

    print(f"\nアンサンブル: {len(models)}モデルの平均で予測（LGB+XGB+CatBoost）")
    if lambda_wrapper:
        print(f"  ※LambdaRankは参考スコアとして別途計算")

        # ── テストデータで評価 ──
    test_df = test_df.copy()
    # target汎用のスコア列（その目的の確率）
    test_df["予測スコア"] = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
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
        out = test_df[["race_id","馬名","着順_num","予測スコア","予測順位","単勝オッズ","人気","勝ち確率","単勝期待値"]].copy()
        out = out.sort_values(["race_id", "予測順位"])
        out.to_csv("model_result.csv", index=False, encoding="utf-8-sig")

    return models, test_df, use_cols, lambda_wrapper


def train_all_targets(csv_path="race_features.csv"):
    """win / place2 / place3 の3モデルを学習し、model.pkl に3セット保存する。"""
    print("特徴量データ読み込み中（共通）...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]

    result = {}
    win_test_df = None
    for target in ["win", "place2", "place3"]:
        models, test_df, use_cols, lambda_wrapper = train_model(
            csv_path=csv_path, target=target, df=df.copy()
        )
        entry = {"models": models, "use_cols": use_cols}
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
        "format":   "multi_v1",
    }
    if "lambda_booster" in result["win"]:
        save_dict["lambda_booster"] = result["win"]["lambda_booster"]

    with open("model.pkl", "wb") as f:
        pickle.dump(save_dict, f)
    print(f"\n✅ 3モデル保存完了 → model.pkl")
    print(f"   win: {len(result['win']['models'])}モデル / "
          f"place2: {len(result['place2']['models'])}モデル / "
          f"place3: {len(result['place3']['models'])}モデル")
    print(f"   ※旧形式互換: model['models']=winモデル も保持")

    return win_test_df


if __name__ == "__main__":
    df = train_all_targets()

    print(f"\n{'='*40}\n🔥 バックテスト（winモデル・生確率ベース）\n{'='*40}")

    # ── 期待値閾値スイープ（生確率での最適閾値を探す） ──
    print(f"\n{'─'*40}\n📊 戦略A 期待値閾値スイープ（最適値を探索）\n{'─'*40}")
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

    print(f"\n{'='*40}\n🏆 クラス別バックテスト（戦略A）\n{'='*40}")
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