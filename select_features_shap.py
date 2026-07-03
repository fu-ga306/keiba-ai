"""
select_features_shap.py
────────────────────────
race_features.csv の全特徴量について SHAP 重要度を算出し、
寄与がほぼ0の列（ノイズ）を特定して除外候補を提案する。

使い方:
    python select_features_shap.py            # SHAP算出 → shap_importance.csv 出力
    python select_features_shap.py 0.0005      # 閾値を指定（平均|SHAP|がこれ未満を除外候補に）

出力:
    shap_importance.csv  … 各特徴量の平均|SHAP|重要度（降順）
    標準出力に除外候補リスト（model.py の FEATURE_COLS から外す候補）

方針:
    - winターゲット（着順==1）で軽量LightGBMを1本学習
    - 2025年テストデータをサンプリングしてSHAP値を計算（高速化）
    - 平均|SHAP| < 閾値 の列を「寄与ほぼ0」として除外候補にする
    - ※コース新特徴量など新規追加分は、実効果を再学習で見るため除外対象から保護する
"""
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

from model import FEATURE_COLS, make_target, LGB_PARAMS

# 新規追加した特徴量は効果検証前なので除外候補から保護する
PROTECTED = {
    "コース好走相対4角", "コース先行勝率", "コース差し複勝率",
    "脚質コース適合", "先行有利コース×先行馬", "差し有利コース×差し馬",
    "回り替わりフラグ", "直線長変化", "坂替わりフラグ", "初コースフラグ",
    "騎手勝率_sm", "調教師勝率_sm", "馬主勝率", "馬主複勝率", "馬主勝率_sm",
    "過去着順_std", "直近5走着順_std", "直近5走着外率",
    # レース内偏差値化・メンバーレベル（今回追加・効果検証前なので保護）
    "過去勝率_レース内偏差", "過去複勝率_レース内偏差", "直近3走平均着順_レース内偏差",
    "過去平均上り_レース内偏差", "過去獲得賞金累計_レース内偏差", "騎手勝率_レース内偏差",
    "父系_今回距離適性_レース内偏差", "直近5走平均着順_レース内偏差",
    "メンバー過去勝率平均", "メンバー過去勝率最大", "過去勝率_対相手",
    "メンバークラス平均", "メンバー賞金平均", "賞金_対相手",
}


def main(threshold=0.0005, sample_n=8000):
    print("特徴量データ読み込み中...")
    df = pd.read_csv("race_features.csv")
    df = df.dropna(subset=["着順_num"])
    df = df[df["着順_num"] >= 1]
    df["年"] = df["race_id"].astype(str).str[:4].astype(int)

    use_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"評価対象特徴量: {len(use_cols)}列")

    train_df = df[df["年"] <= 2024]
    test_df = df[df["年"] == 2025]

    X_train = train_df[use_cols]
    y_train = make_target(train_df["着順_num"], "win")

    print("軽量LightGBMを学習中（SHAP算出用）...")
    model = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=400)
    model.fit(X_train, y_train)

    # SHAP計算（テストデータをサンプリング）
    X_test = test_df[use_cols]
    if len(X_test) > sample_n:
        X_test = X_test.sample(sample_n, random_state=42)
    print(f"SHAP値を計算中（{len(X_test)}行サンプル）...")

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        if isinstance(sv, list):      # 二値分類は[neg, pos]のことがある
            sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs = np.abs(sv).mean(axis=0)
    except Exception as e:
        print(f"  SHAP失敗 → LightGBM gain重要度で代替: {e}")
        mean_abs = model.booster_.feature_importance(importance_type="gain")
        mean_abs = mean_abs / mean_abs.sum()

    imp = pd.DataFrame({"feature": use_cols, "importance": mean_abs})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv("shap_importance.csv", index=False, encoding="utf-8-sig")
    print(f"\n重要度を保存 → shap_importance.csv")

    print(f"\n=== 重要度TOP20 ===")
    for _, r in imp.head(20).iterrows():
        print(f"  {r['importance']:.5f}  {r['feature']}")

    print(f"\n=== 重要度BOTTOM20（除外候補の母集団）===")
    for _, r in imp.tail(20).iterrows():
        prot = " [保護]" if r["feature"] in PROTECTED else ""
        print(f"  {r['importance']:.5f}  {r['feature']}{prot}")

    # 除外候補（閾値未満 かつ 保護対象外）
    drop_cands = imp[(imp["importance"] < threshold) &
                     (~imp["feature"].isin(PROTECTED))]["feature"].tolist()
    print(f"\n=== 除外候補（平均|SHAP| < {threshold}、保護対象を除く）: {len(drop_cands)}列 ===")
    for c in drop_cands:
        print(f"  - {c}")

    # model.py に貼り付けやすい形で出力
    print(f"\n# model.py の除外用リスト（コピペ可）")
    print("SHAP_DROP_COLS = [")
    for c in drop_cands:
        print(f'    "{c}",')
    print("]")

    return drop_cands


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
    main(threshold=th)
