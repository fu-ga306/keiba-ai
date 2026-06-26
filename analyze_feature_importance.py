# -*- coding: utf-8 -*-
"""
analyze_feature_importance.py
─────────────────────────────
学習済みモデル（MF/通常）の特徴量重要度を分析する。
「何が効いていて、何が効いていないか」を可視化し、
足りない特徴量・強化すべき領域を特定する。

入力: model_mf.pkl（MFモデル）または model.pkl（通常モデル）
出力:
  ・重要度ランキング（効いている特徴量top/bottom）
  ・カテゴリ別の重要度合計（どの種類の情報が効いているか）
  → 重要度が低い特徴量 = 改善 or 削除の候補
  → 重要なカテゴリの周辺 = 強化すべき領域

使い方:
  python analyze_feature_importance.py        # MFモデル
  python analyze_feature_importance.py normal # 通常モデル
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_lgb_importance(models, use_cols):
    """アンサンブルの中のLightGBMから重要度を取り出して平均する。"""
    importances = []
    for m in models:
        # CalibratedClassifierCV の場合、中の estimator を取り出す
        booster = None
        if hasattr(m, "calibrated_classifiers_"):
            # 各fold内のestimator
            for cc in m.calibrated_classifiers_:
                est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if est is not None and hasattr(est, "feature_importances_"):
                    booster = est
                    break
        elif hasattr(m, "estimator"):
            est = m.estimator
            if hasattr(est, "feature_importances_"):
                booster = est
        elif hasattr(m, "feature_importances_"):
            booster = m

        if booster is not None and hasattr(booster, "feature_importances_"):
            imp = booster.feature_importances_
            if len(imp) == len(use_cols):
                importances.append(imp)

    if not importances:
        return None
    return np.mean(importances, axis=0)


# 特徴量のカテゴリ分類（足りない領域を見るため）
def categorize(col):
    if any(k in col for k in ["血統", "父", "母父", "種牡馬", "sire"]):
        return "血統"
    if any(k in col for k in ["距離", "延長", "短縮", "経験最長", "経験範囲"]):
        return "距離適性"
    if any(k in col for k in ["騎手"]):
        return "騎手"
    if any(k in col for k in ["調教師"]):
        return "調教師"
    if any(k in col for k in ["上り", "タイム", "先行", "脚質", "ペース"]):
        return "走り・展開"
    if any(k in col for k in ["体重", "斤量", "年齢", "性"]):
        return "馬体・条件"
    if any(k in col for k in ["過去", "勝率", "複勝率", "着順", "改善", "連続"]):
        return "過去成績"
    if any(k in col for k in ["競馬場", "コース", "馬場", "芝", "ダート"]):
        return "コース・馬場"
    if any(k in col for k in ["枠", "馬番", "頭数"]):
        return "枠・頭数"
    if any(k in col for k in ["人気", "オッズ", "期待値"]):
        return "市場情報"
    if any(k in col for k in ["月", "季節"]):
        return "季節"
    if "×" in col:
        return "交互作用"
    return "その他"


def main():
    is_normal = len(sys.argv) > 1 and sys.argv[1] == "normal"
    path = os.path.join(BASE_DIR, "model.pkl" if is_normal else "model_mf.pkl")
    label = "通常モデル" if is_normal else "MFモデル（市場フリー）"

    if not os.path.exists(path):
        print(f"モデルがありません: {path}")
        return

    with open(path, "rb") as f:
        saved = pickle.load(f)

    # 3モデル形式なら win を使う
    if saved.get("format") == "multi_v1":
        models = saved["win"]["models"]
        use_cols = saved["win"]["use_cols"]
        print(f"{label}（win 3モデル形式）")
    else:
        models = saved["models"]
        use_cols = saved["use_cols"]
        print(f"{label}（単一形式）")

    imp = extract_lgb_importance(models, use_cols)
    if imp is None:
        print("重要度を取り出せませんでした（モデル構造が想定と異なる）")
        return

    df = pd.DataFrame({"特徴量": use_cols, "重要度": imp})
    df["重要度%"] = df["重要度"] / df["重要度"].sum() * 100
    df["カテゴリ"] = df["特徴量"].apply(categorize)
    df = df.sort_values("重要度", ascending=False).reset_index(drop=True)

    # ── 重要度 上位20 ──
    print("\n" + "=" * 56)
    print("効いている特徴量 TOP20")
    print("=" * 56)
    for _, r in df.head(20).iterrows():
        bar = "-" * int(r["重要度%"] * 2)
        print(f"  {r['特徴量']:<24} {r['重要度%']:5.2f}% {bar}")

    # ── 重要度 下位15（効いていない＝改善/削除候補）──
    print("\n" + "=" * 56)
    print("効いていない特徴量 BOTTOM15（改善 or 削除の候補）")
    print("=" * 56)
    for _, r in df.tail(15).iterrows():
        print(f"  {r['特徴量']:<24} {r['重要度%']:5.2f}%")

    # ── カテゴリ別の重要度合計 ──
    print("\n" + "=" * 56)
    print("カテゴリ別 重要度合計（どの情報が効いているか）")
    print("=" * 56)
    cat = df.groupby("カテゴリ")["重要度%"].sum().sort_values(ascending=False)
    for name, val in cat.items():
        bar = "-" * int(val / 2)
        print(f"  {name:<12} {val:5.1f}% {bar}")

    # ── 足りない領域の指摘 ──
    print("\n" + "=" * 56)
    print("【足りない特徴量の分析】")
    print("=" * 56)
    present_cats = set(cat.index)
    # 血統があるか
    if "血統" not in present_cats or cat.get("血統", 0) < 1:
        print("  [!] 血統: ほぼ無し → 取得中の血統データで大幅に補える")
    # 距離適性の重要度
    dist_imp = cat.get("距離適性", 0)
    print(f"  距離適性カテゴリの重要度: {dist_imp:.1f}%")
    if dist_imp < 5:
        print("    → まだ低い。今回追加した距離適性特徴量(未学習)で上がる見込み")
    # 騎手トレンドの欠如
    print("  騎手は『通算成績』のみ。『直近の調子(トレンド)』は未実装")
    print("    → 「今乗れている騎手」の特徴量が作れる")
    # 効いていない特徴量の数
    weak = df[df["重要度%"] < 0.3]
    print(f"\n  重要度0.3%未満の特徴量: {len(weak)}個")
    print("    → これらは効いていない。作り直すか、別角度の特徴量に置換")

    # CSV保存
    out = os.path.join(BASE_DIR, "feature_importance.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  詳細を保存 → feature_importance.csv")
    print("\n分析完了")


if __name__ == "__main__":
    main()
