# -*- coding: utf-8 -*-
"""リーク修正後の特徴量再評価＋位置取り予測モデル。

背景: 2026-07-28に時系列リーク（race_id順を時系列順とみなしていた）を修正。
それ以前の「どの特徴量が効くか」の結論はリーク下の判定なので全て無効。
また展開（位置取り）にはオラクル検証で明確な価値が確認された:
  実際の1角位置が事前に分かった場合の単勝ROI（2025・7-9人気×前め）= 155.0%
  過去走の平均から予測した場合（相関0.432）= 102.2%
→ 位置予測の精度を上げれば、その差を取りに行ける。

Part A: place3（複勝）に効く特徴量をクリーンなデータで再評価
Part B: 1角位置の予測モデルを作り、相関0.432をどこまで上げられるか
        →改善した予測位置で人気帯別ROIを再測定
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

TEST_YEAR = 2025


def log(m):
    print(m, flush=True)


def load_base():
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str})
    # 人気・単勝オッズ・着順_num は race_features 側にあるので、通過だけ持ってくる
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "通過"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = rf.merge(rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    d["年"] = d["race_id"].str[:4].astype(int)
    d["着順"] = pd.to_numeric(d["着順_num"], errors="coerce")
    for c in ["単勝オッズ", "人気"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["頭数"] = d.groupby("race_id")["馬名"].transform("count")

    def pos(s, idx):
        try:
            p = [int(x) for x in str(s).split("-") if x.isdigit()]
            return p[idx] if p else np.nan
        except Exception:
            return np.nan

    d["1角"] = d["通過"].map(lambda s: pos(s, 0))
    d["1角相対"] = d["1角"] / d["頭数"]
    d["fuku"] = (d["着順"] <= 3).astype(float)
    d["win"] = (d["着順"] == 1).astype(float)
    return d


def part_a(d):
    log("\n" + "=" * 72)
    log("【Part A】place3(複勝)に効く特徴量 ― リーク修正後のクリーンな再評価")
    log("=" * 72)
    from market_free_model import FEATURE_COLS_MF
    use = [c for c in FEATURE_COLS_MF if c in d.columns]
    tr = d[(d["年"] <= TEST_YEAR - 1) & d["着順"].notna()]
    te = d[(d["年"] == TEST_YEAR) & d["着順"].notna()]
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, num_leaves=63,
                           n_estimators=600, min_child_samples=20, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m.fit(tr[use], tr["fuku"])
    p = m.predict_proba(te[use])[:, 1]
    log(f"  学習{len(tr):,} / 検証{len(te):,} / 特徴量{len(use)}列   AUC={roc_auc_score(te['fuku'], p):.4f}")
    imp = pd.Series(m.booster_.feature_importance("gain"), index=use).sort_values(ascending=False)
    imp = imp / imp.sum() * 100
    log("\n  ― 重要度 上位25 ―")
    for i, (k, v) in enumerate(imp.head(25).items(), 1):
        log(f"   {i:2d}. {k:<28} {v:5.2f}%")
    log("\n  ― 展開・脚質系がどこにいるか ―")
    pace = ["過去平均先行指数", "先行馬フラグ", "想定先行馬数", "想定先行馬率",
            "他馬想定先行馬数", "差し馬×ハイペース想定", "前走4角位置", "過去平均4角位置",
            "前走脚質指数", "脚質スコア", "前走相対位置", "想定逃げ馬数", "先行圧",
            "差し×先行圧", "単騎逃げ", "逃げ争い", "コース脚質バイアス"]
    rank = {k: i + 1 for i, k in enumerate(imp.index)}
    for k in pace:
        if k in rank:
            log(f"   {k:<24} {rank[k]:3d}位 / {len(use)}  {imp[k]:5.2f}%")
    log("\n  ― 重要度ほぼ0（剪定候補）―")
    dead = imp[imp < 0.05]
    log(f"   {len(dead)}列: {', '.join(list(dead.index)[:18])}")
    return imp


def part_b(d):
    log("\n" + "=" * 72)
    log("【Part B】位置取り(1角)予測モデル ― 素朴な過去平均(相関0.432)を超えられるか")
    log("=" * 72)
    # 事前に分かる特徴だけを使う（着順・上がり・通過など当日結果は不使用）
    base = ["枠番", "馬番", "出走頭数", "距離", "is_turf", "クラス_num", "年齢",
            "斤量", "斤量_相対", "馬体重", "体重増減", "騎手勝率", "騎手複勝率",
            "過去出走数", "過去平均先行指数", "先行馬フラグ", "想定先行馬数",
            "想定先行馬率", "他馬想定先行馬数", "脚質スコア", "前走相対位置",
            "想定逃げ馬数", "先行圧", "差し×先行圧", "単騎逃げ", "逃げ争い",
            "前走4角位置", "過去平均4角位置", "前走脚質指数", "前走距離", "距離変化",
            "距離延長幅", "乗り替わり", "休み明け", "連闘", "前走間隔", "競馬場cd",
            "回り_num", "直線長_m", "坂あり", "レース番号", "馬場状態_num"]
    use = [c for c in base if c in d.columns]
    s = d.dropna(subset=["1角相対"])
    tr = s[s["年"] <= TEST_YEAR - 1]
    te = s[s["年"] == TEST_YEAR].copy()
    log(f"  学習{len(tr):,} / 検証{len(te):,} / 特徴量{len(use)}列")

    m = lgb.LGBMRegressor(objective="regression", learning_rate=0.05, num_leaves=63,
                          n_estimators=800, min_child_samples=20, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m.fit(tr[use], tr["1角相対"])
    te["予測位置_model"] = m.predict(te[use])
    # 素朴なベースライン（過去走の平均）
    naive = te["過去平均先行指数"] / te["出走頭数"]
    te["予測位置_naive"] = naive
    c_model = te["予測位置_model"].corr(te["1角相対"])
    c_naive = te["予測位置_naive"].corr(te["1角相対"])
    log(f"\n  相関(実際の1角位置との) : 素朴={c_naive:.3f}  モデル={c_model:.3f}  "
        f"改善{c_model - c_naive:+.3f}")

    imp = pd.Series(m.booster_.feature_importance("gain"), index=use).sort_values(ascending=False)
    imp = imp / imp.sum() * 100
    log("\n  ― 位置予測に効く特徴 上位12 ―")
    for i, (k, v) in enumerate(imp.head(12).items(), 1):
        log(f"   {i:2d}. {k:<26} {v:5.2f}%")

    log("\n  ― 予測位置で切った単勝ROI（2025・上位1/3=前め）―")
    log(f"   {'人気帯':<8}{'予測':<8}{'複勝率':>8}{'勝率':>7}{'単勝ROI':>9}{'n':>7}")
    for lo, hi, nm in [(1, 3, "1-3"), (4, 6, "4-6"), (7, 9, "7-9"), (10, 99, "10-")]:
        sub = te[(te["人気"] >= lo) & (te["人気"] <= hi)].dropna(subset=["予測位置_model"])
        if len(sub) < 300:
            continue
        q = pd.qcut(sub["予測位置_model"], 3, labels=["前", "中", "後"], duplicates="drop")
        for lab in ["前", "後"]:
            ss = sub[q == lab]
            if len(ss) < 80:
                continue
            r = (ss["win"] * ss["単勝オッズ"]).sum() / len(ss) * 100
            log(f"   {nm:<8}{lab:<8}{ss['fuku'].mean()*100:7.1f}%{ss['win'].mean()*100:6.1f}%"
                f"{r:8.1f}%{len(ss):7d}")
        log("")
    log("  参考: 素朴予測での7-9人気×前め=102.2% / オラクル(実位置)=155.0%")


def main():
    d = load_base()
    log(f"データ {len(d):,}行  1角位置あり {d['1角相対'].notna().mean()*100:.1f}%")
    part_a(d)
    part_b(d)


if __name__ == "__main__":
    main()
