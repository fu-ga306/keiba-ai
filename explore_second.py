# -*- coding: utf-8 -*-
"""2-3番人気の激走を拾えるか／1番人気が飛ぶレースを見分けられるか（市場合成モデル）。

ユーザーの仮説: 1番人気の勝率は32.5%＝7割は負ける。2-3番人気が1着になる場面を
                拾えるだけでもプラスになるはず。市場情報を組み込んだ合成モデルで探す。

A. 馬単位: 2-3番人気だけを対象に「勝つか」を学習（市場情報＋能力特徴の合成）
           <=2024学習 / 2025検証。選抜上位の単勝ROIが100%を超えるか。
B. レース単位: 「1番人気が馬券外に飛ぶか」を学習。飛ぶと読んだレースで
           2-3番人気を買った場合のROI。
どちらも「市場の織り込み(人気・オッズ)」を特徴に含めた合成モデルで行う。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def log(m):
    print(m, flush=True)


def load():
    from market_free_model import FEATURE_COLS_MF
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["年"] = d["race_id"].str[:4].astype(int)
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["pop"] = pd.to_numeric(d["人気"], errors="coerce")
    d = d.dropna(subset=["着", "odds", "pop"])
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    # 市場由来の合成特徴
    d["raw"] = 1 / d["odds"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["fav_odds"] = d.groupby("race_id")["odds"].transform("min")
    d["odds比"] = d["odds"] / d["fav_odds"]          # 1番人気との人気差の強さ
    d["q_top"] = d.groupby("race_id")["q"].transform("max")   # 市場の1番人気への確信
    d["q_share"] = d["q"] / d["q_top"]
    d["場"] = d["race_id"].str[4:6].astype(int)
    feats = [c for c in FEATURE_COLS_MF if c in d.columns]
    feats += ["odds", "pop", "q", "odds比", "q_top", "q_share", "fav_odds", "場"]
    return d, list(dict.fromkeys(feats))


def part_a(d, feats):
    log("\n" + "=" * 74)
    log("【A】2-3番人気に特化した合成モデル ― 激走を拾えるか")
    log("=" * 74)
    s = d[(d["pop"] >= 2) & (d["pop"] <= 3)]
    tr, te = s[s["年"] <= 2024], s[s["年"] == 2025].copy()
    base = (te["win"] * te["odds"]).sum() / len(te) * 100
    log(f"  学習{len(tr):,} / 検証{len(te):,}   2-3番人気を全部買った場合のROI = {base:.1f}%")
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                           n_estimators=600, min_child_samples=40, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m.fit(tr[feats], tr["win"])
    te["p"] = m.predict_proba(te[feats])[:, 1]
    log(f"  モデルAUC(この集団内で勝ち馬を当てる力) = {roc_auc_score(te['win'], te['p']):.4f}")
    log(f"\n  {'選抜':<16}{'n':>7}{'勝率':>7}{'平均オッズ':>10}{'単勝ROI':>9}{'複勝率':>8}")
    for frac in [0.05, 0.10, 0.20, 0.33, 0.50]:
        th = te["p"].quantile(1 - frac)
        p = te[te["p"] >= th]
        roi = (p["win"] * p["odds"]).sum() / len(p) * 100
        log(f"  {'上位' + f'{frac:.0%}':<16}{len(p):7d}{p['win'].mean()*100:6.1f}%"
            f"{p['odds'].mean():9.1f}{roi:8.1f}%{p['fuku'].mean()*100:7.1f}%")
    # 市場の中でも「割安な2-3番人気」= モデル確率 > 市場確率 の度合い
    te["edge"] = te["p"] / te["q"].clip(lower=1e-6)
    log(f"\n  {'モデル/市場 比':<16}{'n':>7}{'勝率':>7}{'平均オッズ':>10}{'単勝ROI':>9}")
    for frac in [0.05, 0.10, 0.20]:
        th = te["edge"].quantile(1 - frac)
        p = te[te["edge"] >= th]
        roi = (p["win"] * p["odds"]).sum() / len(p) * 100
        log(f"  {'上位' + f'{frac:.0%}':<16}{len(p):7d}{p['win'].mean()*100:6.1f}%"
            f"{p['odds'].mean():9.1f}{roi:8.1f}%")


def part_b(d, feats):
    log("\n" + "=" * 74)
    log("【B】1番人気が飛ぶレースを見分け、そこで2-3番人気を買う")
    log("=" * 74)
    fav = d[d["pop"] == 1].copy()
    fav["飛ぶ"] = (fav["着"] > 3).astype(float)
    tr, te = fav[fav["年"] <= 2024], fav[fav["年"] == 2025].copy()
    log(f"  学習{len(tr):,} / 検証{len(te):,}   1番人気が馬券外になる率 = {te['飛ぶ'].mean()*100:.1f}%")
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                           n_estimators=600, min_child_samples=40, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m.fit(tr[feats], tr["飛ぶ"])
    te["p_flop"] = m.predict_proba(te[feats])[:, 1]
    log(f"  「飛ぶ」予測のAUC = {roc_auc_score(te['飛ぶ'], te['p_flop']):.4f}")
    d25 = d[d["年"] == 2025]
    sec = d25[(d25["pop"] >= 2) & (d25["pop"] <= 3)]
    log(f"\n  {'1番人気が飛ぶと読んだ上位':<26}{'R数':>6}{'実際に飛んだ率':>14}"
        f"{'2-3人気ROI':>12}{'n':>7}")
    for frac in [0.10, 0.20, 0.33, 0.50]:
        th = te["p_flop"].quantile(1 - frac)
        rids = set(te[te["p_flop"] >= th]["race_id"])
        actual = te[te["p_flop"] >= th]["飛ぶ"].mean() * 100
        s = sec[sec["race_id"].isin(rids)]
        roi = (s["win"] * s["odds"]).sum() / len(s) * 100 if len(s) else np.nan
        log(f"  {'上位' + f'{frac:.0%}':<26}{len(rids):6d}{actual:13.1f}%{roi:11.1f}%{len(s):7d}")
    log(f"  ※参考: 全レースで2-3番人気を買った場合 = "
        f"{(sec['win']*sec['odds']).sum()/len(sec)*100:.1f}%")


def main():
    d, feats = load()
    log(f"データ {len(d):,}行 / 合成特徴 {len(feats)}列（能力特徴＋市場特徴）")
    part_a(d, feats)
    part_b(d, feats)


if __name__ == "__main__":
    main()
