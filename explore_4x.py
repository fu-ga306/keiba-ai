# -*- coding: utf-8 -*-
"""「4倍を3回に1回当てればプラス」の検証。

損益分岐: オッズOの馬の分岐勝率 = 1/O（4倍なら25%）。
→ オッズ2.5-6倍(平均約4倍)の帯で、実勝率が分岐+αを超える条件を総当たりする。
使う武器: 主モデル/place3/MF複勝の順位・全モデル合意・市場歪み補正bias・較正勝率。
検証原則: <=2024学習 / 2025評価 / 期間・場の分割で再現しなければ不採用。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from train_debias import build_market_features, FEATS


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着", "単勝オッズ", "人気"])
    d = build_market_features(d)
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["年"] = d["race_id"].str[:4].astype(int)
    d["bn"] = pd.to_numeric(d["馬番"], errors="coerce").astype("Int64").map(
        lambda x: f"{int(x):02d}" if pd.notna(x) else None)
    m = pd.read_csv("model_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "勝ち確率"]]
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "p3順"})
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF複勝率", "MF複勝順位", "MF勝率順位"]]
    d = (d.merge(m, on=["race_id", "馬名"], how="left")
          .merge(p3, on=["race_id", "馬名"], how="left")
          .merge(mf, on=["race_id", "馬名"], how="left"))
    for c in ["勝ち確率", "p3順", "MF勝率", "MF複勝率", "MF複勝順位", "MF勝率順位"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    return d, tan


def roi(s, tan):
    if not len(s):
        return float("nan")
    return sum(tan.get((r.race_id, r.bn), 0) for r in s.itertuples()) / (len(s) * 100) * 100


def line(s, tan, label, min_n=150):
    if len(s) < min_n:
        return
    mid = s["dt"].median()
    r = roi(s, tan)
    r1, r2 = roi(s[s["dt"] <= mid], tan), roi(s[s["dt"] > mid], tan)
    a = roi(s[s["race_id"].str[4:6] < "06"], tan)
    b = roi(s[s["race_id"].str[4:6] >= "06"], tan)
    wr = s["win"].mean() * 100
    need = 100 / s["単勝オッズ"].mean()
    mk = "◎" if min(r1, r2, a, b) >= 100 else ("○" if min(r1, r2, a, b) >= 90 else "")
    log(f"  {label:<34}{len(s):6d} 勝率{wr:5.1f}%(要{need:4.1f}%) 平均{s['単勝オッズ'].mean():4.1f}倍"
        f" ROI{r:6.1f}% (半{r1:.0f}/{r2:.0f} 場{a:.0f}/{b:.0f}){mk}")


def main():
    d, tan = load()
    tr = d[d["年"] <= 2024]
    te = d[d["年"] == 2025].copy()
    # bias（<=2024学習）
    mdl = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                             n_estimators=700, min_child_samples=100, feature_fraction=0.9,
                             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    mdl.fit(tr[FEATS], tr["win"])
    te["p_adj"] = mdl.predict_proba(te[FEATS])[:, 1]
    te["p_adj"] = te["p_adj"] / te.groupby("race_id")["p_adj"].transform("sum")
    te["bias"] = te["p_adj"] / te["q"].clip(lower=1e-9)
    # 主モデル順位・較正勝率（相互較正）
    te["主順"] = te.groupby("race_id")["勝ち確率"].rank(ascending=False, method="min")
    mid = te["dt"].quantile(0.5)
    te["cal_win"] = np.nan
    for fit_m, ap_m in [(te["dt"] <= mid, te["dt"] > mid), (te["dt"] > mid, te["dt"] <= mid)]:
        f = te[fit_m].dropna(subset=["MF勝率"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(f["MF勝率"], f["win"])
        te.loc[ap_m, "cal_win"] = iso.predict(te.loc[ap_m, "MF勝率"])

    band = te[(te["単勝オッズ"] >= 2.5) & (te["単勝オッズ"] <= 6.0)].copy()
    log(f"対象帯: 2.5-6.0倍  {len(band):,}頭  平均{band['単勝オッズ'].mean():.1f}倍 "
        f"分岐勝率{100/band['単勝オッズ'].mean():.1f}%  実勝率{band['win'].mean()*100:.1f}%")

    log("\n" + "=" * 100)
    log("【オッズ2.5-6倍帯での条件別 勝率とROI】 要=損益分岐勝率")
    log("=" * 100)
    line(band, tan, "帯すべて")
    line(band[band["主順"] == 1], tan, "主モデル1位")
    line(band[(band["主順"] == 1) & (band["人気"] >= 2)], tan, "主1位なのに2番人気以下")
    line(band[(band["主順"] == 1) & (band["p3順"] == 1)], tan, "主1位＋place3-1位")
    line(band[(band["主順"] == 1) & (band["MF複勝順位"] == 1)], tan, "主1位＋MF複勝1位")
    line(band[(band["主順"] == 1) & (band["p3順"] == 1) & (band["MF複勝順位"] == 1)],
         tan, "3モデル合意1位")
    line(band[band["bias"] >= 1.0], tan, "bias≥1.0(市場が過小評価)")
    line(band[band["bias"] >= 1.05], tan, "bias≥1.05")
    line(band[(band["主順"] == 1) & (band["bias"] >= 1.0)], tan, "主1位＋bias≥1.0")
    line(band[(band["主順"] == 1) & (band["p3順"] == 1) & (band["bias"] >= 1.0)],
         tan, "主1位＋place3-1位＋bias≥1.0")
    for th in [0.25, 0.28, 0.30]:
        line(band[band["cal_win"] >= th], tan, f"較正勝率≥{th:.2f}")
    line(band[(band["cal_win"] >= 0.25) & (band["主順"] == 1)], tan, "較正勝率≥0.25＋主1位")
    line(band[(band["cal_win"] >= 0.25) & (band["bias"] >= 1.0)], tan, "較正勝率≥0.25＋bias≥1.0")
    line(band[(band["人気"] == 1)], tan, "1番人気(市場の本命が2.5-6倍)")
    line(band[(band["人気"] == 1) & (band["主順"] == 1)], tan, "同上＋主モデルも1位")

    log("\n" + "=" * 100)
    log("【帯を広げた確認: 2-8倍で同じ勝ち筋があるか】")
    log("=" * 100)
    w = te[(te["単勝オッズ"] >= 2.0) & (te["単勝オッズ"] <= 8.0)]
    line(w[(w["主順"] == 1) & (w["p3順"] == 1) & (w["MF複勝順位"] == 1)], tan, "3モデル合意1位(2-8倍)")
    line(w[(w["主順"] == 1) & (w["bias"] >= 1.0)], tan, "主1位＋bias≥1.0(2-8倍)")
    line(w[(w["cal_win"] >= 0.25)], tan, "較正勝率≥0.25(2-8倍)")


if __name__ == "__main__":
    main()
