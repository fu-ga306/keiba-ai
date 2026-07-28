# -*- coding: utf-8 -*-
"""較正済み確率を使った2案の検証。

案1: 複勝率ベースの期待値（勝率モデルは較正崩壊していたが複勝率は健全だったため）
     複勝の実払戻(JV)で期待値を作り、複勝ROIが100%を超えるか。
案2: オッズ帯を絞った選抜（高オッズ馬の過大評価が原因なら低オッズに限定すれば効くはず）

較正は2025前半で学習し、後半で評価（学習と評価を分ける）。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

UNORD = {"馬連", "ワイド", "3連複", "枠連"}


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "馬番"])
    d = d.merge(rf.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    for c in ["着順_num", "単勝オッズ", "人気", "MF勝率", "MF連対率", "MF複勝率",
              "MF勝率順位", "MF複勝順位", "馬番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "MF複勝率", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    d = d.dropna(subset=["dt"])

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORD:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def fuku_roi(s, pay):
    if not len(s):
        return float("nan")
    inv = ret = 0
    for _, r in s.iterrows():
        inv += 100
        ret += pay.get((r["race_id"], "複勝", r["bn"]), 0)
    return ret / inv * 100


def main():
    d, pay = load()
    mid = d["dt"].quantile(0.5)
    tr, te = d[d["dt"] <= mid], d[d["dt"] > mid].copy()
    log(f"較正学習(前半) {tr['race_id'].nunique()}R / 評価(後半) {te['race_id'].nunique()}R")

    for col, act, total in [("MF勝率", "win", 1.0), ("MF複勝率", "fuku", 3.0)]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(tr[col], tr[act])
        c = iso.predict(te[col])
        te[col + "_c"] = c / te.assign(_c=c).groupby("race_id")["_c"].transform("sum") * total

    # 複勝の市場想定（実払戻から逆算できるのは的中馬のみ。ここでは単勝オッズから
    # ハーヴィル近似で複勝の市場確率を作り、モデルとの比較に使う）
    te["q_win"] = (1 / te["単勝オッズ"])
    te["q_win"] = te["q_win"] / te.groupby("race_id")["q_win"].transform("sum")

    log("\n" + "=" * 72)
    log("【案1】複勝率ベースの選抜（複勝率モデルは較正が健全だった）")
    log("=" * 72)
    base = fuku_roi(te, pay)
    log(f"  全馬を複勝で買った場合 = {base:.1f}%")
    log(f"\n  {'選抜':<30}{'n':>6}{'複勝率':>8}{'複勝ROI':>9}")
    for nm, s in [("MF複勝順位1位", te[te["MF複勝順位"] == 1]),
                  ("MF複勝順位1-2位", te[te["MF複勝順位"] <= 2]),
                  ("MF複勝順位1-3位", te[te["MF複勝順位"] <= 3]),
                  ("較正複勝率 上位10%", te[te["MF複勝率_c"] >= te["MF複勝率_c"].quantile(0.9)]),
                  ("較正複勝率 ≥0.6", te[te["MF複勝率_c"] >= 0.6]),
                  ("較正複勝率 ≥0.5", te[te["MF複勝率_c"] >= 0.5])]:
        if len(s) < 80:
            continue
        log(f"  {nm:<30}{len(s):6d}{s['fuku'].mean()*100:7.1f}%{fuku_roi(s, pay):8.1f}%")

    log("\n  ― 人気帯 × MF複勝順位1位 の複勝ROI ―")
    log(f"  {'人気':<12}{'n':>6}{'複勝率':>8}{'複勝ROI':>9}")
    t1 = te[te["MF複勝順位"] == 1]
    for lo, hi, nm in [(1, 1, "1番人気"), (2, 3, "2-3番人気"), (4, 5, "4-5番人気"),
                       (6, 9, "6-9番人気"), (10, 99, "10番人気-")]:
        s = t1[(t1["人気"] >= lo) & (t1["人気"] <= hi)]
        if len(s) < 50:
            continue
        log(f"  {nm:<12}{len(s):6d}{s['fuku'].mean()*100:7.1f}%{fuku_roi(s, pay):8.1f}%")

    log("\n" + "=" * 72)
    log("【案2】オッズ帯を絞った選抜（高オッズの過大評価を排除）")
    log("=" * 72)
    te["EV単"] = te["MF勝率_c"] * te["単勝オッズ"]
    log(f"  {'オッズ帯':<14}{'選抜':<18}{'n':>6}{'勝率':>7}{'単勝ROI':>9}{'複勝ROI':>9}")
    for lo, hi, nm in [(1.0, 5, "〜5倍"), (5, 10, "5-10倍"), (10, 20, "10-20倍"),
                       (20, 50, "20-50倍")]:
        band = te[(te["単勝オッズ"] >= lo) & (te["単勝オッズ"] < hi)]
        if len(band) < 200:
            continue
        allroi = (band["win"] * band["単勝オッズ"]).sum() / len(band) * 100
        log(f"  {nm:<14}{'帯を全部買う':<18}{len(band):6d}{band['win'].mean()*100:6.1f}%"
            f"{allroi:8.1f}%{fuku_roi(band, pay):8.1f}%")
        for sel_nm, s in [("＋MF複勝1位", band[band["MF複勝順位"] == 1]),
                          ("＋MF勝率1位", band[band["MF勝率順位"] == 1]),
                          ("＋期待値上位20%", band[band["EV単"] >= band["EV単"].quantile(0.8)])]:
            if len(s) < 60:
                continue
            roi = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
            log(f"  {'':<14}{sel_nm:<18}{len(s):6d}{s['win'].mean()*100:6.1f}%"
                f"{roi:8.1f}%{fuku_roi(s, pay):8.1f}%")
        log("")


if __name__ == "__main__":
    main()
