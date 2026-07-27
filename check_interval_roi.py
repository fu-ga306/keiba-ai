# -*- coding: utf-8 -*-
"""MF軸の「前走間隔」別ROIを検証年で測る（2026-07-27に2025で見つけた傾向のOOS確認）。

2025での発見: MF複勝1位の単勝ROIが 連闘(<=3週)86.0% / 通常(4-25週)187.7% / 半年+(>25週)84.8%。
1番人気(市場)では同傾向が出ない＝MF固有の弱点。別年でも再現すれば本物。

KEIBA_TEST_YEAR で対象年を指定（既定2025）。model_mf_result.csv と race_features.csv、
jv_payouts.csv を使う（再学習は不要）。
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

TY = os.environ.get("KEIBA_TEST_YEAR", "2025")


def main():
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "前走間隔"])
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "馬番"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = mf.merge(rf, on=["race_id", "馬名"], how="left").merge(
        rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    d = d[d["race_id"].str.startswith(TY)]
    for c in ["人気", "着順_num", "馬番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["人気", "着順_num", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["fuku"] = (d["着順_num"] <= 3).astype(float)

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(TY)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuku = {(r.race_id, r.組み合わせ): r.払戻金
            for r in jv[jv["券種"] == "複勝"].itertuples()}

    iv = pd.to_numeric(d["前走間隔"], errors="coerce")
    d["区分"] = np.where(iv <= 3, "連闘(<=3週)",
                        np.where(iv > 25, "半年+(>25週)", "通常(4-25週)"))
    d.loc[iv.isna(), "区分"] = "不明"

    def roi(sub, table, amt):
        if not len(sub):
            return float("nan"), 0
        inv = ret = hit = 0
        for _, r in sub.iterrows():
            v = table.get((r["race_id"], r["bn"]), 0)
            inv += amt
            ret += v / 100 * amt
            hit += (v > 0)
        return ret / inv * 100, hit / len(sub) * 100

    print(f"\n=== MF軸の前走間隔別ROI（test{TY}・JV実払戻）===")
    print(f"{'軸の定義':<16}{'区分':<14}{'単勝ROI':>9}{'勝率':>7}{'複勝ROI':>9}{'複勝率':>7}{'R数':>6}")
    for name, mask in [("MF複勝1位(複妙)", d["MF複勝順位"] == 1),
                       ("MF勝率1位(妙)", d["MF勝率順位"] == 1),
                       ("1番人気(対照)", d["人気"] == 1)]:
        for b in ["連闘(<=3週)", "通常(4-25週)", "半年+(>25週)"]:
            s = d[mask & (d["区分"] == b)]
            if len(s) < 50:
                continue
            troi, twin = roi(s, tan, 500)
            froi, _ = roi(s, fuku, 300)
            print(f"{name:<16}{b:<14}{troi:8.1f}%{twin:6.1f}%{froi:8.1f}%"
                  f"{s['fuku'].mean()*100:6.1f}%{len(s):6d}")
        print()
    print("※2025での値: 複妙 連闘86.0% / 通常187.7% / 半年+84.8%、"
          "1番人気は77.3/74.2/89.8(フラット)")


if __name__ == "__main__":
    main()
