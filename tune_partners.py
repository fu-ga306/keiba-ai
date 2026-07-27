# -*- coding: utf-8 -*-
"""印の付け方・相手の選び方を実払戻(JV)で比較する。再学習不要。

モデル自体は天井(2026-07-27に確認)だが、その出力をどう選定に使うかは別問題。
軸(妙/複妙)は現行のまま、相手プールの作り方だけを差し替えてROIを測る。

評価: 妙軸の連系(馬単 妙→相手5点 / 馬連 妙-相手5点)と、複妙軸のワイド(3点)。
帯は妙の人気で分ける(勝負7-9 / 買い4-6)。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}


def load():
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位", "予測スコア"]].rename(
        columns={"予測順位": "主place3順", "予測スコア": "主place3スコア"})
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "馬番"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = mf.merge(p3, on=["race_id", "馬名"], how="inner").merge(
        rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    for c in ["人気", "着順_num", "馬番", "単勝オッズ"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["人気", "着順_num", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["fuku"] = (d["着順_num"] <= 3).astype(float)

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORDERED:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def add_rules(g):
    """レース内で各ルールの相手優先度(小さいほど上位)を作る。市場を使うのは対照のみ。"""
    n = len(g)
    r_mf3 = g["MF複勝順位"].rank(method="min")                 # 現行
    r_mf2 = g["MF連対率"].rank(ascending=False, method="min")
    r_mfw = g["MF勝率"].rank(ascending=False, method="min")
    r_p3 = g["主place3順"].rank(method="min")
    r_pop = g["人気"].rank(method="min")
    return {
        "現行(MF複勝)": r_mf3,
        "MF複勝×主place3": (r_mf3 + r_p3) / 2,
        "MF複勝×MF連対": (r_mf3 + r_mf2) / 2,
        "MF複勝×MF勝率": (r_mf3 + r_mfw) / 2,
        "主place3のみ": r_p3,
        "MF複勝×人気(対照)": (r_mf3 * 0.6 + r_pop * 0.4),
    }


def evaluate(d, pay, n_partner=5):
    myo = d[(d["MF勝率順位"] == 1) & (d["主place3順"] != 1)][["race_id", "人気"]].rename(
        columns={"人気": "妙人気"})
    d = d.merge(myo, on="race_id", how="inner")
    d["帯"] = d["妙人気"].map(
        lambda p: "勝負(7-9)" if 7 <= p <= 9 else ("買い(4-6)" if 4 <= p <= 6 else None))
    d = d[d["帯"].notna()]

    rules = None
    acc = {}
    for (band, rid), g in d.groupby(["帯", "race_id"]):
        ax = g[g["MF勝率順位"] == 1]
        cx = g[g["MF複勝順位"] == 1]
        if not len(ax) or not len(cx):
            continue
        a, c = ax.iloc[0], cx.iloc[0]
        rr = add_rules(g)
        if rules is None:
            rules = list(rr)
        for name, pri in rr.items():
            gg = g.assign(_p=pri.values)
            part = gg[gg["馬名"] != a["馬名"]].nsmallest(n_partner, "_p")
            wpart = gg[gg["馬名"] != c["馬名"]].nsmallest(3, "_p")
            k = (band, name)
            s = acc.setdefault(k, dict(inv=0.0, ret=0.0, hit=0, n=0, cap=0.0, races=0,
                                       winv=0.0, wret=0.0, whit=0, wn=0))
            s["races"] += 1
            s["cap"] += gg.nsmallest(n_partner, "_p")["fuku"].sum()
            for _, p in part.iterrows():
                for kind, combo, amt in (
                        ("馬単", f"{a['bn']}-{p['bn']}", 100),
                        ("馬連", "-".join(sorted([a["bn"], p["bn"]])), 100)):
                    v = pay.get((rid, kind, combo), 0)
                    s["inv"] += amt; s["ret"] += v / 100 * amt
                    s["hit"] += (v > 0); s["n"] += 1
            for _, p in wpart.iterrows():
                v = pay.get((rid, "ワイド", "-".join(sorted([c["bn"], p["bn"]]))), 0)
                s["winv"] += 200; s["wret"] += v / 100 * 200
                s["whit"] += (v > 0); s["wn"] += 1

    print(f"=== 相手の選び方 比較（2025・JV実払戻・相手{n_partner}点）===")
    for band in ["勝負(7-9)", "買い(4-6)"]:
        print(f"\n【{band}】")
        print(f"{'ルール':<20} 連系ROI  的中率  ワイドROI 的中率  捕捉@{n_partner}")
        rows = []
        for name in rules:
            s = acc.get((band, name))
            if not s or not s["n"]:
                continue
            rows.append((name, s["ret"] / s["inv"] * 100, s["hit"] / s["n"] * 100,
                         s["wret"] / s["winv"] * 100, s["whit"] / s["wn"] * 100,
                         s["cap"] / s["races"]))
        for name, roi, hr, wroi, whr, cap in sorted(rows, key=lambda x: -x[1]):
            mark = " ←現行" if name.startswith("現行") else ""
            print(f"{name:<20} {roi:7.1f}% {hr:5.1f}%  {wroi:7.1f}% {whr:5.1f}%  {cap:.3f}{mark}")


if __name__ == "__main__":
    d, pay = load()
    evaluate(d, pay, n_partner=5)
