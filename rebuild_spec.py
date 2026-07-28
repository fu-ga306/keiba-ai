# -*- coding: utf-8 -*-
"""リーク修正・較正修正を踏まえた買い方の再設計と検証（券種を広げる）。

これまでの検証で生き残った素材:
  ・複勝率(place3)モデルは較正が健全 → 軸に使える
  ・勝率モデルは較正崩壊 → isotonicで補正して使う
  ・単勝は5-10倍帯でMF勝率1位が最良(94.4%) / 複勝は較正複勝率≥0.6が最良(90.2%)
本スクリプトは軸と相手を上記に統一したうえで、単勝・複勝・ワイド・馬連・馬単・
3連複まで券種を広げ、実払戻(JV)でROIを測る。

偶然を排除するため、2025を実開催日で前半/後半に割り、
  ①前半で較正→後半で評価  ②後半で較正→前半で評価
の双方向で測り、両方で良い戦略だけを採用候補とする。
"""
import warnings

warnings.filterwarnings("ignore")
import itertools

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
    for c in ["着順_num", "単勝オッズ", "人気", "MF勝率", "MF複勝率",
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


def calibrate(fit_df, apply_df):
    out = apply_df.copy()
    for col, act, total in [("MF勝率", "win", 1.0), ("MF複勝率", "fuku", 3.0)]:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            fit_df[col], fit_df[act])
        c = iso.predict(out[col])
        out[col + "_c"] = c
        s = out.assign(_c=c).groupby("race_id")["_c"].transform("sum")
        out[col + "_c"] = out[col + "_c"] / s.replace(0, np.nan) * total
    return out


def bets_for_race(g):
    """1レース分の買い目候補を返す: [(戦略名, 券種, 組み合わせ)]"""
    out = []
    g = g.sort_values("MF複勝順位")
    f1 = g[g["MF複勝順位"] == 1]
    w1 = g[g["MF勝率順位"] == 1]
    if f1.empty or w1.empty:
        return out
    a_f = f1.iloc[0]           # 複勝軸
    a_w = w1.iloc[0]           # 勝率軸
    part = g[(g["MF複勝順位"] >= 2) & (g["MF複勝順位"] <= 4)]

    # 単勝: 勝率軸が5-10倍のときだけ
    if 5 <= a_w["単勝オッズ"] < 10:
        out.append(("単勝 勝率軸(5-10倍)", "単勝", a_w["bn"]))
    # 複勝: 較正複勝率が高い馬
    for _, r in g[g["MF複勝率_c"] >= 0.6].iterrows():
        out.append(("複勝 較正複勝率≥0.6", "複勝", r["bn"]))
    # 複勝: 複勝軸が4-5番人気のとき
    if 4 <= a_f["人気"] <= 5:
        out.append(("複勝 複勝軸(4-5人気)", "複勝", a_f["bn"]))
    # ワイド/馬連: 複勝軸 - 複勝2,3位
    for _, r in part[part["MF複勝順位"] <= 3].iterrows():
        c = "-".join(sorted([a_f["bn"], r["bn"]]))
        out.append(("ワイド 複勝軸-2,3位", "ワイド", c))
        out.append(("馬連 複勝軸-2,3位", "馬連", c))
    # 馬単: 勝率軸 → 複勝2-4位
    for _, r in part.iterrows():
        if r["bn"] != a_w["bn"]:
            out.append(("馬単 勝率軸→複勝2-4位", "馬単", f"{a_w['bn']}-{r['bn']}"))
    # 3連複: 複勝1-2-3位
    top3 = g[g["MF複勝順位"] <= 3]
    if len(top3) == 3:
        out.append(("3連複 複勝1-2-3位", "3連複", "-".join(sorted(top3["bn"]))))
    # ワイド: 較正複勝率上位3のBOX
    hi = g.nlargest(3, "MF複勝率_c")
    if len(hi) == 3:
        for x, y in itertools.combinations(sorted(hi["bn"]), 2):
            out.append(("ワイド 較正上位3BOX", "ワイド", "-".join(sorted([x, y]))))
    return out


def evaluate(te, pay):
    recs = []
    for rid, g in te.groupby("race_id"):
        for nm, kind, combo in bets_for_race(g):
            recs.append((nm, kind, pay.get((rid, kind, combo), 0)))
    if not recs:
        return pd.DataFrame()
    r = pd.DataFrame(recs, columns=["戦略", "券種", "ret"])
    agg = r.groupby(["戦略", "券種"]).agg(
        点数=("ret", "size"), 的中=("ret", lambda s: (s > 0).sum()),
        払戻=("ret", "sum")).reset_index()
    agg["ROI"] = agg["払戻"] / (agg["点数"] * 100) * 100
    agg["的中率"] = agg["的中"] / agg["点数"] * 100
    return agg


def main():
    d, pay = load()
    mid = d["dt"].quantile(0.5)
    h1, h2 = d[d["dt"] <= mid], d[d["dt"] > mid]
    log(f"前半 {h1['race_id'].nunique()}R / 後半 {h2['race_id'].nunique()}R")

    a = evaluate(calibrate(h1, h2), pay).rename(columns={"ROI": "ROI_後半", "点数": "点数_後半"})
    b = evaluate(calibrate(h2, h1), pay).rename(columns={"ROI": "ROI_前半", "点数": "点数_前半"})
    m = a.merge(b[["戦略", "券種", "ROI_前半", "点数_前半"]], on=["戦略", "券種"], how="outer")
    m["平均ROI"] = (m["ROI_後半"] + m["ROI_前半"]) / 2
    m = m.sort_values("平均ROI", ascending=False)

    log("\n" + "=" * 82)
    log("【券種別の成績】双方向検証（前半で較正→後半評価 / 後半で較正→前半評価）")
    log("=" * 82)
    log(f"  {'戦略':<26}{'券種':<6}{'点数':>7}{'的中率':>8}"
        f"{'ROI後半':>9}{'ROI前半':>9}{'平均':>8}{'判定':>5}")
    for _, r in m.iterrows():
        if pd.isna(r["ROI_後半"]) or pd.isna(r["ROI_前半"]):
            continue
        ok = "◎" if min(r["ROI_後半"], r["ROI_前半"]) >= 100 else (
            "○" if min(r["ROI_後半"], r["ROI_前半"]) >= 88 else "")
        log(f"  {r['戦略']:<26}{r['券種']:<6}{int(r['点数_後半']):7d}{r['的中率']:7.1f}%"
            f"{r['ROI_後半']:8.1f}%{r['ROI_前半']:8.1f}%{r['平均ROI']:7.1f}%{ok:>5}")
    log("\n  ◎=両方100%以上 / ○=両方88%以上（現行75-83%より明確に良い）")

    keep = m[(m["ROI_後半"] >= 88) & (m["ROI_前半"] >= 88)]
    if len(keep):
        tot_pts = keep["点数_後半"].sum()
        tot_ret = (keep["ROI_後半"] * keep["点数_後半"]).sum() / 100
        log(f"\n【採用候補のみで組んだポートフォリオ（後半）】")
        log(f"  {len(keep)}戦略 / {int(tot_pts)}点 / 合成ROI {tot_ret/tot_pts*100:.1f}%")
        for _, r in keep.iterrows():
            log(f"   ・{r['戦略']}（{r['券種']}）")


if __name__ == "__main__":
    main()
