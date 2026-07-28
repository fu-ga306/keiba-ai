# -*- coding: utf-8 -*-
"""時系列リーク修正後、印の付け方〜買い目までを全面的に再検証する。

2026-07-28に「race_id順を時系列順とみなしていた」バグを修正した結果、
MF妙の単勝回収が150.5%→77.2%へ落ちた（リーク由来だった）。現行の買い方は
リークを含む数字の上で設計されているため、クリーンなデータで組み直す必要がある。

出力は「どこにエッジが残っているか」を素の状態で示すことに徹する。
（買い方の再設計はこの結果を見てから行う）

使い方: python revalidate_spec.py   ※<=2024学習/2025検証の成果物が必要
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

TY = os.environ.get("KEIBA_TEST_YEAR", "2025")
UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}


def load():
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "主place3順"})
    main = pd.read_csv("model_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "主勝率順"})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "クラス_num", "前走間隔", "出走頭数"])
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "馬番"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = (mf.merge(p3, on=["race_id", "馬名"], how="left")
           .merge(main, on=["race_id", "馬名"], how="left")
           .merge(rf, on=["race_id", "馬名"], how="left")
           .merge(rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left"))
    d = d[d["race_id"].str.startswith(TY)]
    for c in ["人気", "着順_num", "馬番", "単勝オッズ", "クラス_num", "前走間隔"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["人気", "着順_num", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["win"] = (d["着順_num"] == 1).astype(float)

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(TY)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORDERED:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def roi(sub, pay, kind, amt=100):
    if not len(sub):
        return float("nan"), 0.0
    inv = ret = hit = 0
    for _, r in sub.iterrows():
        v = pay.get((r["race_id"], kind, r["bn"]), 0)
        inv += amt
        ret += v / 100 * amt
        hit += (v > 0)
    return ret / inv * 100, hit / len(sub) * 100


def sec_axis(d, pay):
    print("\n" + "=" * 72)
    print("【1】軸候補の素の成績（単勝・複勝の実払戻）")
    print("=" * 72)
    print(f"{'軸の定義':<20}{'複勝率':>8}{'複勝ROI':>9}{'勝率':>7}{'単勝ROI':>9}{'平均人気':>9}")
    defs = {
        "MF複勝1位(複妙)": d["MF複勝順位"] == 1,
        "MF勝率1位(妙)": d["MF勝率順位"] == 1,
        "主place3-1位": d["主place3順"] == 1,
        "主勝率1位": d["主勝率順"] == 1,
        "1番人気(市場)": d["人気"] == 1,
    }
    for k, m in defs.items():
        s = d[m]
        if not len(s):
            continue
        froi, _ = roi(s, pay, "複勝", 300)
        troi, _ = roi(s, pay, "単勝", 500)
        print(f"{k:<20}{s['fuku'].mean()*100:7.1f}%{froi:8.1f}%"
              f"{s['win'].mean()*100:6.1f}%{troi:8.1f}%{s['人気'].mean():8.1f}")


def sec_band(d, pay):
    print("\n" + "=" * 72)
    print("【2】人気帯別にエッジが残っているか（軸=MF勝率1位／MF複勝1位）")
    print("=" * 72)
    for lab, col in [("妙(MF勝率1位)", "MF勝率順位"), ("複妙(MF複勝1位)", "MF複勝順位")]:
        print(f"\n― {lab} ―")
        print(f"{'人気帯':<10}{'R数':>6}{'複勝率':>8}{'複勝ROI':>9}{'勝率':>7}{'単勝ROI':>9}")
        s = d[d[col] == 1]
        for lo, hi, nm in [(1, 1, "1人気"), (2, 3, "2-3"), (4, 6, "4-6"),
                           (7, 9, "7-9"), (10, 99, "10-")]:
            ss = s[(s["人気"] >= lo) & (s["人気"] <= hi)]
            if len(ss) < 40:
                continue
            froi, _ = roi(ss, pay, "複勝", 300)
            troi, _ = roi(ss, pay, "単勝", 500)
            print(f"{nm:<10}{len(ss):6d}{ss['fuku'].mean()*100:7.1f}%{froi:8.1f}%"
                  f"{ss['win'].mean()*100:6.1f}%{troi:8.1f}%")


def sec_partner(d, pay):
    print("\n" + "=" * 72)
    print("【3】相手プールの質（軸=MF勝率1位、相手5点の馬単・馬連）")
    print("=" * 72)
    myo = d[d["MF勝率順位"] == 1][["race_id", "人気"]].rename(columns={"人気": "妙人気"})
    dd = d.merge(myo, on="race_id", how="inner")
    dd["帯"] = dd["妙人気"].map(lambda p: "妙7-9" if 7 <= p <= 9 else
                               ("妙4-6" if 4 <= p <= 6 else
                                ("妙2-3" if 2 <= p <= 3 else "妙1")))
    rules = ["MF複勝順位", "主place3順", "人気"]
    print(f"{'帯':<8}{'相手ルール':<14}{'馬単ROI':>9}{'馬連ROI':>9}{'捕捉@5':>9}{'R数':>6}")
    for band in ["妙1", "妙2-3", "妙4-6", "妙7-9"]:
        sub = dd[dd["帯"] == band]
        for rule in rules:
            acc = dict(ui=0, ur=0, ri=0, rr=0, cap=0.0, n=0)
            for rid, g in sub.groupby("race_id"):
                ax = g[g["MF勝率順位"] == 1]
                if not len(ax):
                    continue
                a = ax.iloc[0]
                gg = g.assign(_p=g[rule].rank(method="min"))
                acc["cap"] += gg.nsmallest(5, "_p")["fuku"].sum()
                acc["n"] += 1
                for _, p in gg[gg["馬名"] != a["馬名"]].nsmallest(5, "_p").iterrows():
                    acc["ui"] += 100
                    acc["ur"] += pay.get((rid, "馬単", f"{a['bn']}-{p['bn']}"), 0)
                    acc["ri"] += 100
                    acc["rr"] += pay.get((rid, "馬連",
                                          "-".join(sorted([a["bn"], p["bn"]]))), 0)
            if acc["n"] < 30:
                continue
            print(f"{band:<8}{rule:<14}{acc['ur']/acc['ui']*100:8.1f}%"
                  f"{acc['rr']/acc['ri']*100:8.1f}%{acc['cap']/acc['n']:8.3f}{acc['n']:6d}")
        print()


def sec_marks(d, pay):
    print("\n" + "=" * 72)
    print("【4】印(○▲△)の馬券内率＝主place3順の2-4位")
    print("=" * 72)
    print(f"{'印':<6}{'定義':<18}{'勝率':>7}{'連対':>7}{'複勝':>7}{'R数':>7}")
    for mk, r in [("◎", 1), ("○", 2), ("▲", 3), ("△", 4), ("×", 5)]:
        s = d[d["主place3順"] == r]
        if not len(s):
            continue
        ren = (s["着順_num"] <= 2).mean() * 100
        print(f"{mk:<6}{'主place3-'+str(r)+'位':<18}{s['win'].mean()*100:6.1f}%"
              f"{ren:6.1f}%{s['fuku'].mean()*100:6.1f}%{len(s):7d}")


def main():
    d, pay = load()
    print(f"検証年={TY} / {d['race_id'].nunique()}レース {len(d)}頭  ※時系列リーク修正後")
    sec_axis(d, pay)
    sec_band(d, pay)
    sec_partner(d, pay)
    sec_marks(d, pay)
    print("\n" + "=" * 72)
    print("参考(リークあり時の値): 妙の単勝回収150.5% / 複妙の複勝率57.9% / 捕捉@5 1.988")
    print("=" * 72)


if __name__ == "__main__":
    main()
