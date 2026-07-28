# -*- coding: utf-8 -*-
"""買い帯(妙4-6人気×オッズ8倍以上)の中身を分解し、回収率を最大化する構成を探す。

血統リーク除去後、買い帯だけが本番の買い方で黒字になった（2025: 114.4%・
前半107.8/後半120.9%）。ここが唯一の黒字なので中身を最適化してから本番再学習する。

現行の買い帯メニュー: 単勝 / 馬単 妙→複勝上位5 / 馬連 妙-複勝上位5 /
                     3連複 妙◎軸-複勝上位5 / 3連単 妙→複勝3→複勝5
検証軸:
  A. 券種別の寄与（どれが黒字でどれが足を引っ張るか）
  B. 相手の頭数（3/4/5/6）
  C. オッズゲートの水準（8倍/10倍/12倍/15倍）
  D. 妙の人気(4/5/6)とMF自信度(サイズ)による層別
すべて前半/後半の分割で確認し、両方100%以上のものだけ採用候補とする。
"""
import warnings

warnings.filterwarnings("ignore")
import itertools

import os

import numpy as np
import pandas as pd

TY = os.environ.get("KEIBA_TEST_YEAR", "2025")
UNORD = {"馬連", "ワイド", "3連複", "枠連"}


def log(m):
    print(m, flush=True)


def load():
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF勝率順位", "MF複勝順位"]]
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "馬番", "人気", "単勝オッズ", "クラス_num", "着順_num"]]
    d = rf.merge(p3, on=["race_id", "馬名"], how="inner").merge(
        mf, on=["race_id", "馬名"], how="inner")
    for c in ["馬番", "人気", "単勝オッズ", "クラス_num", "着順_num",
              "MF勝率", "MF勝率順位", "MF複勝順位", "place3順"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["馬番", "人気", "単勝オッズ", "着順_num"])
    d = d[d["race_id"].str.startswith(TY)]
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(TY)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORD:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def kai_races(d, odds_gate=8.0, pops=(4, 5, 6)):
    """買い帯に該当するレースを抽出。各レースの軸・相手・文脈を返す。"""
    out = []
    for rid, g in d.groupby("race_id"):
        if len(g) < 4:
            continue
        cls = g["クラス_num"].iloc[0]
        if pd.isna(cls) or cls >= 5:      # OP以上は買い帯対象外
            continue
        fav = g.loc[g["人気"].idxmin()]
        # ◎: 1番人気が2.0倍以下なら人気1位、それ以外はplace3-1位
        hon = fav["馬名"] if fav["単勝オッズ"] <= 2.0 else \
            g.loc[g["place3順"].idxmin(), "馬名"]
        w1 = g[g["MF勝率順位"] == 1]
        if w1.empty:
            continue
        myo = w1.iloc[0]
        if myo["馬名"] == hon:            # 妙なし＝堅実帯
            continue
        if myo["人気"] not in pops:
            continue
        if myo["単勝オッズ"] < odds_gate:
            continue
        f1 = g[g["MF複勝順位"] == 1]
        partners = g[g["馬名"] != myo["馬名"]].sort_values("MF複勝順位")
        out.append(dict(rid=rid, dt=g["dt"].iloc[0], myo=myo, hon=hon,
                        fukumyo=(f1.iloc[0] if len(f1) else myo),
                        partners=partners, g=g, pop=int(myo["人気"]),
                        mfw=myo["MF勝率"], cls=int(cls)))
    return out


def score(bets, pay):
    """bets=[(rid,券種,combo,金額)] → ROI・的中率・前後半"""
    if not bets:
        return None
    rows = [(rid, k, pay.get((rid, k, c if k not in UNORD else
                              "-".join(sorted(c.split("-")))), 0) / 100 * a, a, dt)
            for rid, k, c, a, dt in bets]
    df = pd.DataFrame(rows, columns=["rid", "kind", "ret", "amt", "dt"])
    mid = df["dt"].median()
    inv = df["amt"].sum()
    roi = df["ret"].sum() / inv * 100
    h1 = df[df["dt"] <= mid]
    h2 = df[df["dt"] > mid]
    r1 = h1["ret"].sum() / h1["amt"].sum() * 100 if len(h1) else np.nan
    r2 = h2["ret"].sum() / h2["amt"].sum() * 100 if len(h2) else np.nan
    return roi, (df["ret"] > 0).mean() * 100, len(df), r1, r2, inv


def make_bets(races, kinds, n_part, amt=100):
    """指定券種・相手数で買い目を組む。"""
    bets = []
    for r in races:
        myo, fm, g, dt, rid = r["myo"], r["fukumyo"], r["g"], r["dt"], r["rid"]
        part = r["partners"].head(n_part)
        hon_bn = g[g["馬名"] == r["hon"]]["bn"]
        hon_bn = hon_bn.iloc[0] if len(hon_bn) else None
        for k in kinds:
            if k == "単勝":
                bets.append((rid, "単勝", myo["bn"], 500, dt))
            elif k == "複勝":
                bets.append((rid, "複勝", myo["bn"], 300, dt))
            elif k == "ワイド":
                for _, p in part.head(3).iterrows():
                    bets.append((rid, "ワイド", f"{fm['bn']}-{p['bn']}", 200, dt))
            elif k == "馬単":
                for _, p in part.iterrows():
                    bets.append((rid, "馬単", f"{myo['bn']}-{p['bn']}", amt, dt))
            elif k == "馬連":
                for _, p in part.iterrows():
                    bets.append((rid, "馬連", f"{myo['bn']}-{p['bn']}", amt, dt))
            elif k == "3連複" and hon_bn:
                for _, p in part.iterrows():
                    tri = {myo["bn"], hon_bn, p["bn"]}
                    if len(tri) == 3:
                        bets.append((rid, "3連複", "-".join(sorted(tri)), amt, dt))
            elif k == "3連単":
                sec = part.head(3)
                thi = part.head(5)
                for _, a in sec.iterrows():
                    for _, b in thi.iterrows():
                        if a["bn"] != b["bn"]:
                            bets.append((rid, "3連単",
                                         f"{myo['bn']}-{a['bn']}-{b['bn']}", amt, dt))
    return bets


def show(label, st):
    if not st:
        return
    roi, hit, n, r1, r2, inv = st
    mk = "◎" if min(r1, r2) >= 100 else ("○" if min(r1, r2) >= 90 else "")
    log(f"  {label:<34}{n:6d}点 的中{hit:5.1f}% ROI{roi:7.1f}% (前{r1:6.1f}/後{r2:6.1f}){mk}")


def main():
    d, pay = load()
    races = kai_races(d)
    log(f"買い帯レース: {len(races)}R（妙4-6人気×8倍以上×OP未満）")

    log("\n" + "=" * 84)
    log("【A】券種別の単独成績（相手5頭・現行構成）")
    log("=" * 84)
    for k in ["単勝", "複勝", "ワイド", "馬単", "馬連", "3連複", "3連単"]:
        show(k, score(make_bets(races, [k], 5), pay))

    log("\n" + "=" * 84)
    log("【B】相手の頭数（馬単・馬連・3連複）")
    log("=" * 84)
    for k in ["馬単", "馬連", "3連複"]:
        for n in [3, 4, 5, 6]:
            show(f"{k} 相手{n}頭", score(make_bets(races, [k], n), pay))
        log("")

    log("=" * 84)
    log("【C】オッズゲートの水準（現行8倍）")
    log("=" * 84)
    for gate in [8, 10, 12, 15, 20]:
        rs = kai_races(d, odds_gate=gate)
        show(f"ゲート{gate}倍 現行5券種({len(rs)}R)",
             score(make_bets(rs, ["単勝", "馬単", "馬連", "3連複", "3連単"], 5), pay))

    log("\n" + "=" * 84)
    log("【D】妙の人気別・MF自信度別（現行5券種・相手5）")
    log("=" * 84)
    for p in [4, 5, 6]:
        rs = [r for r in races if r["pop"] == p]
        show(f"妙{p}番人気({len(rs)}R)",
             score(make_bets(rs, ["単勝", "馬単", "馬連", "3連複", "3連単"], 5), pay))
    mfws = [r["mfw"] for r in races if pd.notna(r["mfw"])]
    if mfws:
        th = np.median(mfws)
        for lab, sel in [("MF自信度 上位半分", [r for r in races if r["mfw"] >= th]),
                         ("MF自信度 下位半分", [r for r in races if r["mfw"] < th])]:
            show(f"{lab}({len(sel)}R)",
                 score(make_bets(sel, ["単勝", "馬単", "馬連", "3連複", "3連単"], 5), pay))

    log("\n" + "=" * 84)
    log("【E】券種の組み合わせ最適化（黒字券種だけ残す）")
    log("=" * 84)
    cands = ["単勝", "馬単", "馬連", "3連複", "3連単", "複勝", "ワイド"]
    results = []
    for size in (1, 2, 3, 4):
        for combo in itertools.combinations(cands, size):
            st = score(make_bets(races, list(combo), 5), pay)
            if st and min(st[3], st[4]) >= 100:
                results.append((combo, st))
    results.sort(key=lambda x: -x[1][0])
    if not results:
        log("  両半期100%以上の組み合わせなし")
    for combo, st in results[:12]:
        show("+".join(combo), st)
    verify_new(races, pay)


def verify_new(races, pay):
    """2025で選んだ新構成を、別年でもそのまま当てられるよう定義しておく。
    新構成: 妙5-6番人気(4番人気は62.3%と毒なので除外) × オッズ8倍以上
            馬単 妙→複勝上位6 / 馬連 妙-複勝上位3 / 複勝 妙(的中率確保)"""
    sel = [r for r in races if r["pop"] in (5, 6)]
    log("\n" + "=" * 84)
    log(f"【新構成の検証】妙5-6番人気のみ({len(sel)}R) ※4番人気を除外")
    log("=" * 84)
    bets = []
    for r in sel:
        myo, dt, rid = r["myo"], r["dt"], r["rid"]
        part = r["partners"]
        bets.append((rid, "複勝", myo["bn"], 300, dt))
        for _, p in part.head(6).iterrows():
            bets.append((rid, "馬単", f"{myo['bn']}-{p['bn']}", 100, dt))
        for _, p in part.head(3).iterrows():
            bets.append((rid, "馬連", f"{myo['bn']}-{p['bn']}", 100, dt))
    show("新構成(馬単6+馬連3+複勝)", score(bets, pay))
    for k in ("馬単", "馬連", "複勝"):
        b = [x for x in bets if x[1] == k]
        show(f"  └ {k}単独", score(b, pay))
    log("\n  ― 参考: 現行構成(妙4-6人気・5券種・相手5) ―")
    show("現行", score(make_bets(races, ["単勝", "馬単", "馬連", "3連複", "3連単"], 5), pay))


if __name__ == "__main__":
    main()
