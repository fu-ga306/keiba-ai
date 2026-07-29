# -*- coding: utf-8 -*-
"""なぜROIがこれほど暴れるのか ― ばらつきの正体を定量化し、買い方の設計を見直す。

観測された異常: 同じ買い方で 買い帯 2025=114.4% / 2024=39.9%、勝負帯 53.8% / 138.0%。
仮説A: 単なる偶然（的中率1-2%の券種を年間数十レースで測っているだけ）
仮説B: 予想の仕方に構造的な問題がある

検証:
  1. レース単位のブートストラップでROIの95%信頼区間を出す
     ※点数ではなくレース単位で再標本化する。同一レースの馬単6点は「軸が飛べば全滅」
       で強く相関しており、点数を独立サンプルと見なすのは誤り。
  2. 「回収率100%と80%を区別する」のに必要なレース数を券種ごとに算出
  3. 券種別の「測定可能性」ランキング → 何を買うべきかの再設計へ
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

UNORD = {"馬連", "ワイド", "3連複", "枠連"}
RNG = np.random.default_rng(42)


def log(m):
    print(m, flush=True)


def load(year="2025"):
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
    d = d[d["race_id"].str.startswith(year)]
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(year)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORD:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def build_race_bets(d, pay):
    """各レースについて、券種別に (投資, 払戻) を集計する。
    買い帯相当（妙4-6人気×8倍以上×OP未満）を対象にする。"""
    recs = []
    for rid, g in d.groupby("race_id"):
        if len(g) < 4:
            continue
        cls = g["クラス_num"].iloc[0]
        if pd.isna(cls) or cls >= 5:
            continue
        fav = g.loc[g["人気"].idxmin()]
        hon = fav["馬名"] if fav["単勝オッズ"] <= 2.0 else g.loc[g["place3順"].idxmin(), "馬名"]
        w1 = g[g["MF勝率順位"] == 1]
        if w1.empty:
            continue
        myo = w1.iloc[0]
        if myo["馬名"] == hon or not (4 <= myo["人気"] <= 6) or myo["単勝オッズ"] < 8:
            continue
        part = g[g["馬名"] != myo["馬名"]].sort_values("MF複勝順位")
        f1 = g[g["MF複勝順位"] == 1]
        fm = f1.iloc[0] if len(f1) else myo
        row = {"rid": rid}
        row["単勝"] = (500, pay.get((rid, "単勝", myo["bn"]), 0) / 100 * 500)
        row["複勝"] = (300, pay.get((rid, "複勝", myo["bn"]), 0) / 100 * 300)
        for kind, n, axis in [("馬単", 5, myo), ("馬連", 5, myo)]:
            inv = ret = 0
            for _, p in part.head(n).iterrows():
                c = f"{axis['bn']}-{p['bn']}" if kind == "馬単" else \
                    "-".join(sorted([axis["bn"], p["bn"]]))
                inv += 100
                ret += pay.get((rid, kind, c), 0)
            row[kind] = (inv, ret)
        inv = ret = 0
        for _, p in part.head(3).iterrows():
            c = "-".join(sorted([fm["bn"], p["bn"]]))
            inv += 200
            ret += pay.get((rid, "ワイド", c), 0) / 100 * 200
        row["ワイド"] = (inv, ret)
        recs.append(row)
    return recs


def boot(recs, kind, n_boot=4000):
    """レース単位のブートストラップで ROI の分布を出す。"""
    arr = np.array([r[kind] for r in recs if kind in r], dtype=float)
    if len(arr) < 20:
        return None
    n = len(arr)
    idx = RNG.integers(0, n, size=(n_boot, n))
    inv = arr[:, 0][idx].sum(axis=1)
    ret = arr[:, 1][idx].sum(axis=1)
    rois = ret / inv * 100
    hit = (arr[:, 1] > 0).mean() * 100
    return (arr[:, 1].sum() / arr[:, 0].sum() * 100, hit, n,
            np.percentile(rois, 2.5), np.percentile(rois, 97.5), rois.std())


def need_n(recs, kind, target_gap=20.0):
    """ROIの標準誤差が target_gap/2 になるのに必要なレース数（100%と80%を区別する目安）。"""
    arr = np.array([r[kind] for r in recs if kind in r], dtype=float)
    if len(arr) < 20:
        return None
    per_race_roi = arr[:, 1] / arr[:, 0].mean() * 100
    sd = per_race_roi.std()
    return int((sd / (target_gap / 2)) ** 2)


def main():
    log("=" * 92)
    log("【なぜROIが暴れるのか】レース単位ブートストラップ（買い帯・2025）")
    log("=" * 92)
    d, pay = load("2025")
    recs = build_race_bets(d, pay)
    log(f"対象: {len(recs)}レース\n")
    log(f"  {'券種':<8}{'的中率':>8}{'実測ROI':>9}{'95%信頼区間':>22}{'1σ':>8}"
        f"{'100/80%を区別に必要なR数':>26}")
    for k in ["単勝", "複勝", "ワイド", "馬連", "馬単"]:
        b = boot(recs, k)
        if not b:
            continue
        roi, hit, n, lo, hi, sd = b
        nn = need_n(recs, k)
        log(f"  {k:<8}{hit:7.1f}%{roi:8.1f}%   [{lo:6.1f}% 〜 {hi:7.1f}%]{sd:7.1f}"
            f"{nn:>20,}R")
    log("\n  ※信頼区間が100%をまたぐ＝『儲かるか損か判定できない』。")
    log("    必要R数が実データ量を超える券種は、そもそも検証も運用も成立しない。")

    log("\n" + "=" * 92)
    log("【同じ診断を2024でも】年が変わっても同じ構造か")
    log("=" * 92)
    d24, pay24 = load("2024")
    recs24 = build_race_bets(d24, pay24)
    log(f"対象: {len(recs24)}レース\n")
    log(f"  {'券種':<8}{'的中率':>8}{'実測ROI':>9}{'95%信頼区間':>22}")
    for k in ["単勝", "複勝", "ワイド", "馬連", "馬単"]:
        b = boot(recs24, k)
        if not b:
            continue
        roi, hit, n, lo, hi, sd = b
        log(f"  {k:<8}{hit:7.1f}%{roi:8.1f}%   [{lo:6.1f}% 〜 {hi:7.1f}%]")

    log("\n" + "=" * 92)
    log("【点数と実質サンプルの乖離】同一レース内の買い目は独立ではない")
    log("=" * 92)
    for k in ["馬単", "馬連"]:
        arr = np.array([r[k] for r in recs], dtype=float)
        pts = int(arr[:, 0].sum() / 100)
        hit_races = int((arr[:, 1] > 0).sum())
        log(f"  {k}: 見かけ{pts:,}点 → 実質{len(arr)}レース / 的中したのは{hit_races}レースだけ")
        log(f"       ROIの{(arr[:,1].max()/arr[:,1].sum()*100):.0f}%が最大の1レースに由来")


if __name__ == "__main__":
    main()
