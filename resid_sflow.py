# -*- coding: utf-8 -*-
"""S評価を軸にして、印（gap上位）に流す（2026-08-17）

前回（resid_sgrade.py）との違い
  前回は「軸＝★(gap最大) → 相手＝S評価」で測った。116.8%で現行120.6%に届かず。
  今回は逆向き。「軸＝S評価 → 相手＝印(gap上位)」。

  ワイド・馬連は順不同なので同じ組み合わせに見えるが、**選ばれるレースと
  相手が変わる**ので別物になる。
    前回: gap>=1.5 の馬が居るレースだけ（7,135レース）。相手はS評価の馬
    今回: S評価の馬が居るレースだけ。相手はgapが高い馬
  さらに馬単なら順番そのものが意味を持つ（S→印 と 印→S は別の馬券）。

  狙いは前回と同じ。S評価は馬券内87%だが中央1.5倍で単体では儲からない
  （複勝94.1%）。gapの高い馬は配当が大きいが当たりにくい。
  「確実な1頭」を軸に据えて「妙味のある馬」に流す形なら噛み合うかもしれない。

事前登録（ROIを見る前に固定。あとから足さない）
  軸  : そのレースで評価スコアが最大の馬。ただし S評価であること
        （S が居ないレースは見送り）
  相手: 軸以外で gap が高い順に、
          印○相当 gap>=1.3 / 印△相当 gap>=1.1 / 上位3頭 の3通り（最大3頭）
  券種: ワイド / 馬連 / 馬単(軸→相手) / 馬単(相手→軸)
  → 3(相手) × 4(券種) = 12通り。単勝(軸のS)併用あり/なしで24通り。これで全部。

判定
  ① 現行（軸gap>=1.5 単勝＋ダートならワイド・120.6%）を上回るか
  ② 順列検定（24通りから最良を選ぶことを込みで）
  ③ 的中100本以上

実行: python resid_sflow.py
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
GRADE_TH = [("S", 1.276), ("A", 0.791), ("B", 0.437)]
MATE_MAX = 3
MATE_RULES = [("gap", 1.3), ("gap", 1.1), ("top", 3)]
KINDS = ["ワイド", "馬連", "馬単表", "馬単裏"]
N_PERM = 120
MIN_HIT = 100
EPS = 1e-6
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def logit(v):
    v = np.clip(v, EPS, 1 - EPS)
    return np.log(v / (1 - v))


def load():
    g = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    g["gap"] = g.p1 / g.q
    bc = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                    for y in YEARS], ignore_index=True)
    d = g.merge(bc[["race_id", "bn", "c_win", "c_top3"]], on=["race_id", "bn"],
                how="inner")
    with open("grade_calib.pkl", "rb") as fh:
        cal = pickle.load(fh)
    inv = 1.0 / d.odds.clip(lower=1.01)
    q = inv / inv.groupby(d.race_id).transform("sum")
    lm = logit(q)
    X3 = pd.DataFrame({"lm": lm, "l3": logit(d.c_top3)})
    X3["i3"] = X3.lm * X3.l3
    X1 = pd.DataFrame({"lm": lm, "l1": logit(d.c_win)})
    X1["i1"] = X1.lm * X1.l1
    d["gscore"] = (cal["m3"].predict_proba(X3[cal["F3"]])[:, 1]
                   + cal["m1"].predict_proba(X1[cal["F1"]])[:, 1])
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "複勝", "ワイド", "馬連", "馬単"))].itertuples()}
    races = []
    for rid, x in d.groupby("race_id", sort=False):
        races.append({"rid": rid, "年": int(rid[:4]), "bn": x.bn.values,
                      "gap": x.gap.values.astype(float),
                      "gs": x.gscore.values.astype(float)})
    return races, PAY


def build(races, PAY, mate_rule, kind, with_tan, gaps=None):
    """軸＝S評価の最上位。相手＝gapの高い馬。"""
    typ, par = mate_rule
    rows = []
    for i, r in enumerate(races):
        gs = r["gs"]
        k = int(np.argmax(gs))
        if gs[k] < GRADE_TH[0][1]:          # S が居ないレースは見送り
            continue
        a = r["bn"][k]
        g = r["gap"] if gaps is None else gaps[i]
        order = [j for j in np.argsort(-g) if j != k]
        sel = [j for j in order if g[j] >= par][:MATE_MAX] if typ == "gap" \
            else order[:par]
        if not sel:
            continue
        y = r["年"]
        if with_tan:
            rows.append((y, PAY.get((r["rid"], "単勝", a), 0.0)))
        for j in sel:
            b = r["bn"][j]
            if kind == "ワイド":
                c, kk = f"{min(a,b)}-{max(a,b)}", "ワイド"
            elif kind == "馬連":
                c, kk = f"{min(a,b)}-{max(a,b)}", "馬連"
            elif kind == "馬単表":
                c, kk = f"{a}-{b}", "馬単"
            else:
                c, kk = f"{b}-{a}", "馬単"
            rows.append((y, PAY.get((r["rid"], kk, c), 0.0)))
    return rows


def stat(rows, lab):
    if len(rows) < 200:
        return None
    v = np.array([p for _, p in rows])
    ys = np.array([y for y, _ in rows])
    yr = {y: v[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
    return {"買い方": lab, "点数": len(v), "的中": int((v > 0).sum()),
            "的中率": (v > 0).mean() * 100, "ROI": v.mean(), "年別": yr,
            "100超年": sum(1 for x in yr.values() if x >= 100), "_v": v}


def mlab(mr):
    return f"gap>={mr[1]}" if mr[0] == "gap" else f"上位{mr[1]}"


def main():
    races, PAY = load()
    ns = sum(1 for r in races if r["gs"].max() >= GRADE_TH[0][1])
    log(f"検体 {len(races):,}レース")
    log(f"S評価が居るレース {ns:,}（{ns/len(races)*100:.1f}%）")
    log(f"参考: gap>=1.5 の馬が居るレース 7,135（47.7%）\n")

    res = []
    for mr in MATE_RULES:
        for kind in KINDS:
            for wt in (False, True):
                lab = f"{'単勝+' if wt else ''}{kind} 相手={mlab(mr)}"
                r = stat(build(races, PAY, mr, kind, wt), lab)
                if r:
                    res.append(r)
    log(f"  {'買い方':<26}{'点数':>8}{'的中':>7}{'的中率':>7}{'ROI':>8}{'100超年':>8}  年別")
    for r in sorted(res, key=lambda x: -x["ROI"])[:14]:
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['買い方']:<26}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>6.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5  {yr}")

    log("\n=== 上位5つの95%区間 ===")
    log(f"  {'買い方':<26}{'ROI':>8}{'95%区間':>18}")
    for r in sorted(res, key=lambda x: -x["ROI"])[:5]:
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2500)])
        log(f"  {r['買い方']:<26}{r['ROI']:>7.1f}%"
            f"  [{np.percentile(bs,2.5):>6.1f},{np.percentile(bs,97.5):>7.1f}]")

    log("\n=== 比較 ===")
    log("  現行（軸gap>=1.5 単勝＋ダートならワイド）  10,349点 的中1,178 ROI 120.6%")
    log("  前回（軸=★ → 相手=S・単勝+馬連）          7,432点 的中1,059 ROI 116.8%")

    cand = [r for r in res if r["的中"] >= MIN_HIT]
    if not cand:
        log("\n的中100本以上が無いので順列検定は行いません")
        return
    best = max(cand, key=lambda x: x["ROI"])
    log(f"\n=== 順列検定（{len(MATE_RULES)*len(KINDS)*2}通りから最良を選ぶことを込み）===")
    log(f"  本物の最良値 {best['ROI']:.1f}%（{best['買い方']}）")
    nulls = []
    for i in range(N_PERM):
        gaps = [rng.permutation(r["gap"]) for r in races]
        b = -np.inf
        for mr in MATE_RULES:
            for kind in KINDS:
                for wt in (False, True):
                    r = stat(build(races, PAY, mr, kind, wt, gaps=gaps), "x")
                    if r and r["的中"] >= MIN_HIT and r["ROI"] > b:
                        b = r["ROI"]
        if np.isfinite(b):
            nulls.append(b)
        if (i + 1) % 40 == 0:
            log(f"    {i+1}/{N_PERM}  （偽物の中央値 {np.median(nulls):.1f}%）")
    nulls = np.array(nulls)
    p = float((nulls >= best["ROI"]).mean())
    log(f"\n  偽物の中央値 {np.median(nulls):.1f}%  最大 {nulls.max():.1f}%")
    log(f"  p値 = {p:.4f} → {'✅ 偶然では説明しにくい' if p < 0.05 else '⚠ 偶然の範囲'}")


if __name__ == "__main__":
    main()
