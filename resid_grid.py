# -*- coding: utf-8 -*-
"""券種と組み合わせを幅広く試し、多重性を補正して吟味する（2026-08-17）

「幅広く試す」と「探索で騙される」をどう両立するか
  過去8回の失敗は、たくさん試して一番良いものを選び、**その選んだ回数を
  割り引かずに**採用したことが原因だった。2,509構成から選ぶと翌年-70.5pt落ちる。

  幅広く試すこと自体は悪くない。悪いのは補正しないこと。
  そこでこの検証では、**同じ数だけ試したときに偶然どこまで届くか**を
  順列検定で直接測る。本物の最良値がその分布を超えていれば、
  「幅広く試したうえでも本物」と言える。

試す範囲（この表で全部。あとから足さない）
  軸のgap    : 1.5 / 2.0 / 3.0
  軸のオッズ  : 制限なし / 10倍未満
  券種       : 単勝 / 複勝 / ワイド / 馬連 / 馬単(軸→相手) / 馬単(相手→軸)
  相手の選び方 : 自分のgap>=1.0 / >=1.3 / >=1.6 / 上位2頭 / 上位3頭
               （相手は最大5頭。単勝・複勝では使わない）
  → 単系 2券種 × 3 × 2 = 12通り
     複系 4券種 × 3 × 2 × 5 = 120通り
     合計 132通り

判定
  ① 的中100本以上（それ未満は「測れていない」）
  ② 132通りから最良を選ぶことを込みの順列検定で p<0.05
  ③ 単勝のみ（157.0%）を上回るか

⚠ 払戻は jv_payouts（JRA-VANの確定払戻）。推定オッズは使わない。
⚠ 馬番は2桁ゼロ埋め。

実行: python resid_grid.py
"""
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAPS = [1.5, 2.0, 3.0]
AX_ODDS = [(1.0, 9999.0), (1.0, 10.0)]
SINGLES = ["単勝", "複勝"]
EXOTICS = ["ワイド", "馬連", "馬単表", "馬単裏"]
MATE_RULES = [("gap", 1.0), ("gap", 1.3), ("gap", 1.6), ("top", 2), ("top", 3)]
MATE_MAX = 5
N_PERM = 120
MIN_HIT = 100
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {}
    for r in jv[jv.券種.isin(("単勝", "複勝", "ワイド", "馬連", "馬単"))].itertuples():
        PAY[(r.race_id, r.券種, r.組み合わせ)] = r.払戻金
    # レースごとに配列化しておく（毎回groupbyすると遅い）
    races = []
    for rid, g in d.groupby("race_id", sort=False):
        races.append({"rid": rid, "年": int(rid[:4]),
                      "bn": g.bn.values, "gap": g.gap.values.astype(float),
                      "odds": g.odds.values.astype(float)})
    return races, PAY


def evaluate(races, PAY, ax_gap, ax_odds, kind, mate_rule, gaps=None):
    """1つの構成を評価して (年配列, 払戻配列) を返す。gaps を渡すと順列用。"""
    ys, vs = [], []
    lo, hi = ax_odds
    for i, r in enumerate(races):
        g = r["gap"] if gaps is None else gaps[i]
        o = r["odds"]
        bn = r["bn"]
        order = np.argsort(-g)
        k = order[0]
        if g[k] < ax_gap or not (lo <= o[k] < hi):
            continue
        a = bn[k]
        rid, y = r["rid"], r["年"]
        if kind in ("単勝", "複勝"):
            ys.append(y)
            vs.append(PAY.get((rid, kind, a), 0.0))
            continue
        rest = order[1:]
        typ, par = mate_rule
        if typ == "gap":
            sel = rest[g[rest] >= par]
        else:
            sel = rest[:par]
        if len(sel) == 0:
            continue
        for j in sel[:MATE_MAX]:
            b = bn[j]
            if kind == "ワイド":
                c, kk = f"{min(a,b)}-{max(a,b)}", "ワイド"
            elif kind == "馬連":
                c, kk = f"{min(a,b)}-{max(a,b)}", "馬連"
            elif kind == "馬単表":
                c, kk = f"{a}-{b}", "馬単"
            else:
                c, kk = f"{b}-{a}", "馬単"
            ys.append(y)
            vs.append(PAY.get((rid, kk, c), 0.0))
    return np.array(ys), np.array(vs)


def build_grid():
    grid = []
    for ag, ao in product(AX_GAPS, AX_ODDS):
        for k in SINGLES:
            grid.append((ag, ao, k, None))
        for k, mr in product(EXOTICS, MATE_RULES):
            grid.append((ag, ao, k, mr))
    return grid


def label(c):
    ag, ao, k, mr = c
    od = "" if ao[1] > 100 else " 軸<10倍"
    m = "" if mr is None else (f" 相手gap>={mr[1]}" if mr[0] == "gap"
                               else f" 相手上位{mr[1]}")
    return f"軸gap>={ag}{od} {k}{m}"


def main():
    races, PAY = load()
    grid = build_grid()
    log(f"検体 {len(races):,}レース  試す構成 {len(grid)}通り")
    log("多重性は順列検定で補正する（同じ132通りを偽物でも試す）\n")

    res = []
    for c in grid:
        ys, vs = evaluate(races, PAY, *c)
        if len(vs) < 200:
            continue
        hit = int((vs > 0).sum())
        yr = {y: vs[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
        res.append({"構成": label(c), "点数": len(vs), "的中": hit,
                    "的中率": hit / len(vs) * 100, "ROI": vs.mean(),
                    "100超年": sum(1 for x in yr.values() if x >= 100),
                    "年別": yr, "_v": vs, "_c": c})
    R = sorted(res, key=lambda x: -x["ROI"])
    log(f"評価できた構成 {len(R)}通り\n")

    log("=== 的中100本以上のもの・上位15 ===")
    log(f"  {'構成':<34}{'点数':>8}{'的中':>7}{'的中率':>7}{'ROI':>8}{'100超年':>8}{'95%区間':>18}")
    ok = [r for r in R if r["的中"] >= MIN_HIT]
    for r in ok[:15]:
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(1500)])
        log(f"  {r['構成']:<34}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>6.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5"
            f"  [{np.percentile(bs,2.5):>6.1f},{np.percentile(bs,97.5):>7.1f}]")

    log(f"\n=== 券種ごとの最良（的中{MIN_HIT}本以上）===")
    log(f"  {'券種':<8}{'最良の構成':<32}{'点数':>8}{'的中':>7}{'ROI':>8}")
    for k in SINGLES + EXOTICS:
        cand = [r for r in ok if f" {k}" in r["構成"]]
        if cand:
            b = cand[0]
            log(f"  {k:<8}{b['構成']:<32}{b['点数']:>8,}{b['的中']:>7}{b['ROI']:>7.1f}%")

    pd.DataFrame([{kk: vv for kk, vv in r.items() if not kk.startswith("_")}
                  for r in R]).to_csv("resid_grid_result.csv", index=False,
                                      encoding="utf-8-sig")
    log("\n→ resid_grid_result.csv")

    if not ok:
        log("的中100本以上の構成が無いので順列検定は行いません")
        return
    real = ok[0]["ROI"]
    log(f"\n=== 順列検定（{len(grid)}通りから最良を選ぶことを込み）===")
    log(f"  本物の最良値 {real:.1f}%（{ok[0]['構成']}）")
    log(f"  偽物を{N_PERM}回作ります（1回あたり{len(grid)}通りを全部試す）…")
    nulls = []
    for i in range(N_PERM):
        gaps = [rng.permutation(r["gap"]) for r in races]
        best = -np.inf
        for c in grid:
            ys, vs = evaluate(races, PAY, *c, gaps=gaps)
            if len(vs) >= 200 and (vs > 0).sum() >= MIN_HIT and vs.mean() > best:
                best = vs.mean()
        if np.isfinite(best):
            nulls.append(best)
        if (i + 1) % 20 == 0:
            log(f"    {i+1}/{N_PERM}  （偽物の最良値の中央値 {np.median(nulls):.1f}%）")
    nulls = np.array(nulls)
    p = float((nulls >= real).mean())
    log(f"\n  偽物の中央値 {np.median(nulls):.1f}%  95%点 {np.percentile(nulls,95):.1f}%"
        f"  最大 {nulls.max():.1f}%")
    log(f"  p値 = {p:.4f}")
    log(f"  → {'✅ 132通り試したうえでも偶然では説明しにくい' if p < 0.05 else '⚠ 偶然の範囲。採用しない'}")
    log(f"\n  参考: 単勝のみ(軸gap>=2.0) 1,891点 的中203 ROI 157.0%")


if __name__ == "__main__":
    main()
