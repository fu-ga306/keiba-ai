# -*- coding: utf-8 -*-
"""評価グレード(S〜D)と★(乖離)を指標に、券種を横断して回収率を測る。

背景
  これまで券種拡大は全滅していた（3年通算 馬連88.0/ワイド79.7/3連複67.6%）。
  ただしそれは「軸の選び方」を変えずに券種だけ広げた検証だった。
  評価グレードと★という2つの指標が使えるようになったので、
  軸と相手の選び方を変えたうえで、もう一度全券種を測り直す。

データ
  OOS  : model_mf_result.csv（2025年・3,144レース・本番モデルの正直な出力）
  払戻 : payout_data.csv（2025年の実払戻。単勝〜3連単）
  ※ 2024年は払戻データが無いため2025年のみ。1年分であることに注意。

軸の候補
  EV      … 本番の買い方（乖離>=3・20倍以下・順位別EV）で選ばれる1頭
  ★◎     … ★が付いたMF複勝1位
  S/A     … 評価がS以上／A以上で最も期待値が高い馬
相手の候補
  人気上位N / MF複勝上位N / 評価上位N

⚠ 単勝は「オッズ×100」ではなく実払戻で計算する（同着があるため）。
⚠ 買い目は全て等額。点数が増えるほど1点あたりの負担が増える点に注意。

実行: python exotic_grade.py  → exotic_grade_result.csv
"""
import os
import sys
import warnings
from itertools import combinations, permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
GRADE_TH = [(1.05, "S"), (0.80, "A"), (0.58, "B"), (0.36, "C")]
STAKE = 100          # 1点あたり。回収率は比なので額そのものは結果に影響しない


def load():
    import pickle
    d = pd.read_csv(os.path.join(BASE_DIR, "model_mf_result.csv"),
                    dtype={"race_id": str})
    cal = pickle.load(open(os.path.join(BASE_DIR, "mf_calibrator.pkl"), "rb"))["cal"]
    cp = lambda v, t: (cal[t].predict(np.clip(np.nan_to_num(v, nan=0.0), 0, None))
                       if t in cal else v)
    d["w"] = cp(d["MF勝率"].values, "win")
    d["r"] = cp(d["MF連対率"].values, "place2")
    d["f"] = cp(d["MF複勝率"].values, "place3")
    d["p"] = d.groupby("race_id")["w"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0)
    d["score"] = d.p + d.r + d.f
    d["grade"] = np.select([d.score >= t for t, _ in GRADE_TH],
                           [g for _, g in GRADE_TH], "D")
    d["ev"] = d.p * pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["pop"] = pd.to_numeric(d["人気"], errors="coerce")
    d["mfr"] = pd.to_numeric(d["MF複勝順位"], errors="coerce")
    d["gap"] = d["pop"] - d["mfr"]
    d["star"] = (d.gap >= GAP_MIN) & (pd.to_numeric(d["単勝オッズ"],
                                                    errors="coerce") <= ODDS_MAX)
    d["ev_ok"] = (d.star &
                  (((d.mfr == 1) & (d.ev >= EV_TOP)) |
                   (d.mfr.between(2, 5) & (d.ev >= EV_SUB))))
    # 馬番: model_mf_result には無いので払戻側と突き合わせるために着順から作れない。
    # bet_cache に馬番があるので結合する。
    bc = pd.read_csv(os.path.join(BASE_DIR, "bet_cache_2025.csv"),
                     dtype={"race_id": str, "bn": str},
                     usecols=["race_id", "馬名", "bn", "着"])
    d = d.merge(bc, on=["race_id", "馬名"], how="left")
    d = d[d.bn.notna()].copy()
    d["bn"] = d.bn.astype(str).str.zfill(2)

    pay = pd.read_csv(os.path.join(BASE_DIR, "payout_data.csv"),
                      dtype={"race_id": str, "組み合わせ": str})
    pay["払戻金"] = pd.to_numeric(pay["払戻金"], errors="coerce")
    tbl = {}
    for k, g in pay.groupby("券種"):
        tbl[k] = dict(zip(zip(g.race_id, g.組み合わせ), g.払戻金))
    return d, tbl


def combo_key(kind, bns):
    """払戻表のキーに合わせる。順序ありの券種はそのまま、無い券種は昇順。"""
    if kind in ("馬単", "3連単"):
        return "-".join(bns)
    return "-".join(sorted(bns))


def evaluate(d, tbl, axis_fn, mate_fn, kinds, label):
    """軸と相手の決め方を渡して、券種ごとの点数・回収率を返す。"""
    out = {k: [0, 0.0, 0] for k in kinds}     # 点数, 払戻合計, 的中数
    for rid, g in d.groupby("race_id"):
        ax = axis_fn(g)
        if ax is None:
            continue
        mates = [m for m in mate_fn(g, ax) if m != ax]
        if not mates:
            continue
        for kind in kinds:
            bets = []
            if kind == "単勝":
                bets = [(ax,)]
            elif kind == "複勝":
                bets = [(ax,)]
            elif kind in ("馬連", "ワイド"):
                bets = [(ax, m) for m in mates]
            elif kind == "馬単":
                bets = [(ax, m) for m in mates]
            elif kind == "3連複":
                bets = [(ax, a, b) for a, b in combinations(mates, 2)]
            elif kind == "3連単":
                bets = [(ax, a, b) for a, b in permutations(mates, 2)]
            if not bets:
                continue
            for b in bets:
                out[kind][0] += 1
                amt = tbl.get(kind, {}).get((rid, combo_key(kind, list(b))))
                if amt and not np.isnan(amt):
                    out[kind][1] += amt
                    out[kind][2] += 1
    rows = []
    for k in kinds:
        n, ret, hit = out[k]
        if n == 0:
            continue
        rows.append({"軸と相手": label, "券種": k, "点数": n,
                     "的中": hit, "的中率": round(hit / n * 100, 2),
                     "回収率": round(ret / (n * STAKE) * 100, 1)})
    return rows


def main():
    print("読み込み中...", flush=True)
    d, tbl = load()
    print(f"  {len(d)}頭 / {d.race_id.nunique()}レース / 払戻 {list(tbl)}", flush=True)

    def ax_ev(g):
        c = g[g.ev_ok]
        return c.sort_values("ev", ascending=False).bn.iloc[0] if len(c) else None

    def ax_star_mf1(g):
        c = g[g.star & (g.mfr == 1)]
        return c.sort_values("ev", ascending=False).bn.iloc[0] if len(c) else None

    def ax_grade(th):
        def f(g):
            c = g[g.star & (g.score >= th)]
            return c.sort_values("ev", ascending=False).bn.iloc[0] if len(c) else None
        return f

    def mates_pop(n):
        return lambda g, ax: list(g[g["pop"] <= n].sort_values("pop").bn)

    def mates_mf(n):
        return lambda g, ax: list(g[g.mfr <= n].sort_values("mfr").bn)

    def mates_grade(n):
        return lambda g, ax: list(g.sort_values("score", ascending=False).bn.head(n))

    KINDS = ["単勝", "複勝", "馬連", "馬単", "ワイド", "3連複", "3連単"]
    rows = []
    cases = [
        (ax_ev, mates_pop(3), "EV軸 × 人気上位3"),
        (ax_ev, mates_pop(5), "EV軸 × 人気上位5"),
        (ax_ev, mates_mf(3), "EV軸 × MF複勝上位3"),
        (ax_ev, mates_grade(3), "EV軸 × 評価上位3"),
        (ax_star_mf1, mates_pop(3), "★かつMF1位 × 人気上位3"),
        (ax_grade(0.80), mates_pop(3), "★かつA以上 × 人気上位3"),
        (ax_grade(1.05), mates_pop(3), "★かつS以上 × 人気上位3"),
    ]
    for axf, mf, lbl in cases:
        r = evaluate(d, tbl, axf, mf, KINDS, lbl)
        rows += r
        print(f"\n=== {lbl} ===", flush=True)
        for x in r:
            print(f"  {x['券種']:5s} {x['点数']:>6d}点  的中{x['的中率']:>6.2f}%  "
                  f"回収率 {x['回収率']:>7.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "exotic_grade_result.csv"),
              index=False, encoding="utf-8-sig")
    print("\n" + "=" * 66)
    print("回収率100%超のもの:")
    hit = df[df.回収率 >= 100].sort_values("回収率", ascending=False)
    print(hit.to_string(index=False) if len(hit) else "  なし")
    print(f"\n保存 → exotic_grade_result.csv")
    print("※ 2025年1年のみ。点数が少ないものは偶然の可能性が高い。")


if __name__ == "__main__":
    sys.exit(main())
