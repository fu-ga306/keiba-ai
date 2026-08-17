# -*- coding: utf-8 -*-
"""券種ごとの期待値を出し、レースごとに最良の1点を選んで回収率を測る（2026-08-17）

事前登録（resid_kinds.py に書いたもの。ここでは変えない）
  買うのは「期待値 >= 2.0 の券種のうち、期待値が最大のもの1点」。
  しきい値は券種共通。券種ごとに変えると探索になる。
  券種は 単勝 / 複勝 / 馬連 / ワイド / 馬単 の5つ。三連系は入れない。

当たる確率の組み立て（Harville）
  1着を決めたら、残りの馬で2着を決める、という順に考える。
    P(A→B) = p1(A) × p1(B)/(1-p1(A))
    馬連    = P(A→B) + P(B→A)
    ワイド  = P(2頭とも3着以内)  ← p3 から近似
  複勝は p3 をそのまま使う。

オッズの推定
  馬連などの実オッズは事前に持っていないので、市場確率から推定する。
    推定オッズ = (1 - 控除率) / 市場の当たる確率
  控除率は公表値（単勝複勝20% / 馬連ワイド22.5% / 馬単25%）。
  ⚠ これは近似。実オッズとズレるので、最終判定は実払戻（jv_payouts）で行う。
    ここでの推定オッズは「どの券種を買うか選ぶ」ためだけに使う。

判定
  単勝のみの現行（5年通算157.1%・的中203）を上回らなければ広げない。

実行: python resid_kinds_eval.py
"""
import warnings
from itertools import permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
EV_MIN = 2.0
TAKE = {"単勝": 0.20, "複勝": 0.20, "馬連": 0.225, "ワイド": 0.225, "馬単": 0.25}
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def load_pay():
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    P = {}
    for r in jv.itertuples():
        P.setdefault((r.race_id, r.券種), {})[r.組み合わせ] = r.払戻金
    return P


def race_bets(g, PAY):
    """1レースぶんの候補（券種・組み合わせ・当たる確率・推定オッズ・払戻）を返す。"""
    bn = g.bn.values
    p1 = g.p1.values / max(g.p1.sum(), 1e-9)
    p3 = np.clip(g.p3.values, 1e-6, 0.999)
    q1 = g.q.values / max(g.q.sum(), 1e-9)
    q3 = np.clip(g.q3.values, 1e-6, 0.999)
    odds = g.odds.values
    rid = g.race_id.iloc[0]
    out = []

    def add(kind, combo, p, q):
        if q <= 0:
            return
        est = (1 - TAKE[kind]) / q                       # 推定オッズ
        pay = PAY.get((rid, kind), {}).get(combo, 0.0)
        out.append((kind, combo, p, est, p * est, pay))

    for i in range(len(bn)):
        add("単勝", bn[i], p1[i], q1[i])
        add("複勝", bn[i], p3[i], q3[i])
    n = len(bn)
    if n >= 2:
        # 2頭の組み合わせ。Harville で 1-2着の確率を作る
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pij = p1[i] * p1[j] / max(1 - p1[i], 1e-9)
                qij = q1[i] * q1[j] / max(1 - q1[i], 1e-9)
                a, b = bn[i], bn[j]
                add("馬単", f"{a}-{b}", pij, qij)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                a, b = sorted((bn[i], bn[j]))
                c = f"{a}-{b}"
                if c in seen:
                    continue
                seen.add(c)
                pu = (p1[i] * p1[j] / max(1 - p1[i], 1e-9)
                      + p1[j] * p1[i] / max(1 - p1[j], 1e-9))
                qu = (q1[i] * q1[j] / max(1 - q1[i], 1e-9)
                      + q1[j] * q1[i] / max(1 - q1[j], 1e-9))
                add("馬連", c, pu, qu)
                # ワイド: 2頭とも3着以内（独立近似）
                add("ワイド", c, p3[i] * p3[j] / 1.5, q3[i] * q3[j] / 1.5)
    return out


def main():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    PAY = load_pay()
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")
    log(f"事前登録: 期待値 >= {EV_MIN} の券種のうち最大のもの1点\n")

    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        cands = race_bets(g, PAY)
        if not cands:
            continue
        best = max(cands, key=lambda x: x[4])
        if best[4] < EV_MIN:
            continue
        rows.append({"race_id": rid, "年": int(rid[:4]), "券種": best[0],
                     "組み合わせ": best[1], "確率": best[2], "推定オッズ": best[3],
                     "期待値": best[4], "払戻": best[5]})
    R = pd.DataFrame(rows)
    if R.empty:
        log("買い目が出ませんでした")
        return
    R.to_csv("resid_kinds_result.csv", index=False, encoding="utf-8-sig")

    def stat(df, lab):
        if len(df) < 30:
            return
        roi = df.払戻.sum() / (len(df) * 100) * 100
        h = int((df.払戻 > 0).sum())
        v = df.払戻.values
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
        yr = "  ".join(f"{y}:{g.払戻.sum()/(len(g)*100)*100:.0f}%"
                       for y, g in df.groupby("年") if len(g) > 20)
        log(f"  {lab:<16}{len(df):>7,}{h:>7}{h/len(df)*100:>7.1f}%{roi:>8.1f}%"
            f"  [{np.percentile(bs,2.5):>5.1f},{np.percentile(bs,97.5):>6.1f}]  {yr}")

    log("=== レースごとに最良の券種を1点 ===")
    log(f"  {'区分':<16}{'点数':>7}{'的中':>7}{'的中率':>8}{'ROI':>8}{'95%区間':>17}  年別")
    stat(R, "全体")
    log("")
    for k in ("単勝", "複勝", "馬連", "ワイド", "馬単"):
        stat(R[R.券種 == k], f"うち{k}")

    log("\n=== 選ばれた券種の内訳 ===")
    for k, n in R.券種.value_counts().items():
        log(f"  {k:<8}{n:>6,}点 ({n/len(R)*100:>4.1f}%)")

    log("\n=== 比較：単勝のみ（現行の事前登録）===")
    log("  1,891点  的中203  10.7%  157.1%  [107.7, 219.5]"
        "  2021:178% 2022:92% 2023:227% 2024:266% 2025:86%")


if __name__ == "__main__":
    main()
