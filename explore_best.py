# -*- coding: utf-8 -*-
"""本番仕様を土台に、★・評価グレード・券種・相手の広げ方から最良の条件を探す。

入力（過去の検証と同じものを使う。ここを間違えると全ての数字が狂う）
  bet_cache_2023/2024/2025.csv … EV_tan・mr・乖離・較正済み確率が計算済み
  jv_payouts.csv               … JVの実払戻（単勝〜3連単）

現状の本番仕様（keiba_predict.py から）
  印   : MF複勝順位の上位から ◎○▲△、×は印なしの複勝妙味最大
  ★   : 乖離(人気順位 − MF複勝順位) >= 3 かつ 単勝オッズ <= 20倍
  軸   : ★のうち MF複勝1位ならEV>=1.7 / 2〜5位ならEV>=2.2 を満たす中で期待値最大の1頭
  単勝 : 軸1点 1,000円
  馬単 : 軸 → (MF複勝順位1〜5 かつ 人気3位以内) 各500円
  ※ 3年通算113.6%（単勝のみ・2023 114.6 / 2024 114.8 / 2025 111.5）

評価グレード（合成スコア = 較正済み 勝率 + 連対率 + 複勝率）
  S>=1.05 A>=0.80 B>=0.58 C>=0.36。3年で較正を確認済み。

判定の基準（甘い基準で拾わないため）
  ・3年すべてで100%以上
  ・3年通算のブートストラップ下限が100%以上
  ・的中が3年で30本以上（少数の高配当に依存しない）

⚠ 多数の条件を試すので、良く見えるものが偶然出る。上の3条件を全て満たしたものだけを
   候補とし、それでも「1年分の独立検証が必要」と考えること。

実行: python explore_best.py → explore_best_result.csv
"""
import os
import sys
import warnings
from itertools import combinations, permutations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2023, 2024, 2025]
UNORD = {"馬連", "ワイド", "3連複"}
GRADE_TH = [(1.05, "S"), (0.80, "A"), (0.58, "B"), (0.36, "C")]
rng = np.random.default_rng(0)


def load():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}
    E = {}
    for y in YEARS:
        d = pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                        dtype={"race_id": str, "bn": str})
        d["score"] = d.c_win_n + d.c_top2 + d.c_top3
        d["grade"] = np.select([d.score >= t for t, _ in GRADE_TH],
                               [g for _, g in GRADE_TH], "D")
        E[y] = d
    return E, pay


def axis(g, a=1.7, b=2.2, gmin=3, omax=20, smin=None):
    """本番の軸選び。smin を渡すと評価スコアの下限も課す。"""
    c = g[(g["乖離"] >= gmin) & (g["odds"] <= omax) &
          (((g["mr"] == 1) & (g["EV_tan"] >= a)) |
           (g["mr"].between(2, 5) & (g["EV_tan"] >= b)))]
    if smin is not None:
        c = c[c["score"] >= smin]
    if not len(c):
        return None
    return c.sort_values("EV_tan", ascending=False).bn.iloc[0]


def mates(g, ax, how):
    """相手の広げ方。"""
    if how == "印×人気3":            # 現行の馬単
        pr = g["人気"].rank(method="first")
        s = g[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= 3)]
    elif how == "印":                 # MF複勝1〜5位
        s = g[g.mr.isin([1, 2, 3, 4, 5])]
    elif how.startswith("人気"):
        s = g[g["人気"] <= int(how[2])].sort_values("人気")
    elif how.startswith("評価"):
        s = g.sort_values("score", ascending=False).head(int(how[2]))
    elif how.startswith("MF"):
        s = g[g.mr <= int(how[2])].sort_values("mr")
    else:
        s = g.iloc[0:0]
    return [x for x in s.bn.tolist() if x != ax]


def bets_of(kind, ax, ms):
    if kind in ("単勝", "複勝"):
        return [(ax,)]
    if kind in ("馬連", "ワイド", "馬単"):
        return [(ax, m) for m in ms]
    if kind == "3連複":
        return [(ax, x, y) for x, y in combinations(ms, 2)]
    if kind == "3連単":
        return [(ax, x, y) for x, y in permutations(ms, 2)]
    return []


def run(E, pay, kind, how, axis_kw, label):
    per_year, allpts = {}, []
    for y in YEARS:
        pts = []
        for rid, g in E[y].groupby("race_id", sort=False):
            ax = axis(g, **axis_kw)
            if ax is None:
                continue
            ms = mates(g, ax, how)
            bs = bets_of(kind, ax, ms)
            if not bs:
                continue
            keys = set()
            for b in bs:
                if len(set(b)) != len(b):
                    continue
                keys.add("-".join(sorted(b) if kind in UNORD else list(b)))
            if not keys:
                continue
            cost = len(keys) * 100
            ret = sum(pay.get((rid, kind, k), 0.0) for k in keys)
            pts.append((cost, ret))
        if len(pts) < 30:
            return None
        a = np.array(pts, float)
        per_year[y] = a[:, 1].sum() / a[:, 0].sum() * 100
        allpts += pts
    a = np.array(allpts, float)
    roi = a[:, 1].sum() / a[:, 0].sum() * 100
    idx = rng.integers(0, len(a), (1500, len(a)))
    bs = a[:, 1][idx].sum(1) / a[:, 0][idx].sum(1) * 100
    return {"構成": label, "券種": kind, "相手": how,
            **{f"ROI{y}": round(per_year[y], 1) for y in YEARS},
            "通算": round(roi, 1), "CI下": round(np.percentile(bs, 2.5), 1),
            "最低年": round(min(per_year.values()), 1),
            "的中": int((a[:, 1] > 0).sum()), "R数": len(a),
            "平均点数": round(a[:, 0].sum() / 100 / len(a), 1)}


def main():
    print("読み込み中...", flush=True)
    E, pay = load()
    for y in YEARS:
        print(f"  {y}: {E[y].race_id.nunique():,}レース", flush=True)

    rows = []
    KINDS = ["単勝", "複勝", "馬連", "馬単", "ワイド", "3連複", "3連単"]
    HOWS = ["印×人気3", "印", "人気3", "人気5", "評価3", "MF3"]

    print("\n① 現行の軸（EV 1.7/2.2）で券種と相手を振る", flush=True)
    for kind in KINDS:
        hows = ["印×人気3"] if kind in ("単勝", "複勝") else HOWS
        for how in hows:
            r = run(E, pay, kind, how, dict(), f"現行軸 {kind} × {how}")
            if r:
                rows.append(r)
                print(f"  {r['構成']:26s} {r['通算']:6.1f}%  CI下{r['CI下']:6.1f}"
                      f"  最低年{r['最低年']:6.1f}%  的中{r['的中']:>4d}", flush=True)

    print("\n② 軸に評価スコアの下限を課す（★＋EV＋評価）", flush=True)
    for smin, lbl in [(0.36, "C以上"), (0.58, "B以上"), (0.80, "A以上")]:
        for kind in ["単勝", "馬単", "馬連"]:
            how = "印×人気3"
            r = run(E, pay, kind, how, dict(smin=smin), f"軸{lbl} {kind} × {how}")
            if r:
                rows.append(r)
                print(f"  {r['構成']:26s} {r['通算']:6.1f}%  CI下{r['CI下']:6.1f}"
                      f"  最低年{r['最低年']:6.1f}%  的中{r['的中']:>4d}", flush=True)

    print("\n③ EVのしきい値を振る（単勝・馬単）", flush=True)
    for a in (1.5, 1.7, 1.9):
        for b in (2.0, 2.2, 2.6):
            for kind in ["単勝", "馬単"]:
                r = run(E, pay, kind, "印×人気3", dict(a=a, b=b),
                        f"EV{a}/{b} {kind}")
                if r:
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "explore_best_result.csv"),
              index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("【基準を満たしたもの】3年すべて100%以上 かつ CI下限100%以上 かつ 的中30本以上")
    ok = df[(df.最低年 >= 100) & (df.CI下 >= 100) & (df.的中 >= 30)]
    print(ok.sort_values("通算", ascending=False).to_string(index=False)
          if len(ok) else "  なし")
    print("\n【参考】3年すべて100%以上（CI下限は問わない）")
    ok2 = df[(df.最低年 >= 100) & (df.的中 >= 30)]
    print(ok2.sort_values("通算", ascending=False).head(12).to_string(index=False)
          if len(ok2) else "  なし")
    print("\n保存 → explore_best_result.csv")


if __name__ == "__main__":
    sys.exit(main())
