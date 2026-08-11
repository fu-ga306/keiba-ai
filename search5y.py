# -*- coding: utf-8 -*-
"""5年（2021-2025）で買い方を探索し直す。3年での探索より厳しい基準を課す。

なぜやり直すか（2026-08-11）
  8/10に3年（2023-2025）で100通り以上を探索し「現行が最良」と結論した。
  その後5年に伸ばして現行ルールを測ると:
      2021 90.5 / 2022 114.5 / 2023 122.2 / 2024 116.6 / 2025 134.1
      5年通算 117.0%  95%区間[91.2, 144.6]  100%超の確率 89.4%
  3年（124.7%・89.4%→92.2%）より弱くなった。2021年が足を引っ張っている。
  **データを増やしても確証できなかった**ので、条件そのものを5年で選び直す。

3年探索との違い
  ・検証年が5年 → 偶然が残りにくい
  ・「5年すべて100%超」を必須にする（3年のときは3年すべて）
  ・的中50本以上を必須（3年のときは30本）

⚠ 多数を試すので、基準を満たしても「2026年で再現するか」は別問題。
   採用は再現を確認してから。

実行: python search5y.py → search5y_result.csv
"""
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2021, 2022, 2023, 2024, 2025]
rng = np.random.default_rng(12345)


def load():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}
    cls = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                      usecols=["race_id", "クラス_num", "is_turf", "距離", "出走頭数"],
                      low_memory=False, dtype={"race_id": str}).drop_duplicates("race_id")
    E = {}
    for y in YEARS:
        d = pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                        dtype={"race_id": str, "bn": str})
        d["score"] = d.c_win_n + d.c_top2 + d.c_top3
        E[y] = d.merge(cls, on="race_id", how="left")
    return E, pay


def run(E, pay, *, a=1.7, b=2.2, gap=3.0, omax=20.0, omin=0.0, pop=3,
        smin=None, cls_max=None, turf=None, kinds=("単勝", "馬単")):
    """1つの構成を5年で評価する。単勝1,000円・馬単500円の金額配分。"""
    per, allp = {}, []
    for y in YEARS:
        pts = []
        d = E[y]
        if cls_max is not None:
            d = d[d["クラス_num"] <= cls_max]
        if turf is not None:
            d = d[d["is_turf"] == turf]
        for rid, g in d.groupby("race_id", sort=False):
            c = g[(g["乖離"] >= gap) & (g["odds"] <= omax) & (g["odds"] >= omin) &
                  (((g["mr"] == 1) & (g["EV_tan"] >= a)) |
                   (g["mr"].between(2, 5) & (g["EV_tan"] >= b)))]
            if smin is not None:
                c = c[c["score"] >= smin]
            if not len(c):
                continue
            ax = c.sort_values("EV_tan", ascending=False).bn.iloc[0]
            cost = ret = 0.0
            if "単勝" in kinds:
                cost += 1000
                ret += pay.get((rid, "単勝", ax), 0.0) * 10
            if "馬単" in kinds:
                pr = g["人気"].rank(method="first")
                for m in g[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= pop)].bn:
                    if m == ax:
                        continue
                    cost += 500
                    ret += pay.get((rid, "馬単", f"{ax}-{m}"), 0.0) * 5
            if cost > 0:
                pts.append((cost, ret))
        if len(pts) < 20:
            return None
        arr = np.array(pts, float)
        per[y] = arr[:, 1].sum() / arr[:, 0].sum() * 100
        allp += pts
    arr = np.array(allp, float)
    idx = rng.integers(0, len(arr), (1500, len(arr)))
    bs = arr[:, 1][idx].sum(1) / arr[:, 0][idx].sum(1) * 100
    return {**{f"y{y}": round(per[y], 1) for y in YEARS},
            "通算": round(arr[:, 1].sum() / arr[:, 0].sum() * 100, 1),
            "CI下": round(np.percentile(bs, 2.5), 1),
            "最低年": round(min(per.values()), 1),
            "P100": round(float(np.mean(bs > 100) * 100), 1),
            "的中": int((arr[:, 1] > 0).sum()), "R数": len(arr)}


def main():
    print("読み込み中...", flush=True)
    E, pay = load()
    rows = []

    def add(label, **kw):
        r = run(E, pay, **kw)
        if r:
            rows.append({"構成": label, **r})
            print(f"  {label:30s} 通算{r['通算']:6.1f}% CI下{r['CI下']:6.1f} "
                  f"最低年{r['最低年']:6.1f}% 的中{r['的中']:>3d} R{r['R数']:>5d}",
                  flush=True)
        return r

    print("\n【基準】現行", flush=True)
    base = add("現行(1.7/2.2/20倍/乖離3/人気3)")

    print("\n① EVのしきい値", flush=True)
    for a in (1.5, 1.7, 1.9, 2.1, 2.3):
        for b in (2.0, 2.2, 2.6, 3.0):
            add(f"EV {a}/{b}", a=a, b=b)

    print("\n② オッズの範囲", flush=True)
    for omin, omax in ((0, 12), (0, 15), (0, 25), (0, 30), (5, 20), (7, 20), (5, 15)):
        add(f"オッズ {omin}〜{omax}倍", omin=omin, omax=omax)

    print("\n③ 乖離の下限", flush=True)
    for g in (2, 4, 5, 6):
        add(f"乖離>={g}", gap=g)

    print("\n④ 馬単の相手人気", flush=True)
    for p in (1, 2, 4, 5):
        add(f"相手人気<={p}", pop=p)

    print("\n⑤ クラス・コース", flush=True)
    add("OP以上を除外", cls_max=4)
    add("下級のみ(未勝利1勝)", cls_max=2)
    add("芝のみ", turf=1)
    add("ダートのみ", turf=0)

    print("\n⑥ 券種を分ける", flush=True)
    add("単勝のみ", kinds=("単勝",))
    add("馬単のみ", kinds=("馬単",))

    print("\n⑦ 有望な組み合わせ", flush=True)
    add("相手人気2 × オッズ15", pop=2, omax=15)
    add("相手人気2 × OP除外", pop=2, cls_max=4)
    add("オッズ15 × OP除外", omax=15, cls_max=4)
    add("相手人気2 × オッズ15 × OP除外", pop=2, omax=15, cls_max=4)
    add("EV1.9/2.6 × 相手人気2", a=1.9, b=2.6, pop=2)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "search5y_result.csv"),
              index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("【厳格な基準】5年すべて100%超 かつ CI下限100%超 かつ 的中50本以上")
    ok = df[(df.最低年 >= 100) & (df["CI下"] >= 100) & (df.的中 >= 50)]
    print(ok.sort_values("CI下", ascending=False).to_string(index=False)
          if len(ok) else "  なし")

    print("\n【緩い基準】5年すべて100%超 かつ 的中50本以上")
    ok2 = df[(df.最低年 >= 100) & (df.的中 >= 50)]
    print(ok2.sort_values("CI下", ascending=False).head(10).to_string(index=False)
          if len(ok2) else "  なし")

    print("\n【CI下限の上位10】")
    print(df.nlargest(10, "CI下").to_string(index=False))
    print("\n保存 → search5y_result.csv")


if __name__ == "__main__":
    sys.exit(main())
