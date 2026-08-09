# -*- coding: utf-8 -*-
"""まだ振っていない設定値を3年で検証する。

これまでに試したもの
  ・EVのしきい値（1.5〜2.1 × 2.0〜3.0）→ 現行1.7/2.2が最良
  ・券種7種 × 相手6通り → 単勝・馬単以外は全滅
  ・評価グレードによる足切り → 再現しない
  ・クラス別・新馬戦 → 検体不足か年による崩壊

まだ試していないもの（今回）
  ① オッズ上限   EV_ODDS_MAX = 20倍 を 12/15/20/25/30 で
  ② 乖離の下限   EV_GAP_MIN = 3 を 2/3/4/5 で
  ③ 馬単の相手人気上限 UMATAN_MAX_POP = 3 を 2/3/4/5 で
  ④ 軸の本数     EV_MAX_PICKS = 1 を 1/2 で
  ⑤ 賭け金の配分 均等 / ケリー比例（bet_cacheのkelly列を使う）
  ⑥ 上級クラス(OP以上)の除外

判定の基準（甘い基準で拾わないため）
  3年すべて100%以上 かつ ブートストラップ下限が現行(95.8)を超える かつ 的中30本以上

⚠ 多数の条件を試すので、良く見えるものが偶然出る。現行を上回ったものは
   「半年後に2026年で再現するか確かめる候補」であって、今すぐ変えるものではない。

実行: python tune_knobs.py → tune_knobs_result.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2023, 2024, 2025]
rng = np.random.default_rng(0)

# 現行の設定（keiba_predict.py と一致）
CUR = dict(a=1.7, b=2.2, gap=3.0, omax=20.0, pop=3, picks=1, kelly=False, drop_hi=False)


def load():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}
    cls = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                      usecols=["race_id", "クラス_num"], low_memory=False,
                      dtype={"race_id": str}).drop_duplicates("race_id")
    E = {}
    for y in YEARS:
        d = pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                        dtype={"race_id": str, "bn": str})
        E[y] = d.merge(cls, on="race_id", how="left")
    return E, pay


def run(E, pay, **kw):
    p = dict(CUR, **kw)
    per, allp = {}, []
    for y in YEARS:
        pts = []
        d = E[y]
        if p["drop_hi"]:
            d = d[d["クラス_num"] <= 4]
        for rid, g in d.groupby("race_id", sort=False):
            c = g[(g["乖離"] >= p["gap"]) & (g["odds"] <= p["omax"]) &
                  (((g["mr"] == 1) & (g["EV_tan"] >= p["a"])) |
                   (g["mr"].between(2, 5) & (g["EV_tan"] >= p["b"])))]
            if not len(c):
                continue
            axes = c.sort_values("EV_tan", ascending=False).head(p["picks"])
            cost = ret = 0.0
            pr = g["人気"].rank(method="first")
            for _, ax in axes.iterrows():
                # ケリーを使う場合は賭け金を比例させる（上限は現行の倍まで）
                unit = 1000.0
                if p["kelly"]:
                    k = float(ax.get("kelly", 0) or 0)
                    unit = float(np.clip(k * 20000, 500, 2000))
                cost += unit
                ret += pay.get((rid, "単勝", ax.bn), 0.0) * unit / 100
                for m in g[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= p["pop"])].bn:
                    if m == ax.bn:
                        continue
                    cost += unit / 2
                    ret += pay.get((rid, "馬単", f"{ax.bn}-{m}"), 0.0) * (unit / 2) / 100
            if cost > 0:
                pts.append((cost, ret))
        if len(pts) < 30:
            return None
        a = np.array(pts, float)
        per[y] = a[:, 1].sum() / a[:, 0].sum() * 100
        allp += pts
    a = np.array(allp, float)
    idx = rng.integers(0, len(a), (1500, len(a)))
    bs = a[:, 1][idx].sum(1) / a[:, 0][idx].sum(1) * 100
    return {**{f"ROI{y}": round(per[y], 1) for y in YEARS},
            "通算": round(a[:, 1].sum() / a[:, 0].sum() * 100, 1),
            "CI下": round(np.percentile(bs, 2.5), 1),
            "最低年": round(min(per.values()), 1),
            "的中": int((a[:, 1] > 0).sum()), "R数": len(a)}


def main():
    E, pay = load()
    rows = []

    def add(label, **kw):
        r = run(E, pay, **kw)
        if r:
            rows.append({"設定": label, **r})
            mark = " ←現行" if not kw else ""
            print(f"  {label:24s} 通算{r['通算']:6.1f}%  CI下{r['CI下']:6.1f}  "
                  f"最低年{r['最低年']:6.1f}%  的中{r['的中']:>3d}  R{r['R数']:>4d}{mark}",
                  flush=True)

    print("【基準】現行", flush=True)
    add("現行(1.7/2.2/20倍/乖離3/人気3)")

    print("\n① オッズ上限", flush=True)
    for o in (12, 15, 25, 30, 999):
        add(f"オッズ上限{o}倍", omax=o)

    print("\n② 乖離の下限", flush=True)
    for gp in (2, 4, 5):
        add(f"乖離>={gp}", gap=gp)

    print("\n③ 馬単の相手人気上限", flush=True)
    for pp in (2, 4, 5):
        add(f"相手人気<={pp}", pop=pp)

    print("\n④ 軸の本数", flush=True)
    add("軸2頭", picks=2)

    print("\n⑤ 賭け金をケリー比例に", flush=True)
    add("ケリー配分", kelly=True)

    print("\n⑥ 上級クラス(OP以上)を除外", flush=True)
    add("OP以上を除外", drop_hi=True)

    print("\n⑦ 有望そうな組み合わせ", flush=True)
    add("乖離4 × オッズ15倍", gap=4, omax=15)
    add("乖離4 × 相手人気2", gap=4, pop=2)
    add("オッズ15倍 × OP除外", omax=15, drop_hi=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "tune_knobs_result.csv"),
              index=False, encoding="utf-8-sig")
    base = df[df.設定.str.startswith("現行")].iloc[0]
    print("\n" + "=" * 76)
    print(f"現行: 通算{base.通算}%  CI下{base['CI下']}  最低年{base.最低年}%  的中{base.的中}")
    ok = df[(df.最低年 >= 100) & (df["CI下"] > base["CI下"]) & (df.的中 >= 30)]
    print("\n【現行を上回った設定】3年100%超・CI下限が現行超・的中30本以上")
    print(ok.sort_values("CI下", ascending=False).to_string(index=False)
          if len(ok) else "  なし")
    print("\n保存 → tune_knobs_result.csv")


if __name__ == "__main__":
    sys.exit(main())
