# -*- coding: utf-8 -*-
"""本番の買い目生成が、検証したロジックと一致するかを機械的に確かめる（2026-08-16）

なぜ必要か
  2026-08-16に二重の誤りを出した。
    ① verify34.py の集計が誤っており、A構成を149.2%と報告した（正しくは101.8%）
    ② 構成名の「1-20倍」は軸のオッズ条件なのに、実装で落としていた
  どちらも「検証で見た数字」と「本番が実際に買うもの」がズレていた事故。

  そこで、過去データを本番の関数（keiba_predict._race_bet_plan と
  _build_bet_rows）に通し、そこから出た買い目で回収率を計算する。
  検証スクリプトの数字と一致すれば、実装は正しい。

  ⚠ 特徴量やモデルではなく「買い目の作り方」だけを確認する。
    bet_cache の予測値をそのまま pdf に見立てて渡す。

実行: python check_impl.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import keiba_predict as K

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]

# harness.py で測った値。本番がこれと一致しなければ実装がズレている。
# 買い方を変えたら、この表もいっしょに更新すること。
EXPECT = {"name": "荒れR勝率1位x2 馬単裏 1-10倍",
          "点数": 1525, "的中": 52, "5年": 110.0, "直近2年": 103.0}


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "クラス_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["MF勝率順位"] = g["c_win"].rank(ascending=False, method="first")
    D["MF連対順位"] = g["c_top2"].rank(ascending=False, method="first")
    D["MF複勝順位"] = g["c_top3"].rank(ascending=False, method="first")
    D["単勝オッズ"] = D.odds
    D["人気"] = D.pr
    D["馬番"] = pd.to_numeric(D.bn, errors="coerce")
    D["馬名"] = "馬" + D.bn
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {}
    for r in jv[jv.券種.isin(("馬単", "馬連"))].itertuples():
        PAY[(r.race_id, r.券種, r.組み合わせ)] = r.払戻金
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    log(f"採用構成 {len(K.ARE_PLANS)}件: {[p[0] for p in K.ARE_PLANS]}\n")

    acc = {y: [0.0, 0.0, 0] for y in YEARS}
    nrace = 0
    for rid, pdf in D.groupby("race_id", sort=False):
        pdf = pdf.copy()
        plan = K._race_bet_plan(pdf)
        if plan["判定"] != "買い":
            continue
        rows = K._build_bet_rows(pdf, rid)
        if not rows:
            continue
        nrace += 1
        y = int(rid[:4])
        for r in rows:
            p = PAY.get((rid, r["券種"], r["組み合わせ"]), 0.0)
            acc[y][0] += 100
            acc[y][1] += p
            acc[y][2] += 1 if p > 0 else 0

    tc = sum(acc[y][0] for y in YEARS)
    tr = sum(acc[y][1] for y in YEARS)
    th = sum(acc[y][2] for y in YEARS)
    rc = sum(acc[y][0] for y in RECENT)
    rr = sum(acc[y][1] for y in RECENT)
    rh = sum(acc[y][2] for y in RECENT)
    log("=== 本番の関数が実際に作った買い目での成績 ===")
    log(f"  買うレース {nrace:,}  点数 {int(tc/100):,}（1レース{tc/100/max(nrace,1):.1f}点）")
    log(f"  5年    的中{th:>4}  回収率 {tr/tc*100:6.1f}%")
    log(f"  直近2年 的中{rh:>4}  回収率 {rr/rc*100:6.1f}%")
    log("  年別: " + "  ".join(
        f"{y}:{acc[y][1]/acc[y][0]*100:.1f}%" if acc[y][0] else f"{y}:--" for y in YEARS))
    log("\n=== 検証(harness.py / gradient.py)で測った値 ===")
    log(f"  {EXPECT['name']}: 点数{EXPECT['点数']:,} 的中{EXPECT['的中']}"
        f"  5年{EXPECT['5年']}%  直近2年{EXPECT['直近2年']}%")
    ok = (abs(tr / tc * 100 - EXPECT["5年"]) < 1.0
          and abs(int(tc / 100) - EXPECT["点数"]) <= 5
          and th == EXPECT["的中"])
    log(f"\n  {'✅ 一致。実装は検証どおり' if ok else '⚠ 一致しない。実装を見直すこと'}")


if __name__ == "__main__":
    main()
