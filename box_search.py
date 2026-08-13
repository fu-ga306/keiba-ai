# -*- coding: utf-8 -*-
"""モデルの全頭評価からボックスを組む（2026-08-13）

発想（利用者の指摘）
  我々は全頭にMF複勝順位を付けている。軸を1頭に決める必要はない。
  そして複勝で信号が出ているなら、ワイドはその増幅装置になるはず。
  複勝もワイドも「3着以内に入るか」を当てる同じ土俵で、配当だけ違う。

  複勝 = 1頭が3着以内
  ワイド = 2頭がともに3着以内   ← 複勝の精度がそのまま効く
  馬連 = 2頭が1-2着            ← より厳しい
  3連複 = 3頭が3着以内

やり方
  MF複勝順位の上位K頭でボックスを組む。軸は決めない。
  レース条件別に、券種別に、累計回収率と的中数を出す。

順位付けは累計ROI。ただし**的中数が少ないものは信用しない**。
単勝の人気薄は1点の標準偏差が564%あり、150点・的中9本では
±5ptの精度に48,801点（687年）必要だった。複勝は標準偏差78%で
940点（0.3年）。ワイドはその中間になるはず。

検体: bet_cache_2021〜2025（207,518頭・14,972レース）＋ jv_payouts
実行: python box_search.py → box_search_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {k: {} for k in ("ワイド", "馬連", "馬単", "3連複")}
    for r in jv[jv.券種.isin(PAY)].itertuples():
        PAY[r.券種][(r.race_id, r.組み合わせ)] = r.払戻金
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")

    CONDS = {
        "全体": lambda d: pd.Series(True, index=d.index),
        "長距離2100+": lambda d: d["距離"] >= 2100,
        "芝": lambda d: d.is_turf == 1,
        "ダート": lambda d: d.is_turf == 0,
        "重賞級クラス5+": lambda d: d["クラス_num"] >= 5,
        "道悪": lambda d: d["馬場状態_num"] >= 3,
        "多頭数16+": lambda d: d["出走頭数"] >= 16,
        "少頭数-12": lambda d: d["出走頭数"] <= 12,
    }
    # 人気の縛り（全部人気馬だと配当が付かないので、1つは人気薄を入れる等）
    POPS = {"制限なし": None, "全馬4番人気以下": 4, "全馬3番人気以内": -3}

    rows = []
    for cl, cf in CONDS.items():
        S = D[cf(D)]
        rids = S.race_id.unique()
        log(f"\n=== {cl}  {len(rids):,}レース ===")
        by = {r: g for r, g in S.groupby("race_id", sort=False)}
        for K in (2, 3, 4, 5):
            for pl, pv in POPS.items():
                agg = {k: [0.0, 0.0, 0] for k in ("ワイド", "馬連", "馬単", "3連複")}
                yr = {k: {y: [0.0, 0.0] for y in YEARS} for k in agg}
                nr = 0
                for r, g in by.items():
                    sel = g[g.mr <= K]
                    if pv is not None:
                        sel = sel[sel.pr >= pv] if pv > 0 else sel[sel.pr <= -pv]
                    bs = sorted(sel.bn.tolist())
                    if len(bs) < 2:
                        continue
                    nr += 1
                    y = int(r[:4])
                    for a, b in itertools.combinations(bs, 2):
                        for k in ("ワイド", "馬連"):
                            v = PAY[k].get((r, f"{a}-{b}"), 0.0)
                            agg[k][0] += 100; agg[k][1] += v; agg[k][2] += v > 0
                            yr[k][y][0] += 100; yr[k][y][1] += v
                        for kk in (f"{a}-{b}", f"{b}-{a}"):
                            v = PAY["馬単"].get((r, kk), 0.0)
                            agg["馬単"][0] += 100; agg["馬単"][1] += v
                            agg["馬単"][2] += v > 0
                            yr["馬単"][y][0] += 100; yr["馬単"][y][1] += v
                    if len(bs) >= 3:
                        for c in itertools.combinations(bs, 3):
                            v = PAY["3連複"].get((r, "-".join(c)), 0.0)
                            agg["3連複"][0] += 100; agg["3連複"][1] += v
                            agg["3連複"][2] += v > 0
                            yr["3連複"][y][0] += 100; yr["3連複"][y][1] += v
                for k, v in agg.items():
                    if v[0] < 20000:      # 200点未満は捨てる
                        continue
                    ys = [yr[k][y][1] / yr[k][y][0] * 100 if yr[k][y][0] else np.nan
                          for y in YEARS]
                    if any(np.isnan(x) for x in ys):
                        continue
                    rows.append({"レース条件": cl, "ボックス": f"MF複勝{K}位以内",
                                 "人気": pl, "券種": k, "R数": nr,
                                 "点数": int(v[0] / 100), "的中": v[2],
                                 "的中率": round(v[2] / (v[0] / 100) * 100, 1),
                                 "累計ROI": round(v[1] / v[0] * 100, 1),
                                 "最悪年": round(min(ys), 1),
                                 "100%超年数": sum(1 for x in ys if x >= 100)})
    R = pd.DataFrame(rows).sort_values("累計ROI", ascending=False)
    R.to_csv("box_search_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n\n構成 {len(R):,}件を評価")
    log(f"累計100%超: {int((R.累計ROI >= 100).sum())}件\n")
    log("=== 累計ROI 上位20 ===")
    log(R.head(20).to_string(index=False))
    log("\n=== そのうち的中100本以上（信用できる規模）===")
    log(R[(R.累計ROI >= 100) & (R.的中 >= 100)].to_string(index=False)
        if len(R[(R.累計ROI >= 100) & (R.的中 >= 100)]) else "  なし")


if __name__ == "__main__":
    main()
