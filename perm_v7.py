# -*- coding: utf-8 -*-
"""第7世代の網で順列検定する（2026-08-16・最後の関門）

なぜ必要か
  第7世代は15,876構成を探索した。第6世代（1,797構成）の8.8倍なので、
  偶然の当たりも8.8倍出やすい。第6世代で測った p=0.1575 はこの網には使えない。
  同じ網でシャッフルして、実力ゼロでもどこまで行くかを測り直す。

やること
  レース内でモデルの3順位（勝率・連対・複勝）をシャッフルして実力ゼロの
  データを作り、第7世代と同じグリッドで最良ROIを記録する。
  それを何百回も繰り返し、実測の最良ROIがその分布のどこに位置するかを見る。

⚠ セルの足切りには必ず的中数の下限を設ける（2026-08-16の教訓）。
  点数だけで切ると、的中2-5本のノイズセルが最大値を取って検定が無意味になる。
  実際、足切りを的中30本にしたら帰無の中央値が160.8%→115.1%に変わった。

⚠ 検定するのは「直近2年の馬単裏ROI」。採用候補の中心が馬単裏なので揃える。

実行: python perm_v7.py → perm_v7_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RECENT = [2024, 2025]
N_PERM = 300
MIN_PT = 100
MIN_HIT = 30
rng = np.random.default_rng(20260816)


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in RECENT], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    UMA = {}
    for r in jv[jv.券種 == "馬単"].itertuples():
        UMA[(r.race_id, r.組み合わせ)] = r.払戻金
    log(f"検体（直近2年） {len(D):,}頭 / {D.race_id.nunique():,}レース")

    races = []
    for rid, g in D.groupby("race_id", sort=False):
        n = len(g)
        bn = g.bn.values
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = UMA.get((rid, f"{bn[i]}-{bn[j]}"), 0.0)
        races.append({
            "odds": g.odds.values.astype(float),
            "cw": g.c_win.values, "c2": g.c_top2.values, "c3": g.c_top3.values,
            "pr": g.pr.values.astype(float),
            "dist": float(g["距離"].iloc[0]) if pd.notna(g["距離"].iloc[0]) else 0.0,
            "turf": float(g["is_turf"].iloc[0]) if pd.notna(g["is_turf"].iloc[0]) else 0.0,
            "cls": float(g["クラス_num"].iloc[0]) if pd.notna(g["クラス_num"].iloc[0]) else 0.0,
            "baba": float(g["馬場状態_num"].iloc[0]) if pd.notna(g["馬場状態_num"].iloc[0]) else 0.0,
            "tosu": float(g["出走頭数"].iloc[0]) if pd.notna(g["出走頭数"].iloc[0]) else 0.0,
            "uma": M,
        })
    log(f"  展開完了 {len(races):,}レース\n")

    # 第7世代と同じグリッド
    ODDS = [(1, 99), (1, 20), (5, 30), (3, 15), (10, 40), (1, 10)]
    AXN, MATEN = [1, 2, 3], [2, 3, 4, 5, 6]

    def rank_desc(v):
        o = np.argsort(-v, kind="stable")
        r = np.empty(len(v))
        r[o] = np.arange(1, len(v) + 1)
        return r

    CONDN = ("全体", "堅いR", "荒れR", "長距離", "中距離", "短距離", "芝", "ダート",
             "芝長距離", "ダ短距離", "道悪", "少頭数", "中頭数", "上級クラス")

    def best_roi(shuffle):
        keys = [(c, b, o, a, m) for c in CONDN for b in (0, 1, 2)
                for o in range(len(ODDS)) for a in AXN for m in MATEN if m > a]
        cost = {k: 0.0 for k in keys}
        ret = {k: 0.0 for k in keys}
        hit = {k: 0 for k in keys}
        for R in races:
            v = (R["cw"], R["c2"], R["c3"])
            if shuffle:
                p = rng.permutation(len(R["odds"]))
                rk = [rank_desc(x[p]) for x in v]
            else:
                rk = [rank_desc(x) for x in v]
            fav = np.nan
            m1 = R["pr"] == 1
            if m1.any():
                fav = rk[2][m1][0]
            fl = {
                "全体": True,
                "堅いR": (not np.isnan(fav)) and fav == 1,
                "荒れR": (not np.isnan(fav)) and fav >= 4,
                "長距離": R["dist"] >= 1900,
                "中距離": 1600 <= R["dist"] <= 1800,
                "短距離": R["dist"] <= 1400,
                "芝": R["turf"] == 1,
                "ダート": R["turf"] == 0,
                "芝長距離": R["turf"] == 1 and R["dist"] >= 1900,
                "ダ短距離": R["turf"] == 0 and R["dist"] <= 1400,
                "道悪": R["baba"] >= 3,
                "少頭数": R["tosu"] <= 12,
                "中頭数": 13 <= R["tosu"] <= 15,
                "上級クラス": R["cls"] >= 4,
            }
            for oi, (olo, ohi) in enumerate(ODDS):
                om = (R["odds"] >= olo) & (R["odds"] < ohi)
                for bi in (0, 1, 2):
                    r = rk[bi]
                    for av in AXN:
                        ax = np.where((r <= av) & om)[0]
                        if not len(ax):
                            continue
                        i = ax[np.argmin(r[ax])]
                        for mn in MATEN:
                            if mn <= av:
                                continue
                            mate = np.where((r <= mn) & (np.arange(len(r)) != i))[0]
                            if not len(mate):
                                continue
                            # 馬単裏（相手→軸）で評価する
                            vals = R["uma"][mate, i]
                            c = 100.0 * len(mate)
                            s = vals.sum()
                            h = int((vals > 0).sum())
                            for cn, ok in fl.items():
                                if ok:
                                    k = (cn, bi, oi, av, mn)
                                    cost[k] += c
                                    ret[k] += s
                                    hit[k] += h
        best = -1.0
        for k in keys:
            if cost[k] >= MIN_PT * 100 and hit[k] >= MIN_HIT:
                best = max(best, ret[k] / cost[k] * 100)
        return best

    log("実測の最良ROIを計算中...")
    obs = best_roi(False)
    log(f"  実測 最良ROI = {obs:.1f}%\n")

    log(f"シャッフル {N_PERM}回...")
    null = []
    for k in range(N_PERM):
        null.append(best_roi(True))
        if (k + 1) % 25 == 0:
            log(f"  {k+1}/{N_PERM}  帰無の中央 {np.median(null):.1f}%")
    null = np.array(null)
    p = float(np.mean(null >= obs))
    log("\n=== 結果 ===")
    log(f"  探索した構成      15,876（第6世代の8.8倍）")
    log(f"  実測 最良ROI      {obs:.1f}%")
    log(f"  帰無の中央値      {np.median(null):.1f}%")
    log(f"  帰無の95%上限     {np.percentile(null, 95):.1f}%")
    log(f"  p値（探索込み）   {p:.4f}")
    log("  " + ("→ 新基準(p<0.20)を満たす。探索の産物では説明できない" if p < 0.20
                else "→ 新基準を満たさない。実力ゼロでも同じ数字が出る"))
    pd.DataFrame([{"世代": "v7", "探索数": 15876, "実測最良ROI": round(obs, 1),
                   "帰無中央": round(float(np.median(null)), 1),
                   "帰無95%上限": round(float(np.percentile(null, 95)), 1),
                   "p値": round(p, 4), "試行": N_PERM}]) \
        .to_csv("perm_v7_result.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
