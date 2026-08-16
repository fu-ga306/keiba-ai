# -*- coding: utf-8 -*-
"""順列検定に意味があるのかを実験で確かめる（2026-08-16）

問い（利用者）
  「①直近2年とも100%超 ②的中30本以上 ③通算100%超 ④スリッページ通過」
  まで通れば十分ではないか。⑤順列検定は本当に必要か。

実験の設計
  順列検定の主張は「探索を広げるほど偶然の当たりが増え、それは翌年に再現しない」。
  これが正しいかは、過去のある時点で①〜③を満たした構成を選び、
  **その後の年**でどうなったかを見れば分かる。

    選抜期間 2021-2023 … ここだけを見て①〜③を満たす構成を選ぶ
    検証期間 2024-2025 … 一度も見ていない年での実際の成績

  ①〜③で選んだものが検証期間でも100%を超えるなら、順列検定は不要。
  検証期間で崩れるなら、順列検定が捉えていた「偶然」は実在する。

  さらに、選抜時の構成数（探索の広さ）を変えて、
  「広く探すほど崩れ方が大きくなるか」も見る。これが順列検定の核心。

⚠ 検証期間のデータは選抜に一切使わない。

実行: python perm_value.py → perm_value_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN = [2021, 2022, 2023]
TEST = [2024, 2025]
SEL_YEARS = [2022, 2023]      # 選抜期間の「直近2年」
MIN_HIT = 30


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in TRAIN + TEST], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False)
    D["r2"] = g["c_top2"].rank(ascending=False)
    D["r3"] = g["c_top3"].rank(ascending=False)
    fav = D[D.pr == 1][["race_id", "r3"]].rename(columns={"r3": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    UMA = {}
    for r in jv[jv.券種 == "馬単"].itertuples():
        UMA[(r.race_id, r.組み合わせ)] = r.払戻金
    return D, UMA


CONDS = {
    "全体": lambda d: pd.Series(True, index=d.index),
    "堅いR": lambda d: d.fav_mr == 1,
    "荒れR": lambda d: d.fav_mr >= 4,
    "長距離1900+": lambda d: d["距離"] >= 1900,
    "中距離1600-1800": lambda d: (d["距離"] >= 1600) & (d["距離"] <= 1800),
    "短距離-1400": lambda d: d["距離"] <= 1400,
    "芝": lambda d: d.is_turf == 1,
    "ダート": lambda d: d.is_turf == 0,
    "芝長距離": lambda d: (d.is_turf == 1) & (d["距離"] >= 1900),
    "ダ短距離": lambda d: (d.is_turf == 0) & (d["距離"] <= 1400),
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
    "中頭数13-15": lambda d: (d["出走頭数"] >= 13) & (d["出走頭数"] <= 15),
    "上級クラス4+": lambda d: d["クラス_num"] >= 4,
}
BASIS = {"勝率": "r1", "連対": "r2", "複勝": "r3"}
ODDS = [(1, 99), (1, 20), (5, 30), (3, 15), (10, 40), (1, 10)]
AXN, MATEN = [1, 2, 3], [2, 3, 4, 5, 6]


def main():
    D, UMA = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    races = {r: g for r, g in D.groupby("race_id", sort=False)}

    rows = []
    n = 0
    for cl, cf in CONDS.items():
        rid = list(D[cf(D)].race_id.unique())
        for bl, bcol in BASIS.items():
            for olo, ohi in ODDS:
                for av, mn in itertools.product(AXN, MATEN):
                    if mn <= av:
                        continue
                    n += 1
                    acc = {y: [0.0, 0.0, 0] for y in TRAIN + TEST}
                    for r in rid:
                        g = races[r]
                        y = int(r[:4])
                        ax = g[(g[bcol] <= av) & (g.odds >= olo) & (g.odds < ohi)]
                        if ax.empty:
                            continue
                        a0 = ax.sort_values(bcol).bn.iloc[0]
                        for b in [x for x in g[g[bcol] <= mn].bn if x != a0]:
                            v = UMA.get((r, f"{b}-{a0}"), 0.0)   # 馬単裏
                            acc[y][0] += 100
                            acc[y][1] += v
                            acc[y][2] += 1 if v > 0 else 0
                    tr_c = sum(acc[y][0] for y in TRAIN)
                    tr_h = sum(acc[y][2] for y in TRAIN)
                    te_c = sum(acc[y][0] for y in TEST)
                    if tr_c < 10000 or te_c < 10000:
                        continue
                    sel_ok = all(acc[y][0] >= 1500 and acc[y][1] / acc[y][0] * 100 >= 100
                                 for y in SEL_YEARS)
                    rows.append({
                        "条件": f"{cl}/{bl}{av}位軸x上位{mn}/{olo}-{ohi}倍",
                        "選抜期間ROI": round(sum(acc[y][1] for y in TRAIN) / tr_c * 100, 1),
                        "選抜期間的中": tr_h,
                        "選抜条件を満たす": sel_ok,
                        "検証期間ROI": round(sum(acc[y][1] for y in TEST) / te_c * 100, 1),
                        "検証期間点数": int(te_c / 100),
                    })
        log(f"  {cl} 完了（{n:,}構成）")

    R = pd.DataFrame(rows)
    R.to_csv("perm_value_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索 {n:,}構成 / 評価できた {len(R):,}件\n")

    ok = R[(R.選抜条件を満たす) & (R.選抜期間的中 >= MIN_HIT) & (R.選抜期間ROI >= 100)]
    log("=== ①〜③だけで選んだ構成が、その後の年でどうなったか ===")
    log(f"  選抜期間(2021-2023)で条件を満たした構成: {len(ok)}件")
    if len(ok):
        log(f"  選抜期間の平均ROI: {ok.選抜期間ROI.mean():.1f}%")
        log(f"  検証期間の平均ROI: {ok.検証期間ROI.mean():.1f}%"
            f"（{ok.検証期間ROI.mean() - ok.選抜期間ROI.mean():+.1f}pt）")
        log(f"  検証期間でも100%を超えた: {int((ok.検証期間ROI >= 100).sum())}/{len(ok)}件"
            f"（{(ok.検証期間ROI >= 100).mean()*100:.0f}%）")
        log("\n  選抜期間の成績が良い順・上位15")
        log(ok.sort_values("選抜期間ROI", ascending=False).head(15).to_string(index=False))

    log("\n=== 探索を広げるほど崩れるか（順列検定の核心）===")
    log("  構成をランダムに N 件だけ見た場合の『最良構成』が、検証期間でどうなるか")
    log(f"{'探索した構成数':<16}{'選抜期間の最良':>14}{'検証期間の成績':>14}{'落差':>10}")
    rng = np.random.default_rng(20260816)
    for k in (50, 200, 1000, 3000, len(R)):
        if k > len(R):
            continue
        drops = []
        for _ in range(200):
            s = R.sample(min(k, len(R)), random_state=int(rng.integers(1 << 30)))
            s = s[s.選抜期間的中 >= MIN_HIT]
            if s.empty:
                continue
            b = s.loc[s.選抜期間ROI.idxmax()]
            drops.append((b.選抜期間ROI, b.検証期間ROI))
        if not drops:
            continue
        a = np.array(drops)
        log(f"{k:>10,}件      {a[:,0].mean():>12.1f}%{a[:,1].mean():>13.1f}%"
            f"{a[:,1].mean()-a[:,0].mean():>+9.1f}pt")


if __name__ == "__main__":
    main()
