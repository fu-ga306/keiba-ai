# -*- coding: utf-8 -*-
"""探索で見つけた構成は、探索に使っていない年でも通用するか（2026-08-15）

争点
  「124.8%は実際に出た数字だから使えるのでは？」という指摘は正当。
  過去にその通り買っていれば実際に儲かったのは事実。
  問題は「これから先も通用するか」で、そこは議論ではなく実験で決まる。

やり方（これが唯一の決着方法）
  探索期間  2021-2023 … ここだけを見て「最良の構成」を選ぶ
  検証期間  2024-2025 … 選んだ構成を、一度も見ていない年で試す

  探索期間で良かった構成が検証期間でも良ければ「使える」。
  探索期間だけ良くて検証期間で崩れるなら「探索の産物」。

⚠ 検証期間のデータは選定に一切使わない。使った時点で実験が壊れる。

実行: python holdout_test.py → holdout_test_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN = [2021, 2022, 2023]
TEST = [2024, 2025]
MIN_HITS_TRAIN = 20      # 探索期間での最低的中数（緩めにして候補を広く取る）
rng = np.random.default_rng(20260815)


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in TRAIN + TEST], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    FK = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "複勝"].itertuples()}
    D["fuku"] = [FK.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]
    D["tan"] = D.win * D.odds * 100
    D["mr"] = D.groupby("race_id")["c_top3"].rank(ascending=False)
    fav = D[D.pr == 1][["race_id", "mr"]].rename(columns={"mr": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    tr, te = D[D.年.isin(TRAIN)], D[D.年.isin(TEST)]
    log(f"探索期間 {TRAIN} {len(tr):,}頭 / 検証期間 {TEST} {len(te):,}頭\n")

    CONDS = {
        "全体": lambda d: pd.Series(True, index=d.index),
        "堅いR": lambda d: d.fav_mr == 1,
        "荒れR": lambda d: d.fav_mr >= 4,
        "長距離1900+": lambda d: d["距離"] >= 1900,
        "短距離-1400": lambda d: d["距離"] <= 1400,
        "芝": lambda d: d.is_turf == 1,
        "ダート": lambda d: d.is_turf == 0,
        "道悪": lambda d: d["馬場状態_num"] >= 3,
        "少頭数-12": lambda d: d["出走頭数"] <= 12,
        "多頭数16+": lambda d: d["出走頭数"] >= 16,
    }
    MRS = [1, 2, 3, 5]
    ODDS = [(1, 5), (3, 10), (5, 15), (10, 20), (1, 10), (1, 20), (5, 30), (20, 50)]
    POPS = [(1, 3), (4, 6), (1, 6), (4, 99), (1, 99)]

    def pick(d, cf, mv, olo, ohi, plo, phi):
        return d[cf(d) & (d.mr <= mv) & (d.odds >= olo) & (d.odds < ohi)
                 & (d.pr >= plo) & (d.pr <= phi)]

    rows = []
    for (cl, cf), mv, (olo, ohi), (plo, phi) in itertools.product(
            CONDS.items(), MRS, ODDS, POPS):
        for bet, col in (("単勝", "tan"), ("複勝", "fuku")):
            a = pick(tr, cf, mv, olo, ohi, plo, phi)
            if int((a[col] > 0).sum()) < MIN_HITS_TRAIN:
                continue
            b = pick(te, cf, mv, olo, ohi, plo, phi)
            if len(b) < 30:
                continue
            rows.append({
                "券種": bet, "条件": f"{cl}/MF≤{mv}/{olo}-{ohi}倍/人気{plo}-{phi}",
                "探索_点数": len(a), "探索_的中": int((a[col] > 0).sum()),
                "探索ROI": round(float(a[col].mean()), 1),
                "検証_点数": len(b), "検証_的中": int((b[col] > 0).sum()),
                "検証ROI": round(float(b[col].mean()), 1)})
    R = pd.DataFrame(rows)
    R["差"] = (R.検証ROI - R.探索ROI).round(1)
    R.to_csv("holdout_test_result.csv", index=False, encoding="utf-8-sig")
    log(f"候補 {len(R):,}件（探索期間で的中{MIN_HITS_TRAIN}本以上）\n")

    log("=== 探索期間で成績が良かった上位20を、検証期間で試す ===")
    top = R.sort_values("探索ROI", ascending=False).head(20)
    log(top[["券種", "条件", "探索_点数", "探索_的中", "探索ROI",
             "検証_点数", "検証_的中", "検証ROI", "差"]].to_string(index=False))
    log(f"\n  上位20の平均: 探索 {top.探索ROI.mean():.1f}% → 検証 {top.検証ROI.mean():.1f}%"
        f"（{top.検証ROI.mean()-top.探索ROI.mean():+.1f}pt）")
    log(f"  検証でも100%を超えた: {int((top.検証ROI >= 100).sum())}/20件")

    log("\n=== 参考: 探索期間の成績と検証期間の成績の相関 ===")
    from scipy import stats
    c = stats.spearmanr(R.探索ROI, R.検証ROI)
    log(f"  順位相関 {c.correlation:+.4f} (p={c.pvalue:.3f})")
    log("  → 正なら『探索で良い構成は本当に良い』。ゼロなら探索に意味が無い")


if __name__ == "__main__":
    main()
