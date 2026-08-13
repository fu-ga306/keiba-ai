# -*- coding: utf-8 -*-
"""最も「安定して」回収率が出る条件を総当たりで探す（2026-08-13）

利用者の指示により、検体不足の制約を外して探索する。
順位付けは点推定ではなく**最悪年の回収率**で行う。
5年のうち一番悪い年でも100%を超えているものだけが「安定」に値する。

⚠ この探索は過去に何度も幻を生んでいる（芝限定148.9%、穴馬124.8%）。
  出てきた構成は「安定して見える」だけで、証明ではない。順列検定を併記する。

検体: bet_cache_2021〜2025.csv（walk-forward OOS・207,518頭・14,972レース）
      jv_payouts.csv（実払戻）
実行: python stable_search.py → stable_search_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_N = 150            # これ未満は対象外（それでも小さい）


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    fk = {(r.race_id, r.組み合わせ): r.払戻金
          for r in jv[jv.券種 == "複勝"].itertuples()}
    D["tan"] = D.win * D.odds * 100
    D["fuku"] = [fk.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]
    return D


CONDS = {
    "全体": lambda d: pd.Series(True, index=d.index),
    "芝": lambda d: d.is_turf == 1,
    "ダート": lambda d: d.is_turf == 0,
    "長距離2100+": lambda d: d["距離"] >= 2100,
    "短距離-1400": lambda d: d["距離"] <= 1400,
    "重賞級クラス5+": lambda d: d["クラス_num"] >= 5,
    "下級クラス1-2": lambda d: d["クラス_num"] <= 2,
    "道悪": lambda d: d["馬場状態_num"] >= 3,
    "良馬場": lambda d: d["馬場状態_num"] <= 1,
    "多頭数16+": lambda d: d["出走頭数"] >= 16,
    "少頭数-12": lambda d: d["出走頭数"] <= 12,
}
MRS = [(1, "MF複勝1位"), (2, "MF複勝2位以内"), (3, "MF複勝3位以内"), (5, "MF複勝5位以内")]
ODDS = [(1, 5, "〜5倍"), (5, 10, "5-10倍"), (10, 20, "10-20倍"),
        (20, 50, "20-50倍"), (1, 10, "〜10倍"), (1, 20, "〜20倍")]
POPS = [(1, 99, "人気不問"), (4, 99, "4番人気以下"), (6, 99, "6番人気以下"),
        (1, 3, "3番人気以内")]


def main():
    D = load()
    print(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  1着{int(D.win.sum()):,}頭\n")
    rows = []
    for (cl, cf), (mv, ml), (olo, ohi, od), (plo, phi, pl) in itertools.product(
            CONDS.items(), MRS, ODDS, POPS):
        s = D[cf(D) & (D.mr <= mv) & (D.odds >= olo) & (D.odds < ohi) &
              (D.pr >= plo) & (D.pr <= phi)]
        if len(s) < MIN_N:
            continue
        for bet, col in (("単勝", "tan"), ("複勝", "fuku")):
            yr = [s[s.年 == y][col].mean() for y in YEARS]
            if any(np.isnan(v) for v in yr):
                continue
            rows.append({"券種": bet, "レース条件": cl, "馬の条件": ml,
                         "オッズ": od, "人気": pl, "点数": len(s),
                         "的中": int((s[col] > 0).sum()),
                         "回収率": round(s[col].mean(), 1),
                         "最悪年": round(min(yr), 1), "最良年": round(max(yr), 1),
                         "100%超の年数": sum(1 for v in yr if v >= 100)})
    R = pd.DataFrame(rows)
    R.to_csv("stable_search_result.csv", index=False, encoding="utf-8-sig")
    print(f"探索した構成 {len(R):,}件\n")
    top = R[(R.最悪年 >= 100)].sort_values("最悪年", ascending=False)
    print(f"=== 5年すべてで100%超えた構成: {len(top)}件 ===")
    if len(top):
        print(top.head(20).to_string(index=False))
    print(f"\n=== 参考: 最悪年が高い順（100%未満も含む）上位12 ===")
    print(R.sort_values("最悪年", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
