# -*- coding: utf-8 -*-
"""これまでの知見を織り込んだ買い目探索（2026-08-15・第3世代）

これまでに分かっていること（すべてクリーンデータ）
  ・実運用では7分前で選ぶので -27pt 落ちる。確定オッズ基準の数字は当てにならない
  ・EVと実払戻の順位相関は5年とも負（オプティマイザの呪い）
  ・呪いの強さはレースによって違う。1番人気が弱いレースでは 0.423 → 0.676 に緩和
  ・信頼度の判別力は1〜4番人気まで。5番人気以下では消える
  ・穴（10番人気以下）は拾えない（順列検定 p=0.598）
  ・的中が十分ある構成は例外なく85〜95%に収束する（2,000構成以上で確認）

そこで今回は
  ① レース信頼度（1番人気に対するモデル評価）を軸に加える ← 新規
  ② 探索した構成数を数え、最良構成に family-wise 順列検定をかける ← 自動化
  ③ 的中50本未満は最初から除外する（判定不能なものを候補にしない）
  ④ 評価は「95%区間の下限」。点推定が高いだけの構成は拾わない

⚠ これで出なければ、買い方の探索は打ち止めにする。

検体: bet_cache_2021〜2025（クリーン・walk-forward OOS）
実行: python search_v3.py → search_v3_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_HITS = 50           # 的中がこれ未満の構成は判定不能として捨てる
rng = np.random.default_rng(20260815)


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
    FK = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "複勝"].itertuples()}
    D["fuku"] = [FK.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]
    D["tan"] = D.win * D.odds * 100
    D["mr"] = D.groupby("race_id")["c_top3"].rank(ascending=False)
    D["r1"] = D.groupby("race_id")["c_win"].rank(ascending=False)
    fav = D[D.pr == 1][["race_id", "mr"]].rename(columns={"mr": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")

    # 軸（レース条件）… 新規に「レース信頼度」を追加
    CONDS = {
        "全体": pd.Series(True, index=D.index),
        "堅いR(1人気=MF1位)": D.fav_mr == 1,
        "荒れR(1人気=MF4位以下)": D.fav_mr >= 4,
        "長距離1900+": D["距離"] >= 1900,
        "短距離-1400": D["距離"] <= 1400,
        "芝": D.is_turf == 1,
        "ダート": D.is_turf == 0,
        "道悪": D["馬場状態_num"] >= 3,
        "少頭数-12": D["出走頭数"] <= 12,
        "多頭数16+": D["出走頭数"] >= 16,
    }
    MRS = [(1, "MF複勝1位"), (2, "MF複勝2位以内"), (3, "MF複勝3位以内"), (5, "MF複勝5位以内")]
    ODDS = [(1, 5), (3, 10), (5, 15), (10, 20), (1, 10), (1, 20), (5, 30)]
    POPS = [(1, 3, "人気1-3"), (4, 6, "人気4-6"), (1, 6, "人気1-6"),
            (4, 99, "人気4以下"), (1, 99, "人気不問")]

    rows = []
    n_try = 0
    for (cl, cf), (mv, ml), (olo, ohi), (plo, phi, pl) in itertools.product(
            CONDS.items(), MRS, ODDS, POPS):
        s = D[cf & (D.mr <= mv) & (D.odds >= olo) & (D.odds < ohi)
              & (D.pr >= plo) & (D.pr <= phi)]
        for bet, col in (("単勝", "tan"), ("複勝", "fuku")):
            n_try += 1
            hits = int((s[col] > 0).sum())
            if hits < MIN_HITS:
                continue
            v = s[col].values
            b = np.array([rng.choice(v, len(v)).mean() for _ in range(1500)])
            lo = float(np.percentile(b, 2.5))
            yr = [s[s.年 == y][col].mean() for y in YEARS]
            rows.append({"券種": bet, "レース条件": cl, "馬": ml,
                         "オッズ": f"{olo}-{ohi}", "人気": pl,
                         "点数": len(s), "的中": hits,
                         "ROI": round(float(v.mean()), 1), "CI下": round(lo, 1),
                         "最悪年": round(float(min(yr)), 1),
                         "年100超": sum(1 for x in yr if not np.isnan(x) and x >= 100)})
    R = pd.DataFrame(rows).sort_values("CI下", ascending=False)
    R.to_csv("search_v3_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n探索した構成 {n_try:,}件 / 的中{MIN_HITS}本以上で評価できた {len(R):,}件")
    log(f"  ROI100%超: {int((R.ROI >= 100).sum())}件")
    log(f"  95%下限が100%超: {int((R.CI下 >= 100).sum())}件  ← これが本命\n")
    log("=== 95%下限が高い順 上位15 ===")
    log(R.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
