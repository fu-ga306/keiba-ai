# -*- coding: utf-8 -*-
"""全年代を横並びで比べて、良い年と悪い年の違いを探す（2026-08-18）

なぜこうするか
  2025年だけを掘っても原因が出なかった。個別の仮説（古さ・血統・市場効率）は
  すべて否定された。ならば「2025年は何が違うのか」ではなく
  「良い年と悪い年は何が違うのか」を全年で並べて見るべき。

  年ごとのモデルの上乗せ（ΔR²）には大きな差がある。
    2021 0.0043 / 2022 0.0043 / 2023 0.0060 / 2024 0.0115 / 2025 0.0006
  この差と一緒に動いている指標があれば、それが手がかりになる。

並べる指標（いずれも回収率とは独立に測れるもの）
  レース側 : レース数・平均頭数・芝ダ比率・クラス構成
  市場側   : 1番人気の勝率・上位人気の集中度・オッズ分布・市場R²
  結果側   : 波乱の起きやすさ（1番人気が飛ぶ率）
  モデル側 : ΔR²・gap分布・軸の的中率

  そのうえで、各指標とΔR²・ROIの関係を見る。
  ⚠ 6年しかないので相関は弱い。傾向を見るだけで、断定はしない。

実行: python diag_years.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import model_diag as M

EPS = 1e-9
AX_GAP, MATE_GAP, MATE_MAX = 1.5, 1.3, 3
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["年"] = d.race_id.str[:4].astype(int)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf", "距離", "出走頭数", "クラス_num",
                              "馬場状態_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    YS = sorted(d.年.unique())
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース  年 {YS}\n")

    rec = {}
    for y in YS:
        s = d[d.年 == y].copy()
        r = {}
        r["レース数"] = s.race_id.nunique()
        r["平均頭数"] = s.groupby("race_id").size().mean()
        r["芝率"] = s.groupby("race_id")["is_turf"].first().mean() * 100
        # 市場側
        fav = s[s.人気 == 1]
        r["1番人気勝率"] = fav.win.mean() * 100
        r["1番人気中央オッズ"] = fav.odds.median()
        top3 = s[s.人気 <= 3]
        r["上位3人気の勝率計"] = top3.win.mean() * 3 * 100
        r["市場集中度"] = s.groupby("race_id").q.max().mean() * 100
        # 波乱
        r["10番人気以下が勝つ率"] = s[s.人気 >= 10].win.mean() * 100
        win = s[s.win == 1]
        r["勝ち馬の中央オッズ"] = win.odds.median()
        r["勝ち馬の平均人気"] = win.人気.mean()
        # 市場R² と ΔR²
        t = s.copy()
        t["_rc"] = pd.factorize(t.race_id)[0]
        t["lq"] = np.log(t.q.clip(EPS))
        t["lp"] = np.log((t.p1 / t.groupby("race_id").p1.transform("sum")).clip(EPS))
        l0 = M.null_ll(t)
        _, lm = M.clogit(t, ["lq"])
        _, lb = M.clogit(t, ["lq", "lp"])
        r["市場R2"] = 1 - lm / l0
        r["ΔR2"] = (1 - lb / l0) - (1 - lm / l0)
        # モデル側
        r["gap中央"] = s.gap.median()
        r["gap>=1.5率"] = (s.gap >= 1.5).mean() * 100
        # 買い目
        rows = []
        for rid, g in s.groupby("race_id", sort=False):
            gv = g.gap.values
            k = int(np.argmax(gv))
            if gv[k] < AX_GAP:
                continue
            a = g.bn.values[k]
            rows.append(PAY.get((rid, "単勝", a), 0.0))
            if pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0:
                for j in [x for x in np.argsort(-gv) if x != k and gv[x] >= MATE_GAP][:MATE_MAX]:
                    b = g.bn.values[j]
                    rows.append(PAY.get((rid, "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0))
        v = np.array(rows)
        r["点数"] = len(v)
        r["的中率"] = (v > 0).mean() * 100
        r["ROI"] = v.mean()
        rec[y] = r
    R = pd.DataFrame(rec)

    log("=== 全年代の横並び ===")
    order = ["レース数", "平均頭数", "芝率", "1番人気勝率", "1番人気中央オッズ",
             "上位3人気の勝率計", "市場集中度", "10番人気以下が勝つ率",
             "勝ち馬の中央オッズ", "勝ち馬の平均人気", "市場R2", "ΔR2",
             "gap中央", "gap>=1.5率", "点数", "的中率", "ROI"]
    log(f"  {'指標':<22}" + "".join(f"{y:>10}" for y in YS))
    for k in order:
        vals = R.loc[k]
        fmt = "{:>10.4f}" if k in ("市場R2", "ΔR2") else (
            "{:>10.0f}" if k in ("レース数", "点数") else "{:>10.2f}")
        log(f"  {k:<22}" + "".join(fmt.format(vals[y]) for y in YS))

    log("\n=== ΔR²（モデルの上乗せ）と一緒に動いている指標 ===")
    log("  6年しかないので参考。|相関|が0.7以上のものだけ出す。")
    from scipy.stats import spearmanr
    tgt = R.loc["ΔR2"]
    out = []
    for k in order:
        if k in ("ΔR2",):
            continue
        rho, p = spearmanr(R.loc[k].values, tgt.values)
        if abs(rho) >= 0.7:
            out.append((abs(rho), k, rho, p))
    for _, k, rho, p in sorted(out, reverse=True):
        log(f"  {k:<22}rho={rho:+.3f}  (p={p:.3f})")
    if not out:
        log("  該当なし")

    log("\n=== ROIと一緒に動いている指標 ===")
    tgt = R.loc["ROI"]
    out = []
    for k in order:
        if k == "ROI":
            continue
        rho, p = spearmanr(R.loc[k].values, tgt.values)
        if abs(rho) >= 0.7:
            out.append((abs(rho), k, rho, p))
    for _, k, rho, p in sorted(out, reverse=True):
        log(f"  {k:<22}rho={rho:+.3f}  (p={p:.3f})")
    if not out:
        log("  該当なし")


if __name__ == "__main__":
    main()
