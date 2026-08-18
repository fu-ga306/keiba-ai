# -*- coding: utf-8 -*-
"""2025年の異常をさらに掘る（2026-08-18・第2段）

第1段でわかったこと
  ・市場の精度は変わっていない（市場R² 0.2532 で他年と同水準）
  ・モデルの上乗せだけが消えた（ΔR² 0.0115 → 0.0006）
  ・的中率そのものが落ちた（13.2% → 9.3%）。運では説明できない（p=0.0013）
  ・落ち方はほぼ全区分で均一（-40〜-77pt）。上級クラスだけ無傷

  そして最大の手がかり:
    **2025年は平均gapが最も高い（1.86）のに的中率が最も低い（9.3%）**
    モデルは他の年より自信を持っていたのに、当たらなかった。
    これは「較正が崩れた」形で、学習時と検証時で何かが変わったことを示す。

この段で調べること
  ① 特徴量の欠損率が2025年に変わっていないか（データ側の事故）
  ② gapの分布そのものが2025年にずれていないか
  ③ gap帯ごとの的中率（較正が崩れているのはどの帯か）
  ④ 2025年の中で時期による差はあるか
  ⑤ 学習データから2024年を抜いたら2025年は良くなるか
     （2024はΔR²0.0115と異常に良い年。そこに引きずられた可能性）

実行: python diag_2025b.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import model_diag as M
from market_free_model import FEATURE_COLS_MF

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
SEEDS = [42, 7, 123]
N_ROUNDS = 600
AX_GAP = 1.5
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


def params(seed):
    return dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed)


def load():
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(["race_id", "馬名", "馬番", "着順_num", "人気",
                              "単勝オッズ", "is_turf", "距離"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[(D.odds > 0) & D["着"].notna()].copy()
    D["win"] = (D["着"] == 1).astype(float)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy().reset_index(drop=True)
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    D["lq"] = np.log(D.q.clip(EPS))
    return D, BASE


def fit_predict(tr, te, cols):
    f = np.mean([lgb.train(params(sd),
                           lgb.Dataset(tr[cols], tr.win, init_score=tr.lq.values),
                           num_boost_round=N_ROUNDS).predict(te[cols], raw_score=True)
                 for sd in SEEDS], axis=0)
    d = te.copy()
    sc = f + te.lq.values
    d["_sc"] = sc
    e = np.exp(d._sc - d.groupby("race_id")._sc.transform("max"))
    d["p"] = e / e.groupby(d.race_id).transform("sum")
    d["gap"] = d.p / d.q
    return d


def dr2(d):
    s = d.copy()
    s["_rc"] = pd.factorize(s.race_id)[0]
    s["lp"] = np.log(s.p.clip(EPS))
    l0 = M.null_ll(s)
    _, lm = M.clogit(s, ["lq"])
    _, lb = M.clogit(s, ["lq", "lp"])
    return (1 - lb / l0) - (1 - lm / l0)


def main():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量{len(BASE)}列\n")

    # ── ① 欠損率 ────────────────────────────────────
    log("=== ① 特徴量の欠損率は2025年に変わったか ===")
    miss = {}
    for y in YEARS:
        s = D[D.年 == y]
        miss[y] = s[BASE].isna().mean()
    mdf = pd.DataFrame(miss)
    log(f"  全特徴量の平均欠損率: " + "  ".join(
        f"{y}:{mdf[y].mean()*100:.1f}%" for y in YEARS))
    ch = (mdf[2025] - mdf[[2021, 2022, 2023, 2024]].mean(axis=1)).sort_values(
        ascending=False)
    log(f"\n  2025年に欠損が増えた列 上位8:")
    for k, v in ch.head(8).items():
        if v > 0.005:
            log(f"    {k[:34]:<36}{mdf[2025][k]*100:>6.1f}%  (他年平均"
                f"{mdf[[2021,2022,2023,2024]].mean(axis=1)[k]*100:.1f}% / 差{v*100:+.1f}pt)")
    big = ch[ch > 0.05]
    log(f"  → 5pt以上増えた列: {len(big)}件 {'⚠ 要注意' if len(big) else '（なし）'}")

    # ── ②③ gapの分布と較正 ──────────────────────────
    log("\n=== ② gapの分布は2025年にずれたか ===")
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["年"] = d.race_id.str[:4].astype(int)
    log(f"  {'年':<8}{'gap中央':>9}{'gap90%点':>10}{'gap>=1.5の割合':>15}{'gap>=2.0':>10}")
    for y in YEARS:
        s = d[d.年 == y]
        log(f"  {y:<8}{s.gap.median():>9.3f}{s.gap.quantile(.9):>10.3f}"
            f"{(s.gap>=1.5).mean()*100:>14.1f}%{(s.gap>=2.0).mean()*100:>9.1f}%")

    log("\n=== ③ gap帯ごとの実際の勝率（較正が崩れているか）===")
    log("  gapが高いほど勝率も高いはず。2025だけ崩れていないか見る。")
    log(f"  {'gap帯':<12}" + "".join(f"{y:>10}" for y in YEARS))
    for lo, hi in [(0, .8), (.8, 1.2), (1.2, 1.6), (1.6, 2.2), (2.2, 3.5), (3.5, 99)]:
        row = f"  {lo}-{hi if hi < 90 else '∞'}".ljust(14)
        for y in YEARS:
            s = d[(d.年 == y) & (d.gap >= lo) & (d.gap < hi)]
            row += f"{s.win.mean()*100:>9.1f}%" if len(s) > 200 else f"{'--':>10}"
        log(row)

    # ── ④ 2025年内の時期差 ─────────────────────────
    log("\n=== ④ 2025年の中で時期による差 ===")
    s25 = d[d.年 == 2025].copy()
    s25["kai"] = s25.race_id.str[6:8]
    sel = s25.loc[s25.groupby("race_id").gap.idxmax()]
    sel = sel[sel.gap >= AX_GAP]
    log(f"  開催回ごとの軸の勝率（{len(sel)}レース）")
    log(f"  {'開催回':<8}{'レース':>7}{'勝率':>8}")
    for k, g in sel.groupby("kai"):
        if len(g) >= 30:
            log(f"  {k:<8}{len(g):>7}{g.win.mean()*100:>7.1f}%")

    # ── ⑤ 2024年を学習から抜いたら ────────────────────
    log("\n=== ⑤ 学習から2024年を抜いたら2025年は良くなるか ===")
    log("  2024はΔR²0.0115と異常に良い年。そこに引きずられた可能性を確かめる。")
    te = D[D.年 == 2025]
    for lab, tr in (("2019-2024（通常）", D[D.年 <= 2024]),
                    ("2019-2023（2024を除く）", D[D.年 <= 2023]),
                    ("2019-2022（2023-24を除く）", D[D.年 <= 2022])):
        log(f"  {lab} 学習中…")
        r = fit_predict(tr, te, BASE)
        s = r.loc[r.groupby("race_id").gap.idxmax()]
        s = s[s.gap >= AX_GAP]
        log(f"    ΔR² {dr2(r):.4f}  軸{len(s):,}レース 勝率{s.win.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
