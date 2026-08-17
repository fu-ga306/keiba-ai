# -*- coding: utf-8 -*-
"""2026年を完全な未見データとして検証する（2026-08-17）

なぜこれが重要か
  これまでの検証は 2021〜2025 で行い、**2026年は一度も使っていない**。
  学習を2025年までで止めて2026年を予想すれば、本当の意味の未見データになる。

  しかも直近の疑問に直接答えが出る。
    ・2025年が69%と弱かった。効果が古くなっているのか？
    ・それとも2025年がたまたま悪かっただけか？
  2026年で成立すれば「2025がたまたま」、崩れれば「効果が古くなっている」。

  ⚠ この検証は1回だけ。結果を見てから条件を変えることはしない。
    条件は resid_io.py に確定済みのものをそのまま使う。

事前登録（resid_io.py に実装済み・変更しない）
  軸  : 残差モデルの gap が最大の1頭・gap>=2.0 → 単勝1点
  ダートなら 相手（軸以外で gap>=1.3・最大3頭）にワイドを追加
  芝は単勝のみ

比較対象（2021-2025の walk-forward）
  2,926点  的中236  ROI 163.3%  95%区間[112.5, 226.2]  100%超 4/5年
  年別 210 / 106 / 154 / 337 / 69%

実行: python test_2026.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import resid_io
from market_free_model import FEATURE_COLS_MF

EPS = 1e-9
SEEDS = [42, 7, 123]
N_ROUNDS = 600
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def params(seed):
    return dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
                num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                seed=seed, bagging_seed=seed, feature_fraction_seed=seed)


def main():
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

    tr = D[D.年 <= 2025]
    te = D[D.年 == 2026].copy()
    log(f"学習 {len(tr):,}頭（〜2025年） → 検証 {len(te):,}頭 / "
        f"{te.race_id.nunique():,}レース（2026年）")
    if te.empty:
        log("2026年のデータがありません")
        return
    log(f"2026年の期間: {te.race_id.min()[:8]} 〜 {te.race_id.max()[:8]}\n")

    log("学習中（3シード）…")
    f = np.mean([lgb.train(params(sd),
                           lgb.Dataset(tr[BASE], tr.win, init_score=tr.lq.values),
                           num_boost_round=N_ROUNDS).predict(te[BASE], raw_score=True)
                 for sd in SEEDS], axis=0)
    sc = f + te.lq.values
    te["_sc"] = sc
    e = np.exp(te._sc - te.groupby("race_id")["_sc"].transform("max"))
    te["p"] = e / e.groupby(te.race_id).transform("sum")
    te["gap"] = te.p / te.q
    te.to_csv("test_2026_pred.csv", index=False, encoding="utf-8-sig",
              columns=["race_id", "馬名", "馬番", "人気", "odds", "着", "win",
                       "is_turf", "q", "p", "gap"])

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    have = {r.race_id for r in jv.itertuples()}

    rows, nrace = [], 0
    m = {"gap_min": resid_io.AX_GAP}
    for rid, g in te.groupby("race_id", sort=False):
        if rid not in have:
            continue                       # 払戻が無いレースは判定できない
        bets = resid_io.pick_bets(g, model=m)
        if not bets:
            continue
        nrace += 1
        for b in bets:
            rows.append({"race_id": rid, "月": rid[4:6] if len(rid) > 6 else "",
                         "券種": b["券種"],
                         "払戻": PAY.get((rid, b["券種"], b["組み合わせ"]), 0.0),
                         "オッズ": b["単勝オッズ"], "gap": b["gap"]})
    R = pd.DataFrame(rows)
    if R.empty:
        log("買い目が出ませんでした")
        return

    roi = R.払戻.sum() / (len(R) * 100) * 100
    hit = int((R.払戻 > 0).sum())
    v = R.払戻.values
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(4000)])
    log("=" * 60)
    log("=== 2026年（完全な未見データ）の成績 ===")
    log("=" * 60)
    log(f"  買うレース {nrace:,}  点数 {len(R):,}  的中 {hit}（{hit/len(R)*100:.1f}%）")
    log(f"  回収率 {roi:.1f}%   95%区間 [{np.percentile(bs,2.5):.1f}, "
        f"{np.percentile(bs,97.5):.1f}]")
    log(f"\n  券種別:")
    for k, g in R.groupby("券種"):
        log(f"    {k:<6}{len(g):>6,}点  的中{int((g.払戻>0).sum()):>4}"
            f"  ROI {g.払戻.sum()/(len(g)*100)*100:>6.1f}%")

    log(f"\n  月別:")
    for mo, g in R.groupby("月"):
        if len(g) >= 20:
            log(f"    {mo}月  {len(g):>5,}点  的中{int((g.払戻>0).sum()):>3}"
                f"  ROI {g.払戻.sum()/(len(g)*100)*100:>6.1f}%")

    log("\n" + "=" * 60)
    log("=== 過去5年との比較 ===")
    log(f"  {'期間':<14}{'点数':>8}{'的中':>7}{'的中率':>8}{'ROI':>9}")
    log(f"  {'2021-2025':<14}{2926:>8,}{236:>7}{8.1:>7.1f}%{163.3:>8.1f}%")
    log(f"  {'2026(未見)':<14}{len(R):>8,}{hit:>7}{hit/len(R)*100:>7.1f}%{roi:>8.1f}%")
    log(f"\n  買い率: 過去12.6% / 2026 {nrace/te.race_id.nunique()*100:.1f}%")
    log("\n=== 判定 ===")
    if roi >= 100:
        log(f"  ✅ 未見の2026年でも100%を超えた（{roi:.1f}%）")
        log("     2025年の69%は、効果の減衰ではなく年による振れの可能性が高い")
    else:
        log(f"  ⚠ 未見の2026年では100%を割った（{roi:.1f}%）")
        log("     効果が古くなっている可能性がある。前向き検証を続けて確かめる")
    if hit < 50:
        log(f"  ※ 的中{hit}本。まだ標本が少なく、この1年だけで結論は出せない")


if __name__ == "__main__":
    main()
