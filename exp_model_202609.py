# -*- coding: utf-8 -*-
"""事前登録した構成を全部試す（2026-09-03）

⚠ 事前登録_202609.md に書いたものだけを試す。あとから足さない。
  結果は**全部**出す。良いものだけ載せない。

枠組み
  開発  2021-2024   ここで全部の試行を行う
  評価  2025        このスクリプトでは**触らない**

節約
  B（券種）と C（しきい値）は、学習し直さずに同じ予測から評価できる。
  学習が要るのは A（特徴量）と E（学習量）と D（シード数）だけ。

実行
  python exp_model_202609.py            全部
  python exp_model_202609.py --quick    Aだけ（動作確認用）
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_YEARS = [2021, 2022, 2023, 2024]      # 2025は触らない
EPS = 1e-9
RNG = np.random.default_rng(20260903)


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


# ── 特徴量の系統。A で外す対象 ─────────────────────────────────────
def group_of(col):
    if re.search(r"クラス", col):
        return "クラス"
    if re.search(r"賞金", col):
        return "賞金"
    if re.search(r"回り", col):
        return "回り"
    return None


def load():
    from train_resid import FEATURE_COLS_MF, market_prob
    head = pd.read_csv("race_features.csv", nrows=1)
    BASE = [c for c in FEATURE_COLS_MF if c in head.columns]
    use = list(dict.fromkeys(
        ["race_id", "馬名", "馬番", "着順_num", "人気", "単勝オッズ", "is_turf"] + BASE))
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
    D["q"] = market_prob(D)
    D["lq"] = np.log(D.q.clip(EPS))
    return D, BASE


def fit_predict(D, cols, years, seeds, rounds):
    """年ごとに、その年より前で学習してその年を予測する。"""
    out = []
    for y in years:
        tr, te = D[D.年 < y], D[D.年 == y]
        if len(tr) < 5000 or te.empty:
            continue
        X, ytr, init = tr[cols], tr["win"].values, tr["lq"].values
        preds = []
        for sd in seeds:
            p = dict(objective="binary", metric="binary_logloss",
                     learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                     seed=sd, verbose=-1, num_threads=4)
            ds = lgb.Dataset(X, ytr, init_score=init, free_raw_data=False)
            m = lgb.train(p, ds, num_boost_round=rounds)
            preds.append(m.predict(te[cols], raw_score=True))
        f = np.mean(preds, axis=0)
        t = te[["race_id", "馬番", "年", "odds", "人気", "着", "is_turf"]].copy()
        sc = f + te["lq"].values
        e = np.exp(sc - pd.Series(sc, index=te.index).groupby(te.race_id).transform("max"))
        t["p"] = (e / e.groupby(te.race_id.values).transform("sum")).values
        t["gap"] = t["p"] / te["q"].values
        out.append(t)
    return pd.concat(out, ignore_index=True)


def payouts():
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金
            for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}


def evaluate(P, PAY, gap_min, with_wide):
    """買い目を作って回収率を出す。resid_io.pick_bets と同じ買い方。"""
    ret = []
    for rid, g in P.groupby("race_id", sort=False):
        g = g.sort_values("gap", ascending=False)
        ax = g.iloc[0]
        if ax["gap"] < gap_min:
            continue
        bn = str(int(ax["馬番"])).zfill(2)
        ret.append(PAY.get((rid, "単勝", bn), 0.0))
        if with_wide and not bool(ax["is_turf"]):
            mates = g.iloc[1:][g.iloc[1:]["gap"] >= 1.3].head(3)
            for _, m in mates.iterrows():
                mb = str(int(m["馬番"])).zfill(2)
                combo = "-".join(sorted([bn, mb]))
                ret.append(PAY.get((rid, "ワイド", combo), 0.0))
    return np.array(ret)


def stat(ret):
    if len(ret) < 50:
        return dict(n=len(ret), roi=float("nan"), lo=float("nan"), hi=float("nan"))
    s = RNG.choice(ret, size=(3000, len(ret))).mean(axis=1)
    return dict(n=len(ret), roi=ret.mean(),
                lo=np.percentile(s, 2.5), hi=np.percentile(s, 97.5))


def row(label, st):
    return ("  %-34s %6d点  %6.1f%%  95%%[%5.1f, %5.1f]  %s"
            % (label, st["n"], st["roi"], st["lo"], st["hi"],
               "○100%超" if st["lo"] > 100 else ""))


def main():
    quick = "--quick" in sys.argv
    log("読み込み")
    D, BASE = load()
    PAY = payouts()
    log(f"  {len(D):,}頭 / {D.race_id.nunique():,}レース / 特徴量{len(BASE)}列")
    log(f"  開発に使う年 {DEV_YEARS}（2025は触らない）")

    groups = {}
    for c in BASE:
        g = group_of(c)
        if g:
            groups.setdefault(g, []).append(c)
    log("  系統ごとの列数: " + str({k: len(v) for k, v in groups.items()}))
    print(flush=True)

    results = []

    # ── A: 直した3列は効いているか ──────────────────────────────
    log("=== A 特徴量の系統を外す ===")
    variants = [("A-1 いまの全特徴量（基準）", BASE)]
    for g in ("クラス", "賞金", "回り"):
        variants.append((f"A-{'234'[('クラス','賞金','回り').index(g)]} {g}系を除く",
                         [c for c in BASE if group_of(c) != g]))
    preds = {}
    for lab, cols in variants:
        t0 = datetime.now()
        P = fit_predict(D, cols, DEV_YEARS, [42, 7, 123], 600)
        preds[lab] = P
        st = stat(evaluate(P, PAY, 1.5, True))
        results.append((lab, st))
        log(row(lab, st) + f"   {(datetime.now()-t0).total_seconds()/60:.0f}分")
    if quick:
        log("--quick のためここで終わります")
        return

    base_P = preds[variants[0][0]]

    # ── B: 券種（学習し直さない） ────────────────────────────────
    print(flush=True)
    log("=== B 券種 ===")
    for lab, ww in (("B-1 単勝＋ワイド（基準）", True), ("B-2 単勝のみ", False)):
        st = stat(evaluate(base_P, PAY, 1.5, ww))
        results.append((lab, st))
        log(row(lab, st))

    # ── C: しきい値（学習し直さない） ────────────────────────────
    print(flush=True)
    log("=== C しきい値（一番良い値は選ばない。単調かどうかを見る） ===")
    for gm in (1.3, 1.5, 1.7, 2.0):
        st = stat(evaluate(base_P, PAY, gm, True))
        results.append((f"C gap>={gm}", st))
        log(row(f"C gap>={gm}", st))

    # ── D: シード数 ──────────────────────────────────────────
    print(flush=True)
    log("=== D シード数 ===")
    st = stat(evaluate(base_P, PAY, 1.5, True))
    results.append(("D-1 3シード（基準）", st))
    log(row("D-1 3シード（基準）", st))
    t0 = datetime.now()
    P7 = fit_predict(D, BASE, DEV_YEARS, [42, 7, 123, 2024, 99, 555, 31], 600)
    st7 = stat(evaluate(P7, PAY, 1.5, True))
    results.append(("D-2 7シード", st7))
    log(row("D-2 7シード", st7) + f"   {(datetime.now()-t0).total_seconds()/60:.0f}分")

    # ── E: 学習量 ────────────────────────────────────────────
    print(flush=True)
    log("=== E 学習量（一番良い値は選ばない） ===")
    for nr in (300, 600, 1200):
        if nr == 600:
            st = stat(evaluate(base_P, PAY, 1.5, True))
        else:
            t0 = datetime.now()
            P = fit_predict(D, BASE, DEV_YEARS, [42, 7, 123], nr)
            st = stat(evaluate(P, PAY, 1.5, True))
        results.append((f"E rounds={nr}", st))
        log(row(f"E rounds={nr}", st))

    # ── まとめ ───────────────────────────────────────────────
    print(flush=True)
    log("=== 全結果（2021-2024。2025は見ていない） ===")
    for lab, st in results:
        log(row(lab, st))
    ok = [l for l, s in results if s["lo"] > 100]
    print(flush=True)
    log(f"  95%区間が100%を上回った構成: {len(ok)} / {len(results)}")
    if ok:
        log("    " + " / ".join(ok))
    else:
        log("    無し。**採用規則により、何も変えない。**")

    pd.DataFrame([{"構成": l, **s} for l, s in results]).to_csv(
        "exp_model_202609_result.csv", index=False, encoding="utf-8-sig")
    log("  exp_model_202609_result.csv に保存")


if __name__ == "__main__":
    main()
