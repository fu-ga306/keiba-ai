# -*- coding: utf-8 -*-
"""残差モデルの結果が脆くないかを確かめる（2026-08-17）

確かめること
  ① 乱数シードを変えても同じ結果になるか
     → 1つのシードでしか出ない結果なら、それは偶然の産物
  ② 学習回数を変えても同じか
     → 特定の回数でだけ良いなら、過学習の可能性
  ③ 年を1つ抜いても成立するか（leave-one-year-out）
     → 2024（227%）に支えられていないか
  ④ 前半3年で決めた形が、後半2年で成立するか
     → 本当の意味の前向き検証に一番近い

これまで8回、バックテストで良く見えたものが崩れている。
崩れ方の共通点は「条件を少し変えると消える」だった。だから変えて試す。

事前登録（この時点で固定・あとから変えない）
  買い方: 各レースで gap（モデル予測確率÷市場確率）が最大の1頭を単勝1点。
          ただし gap >= 2.0 のときだけ買う。ほかの条件は付けない。
  なぜ2.0か: 順列検定の偽物の最大値が114.6%で、gap>=1.5の113.0%はそれを
             超えられない。統計的に区別できるのは gap>=2.0 だけ。

実行: python resid_robust.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from market_free_model import FEATURE_COLS_MF

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
GAP_MIN = 2.0                      # 事前登録
SEEDS = [42, 7, 123, 2024, 99]
rng = np.random.default_rng(20260817)


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
    use = list(dict.fromkeys(["race_id", "馬名", "着順_num", "人気", "単勝オッズ"] + BASE))
    D = pd.read_csv("race_features.csv", usecols=use, dtype={"race_id": str},
                    low_memory=False)
    D["race_id"] = D["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[D["着"].notna() & (D.odds > 0) & D["年"].between(2019, 2025)].copy()
    D["win"] = (D["着"] == 1).astype(int)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy().reset_index(drop=True)
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    D["lq"] = np.log(D.q.clip(EPS))
    return D, BASE


def score(D, BASE, seed, rounds, test_years, train_upto=None):
    """walk-forward で予測。train_upto を指定するとその年までで固定学習。"""
    out = []
    for y in test_years:
        cut = train_upto if train_upto else y - 1
        tr = D[D.年 <= cut]
        te = D[D.年 == y].copy()
        if len(tr) < 5000 or te.empty:
            continue
        m = lgb.train(params(seed), lgb.Dataset(tr[BASE], tr.win,
                                                init_score=tr.lq.values),
                      num_boost_round=rounds)
        te["sc"] = m.predict(te[BASE], raw_score=True) + te.lq.values
        out.append(te)
    return pd.concat(out) if out else pd.DataFrame()


def roi(te):
    """gap最大の1頭を選び、gap>=GAP_MIN なら買う。"""
    if te.empty:
        return None
    e = np.exp(te.sc - te.groupby("race_id")["sc"].transform("max"))
    p = e / te.groupby("race_id")["e"].transform("sum") if False else \
        e / e.groupby(te.race_id).transform("sum")
    g = p / te.q
    t = te.assign(_g=g.values)
    sel = t.loc[t.groupby("race_id")["_g"].idxmax()]
    sel = sel[sel._g >= GAP_MIN]
    if len(sel) < 50:
        return None
    return {"点数": len(sel), "的中": int(sel.win.sum()),
            "ROI": (sel.win * sel.odds).sum() / len(sel) * 100,
            "年別": {y: (x.win * x.odds).sum() / len(x) * 100
                    for y, x in sel.groupby("年") if len(x) > 20}}


def main():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量 {len(BASE)}列")
    log(f"事前登録: gap>={GAP_MIN} の1頭を単勝1点。他の条件なし。\n")

    log("=== ① シードを変える（5通り）===")
    log(f"  {'シード':<10}{'点数':>8}{'的中':>7}{'ROI':>9}")
    rs = []
    for sd in SEEDS:
        r = roi(score(D, BASE, sd, 600, YEARS))
        if r:
            rs.append(r["ROI"])
            log(f"  {sd:<10}{r['点数']:>8,}{r['的中']:>7}{r['ROI']:>8.1f}%")
    if rs:
        log(f"  → 中央値 {np.median(rs):.1f}%  範囲 {min(rs):.1f}〜{max(rs):.1f}%"
            f"  ばらつき {np.std(rs):.1f}pt")
        log(f"  {'✅ シードに依存しない' if min(rs) >= 100 else '⚠ シードによっては100%割れ'}")

    log("\n=== ② 学習回数を変える ===")
    log(f"  {'回数':<10}{'点数':>8}{'的中':>7}{'ROI':>9}")
    rr = []
    for nb in (200, 400, 600, 900):
        r = roi(score(D, BASE, 42, nb, YEARS))
        if r:
            rr.append(r["ROI"])
            log(f"  {nb:<10}{r['点数']:>8,}{r['的中']:>7}{r['ROI']:>8.1f}%")
    if rr:
        log(f"  → 範囲 {min(rr):.1f}〜{max(rr):.1f}%"
            f"  {'✅ 回数に依存しない' if min(rr) >= 100 else '⚠ 回数で100%割れ'}")

    log("\n=== ③ 年を1つ抜く（2024の227%に支えられていないか）===")
    r = roi(score(D, BASE, 42, 600, YEARS))
    log(f"  {'除く年':<10}{'点数':>8}{'的中':>7}{'ROI':>9}")
    if r:
        full = score(D, BASE, 42, 600, YEARS)
        e = np.exp(full.sc - full.groupby("race_id")["sc"].transform("max"))
        p = e / e.groupby(full.race_id).transform("sum")
        full = full.assign(_g=(p / full.q).values)
        sel = full.loc[full.groupby("race_id")["_g"].idxmax()]
        sel = sel[sel._g >= GAP_MIN]
        for y in [None] + YEARS:
            s = sel if y is None else sel[sel.年 != y]
            v = (s.win * s.odds * 100).values
            bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
            lab = "なし(全年)" if y is None else str(y)
            log(f"  {lab:<10}{len(s):>8,}{int(s.win.sum()):>7}{v.mean():>8.1f}%"
                f"   95%区間[{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]")

    log("\n=== ④ 前半3年で学習を固定し、後半2年を当てる（前向きに一番近い）===")
    r = roi(score(D, BASE, 42, 600, [2024, 2025], train_upto=2023))
    if r:
        log(f"  2023までで学習 → 2024-2025を予測")
        log(f"  {r['点数']:,}点  的中{r['的中']}  ROI {r['ROI']:.1f}%")
        log("  年別: " + "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items()))
    else:
        log("  標本不足")


if __name__ == "__main__":
    main()
