# -*- coding: utf-8 -*-
"""モデルはどれだけ古くなると使えなくなるか（2026-08-18）

なぜ調べるか
  2023年までで学習を止めて2025年を当てたとき、ROIが70%まで落ちた
  （毎年学習し直せば136%）。効果が古くなる可能性がある。
  だが「1年古い」と「1か月古い」では話が違う。運用で決めたいのは
  「週次の再学習で足りるのか、もっと頻繁にすべきか」。

やり方
  学習を2024年末で止めて2025年を予測し、**2025年の中で月が進むほど
  成績が落ちるか**を見る。落ちるなら、落ち始める時期が再学習の目安になる。

    学習: 〜2024年12月（固定）
    検証: 2025年1月〜12月を、経過月数ごとに区切って成績を出す

  比べるために、毎年学習し直した場合（通常のwalk-forward）の同じ期間も出す。

  ⚠ 月ごとに区切ると1区間の的中が数十本になる。単月の上下は誤差なので、
    傾き（だんだん落ちているか）だけを見る。

事前登録
  買い方は確定したもの（軸gap>=1.5 単勝＋ダートならワイド）。変更しない。
  見るのは経過月数とROI・ΔR²の関係だけ。

実行: python resid_stale.py
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import model_diag as M
from features import attach_race_date
from market_free_model import FEATURE_COLS_MF

EPS = 1e-9
SEEDS = [42, 7, 123]
N_ROUNDS = 600
AX_GAP, MATE_GAP, MATE_MAX = 1.5, 1.3, 3
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
    D = attach_race_date(D)
    D["年"] = D["race_id"].str[:4].astype(int)
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["odds"] = pd.to_numeric(D["単勝オッズ"], errors="coerce")
    D = D[(D.odds > 0) & D["着"].notna() & D["_race_dt"].notna()].copy()
    D["win"] = (D["着"] == 1).astype(float)
    D["頭数"] = D.groupby("race_id")["race_id"].transform("size")
    D = D[D["頭数"] >= 8].copy().reset_index(drop=True)
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    D["lq"] = np.log(D.q.clip(EPS))
    D["bn"] = pd.to_numeric(D["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    return D, BASE


def fit(tr, te, cols):
    return np.mean([lgb.train(params(sd),
                              lgb.Dataset(tr[cols], tr.win, init_score=tr.lq.values),
                              num_boost_round=N_ROUNDS).predict(te[cols], raw_score=True)
                    for sd in SEEDS], axis=0)


def score(te, f):
    d = te.copy()
    d["f"] = f
    sc = d.f + d.lq
    e = np.exp(sc - sc.groupby(d.race_id).transform("max"))
    d["p"] = e / e.groupby(d.race_id).transform("sum")
    d["gap"] = d.p / d.q
    return d


def bets(d, PAY):
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        a = g.bn.values[k]
        dt = g["_race_dt"].iloc[0]
        rows.append((dt, PAY.get((rid, "単勝", a), 0.0)))
        if pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0:
            for j in [x for x in np.argsort(-gv) if x != k and gv[x] >= MATE_GAP][:MATE_MAX]:
                b = g.bn.values[j]
                rows.append((dt, PAY.get((rid, "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0)))
    return pd.DataFrame(rows, columns=["dt", "払戻"])


def dr2(d):
    dd = d.copy()
    dd["_rc"] = pd.factorize(dd.race_id)[0]
    dd["lp"] = np.log(dd.p.clip(EPS))
    l0 = M.null_ll(dd)
    _, lm = M.clogit(dd, ["lq"])
    _, lb = M.clogit(dd, ["lq", "lp"])
    return (1 - lb / l0) - (1 - lm / l0)


def main():
    D, BASE = load()
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    log(f"日付 {D._race_dt.min():%Y-%m-%d} 〜 {D._race_dt.max():%Y-%m-%d}\n")

    tr = D[D.年 <= 2024]
    te = D[D.年 == 2025].copy()
    log(f"学習 〜2024年（{len(tr):,}頭・固定） → 検証 2025年（{len(te):,}頭）")
    log("学習中…")
    d = score(te, fit(tr, te, BASE))
    b = bets(d, PAY)
    b["経過月"] = ((b.dt - pd.Timestamp("2025-01-01")).dt.days // 30).clip(0, 11)

    log("\n=== 学習を2024年末で止めたまま、2025年を進んでいくと ===")
    log(f"  {'経過':<10}{'点数':>7}{'的中':>6}{'ROI':>9}{'ΔR2':>9}")
    for m3, g in b.groupby(b["経過月"] // 3):
        lab = f"{int(m3)*3}-{int(m3)*3+2}か月"
        sub = d[(d._race_dt >= pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(m3) * 90))
                & (d._race_dt < pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(m3) * 90 + 90))]
        r2 = dr2(sub) if len(sub) > 3000 else np.nan
        log(f"  {lab:<10}{len(g):>7,}{int((g.払戻>0).sum()):>6}"
            f"{g.払戻.mean():>8.1f}%{r2:>9.4f}")

    # 傾きを見る（月ごとのROIが下がっていくか）
    mon = b.groupby("経過月").払戻.agg(["mean", "size", lambda x: (x > 0).sum()])
    mon.columns = ["ROI", "点数", "的中"]
    log(f"\n  {'経過月':<8}{'点数':>7}{'的中':>6}{'ROI':>9}")
    for m2, r in mon.iterrows():
        log(f"  {int(m2):>3}か月{'':<2}{int(r.点数):>7,}{int(r.的中):>6}{r.ROI:>8.1f}%")
    from scipy.stats import spearmanr
    rho, p = spearmanr(mon.index, mon.ROI)
    log(f"\n  経過月とROIの順位相関 rho={rho:+.3f} (p={p:.3f})")
    log(f"  → {'⚠ 古くなるほど落ちる傾向' if rho < -0.4 and p < 0.2 else '○ はっきりした劣化の傾向は無い'}")

    log("\n=== 比較: 毎年学習し直した場合（2025年全体）===")
    log("  10,349点 的中1,178 ROI 120.6%（うち2025年は79%）")
    log(f"  今回（2024年末で固定・2025年全体）: {len(b):,}点 "
        f"的中{int((b.払戻>0).sum())} ROI {b.払戻.mean():.1f}%")


if __name__ == "__main__":
    main()
