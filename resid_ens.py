# -*- coding: utf-8 -*-
"""残差モデルをアンサンブル化して効果を測る（2026-08-18）

なぜ試すか
  いまの残差モデルは LightGBM のみ（3シード平均）。
  本番のMFモデルは LGB5シード + XGB2 + CatBoost2 のアンサンブル。
  種類を混ぜると学習のブレが減り、精度が上がることがある。

  期待は大きくない。本番MFで測ったときアンサンブルは単一LGBよりΔR²が低く
  （0.0001 vs 0.0003）、距離分割も残差モデルには効かなかった
  （ΔR² 0.0044 → 0.0025）。残差モデルは「小さな信号を広く集める」性質なので
  手を加えると壊れやすい。

  ただしアンサンブルは距離分割と違い**データを分けない**ので、
  「信号が薄まる」害は無いはず。測る価値はある。

市場を出発点に固定する方法（3種で書き方が違う）
  LightGBM : Dataset(init_score=...)
  XGBoost  : DMatrix(base_margin=...)
  CatBoost : fit(baseline=...)
  いずれも「log(市場確率)からのズレだけを学ぶ」形になる。
  予測時は出発点を0にして、特徴量ぶんの寄与だけを取り出す。

事前登録（ROIを見る前に固定。あとから足さない）
  (1) LGB 3シード（現行）
  (2) LGB 3シード + XGB 2シード
  (3) LGB 3シード + XGB 2シード + CatBoost 2シード
  合成は raw score の単純平均。重み付けはしない（重みを探すと探索になる）。

判定
  ΔR²とROIの**両方**が現行を上回ったときだけ採用する。

実行: python resid_ens.py
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
SEEDS_LGB = [42, 7, 123]
SEEDS_XC = [42, 7]
N_ROUNDS = 600
AX_GAP, MATE_GAP, MATE_MAX = 1.5, 1.3, 3
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


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
    D["bn"] = pd.to_numeric(D["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    return D, BASE


def f_lgb(tr, te, cols):
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
             num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    out = []
    for sd in SEEDS_LGB:
        q = dict(p, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
        m = lgb.train(q, lgb.Dataset(tr[cols], tr.win, init_score=tr.lq.values),
                      num_boost_round=N_ROUNDS)
        out.append(m.predict(te[cols], raw_score=True))
    return out


def f_xgb(tr, te, cols):
    import xgboost as xgb
    dtr = xgb.DMatrix(tr[cols], label=tr.win.values, base_margin=tr.lq.values)
    dte = xgb.DMatrix(te[cols], base_margin=np.zeros(len(te)))
    out = []
    for sd in SEEDS_XC:
        p = dict(objective="binary:logistic", eta=0.03, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8, seed=sd, verbosity=0)
        b = xgb.train(p, dtr, num_boost_round=N_ROUNDS)
        out.append(b.predict(dte, output_margin=True))
    return out


def f_cat(tr, te, cols):
    from catboost import CatBoostClassifier, Pool
    out = []
    for sd in SEEDS_XC:
        m = CatBoostClassifier(iterations=N_ROUNDS, learning_rate=0.03, depth=5,
                               loss_function="Logloss", verbose=False,
                               random_seed=sd, allow_writing_files=False)
        m.fit(Pool(tr[cols], tr.win, baseline=tr.lq.values))
        r = m.predict(Pool(te[cols], baseline=np.zeros(len(te))),
                      prediction_type="RawFormulaVal")
        out.append(np.asarray(r).ravel())
    return out


def run(D, BASE, mode):
    out = pd.Series(np.nan, index=D.index)
    for y in YEARS:
        tr, tem = D[D.年 < y], D.年 == y
        if len(tr) < 5000 or not tem.any():
            continue
        te = D[tem]
        fs = f_lgb(tr, te, BASE)
        if mode in ("lx", "lxc"):
            fs = fs + f_xgb(tr, te, BASE)
        if mode == "lxc":
            fs = fs + f_cat(tr, te, BASE)
        out[tem] = np.mean(fs, axis=0)
        log(f"    {y} 完了（{len(fs)}モデル平均）")
    return out


def evaluate(D, f, PAY, lab):
    d = D.copy()
    d["f"] = f
    d = d[d.f.notna()].copy()
    sc = d.f + d.lq
    e = np.exp(sc - sc.groupby(d.race_id).transform("max"))
    d["p"] = e / e.groupby(d.race_id).transform("sum")
    d["gap"] = d.p / d.q
    dd = d.copy()
    dd["_rc"] = pd.factorize(dd.race_id)[0]
    dd["lp"] = np.log(dd.p.clip(EPS))
    l0 = M.null_ll(dd)
    _, lm = M.clogit(dd, ["lq"])
    _, lb = M.clogit(dd, ["lq", "lp"])
    dr2 = (1 - lb / l0) - (1 - lm / l0)
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        a = g.bn.values[k]
        y = int(rid[:4])
        rows.append((y, PAY.get((rid, "単勝", a), 0.0)))
        if pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0:
            for j in [x for x in np.argsort(-gv) if x != k and gv[x] >= MATE_GAP][:MATE_MAX]:
                b = g.bn.values[j]
                rows.append((y, PAY.get((rid, "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0)))
    v = np.array([r[1] for r in rows])
    ys = np.array([r[0] for r in rows])
    bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
    return {"作り方": lab, "ΔR2": dr2, "Benter比": dr2 / 0.0178 * 100,
            "点数": len(v), "的中": int((v > 0).sum()), "ROI": v.mean(),
            "下": np.percentile(bs, 2.5), "上": np.percentile(bs, 97.5),
            "年別": {y: v[ys == y].mean() for y in YEARS},
            "超年": sum(1 for y in YEARS if v[ys == y].mean() >= 100)}


def main():
    D, BASE = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース  特徴量{len(BASE)}列\n")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    res = []
    for mode, lab in (("l", "(1) LGB3シード（現行）"),
                      ("lx", "(2) LGB3 + XGB2"),
                      ("lxc", "(3) LGB3 + XGB2 + Cat2")):
        log(f"{lab} 学習中…")
        r = evaluate(D, run(D, BASE, mode), PAY, lab)
        res.append(r)
        log(f"   ΔR² {r['ΔR2']:.4f}（Benter比{r['Benter比']:.1f}%）"
            f"  {r['点数']:,}点 的中{r['的中']} ROI {r['ROI']:.1f}%\n")
    log("=" * 66)
    log(f"  {'作り方':<26}{'ΔR2':>8}{'Benter比':>9}{'点数':>8}{'的中':>7}{'ROI':>8}{'100超年':>8}")
    for r in res:
        log(f"  {r['作り方']:<26}{r['ΔR2']:>8.4f}{r['Benter比']:>8.1f}%"
            f"{r['点数']:>8,}{r['的中']:>7}{r['ROI']:>7.1f}%{r['超年']:>6}/5")
    log(f"\n  {'作り方':<26}{'95%区間':>18}  年別")
    for r in res:
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['作り方']:<26}[{r['下']:>6.1f},{r['上']:>7.1f}]  {yr}")
    base = res[0]
    log("\n=== 判定（ΔR²とROIの両方が現行を上回ったときだけ採用）===")
    for r in res[1:]:
        a, b = r["ΔR2"] > base["ΔR2"], r["ROI"] > base["ROI"]
        log(f"  {r['作り方']:<26} ΔR² {'○' if a else '×'}  ROI {'○' if b else '×'}"
            f"  → {'✅ 採用候補' if a and b else '不採用'}")


if __name__ == "__main__":
    main()
