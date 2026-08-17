# -*- coding: utf-8 -*-
"""残差モデルに距離分割を入れて効果を測る（2026-08-18）

なぜ試すか
  本番のMFモデルは1900mを境に長距離／短距離で別モデルにして、
  クリーンデータのwalk-forward検証で全体1.6倍に改善した。
  残差モデルにはこの分割が入っていない。同じ効果が出るかもしれない。

  短距離では騎手・厩舎の列を落とすのが本番の設計（mf_cols_for）。
  短距離は展開の影響が大きく、騎手厩舎の累積成績がノイズになるため。

事前登録（ROIを見る前に固定。あとから足さない）
  ① 分割なし（現行）
  ② 1900mで分割・両方とも全特徴量
  ③ 1900mで分割・短距離は騎手厩舎を除く（本番MFと同じ設計）

  評価は2つ。
    ΔR²  … Benterと同じ物差し（市場だけのときからの改善）。基準0.0178
    ROI  … 確定した買い方（軸gap>=1.5 単勝＋ダートならワイド）での回収率

判定
  ΔR²とROIの**両方**が現行を上回ったときだけ採用する。
  片方だけなら採らない（片方だけの改善は偶然で起きる）。

実行: python resid_split.py
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
SPLIT = 1900
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
    try:
        from market_free_model import mf_cols_for
        SHORT = [c for c in mf_cols_for(0) if c in head.columns]
    except Exception:
        SHORT = BASE
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
    D["_long"] = pd.to_numeric(D["距離"], errors="coerce") >= SPLIT
    return D, BASE, SHORT


def fit_all(tr, te, cols):
    f = np.mean([lgb.train(params(sd),
                           lgb.Dataset(tr[cols], tr.win, init_score=tr.lq.values),
                           num_boost_round=N_ROUNDS).predict(te[cols], raw_score=True)
                 for sd in SEEDS], axis=0)
    return f


def run(D, BASE, SHORT, mode):
    """walk-forward で f（特徴量ぶんのスコア）を出す。"""
    out = pd.Series(np.nan, index=D.index)
    for y in YEARS:
        tr, tem = D[D.年 < y], D.年 == y
        if len(tr) < 5000 or not tem.any():
            continue
        te = D[tem]
        if mode == "none":
            out[tem] = fit_all(tr, te, BASE)
        else:
            cs = BASE if mode == "same" else SHORT
            for is_long, cols in ((True, BASE), (False, cs)):
                trm = tr[tr._long == is_long]
                tem2 = te[te._long == is_long]
                if len(trm) < 2000 or tem2.empty:
                    continue
                out[tem2.index] = fit_all(trm, tem2, cols)
    return out


def evaluate(D, f, PAY, lab):
    d = D.copy()
    d["f"] = f
    d = d[d.f.notna()].copy()
    sc = d.f + d.lq
    e = np.exp(sc - sc.groupby(d.race_id).transform("max"))
    d["p"] = e / e.groupby(d.race_id).transform("sum")
    d["gap"] = d.p / d.q
    # ΔR²
    dd = d.copy()
    dd["_rc"] = pd.factorize(dd.race_id)[0]
    dd["lp"] = np.log(dd.p.clip(EPS))
    l0 = M.null_ll(dd)
    _, lm = M.clogit(dd, ["lq"])
    _, lb = M.clogit(dd, ["lq", "lp"])
    dr2 = (1 - lb / l0) - (1 - lm / l0)
    # ROI（確定した買い方）
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
    D, BASE, SHORT = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")
    log(f"特徴量 全{len(BASE)}列 / 短距離用{len(SHORT)}列（差{len(BASE)-len(SHORT)}列）")
    log(f"{SPLIT}m以上 {D._long.sum():,}頭 / 未満 {(~D._long).sum():,}頭\n")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}

    res = []
    for mode, lab in (("none", "① 分割なし（現行）"),
                      ("same", "② 1900m分割・全特徴量"),
                      ("short", "③ 1900m分割・短距離は騎手厩舎除く")):
        log(f"{lab} 学習中…")
        f = run(D, BASE, SHORT, mode)
        r = evaluate(D, f, PAY, lab)
        res.append(r)
        log(f"   ΔR² {r['ΔR2']:.4f}（Benter比{r['Benter比']:.1f}%）  "
            f"{r['点数']:,}点 的中{r['的中']} ROI {r['ROI']:.1f}%")

    log("\n" + "=" * 68)
    log(f"  {'作り方':<30}{'ΔR2':>8}{'Benter比':>9}{'点数':>8}{'的中':>7}{'ROI':>8}{'100超年':>8}")
    for r in res:
        log(f"  {r['作り方']:<30}{r['ΔR2']:>8.4f}{r['Benter比']:>8.1f}%"
            f"{r['点数']:>8,}{r['的中']:>7}{r['ROI']:>7.1f}%{r['超年']:>6}/5")
    log(f"\n  {'作り方':<30}{'95%区間':>18}  年別")
    for r in res:
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['作り方']:<30}[{r['下']:>6.1f},{r['上']:>7.1f}]  {yr}")

    base = res[0]
    log("\n=== 判定（ΔR²とROIの両方が現行を上回ったときだけ採用）===")
    for r in res[1:]:
        a = r["ΔR2"] > base["ΔR2"]
        b = r["ROI"] > base["ROI"]
        log(f"  {r['作り方']:<30} ΔR² {'○' if a else '×'}  ROI {'○' if b else '×'}"
            f"  → {'✅ 採用候補' if a and b else '不採用'}")


if __name__ == "__main__":
    main()
