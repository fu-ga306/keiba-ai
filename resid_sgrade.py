# -*- coding: utf-8 -*-
"""★軸とS評価を組み合わせる（2026-08-17）

なぜ組み合わせが効きそうか
  この2つは**違うものを測っている**。

    S評価  : 絶対的に来る確率が高い馬。馬券内87.0%・勝率57.5%・中央オッズ1.5倍。
             市場込みで作っているので、ほぼ人気馬になる。
             単体で買っても複勝85%程度で儲からない（オッズが低すぎる）。

    ★軸(gap): 市場が過小評価している馬。gap＝モデルの予測確率÷市場の確率。
             中穴が多く、当たりにくいが配当が大きい。

  つまり「確実に来る馬」と「妙味のある馬」。この2頭を結ぶワイド・馬連なら、
  当たりやすさ（S側）と配当（★側）を両取りできる可能性がある。

事前登録（ROIを見る前に固定。あとから足さない）
  軸  : 残差モデルの gap が最大の1頭・gap>=1.5（現行と同じ）
  相手: 評価が S / A以上 / B以上 の馬（軸を除く・最大3頭）
        評価は grade_calib.pkl（市場＋モデルの2次元較正）で付ける
  券種: ワイド / 馬連
  → 3(評価) × 2(券種) = 6通り。単勝併用あり/なしで12通り。これで全部。

  参考として次も測る（比べる基準）
    ・S評価の複勝だけ（S単体では儲からないことの確認）
    ・現行（軸の単勝＋ダートならワイド）

判定
  ① 現行（軸gap>=1.5 で 120.6%）を上回るか
  ② 順列検定（12通りから最良を選ぶことを込みで）
  ③ 的中100本以上

⚠ 払戻は jv_payouts の実払戻。馬番は2桁ゼロ埋め。

実行: python resid_sgrade.py
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAP = 1.5
MATE_MAX = 3
GRADE_TH = [("S", 1.276), ("A", 0.791), ("B", 0.437)]
N_PERM = 150
MIN_HIT = 100
EPS = 1e-6
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def logit(v):
    v = np.clip(v, EPS, 1 - EPS)
    return np.log(v / (1 - v))


def load():
    # gap は残差モデル、評価は bet_cache の較正済み確率から作る
    g = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    g["gap"] = g.p1 / g.q
    bc = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                    for y in YEARS], ignore_index=True)
    d = g.merge(bc[["race_id", "bn", "c_win", "c_top3", "odds"]].rename(
        columns={"odds": "odds_bc"}), on=["race_id", "bn"], how="inner")
    with open("grade_calib.pkl", "rb") as fh:
        cal = pickle.load(fh)
    inv = 1.0 / d.odds.clip(lower=1.01)
    q = inv / inv.groupby(d.race_id).transform("sum")
    lm = logit(q)
    X3 = pd.DataFrame({"lm": lm, "l3": logit(d.c_top3)})
    X3["i3"] = X3.lm * X3.l3
    X1 = pd.DataFrame({"lm": lm, "l1": logit(d.c_win)})
    X1["i1"] = X1.lm * X1.l1
    d["gscore"] = (cal["m3"].predict_proba(X3[cal["F3"]])[:, 1]
                   + cal["m1"].predict_proba(X1[cal["F1"]])[:, 1])
    d["grade"] = np.select([d.gscore >= GRADE_TH[0][1], d.gscore >= GRADE_TH[1][1],
                            d.gscore >= GRADE_TH[2][1]], ["S", "A", "B"], "D")
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "複勝", "ワイド", "馬連"))].itertuples()}
    return d, PAY


RANK = {"S": 0, "A": 1, "B": 2, "D": 3}


def build(d, PAY, mate_grade, kind, with_tan, gcol="gap"):
    """軸(gap最大) × 相手(評価がmate_grade以上) の買い目を作る。"""
    lim = RANK[mate_grade]
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = pd.to_numeric(g[gcol], errors="coerce")
        if not gv.notna().any() or float(gv.max()) < AX_GAP:
            continue
        k = gv.idxmax()
        a = g.loc[k, "bn"]
        y = int(rid[:4])
        if with_tan:
            rows.append((y, PAY.get((rid, "単勝", a), 0.0)))
        rest = g.drop(index=k)
        rest = rest[rest.grade.map(RANK) <= lim]
        if rest.empty:
            continue
        for b in rest.nlargest(min(MATE_MAX, len(rest)), "gscore").bn:
            c = f"{min(a,b)}-{max(a,b)}"
            rows.append((y, PAY.get((rid, kind, c), 0.0)))
    return rows


def stat(rows, lab):
    if len(rows) < 150:
        return None
    v = np.array([p for _, p in rows])
    ys = np.array([y for y, _ in rows])
    yr = {y: v[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
    return {"買い方": lab, "点数": len(v), "的中": int((v > 0).sum()),
            "的中率": (v > 0).mean() * 100, "ROI": v.mean(), "年別": yr,
            "100超年": sum(1 for x in yr.values() if x >= 100), "_v": v}


def main():
    d, PAY = load()
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")
    log(f"評価の分布: " + "  ".join(
        f"{k}{v:,}({v/len(d)*100:.1f}%)" for k, v in d.grade.value_counts().items()))
    log(f"軸gap>={AX_GAP} を満たすレース "
        f"{(d.groupby('race_id').gap.max() >= AX_GAP).sum():,}\n")

    log("=== ★軸 × 評価の相手（ワイド・馬連）===")
    res = []
    for mg in ("S", "A", "B"):
        for kind in ("ワイド", "馬連"):
            for wt in (False, True):
                lab = f"{'単勝+' if wt else ''}{kind} 相手={mg}以上"
                r = stat(build(d, PAY, mg, kind, wt), lab)
                if r:
                    res.append(r)
    log(f"  {'買い方':<26}{'点数':>8}{'的中':>7}{'的中率':>7}{'ROI':>8}{'100超年':>8}  年別")
    for r in sorted(res, key=lambda x: -x["ROI"]):
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['買い方']:<26}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>6.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5  {yr}")

    log("\n=== 上位5つの95%区間 ===")
    log(f"  {'買い方':<26}{'ROI':>8}{'95%区間':>18}")
    for r in sorted(res, key=lambda x: -x["ROI"])[:5]:
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2500)])
        log(f"  {r['買い方']:<26}{r['ROI']:>7.1f}%"
            f"  [{np.percentile(bs,2.5):>6.1f},{np.percentile(bs,97.5):>7.1f}]")

    log("\n=== 比較用 ===")
    s = d[d.grade == "S"]
    sv = np.array([PAY.get((r.race_id, "複勝", r.bn), 0.0) for r in s.itertuples()])
    log(f"  S評価の複勝だけ            {len(sv):,}点 的中{int((sv>0).sum())}"
        f"（{(sv>0).mean()*100:.1f}%）ROI {sv.mean():.1f}%  中央オッズ{s.odds.median():.1f}倍")
    log("  現行（軸gap>=1.5 単勝＋ダートならワイド）  10,349点 的中1,178 ROI 120.6%")

    log("\n=== ★軸とS評価は同じ馬になるか ===")
    same = 0
    n = 0
    for rid, g in d.groupby("race_id", sort=False):
        gv = pd.to_numeric(g["gap"], errors="coerce")
        if not gv.notna().any() or float(gv.max()) < AX_GAP:
            continue
        n += 1
        if g.loc[gv.idxmax(), "grade"] == "S":
            same += 1
    log(f"  軸がS評価でもある: {same:,}/{n:,} = {same/max(n,1)*100:.1f}%")
    log("  → 低いほど『別のものを見ている』＝組み合わせる意味がある")

    cand = [r for r in res if r["的中"] >= MIN_HIT]
    if not cand:
        log("\n的中100本以上の構成が無いので順列検定は行いません")
        return
    best = max(cand, key=lambda x: x["ROI"])
    log(f"\n=== 順列検定（12通りから最良を選ぶことを込み）===")
    log(f"  本物の最良値 {best['ROI']:.1f}%（{best['買い方']}）")
    nulls = []
    d2 = d.copy()
    for i in range(N_PERM):
        d2["gap"] = d.groupby("race_id")["gap"].transform(
            lambda s: rng.permutation(s.values))
        b = -np.inf
        for mg in ("S", "A", "B"):
            for kind in ("ワイド", "馬連"):
                for wt in (False, True):
                    r = stat(build(d2, PAY, mg, kind, wt), "x")
                    if r and r["的中"] >= MIN_HIT and r["ROI"] > b:
                        b = r["ROI"]
        if np.isfinite(b):
            nulls.append(b)
        if (i + 1) % 50 == 0:
            log(f"    {i+1}/{N_PERM}")
    nulls = np.array(nulls)
    p = float((nulls >= best["ROI"]).mean())
    log(f"\n  偽物の中央値 {np.median(nulls):.1f}%  最大 {nulls.max():.1f}%")
    log(f"  p値 = {p:.4f} → {'✅ 偶然では説明しにくい' if p < 0.05 else '⚠ 偶然の範囲'}")


if __name__ == "__main__":
    main()
