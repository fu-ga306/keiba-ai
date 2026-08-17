# -*- coding: utf-8 -*-
"""取れそうなレースだけ点数を増やす（2026-08-17）

考え方
  モデルはどのレースでも同じように効くわけではない。model_diag.py で
  Benter基準に対する寄与を区分別に測ってある（ROIとは独立に測った値）。

    長距離(2200m+)      10.1%   ← 最も効く
    ダート               6.3%
    マイル(1400-1799)    5.9%
    1-2勝クラス          4.8%
    少頭数(<=12)         4.3%
    多頭数(13+)          2.5%
    新馬・未勝利          1.9%
    芝                  1.3%
    中距離(1800-2199)    0.6%   ← ほぼ効かない

  この差はROIを見て決めたものではないので、判定条件として使ってよい。
  効くレースでは点数を増やし、効かないレースでは減らす（または買わない）。

事前登録（ROIを見る前に固定）
  判定に使うもの（model_diag で寄与が大きかった順に3つ ＋ モデル自身の自信）
    G1 ダート
    G2 距離2200m以上
    G3 出走12頭以下
    G4 軸のgapが3.0以上（モデルが特に強く見ている）
    G5 相手にもgap1.3以上の馬が居る（レース全体が読めている）

  点数の増やし方（2段階のみ。細かく刻むと探索になる）
    判定に通らない : 単勝1点のみ
    判定に通る     : 単勝1点 ＋ ワイド（相手gap>=1.3・最大3頭）

  比べるもの
    ① 判定なしで全部単勝1点（＝現行）
    ② 判定なしで全部 単勝＋ワイド
    ③ 各判定ごとに「通ったら増やす」
    ④ 判定を2つ以上満たしたら増やす

判定
  ・的中100本以上
  ・順列検定（判定5つ＋組み合わせを込みで）
  ・現行（157.0%）と比べてどうか

実行: python resid_gate.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAP, MATE_GAP, MATE_MAX = 2.0, 1.3, 3
N_PERM = 150
MIN_HIT = 100
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf", "距離", "出走頭数",
                              "クラス_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {}
    for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples():
        PAY[(r.race_id, r.券種, r.組み合わせ)] = r.払戻金
    races = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values.astype(float)
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        n = len(g)
        head = pd.to_numeric(g["出走頭数"], errors="coerce").iloc[0]
        races.append({
            "rid": rid, "年": int(rid[:4]), "bn": g.bn.values, "gap": gv, "ax": k,
            "dirt": bool(pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0),
            "long": bool(pd.to_numeric(g["距離"], errors="coerce").iloc[0] >= 2200),
            "few": bool((head if pd.notna(head) else n) <= 12),
        })
    return races, PAY


GATES = {
    "G1 ダート": lambda r: r["dirt"],
    "G2 2200m以上": lambda r: r["long"],
    "G3 12頭以下": lambda r: r["few"],
    "G4 軸gap>=3.0": lambda r: r["gap"][r["ax"]] >= 3.0,
    "G5 相手あり": lambda r: bool(((r["gap"] >= MATE_GAP).sum() - 1) > 0),
}


def play(races, PAY, gate=None, always_wide=False, gaps=None, n_need=1):
    """判定に通ったレースだけワイドを足す。通らなければ単勝1点。"""
    ys, vs, ext = [], [], 0
    for i, r in enumerate(races):
        g = r["gap"] if gaps is None else gaps[i]
        k = int(np.argmax(g))
        if g[k] < AX_GAP:
            continue
        a = r["bn"][k]
        y = r["rid"], r["年"]
        ys.append(r["年"])
        vs.append(PAY.get((r["rid"], "単勝", a), 0.0))
        if always_wide:
            on = True
        elif gate is None:
            on = False
        elif isinstance(gate, list):
            on = sum(1 for f in gate if f(r)) >= n_need
        else:
            on = gate(r)
        if not on:
            continue
        rest = [j for j in np.argsort(-g) if j != k and g[j] >= MATE_GAP][:MATE_MAX]
        for j in rest:
            b = r["bn"][j]
            ys.append(r["年"])
            vs.append(PAY.get((r["rid"], "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0))
            ext += 1
    return np.array(ys), np.array(vs), ext


def stat(ys, vs, lab, ext=0):
    if len(vs) < 200:
        return None
    yr = {y: vs[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
    return {"買い方": lab, "点数": len(vs), "追加": ext, "的中": int((vs > 0).sum()),
            "的中率": (vs > 0).mean() * 100, "ROI": vs.mean(),
            "100超年": sum(1 for x in yr.values() if x >= 100), "年別": yr, "_v": vs}


def main():
    races, PAY = load()
    log(f"軸gap>={AX_GAP} を満たす {len(races):,}レースが対象")
    log(f"判定に通ったら ワイド（相手gap>={MATE_GAP}・最大{MATE_MAX}頭）を追加\n")

    res = []
    res.append(stat(*play(races, PAY)[:2], "① 判定なし・単勝のみ（現行）"))
    ys, vs, e = play(races, PAY, always_wide=True)
    res.append(stat(ys, vs, "② 判定なし・全部ワイド追加", e))
    for name, f in GATES.items():
        ys, vs, e = play(races, PAY, gate=f)
        r = stat(ys, vs, f"③ {name} で追加", e)
        if r:
            res.append(r)
    for n in (2, 3):
        ys, vs, e = play(races, PAY, gate=list(GATES.values()), n_need=n)
        r = stat(ys, vs, f"④ 判定{n}つ以上で追加", e)
        if r:
            res.append(r)

    log(f"  {'買い方':<30}{'点数':>7}{'追加':>6}{'的中':>7}{'的中率':>7}{'ROI':>8}{'100超年':>8}")
    for r in res:
        if r:
            log(f"  {r['買い方']:<30}{r['点数']:>7,}{r['追加']:>6,}{r['的中']:>7}"
                f"{r['的中率']:>6.1f}%{r['ROI']:>7.1f}%{r['100超年']:>6}/5")

    log(f"\n  {'買い方':<30}{'95%区間':>18}  年別")
    for r in res:
        if not r:
            continue
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
        yr = "  ".join(f"{y}:{x:.0f}%" for y, x in r["年別"].items())
        log(f"  {r['買い方']:<30}[{np.percentile(bs,2.5):>6.1f},"
            f"{np.percentile(bs,97.5):>7.1f}]  {yr}")

    log("\n=== 判定そのものの効き目（追加したワイドだけの成績）===")
    log(f"  {'判定':<20}{'ワイド点数':>10}{'的中':>7}{'ROI':>8}")
    for name, f in GATES.items():
        ysa, vsa, _ = play(races, PAY, gate=f)
        ysb, vsb, _ = play(races, PAY)
        # 単勝は共通なので差分がワイド部分
        n = len(vsa) - len(vsb)
        if n > 100:
            w = vsa.sum() - vsb.sum()
            log(f"  {name:<20}{n:>10,}{'':>7}{w/(n*100)*100:>7.1f}%")

    ok = [r for r in res if r and r["的中"] >= MIN_HIT]
    if not ok:
        return
    best = max(ok, key=lambda x: x["ROI"])
    log(f"\n=== 順列検定（判定{len(GATES)}つ＋組み合わせを込み）===")
    log(f"  本物の最良値 {best['ROI']:.1f}%（{best['買い方']}）")
    nulls = []
    for i in range(N_PERM):
        gaps = [rng.permutation(r["gap"]) for r in races]
        b = -np.inf
        for f in list(GATES.values()) + [None]:
            ys, vs, _ = play(races, PAY, gate=f, gaps=gaps)
            if len(vs) >= 200 and (vs > 0).sum() >= MIN_HIT:
                b = max(b, vs.mean())
        for n in (2, 3):
            ys, vs, _ = play(races, PAY, gate=list(GATES.values()), n_need=n, gaps=gaps)
            if len(vs) >= 200 and (vs > 0).sum() >= MIN_HIT:
                b = max(b, vs.mean())
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
