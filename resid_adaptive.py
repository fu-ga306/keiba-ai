# -*- coding: utf-8 -*-
"""相手の評価に応じて買う点数を変える（2026-08-17）

前回（resid_nagashi.py）との違い
  前回は「相手＝2着以内確率の上位N頭」と**頭数を固定**した。N=2,3,5 のどれでも
  単勝のみ（157.0%）を超えられなかった。最良は 単勝+馬連 相手=p2上位3 の128.6%で、
  95%区間は[95.1, 173.9]と100%を跨いだ。

  頭数を固定すると、モデルが評価していない馬まで無理に相手に入れてしまう。
  評価の高い相手が1頭しかいないレースでも3頭買うので、余計な2点が入る。

  今回は**相手にも評価の条件を課す**。相手自身の gap がしきい値を超えた馬だけを
  買う。条件を満たす相手が居なければ流さない（単勝だけ、または見送り）。
  つまりレースごとに点数が変わる。

事前登録（ROIを見る前に固定。あとから足さない）
  軸  : 残差モデルの gap が最大の1頭。gap >= 2.0（現行の単勝と同じ）
  相手: 軸以外で、自分の gap >= しきい値 の馬すべて。上限5頭（点数の暴走を防ぐ）
        しきい値は 1.0 / 1.3 / 1.6 の3通り
  券種: ワイド または 馬連
  単勝: 併用する / しない の2通り
  → 3 × 2 × 2 = 12通り。これで全部。

判定（3つとも満たすこと）
  ① 単勝のみ（1,891点・的中203・157.0%）を上回る
  ② 順列検定を通る（12通りから最良を選ぶことを込みで）
  ③ 的中が100本以上

⚠ 払戻は jv_payouts（JRA-VANの確定払戻）で照合する。推定オッズは使わない。
⚠ 馬番は2桁ゼロ埋め。揃えないと1桁馬番の買い目が照合できない。

実行: python resid_adaptive.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAP = 2.0
MATE_THS = [1.0, 1.3, 1.6]
KINDS = ["ワイド", "馬連"]
WITH_TAN = [False, True]
MATE_MAX = 5
N_PERM = 300
MIN_HIT = 100
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {}
    for r in jv[jv.券種.isin(("単勝", "ワイド", "馬連"))].itertuples():
        PAY[(r.race_id, r.券種, r.組み合わせ)] = r.払戻金
    return d, PAY


def build(d, PAY, mate_th, kind, with_tan, gcol="gap"):
    """相手の評価で点数が変わる買い方。払戻の列を返す。"""
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = pd.to_numeric(g[gcol], errors="coerce")
        if not gv.notna().any() or float(gv.max()) < AX_GAP:
            continue
        ax = g.loc[gv.idxmax()]
        a = ax.bn
        rest = g[(g.bn != a) & (gv >= mate_th)]
        if rest.empty:
            continue                        # 条件を満たす相手が居なければ買わない
        mates = rest.nlargest(min(MATE_MAX, len(rest)), gcol).bn.tolist()
        y = int(rid[:4])
        if with_tan:
            rows.append((y, PAY.get((rid, "単勝", a), 0.0)))
        for b in mates:
            c = f"{min(a,b)}-{max(a,b)}"
            rows.append((y, PAY.get((rid, kind, c), 0.0)))
    return rows


def stat(rows, lab):
    if len(rows) < 100:
        return None
    v = np.array([p for _, p in rows])
    ys = np.array([y for y, _ in rows])
    yr = {y: v[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
    return {"買い方": lab, "点数": len(v), "的中": int((v > 0).sum()),
            "的中率": (v > 0).mean() * 100, "ROI": v.mean(),
            "年別": yr, "100超年": sum(1 for x in yr.values() if x >= 100), "_v": v}


def main():
    d, PAY = load()
    nr = d.race_id.nunique()
    log(f"検体 {len(d):,}頭 / {nr:,}レース")
    log(f"軸: gap>={AX_GAP}（現行と同じ）／相手: 自分の gap >= しきい値・最大{MATE_MAX}頭")
    log(f"しきい値 {MATE_THS} × 券種 {KINDS} × 単勝併用 {WITH_TAN} = 12通り\n")

    res = []
    for th in MATE_THS:
        for kind in KINDS:
            for wt in WITH_TAN:
                lab = f"{'単勝+' if wt else ''}{kind} 相手gap>={th}"
                r = stat(build(d, PAY, th, kind, wt), lab)
                if r:
                    res.append(r)
    log(f"  {'買い方':<26}{'点数':>8}{'的中':>7}{'的中率':>8}{'ROI':>8}{'100超年':>8}  年別")
    for r in sorted(res, key=lambda x: -x["ROI"]):
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['買い方']:<26}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>7.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5  {yr}")

    log("\n=== 1レースあたり何点になるか ===")
    log(f"  {'しきい値':<12}{'買うレース':>10}{'総点数':>9}{'1R点数':>9}")
    for th in MATE_THS:
        rows = build(d, PAY, th, "ワイド", False)
        nrace = sum(1 for _ in {rid for rid, g in d.groupby("race_id", sort=False)
                                if pd.to_numeric(g.gap, errors="coerce").max() >= AX_GAP
                                and ((g.gap >= th) & (g.bn != g.loc[
                                    pd.to_numeric(g.gap, errors='coerce').idxmax()].bn)).any()})
        if rows:
            log(f"  gap>={th:<8}{nrace:>10,}{len(rows):>9,}{len(rows)/max(nrace,1):>9.1f}")

    log("\n=== 上位5つの95%区間 ===")
    log(f"  {'買い方':<26}{'ROI':>8}{'95%区間':>18}")
    for r in sorted(res, key=lambda x: -x["ROI"])[:5]:
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(3000)])
        log(f"  {r['買い方']:<26}{r['ROI']:>7.1f}%"
            f"  [{np.percentile(bs,2.5):>6.1f},{np.percentile(bs,97.5):>7.1f}]")

    log("\n=== 比較: 単勝のみ（現行）===")
    log("  1,891点  的中203  10.7%  ROI 157.0%  95%区間[106.6, 220.5]  100超年 3/5")

    cand = [r for r in res if r["的中"] >= MIN_HIT]
    real = max((r["ROI"] for r in cand), default=np.nan)
    if not np.isfinite(real):
        log("\n的中100本以上の構成が無いので順列検定は行いません")
        return
    log(f"\n=== 順列検定（12通りから最良を選ぶことを込み・的中{MIN_HIT}本以上）===")
    log(f"  本物の最良値 {real:.1f}%  偽物を{N_PERM}回…")
    d2 = d.copy()
    nulls = []
    for i in range(N_PERM):
        d2["gap"] = d.groupby("race_id")["gap"].transform(
            lambda s: rng.permutation(s.values))
        best = -np.inf
        for th in MATE_THS:
            for kind in KINDS:
                for wt in WITH_TAN:
                    r = stat(build(d2, PAY, th, kind, wt), "x")
                    if r and r["的中"] >= MIN_HIT and r["ROI"] > best:
                        best = r["ROI"]
        if np.isfinite(best):
            nulls.append(best)
        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{N_PERM}")
    if nulls:
        nulls = np.array(nulls)
        p = float((nulls >= real).mean())
        log(f"\n  偽物の中央値 {np.median(nulls):.1f}%  最大 {nulls.max():.1f}%")
        log(f"  p値 = {p:.4f} → {'✅ 偶然では説明しにくい' if p < 0.05 else '⚠ 偶然の範囲'}")


if __name__ == "__main__":
    main()
