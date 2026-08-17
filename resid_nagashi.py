# -*- coding: utf-8 -*-
"""残差モデルの軸から流す買い方を検証する（2026-08-17）

前回(resid_kinds.py)との違い
  前回は「券種を期待値で横並びに比べて最良を選ぶ」形にして失敗した。
  期待値だけで比べると必ず分散が最大の券種（馬連・馬単）が選ばれ、
  単勝・複勝は5年で各1回しか選ばれなかった。実質「常に馬単を買う」になった。

  今回は違う。**軸は今と同じ方法で決める**（残差モデルのgapが最大で2.0以上）。
  そこから相手に流すだけなので、券種同士を比べる必要がない。
  さらに、買う組み合わせが決まっているので**実際の払戻でそのまま測れる**。
  馬連・ワイドの事前オッズが無くても検証できる（前回はここで詰まった）。

事前登録（ROIを見る前に固定。あとから足さない）
  軸 : 残差モデルの gap が最大の1頭。gap >= 2.0 のときだけ買う（現行と同じ）
  相手: 残差モデルの「2着以内確率(p2)」または「3着以内確率(p3)」の上位N頭
        （軸を除く）。N は 2 / 3 / 5 の3通り
  券種: ワイド または 馬連
  → 2(相手基準) × 3(N) × 2(券種) = 12通り。これで全部。増やさない。

  比較用に、単勝との併用も見る（軸の単勝1点 ＋ 流し）。

判定
  ① 単勝のみ（1,891点・的中203・157.1%）を上回るか
  ② 順列検定（12通りから最良を選ぶことを込みで）を通るか
  ③ 的中が100本以上あるか

⚠ 12通りから最良を選ぶので、その分を割り引かないと意味がない。
  順列検定は「12通り全部を試して最良を取る」ところまで模擬する。

実行: python resid_nagashi.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_MIN = 2.0
N_PERM = 300
MIN_HIT = 100
rng = np.random.default_rng(20260817)

MATES = ["p2", "p3"]
NS = [2, 3, 5]
KINDS = ["ワイド", "馬連"]


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


def build(d, PAY, mate, n, kind, with_tan=False, gap_col="gap"):
    """事前登録した買い方で、実際の払戻を並べる。

    馬番は2桁ゼロ埋め。jv_payouts が "09-14" 形式なので揃えないと照合できない
    （2026-08-16に的中54本が11本に見えた事故の再発防止）。
    """
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = pd.to_numeric(g[gap_col], errors="coerce")
        if not gv.notna().any():
            continue
        ax = g.loc[gv.idxmax()]
        if float(gv.max()) < GAP_MIN:
            continue
        a = ax.bn
        rest = g[g.bn != a]
        if rest.empty:
            continue
        mates = rest.nlargest(min(n, len(rest)), mate).bn.tolist()
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
    hit = int((v > 0).sum())
    roi = v.mean()
    yr = {y: v[ys == y].mean() for y in YEARS if (ys == y).sum() > 20}
    return {"買い方": lab, "点数": len(v), "的中": hit, "的中率": hit / len(v) * 100,
            "ROI": roi, "年別": yr, "100超年": sum(1 for x in yr.values() if x >= 100),
            "_v": v}


def main():
    d, PAY = load()
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")
    log(f"軸: gap>={GAP_MIN} の1頭（現行の単勝と同じ選び方）")
    log(f"相手: {MATES} の上位 {NS} 頭 / 券種: {KINDS} → {len(MATES)*len(NS)*len(KINDS)}通り\n")

    res = []
    for mate in MATES:
        for n in NS:
            for kind in KINDS:
                lab = f"{kind} 相手={mate}上位{n}"
                r = stat(build(d, PAY, mate, n, kind), lab)
                if r:
                    res.append(r)
    log("=== 流しのみ ===")
    log(f"  {'買い方':<24}{'点数':>8}{'的中':>7}{'的中率':>8}{'ROI':>8}{'100超年':>8}  年別")
    for r in sorted(res, key=lambda x: -x["ROI"]):
        yr = "  ".join(f"{y}:{v:.0f}%" for y, v in r["年別"].items())
        log(f"  {r['買い方']:<24}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>7.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5  {yr}")

    log("\n=== 単勝＋流し（軸の単勝も一緒に買う）===")
    res2 = []
    for mate in MATES:
        for n in NS:
            for kind in KINDS:
                lab = f"単勝+{kind} 相手={mate}上位{n}"
                r = stat(build(d, PAY, mate, n, kind, with_tan=True), lab)
                if r:
                    res2.append(r)
    log(f"  {'買い方':<26}{'点数':>8}{'的中':>7}{'的中率':>8}{'ROI':>8}{'100超年':>8}")
    for r in sorted(res2, key=lambda x: -x["ROI"]):
        log(f"  {r['買い方']:<26}{r['点数']:>8,}{r['的中']:>7}{r['的中率']:>7.1f}%"
            f"{r['ROI']:>7.1f}%{r['100超年']:>6}/5")

    log("\n=== 比較: 単勝のみ（現行）===")
    tan = stat([(int(r[0]), r[1]) for r in
                [(rid[:4], PAY.get((rid, "単勝",
                  g.loc[pd.to_numeric(g.gap, errors="coerce").idxmax()].bn), 0.0))
                 for rid, g in d.groupby("race_id", sort=False)
                 if pd.to_numeric(g.gap, errors="coerce").max() >= GAP_MIN]], "単勝のみ")
    if tan:
        log(f"  {tan['点数']:,}点  的中{tan['的中']}  {tan['的中率']:.1f}%"
            f"  ROI {tan['ROI']:.1f}%  100超年 {tan['100超年']}/5")

    # ── 95%区間 ─────────────────────────────────────
    log("\n=== 上位5つの95%区間 ===")
    log(f"  {'買い方':<26}{'ROI':>8}{'95%区間':>18}")
    for r in sorted(res + res2, key=lambda x: -x["ROI"])[:5]:
        v = r["_v"]
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(3000)])
        log(f"  {r['買い方']:<26}{r['ROI']:>7.1f}%"
            f"  [{np.percentile(bs,2.5):>6.1f},{np.percentile(bs,97.5):>7.1f}]")

    # ── 順列検定（12通りから最良を選ぶことを込みで）────────────
    log(f"\n=== 順列検定（{len(MATES)*len(NS)*len(KINDS)}通りから最良を選ぶことを込みで）===")
    real = max((r["ROI"] for r in res if r["的中"] >= MIN_HIT), default=np.nan)
    log(f"  本物の最良値（的中{MIN_HIT}本以上）: {real:.1f}%")
    log(f"  偽物を{N_PERM}回作ります…")
    d2 = d.copy()
    nulls = []
    for i in range(N_PERM):
        # レース内でモデルのスコアをシャッフル（モデルが何も知らない状態）
        for c in ("gap", "p2", "p3"):
            d2[c] = d.groupby("race_id")[c].transform(
                lambda s: rng.permutation(s.values))
        best = -np.inf
        for mate in MATES:
            for n in NS:
                for kind in KINDS:
                    r = stat(build(d2, PAY, mate, n, kind), "x")
                    if r and r["的中"] >= MIN_HIT and r["ROI"] > best:
                        best = r["ROI"]
        if np.isfinite(best):
            nulls.append(best)
        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{N_PERM}")
    if nulls:
        nulls = np.array(nulls)
        p = float((nulls >= real).mean())
        log(f"\n  偽物の中央値 {np.median(nulls):.1f}%  95%点 {np.percentile(nulls,95):.1f}%"
            f"  最大 {nulls.max():.1f}%")
        log(f"  p値 = {p:.4f}")
        log(f"  → {'✅ 偶然では説明しにくい' if p < 0.05 else '⚠ 偶然の範囲。採用しない'}")


if __name__ == "__main__":
    main()
