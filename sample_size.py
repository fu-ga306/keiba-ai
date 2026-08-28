# -*- coding: utf-8 -*-
"""その回収率を主張するのに何点必要かを計算する（2026-08-28）

なぜ要るか
  バックテストで120%が出ても、それが実力なのか偶然なのかは
  点数によって決まります。単勝は外れが大半で、たまに大きく返ってくるので
  1点あたりのばらつきが平均より遥かに大きい。この形の分布では、
  **数百点では何も言えません。**

  「何点あれば言えるか」を先に出しておくと、
  途中経過の数字に振り回されなくなります。

出すもの
  ・1点あたりの平均と標準偏差（実データから）
  ・95%下限が100%を超えるのに必要な点数（正規近似とブートストラップ）
  ・それが何か月分か
  ・いまの実測点数で、何が言えて何が言えないか

実行
  python sample_size.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

import resid_io

Z95 = 1.959963985           # 両側95%
BOOT = 4000
RNG = np.random.default_rng(20260828)


def log(m):
    print(m, flush=True)


def build():
    """check_resid.py と同じ手順で買い目の払戻を並べる。
    **同じ入力・同じ関数**を使うこと。ここがずれると全部の数字が意味を失う。"""
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["馬番"] = pd.to_numeric(d["bn"], errors="coerce")
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}

    m = {"gap_min": resid_io.AX_GAP}
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        for b in resid_io.pick_bets(g, model=m):
            rows.append({"年": int(rid[:4]), "券種": b["券種"],
                         "払戻": PAY.get((rid, b["券種"], b["組み合わせ"]), 0.0)})
    return pd.DataFrame(rows)


def need_n_normal(mu, sd, target=100.0, z=Z95):
    """正規近似での必要点数。**この分布では過大に出ます。**
    左右対称を仮定するが、実際は右に長い裾を持つため、
    下側の percentile は正規近似より内側（高い側）に来るからです。
    保守側の目安として併記するだけにします。"""
    if mu <= target:
        return None
    return int(np.ceil((z * sd / (mu - target)) ** 2))


def need_n_boot(x, target=100.0, hi=30000):
    """ブートストラップで必要点数を二分探索する。**こちらを主に使います。**
    分布の形をそのまま使うので、歪んでいても正しく出ます。"""
    if x.mean() <= target:
        return None
    if boot_lo(x, hi)[0] <= target:
        return None
    lo_n, hi_n = 100, hi
    while lo_n < hi_n:
        mid = (lo_n + hi_n) // 2
        if boot_lo(x, mid)[0] > target:
            hi_n = mid
        else:
            lo_n = mid + 1
    return lo_n


def boot_lo(x, n, reps=BOOT):
    """n点を復元抽出して、95%下限の中央的な値を見る。
    正規近似が効くかどうかの確認用（歪んだ分布では効かないことがある）。"""
    idx = RNG.integers(0, len(x), size=(reps, n))
    means = x[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def section(R, label, x, months_span):
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    log(f"\n  【{label}】")
    log(f"    点数 {n:,}   平均 {mu:.1f}%   標準偏差 {sd:.0f}%")
    log(f"    → 1点のばらつきは平均の {sd/max(mu,1e-9):.0f}倍")

    lo, hi = boot_lo(x, n)
    log(f"    いまの{n:,}点での95%区間: [{lo:.0f}%, {hi:.0f}%]")

    need = need_n_boot(x)
    if need is None:
        log("    95%下限が100%を超える点数が見つかりません。")
        log("    平均が低いか、ばらつきが大きすぎます。")
        return
    pm = n / months_span if months_span else 0
    log(f"    95%下限が100%を超えるのに必要: **{need:,}点**（ブートストラップ）")
    if pm > 0:
        log(f"    この買い方は月あたり約{pm:.0f}点 → **{need/pm:.0f}か月**")
    nn = need_n_normal(mu, sd)
    if nn:
        log(f"    参考: 正規近似だと {nn:,}点。**{nn/need:.1f}倍に見積もりすぎます**")
        log("          （右に長い裾があるので左右対称の仮定が効きません）")


def main():
    R = build()
    R["ret"] = R.払戻  # 100円賭けたときの戻り（円）＝そのまま%
    years = sorted(R.年.unique())
    months = (len(years)) * 12.0     # 年ごとに丸1年ぶんある前提
    log("=" * 60)
    log("  その回収率、何点あれば言えるか")
    log("=" * 60)
    log(f"\n  対象: {years[0]}〜{years[-1]} の{len(years)}年 = {months:.0f}か月")
    log(f"  全体 {len(R):,}点  的中 {int((R.ret>0).sum()):,}"
        f"  回収率 {R.ret.mean():.1f}%")

    section(R, "全体（単勝＋ワイド）", R.ret.values, months)
    for k, g in R.groupby("券種"):
        section(R, k, g.ret.values, months)

    log("\n" + "=" * 60)
    log("  読み方")
    log("=" * 60)
    log("""
  ・標準偏差が平均の何倍もあるのは、外れが大半でたまに大きく返るからです。
    この形では平均が落ち着くのに非常に多くの点数が要ります。
  ・「必要点数」に届くまでは、途中の回収率が100%を割っても超えても、
    **それは判断材料になりません。**
  ・逆に、必要点数が現実的でない（何十か月もかかる）と分かったら、
    回収率を主要な指標にすること自体をやめる、という判断ができます。
""")


if __name__ == "__main__":
    main()
