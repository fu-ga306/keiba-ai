# -*- coding: utf-8 -*-
"""残差モデルの結果が偶然でないかを順列検定で確かめる（2026-08-17）

測ったもの（resid_pred.csv）
  gap>=2.0 で 2,806点・的中276・ROI 137.5%・95%区間[102.4, 180.2]
  年別 128/112/107/227/136 で5年すべて100%超。

  数字は良いが、この形で8回騙されている。採用の前に順列検定を通す。

検定のやり方
  モデルのスコア（gap）をレース内でシャッフルする。つまり「モデルは
  何も知らない」状態を作る。特徴量と着順の対応だけを壊し、オッズ分布や
  頭数やレース構成はそのまま残す。

  その状態で**同じ5段階のしきい値をすべて試し、最良のROI**を記録する。
  これを何度も繰り返して、本物の最良値がどのくらい珍しいかを見る。

  ⚠ 「最良を選ぶ」ところまで含めて模擬するのが要点。本物側でも5段階から
    最良を選んでいるので、偽物側も同じ選び方をしないと不公平になる。
  ⚠ 的中数の下限を設ける。的中が数本のマスは偶然だけで極端な値を出すので、
    これを入れないと偽物側の最良値が跳ね上がって検定が甘くなる
    （2026-08-16に実際に起きた）。

実行: python resid_perm.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_PERM = 400
MIN_HIT = 100          # これ未満の的中数のマスは候補にしない
THS = [1.0, 1.2, 1.5, 2.0, 3.0]
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def best_of_grid(gap, odds, win, rc, nr):
    """レースごとに gap 最大の1頭を選び、しきい値ごとのROIの最良値を返す。"""
    order = np.lexsort((-gap, rc))
    first = np.ones(len(rc), dtype=bool)
    first[1:] = rc[order][1:] != rc[order][:-1]
    idx = order[first]                      # 各レースの gap 最大の行
    g, o, w = gap[idx], odds[idx], win[idx]
    best = -np.inf
    best_th = None
    for th in THS:
        m = g >= th
        n = int(m.sum())
        h = int(w[m].sum())
        if n < 200 or h < MIN_HIT:
            continue
        roi = (w[m] * o[m]).sum() / n * 100
        if roi > best:
            best, best_th = roi, th
    return best, best_th


def main():
    d = pd.read_csv("resid_pred.csv", dtype={"race_id": str})
    rc, _ = pd.factorize(d.race_id)
    d["_rc"] = rc
    nr = rc.max() + 1
    gap = d.gap.values.astype(float)
    odds = d.odds.values.astype(float)
    win = d.win.values.astype(float)
    log(f"検体 {len(d):,}頭 / {nr:,}レース")
    log(f"しきい値 {THS} の中から最良を選ぶ。的中{MIN_HIT}本未満は候補外。\n")

    real, real_th = best_of_grid(gap, odds, win, rc, nr)
    log(f"=== 本物 ===")
    log(f"  最良 ROI {real:.1f}%（gap>={real_th}）\n")

    # レース内でシャッフル（モデルが何も知らない状態）
    log(f"=== 偽物を{N_PERM}回作る ===")
    order = np.argsort(rc, kind="stable")
    bounds = np.searchsorted(rc[order], np.arange(nr + 1))
    nulls = []
    for i in range(N_PERM):
        sh = gap.copy()
        gs = gap[order]
        out = np.empty_like(gs)
        for k in range(nr):
            a, b = bounds[k], bounds[k + 1]
            out[a:b] = rng.permutation(gs[a:b])
        sh[order] = out
        r, _t = best_of_grid(sh, odds, win, rc, nr)
        if np.isfinite(r):
            nulls.append(r)
        if (i + 1) % 100 == 0:
            log(f"  {i+1}/{N_PERM} 完了")
    nulls = np.array(nulls)
    p = float((nulls >= real).mean())
    log(f"\n=== 結果 ===")
    log(f"  本物の最良値        {real:.1f}%")
    log(f"  偽物の最良値の中央値  {np.median(nulls):.1f}%")
    log(f"  偽物の95パーセンタイル {np.percentile(nulls,95):.1f}%")
    log(f"  偽物の最大値         {nulls.max():.1f}%")
    log(f"  p値 = {p:.4f}  （{int((nulls>=real).sum())}/{len(nulls)} が本物以上）")
    log(f"\n  → {'✅ 偶然では説明しにくい（p<0.05）' if p < 0.05 else '⚠ 偶然の範囲。採用しない'}")

    log("\n=== 参考: 各しきい値を単独で見たときの偽物分布 ===")
    log(f"  {'しきい値':<10}{'本物':>8}{'偽物中央':>10}{'偽物95%':>10}{'p値':>8}")
    order2 = np.lexsort((-gap, rc))
    first = np.ones(len(rc), dtype=bool)
    first[1:] = rc[order2][1:] != rc[order2][:-1]
    idx = order2[first]
    for th in THS:
        m = gap[idx] >= th
        if m.sum() < 200 or win[idx][m].sum() < MIN_HIT:
            continue
        r = (win[idx][m] * odds[idx][m]).sum() / m.sum() * 100
        ns = []
        for _ in range(200):
            sh = gap.copy()
            gs = gap[order]
            out = np.empty_like(gs)
            for k in range(nr):
                a, b = bounds[k], bounds[k + 1]
                out[a:b] = rng.permutation(gs[a:b])
            sh[order] = out
            o2 = np.lexsort((-sh, rc))
            f2 = np.ones(len(rc), dtype=bool)
            f2[1:] = rc[o2][1:] != rc[o2][:-1]
            i2 = o2[f2]
            m2 = sh[i2] >= th
            if m2.sum() >= 200 and win[i2][m2].sum() >= MIN_HIT:
                ns.append((win[i2][m2] * odds[i2][m2]).sum() / m2.sum() * 100)
        if ns:
            ns = np.array(ns)
            log(f"  gap>={th:<6}{r:>7.1f}%{np.median(ns):>9.1f}%"
                f"{np.percentile(ns,95):>9.1f}%{(ns>=r).mean():>8.4f}")


if __name__ == "__main__":
    main()
