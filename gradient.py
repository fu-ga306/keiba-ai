# -*- coding: utf-8 -*-
"""勾配テスト：オッズ上限を絞るほど良くなるのは本物か（2026-08-16・第2ラウンド）

第1ラウンド(prereg.py)で出たこと
    軸1-10倍 110.0%  /  1-20倍 100.2%  /  1-40倍 97.6%
  絞るほど良い。しかも事実2（市場は60倍超を買いすぎる）が予言した向きと一致した。

なぜこれが「探索して一番良いものを選んだ」のと違うか
  探索は、たくさんのマスの中から一番高いマスを拾う。どのマスが高いかは事前に
  わからないので、偶然高いマスを拾ってしまう。
  今回は「絞るほど良くなるはず」という向きを先に決めていて、実際にその向きに
  3点とも並んだ。偶然に3点が予言どおりの順に並ぶ確率は 1/6。

本物なら成り立つはずのこと（これが今回の予言）
  予言1: さらに絞っても（7倍、5倍）改善が続く、または高止まりする
  予言2: 逆に緩めると（60倍、上限なし）単調に悪化し続ける
  予言3: 効果は「大穴を切ったこと」から来るので、切った馬たちの回収率は
         100%を大きく下回っているはず（＝切って正解だった、が直接見える）

  予言2と3が外れたら、1-10倍の110%は偶然。採用しない。

⚠ 的中52本しかない。CI下限は50%台。どう転んでも「黒字が確定した」とは言えない。
  ここで見ているのは「現行より良いと考える理由があるか」だけ。

実行: python gradient.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from harness import Strategy, are_race, evaluate, load, report


def log(m):
    print(m, flush=True)


# 予言1と2: 上限を動かしたときの単調性
LADDER = [3, 5, 7, 10, 15, 20, 30, 40, 60, 9999]
ROUND2 = [Strategy(f"軸 1-{'∞' if u > 100 else u}倍", "馬単裏", "r1", 1, 2,
                   (1.0, float(u)), are_race) for u in LADDER]
# 相手側にも同じ理屈を効かせる（事実2は相手にも当てはまるはず）
ROUND2 += [
    Strategy("軸1-10倍 + 相手40倍以下", "馬単裏", "r1", 1, 2, (1.0, 10.0),
             are_race, mate_odds=(1.0, 40.0)),
    Strategy("軸1-10倍 + 相手20倍以下", "馬単裏", "r1", 1, 2, (1.0, 10.0),
             are_race, mate_odds=(1.0, 20.0)),
]


def prediction3():
    """予言3: 切り捨てた馬（軸オッズ10倍超）の買い目は、実際に負けているか。

    もし切った側も100%前後なら、絞った効果は「大穴が悪い」からではなく
    たまたま。切った側がはっきり100%を下回っていて初めて理屈が通る。
    """
    from harness import _bets, load as L
    D, PAY, races = L()
    bands = [(1, 10), (10, 20), (20, 40), (40, 9999)]
    log("\n=== 予言3: 軸のオッズ帯ごとの成績（切った側は本当に負けているか）===")
    log(f"  {'軸オッズ帯':<14}{'点数':>7}{'的中':>6}{'的中率':>8}{'5年ROI':>9}")
    for lo, hi in bands:
        s = Strategy("", "馬単裏", "r1", 1, 2, (float(lo), float(hi)), are_race)
        c = r_ = 0.0
        h = 0
        for rid, g in races.items():
            for p in _bets(g, rid, s, PAY).values():
                c += 100
                r_ += p
                h += 1 if p > 0 else 0
        if c:
            lab = f"{lo}-{'∞' if hi > 100 else hi}倍"
            log(f"  {lab:<14}{int(c/100):>7,}{h:>6}{h/(c/100)*100:>7.1f}%{r_/c*100:>8.1f}%")


def main():
    load()
    log("第2ラウンド。第1ラウンドで見つけた単調性が本物かを確かめる。")
    log("予言1: さらに絞っても改善が続く / 予言2: 緩めると単調に悪化")
    log("予言3: 切った側（10倍超が軸）は、はっきり100%を下回っている\n")

    rows = [evaluate(s, n_sim=12) for s in ROUND2]
    R = report(rows)
    R.to_csv("gradient_result.csv", index=False, encoding="utf-8-sig")

    log("\n=== 予言1・2の判定（上限を緩めるほど悪化しているか）===")
    lad = R[R.name.str.startswith("軸 1-")].copy()
    lad["上限"] = [float(x) for x in LADDER][:len(lad)]
    lad = lad.sort_values("上限")
    v5 = lad["5年ROI"].values
    # 順位相関（上限が大きいほどROIが下がるなら負になる）
    from scipy.stats import spearmanr
    rho, p = spearmanr(lad["上限"].values, v5)
    log(f"  上限とROIの順位相関 rho={rho:+.3f} (p={p:.4f})")
    log(f"  → {'✅ 緩めるほど悪化。理屈どおり' if rho < -0.5 else '⚠ 単調でない。偶然の可能性が高い'}")
    log("  " + "  ".join(f"{int(u) if u<100 else '∞'}倍:{r:.1f}%"
                         for u, r in zip(lad["上限"], v5)))

    prediction3()

    log("\n=== 総合判定 ===")
    best = R[(R["5年ROI"] >= 100) & (R["直近2年ROI"] >= 100) & (R["7分前ROI"] >= 100)]
    if len(best):
        log(best[["name", "点数", "的中", "5年ROI", "直近2年ROI", "7分前ROI",
                  "CI下", "100超年"]].to_string(index=False))
    else:
        log("  3条件すべてを満たすものは なし")


if __name__ == "__main__":
    main()
