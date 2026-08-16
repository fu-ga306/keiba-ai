# -*- coding: utf-8 -*-
"""残差モデルが「実際に買える時刻」でも成立するかを測る（2026-08-17）

なぜこれが決定的か
  残差モデルは市場（オッズ）を入力に使う。従来のMFは市場を見なかったので
  オッズが動いても予測は変わらなかったが、今回は**オッズが動けば予測も動く**。

  確定オッズで測った 101.7% は、確定オッズを知っている前提の数字。
  実際に買うのは締切7分前で、そのときのオッズは違う。二重に効く。
    ① 入力が変わる  → モデルの予測そのものが変わる
    ② 選択が変わる  → 選ぶ馬が変わる
  払戻は確定オッズなので、そこは変わらない（パリミュチュエル）。

模擬の仕方
  残差モデルのスコアは  score = f(特徴量) + log(市場確率)  の形をしている。
  f は特徴量だけの関数なのでオッズが動いても変わらない。
  保存済みの gap = 予測確率/市場確率 が exp(f) に比例するので、
      7分前の予測 ∝ gap × (7分前の市場確率)
  として、オッズだけ差し替えれば7分前の予測が再現できる。

  7分前オッズは、実測のドリフト（odds_history から作った人気帯×勝敗別の分布）を
  確定オッズに逆向きに当てて作る。harness.py と同じ方法。

  ⚠ 勝ち馬ほど締切間際にオッズが下がるので、勝敗別に分けないと
    「7分前のほうが儲かる」という誤った結果になる。

実行: python resid_slip.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
N_SIM = 30
rng = np.random.default_rng(20260817)


def log(m):
    print(m, flush=True)


def drift_pools():
    """実測のドリフト分布。log(確定/7分前) を人気帯×勝敗で集める。"""
    o = pd.read_csv("odds_history.csv", dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    o["単勝オッズ"] = pd.to_numeric(o["単勝オッズ"], errors="coerce")
    o = o[o["単勝オッズ"] > 0].sort_values("t")
    last = (o.groupby(["race_id", "馬名"]).tail(1)[["race_id", "馬名", "単勝オッズ"]]
            .rename(columns={"単勝オッズ": "pre"}))
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"])
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rf = rf.rename(columns={"単勝オッズ": "fin", "人気": "ninki", "着順_num": "着"})
    for c in ("fin", "ninki", "着"):
        rf[c] = pd.to_numeric(rf[c], errors="coerce")
    m = last.merge(rf, on=["race_id", "馬名"], how="inner").dropna(
        subset=["pre", "fin", "ninki", "着"])
    m = m[(m.pre > 0) & (m.fin > 0)].copy()
    m["lr"] = np.log(m.fin / m.pre)
    m["w"] = (m["着"] == 1).astype(int)
    m["band"] = np.select([m.ninki <= 1, m.ninki <= 3, m.ninki <= 5, m.ninki <= 7],
                          [0, 1, 2, 3], 4)
    pools = {}
    for b in range(5):
        for w in (0, 1):
            v = m[(m.band == b) & (m.w == w)].lr.values
            pools[(b, w)] = v
    # 勝ち馬の標本が少ない帯はまとめる
    merged = np.concatenate([pools[(b, 1)] for b in (2, 3, 4) if len(pools[(b, 1)])])
    for b in (2, 3, 4):
        if len(pools[(b, 1)]) < 30:
            pools[(b, 1)] = merged
    log(f"  ドリフト標本 {len(m):,}頭 / {m.race_id.nunique()}レース")
    log("  人気帯ごとの中央比（確定/7分前）: "
        + "  ".join(f"{['1番','2-3番','4-5番','6-7番','8番以下'][b]}"
                    f"{np.median(np.exp(np.concatenate([pools[(b,0)],pools[(b,1)]]))):.3f}"
                    for b in range(5)))
    return pools


def main():
    d = pd.read_csv("resid_pred.csv", dtype={"race_id": str, "bn": str})
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")
    pools = drift_pools()

    d["band"] = np.select([d.人気 <= 1, d.人気 <= 3, d.人気 <= 5, d.人気 <= 7],
                          [0, 1, 2, 3], 4).astype(int)
    d["w"] = d.win.astype(int)
    rc, _ = pd.factorize(d.race_id)
    d["_rc"] = rc
    nr = d._rc.max() + 1

    def evaluate(odds, tag_year=True):
        """与えたオッズで予測し直し、期待値最大の1頭を選んだときの回収率。"""
        q = 1.0 / np.clip(odds, 1.01, None)
        s = np.zeros(nr)
        np.add.at(s, d._rc.values, q)
        q = q / s[d._rc.values]
        sc = d.gap.values * q                 # 予測 ∝ exp(f) × 市場確率
        t = np.zeros(nr)
        np.add.at(t, d._rc.values, sc)
        p = sc / t[d._rc.values]
        ev = p * odds
        # レースごとに期待値最大の1頭
        best = pd.Series(ev).groupby(d._rc.values).idxmax().values
        sel = d.iloc[best]
        roi = (sel.win * sel.odds).sum() / len(sel) * 100   # 払戻は確定オッズ
        yr = {y: (g.win * g.odds).sum() / len(g) * 100
              for y, g in sel.groupby("年")} if tag_year else {}
        return roi, sel.win.mean() * 100, yr, sel

    log("\n=== ① 確定オッズを知っている前提（前回の測定）===")
    roi0, hit0, yr0, sel0 = evaluate(d.odds.values)
    log(f"  期待値最大の1頭  {len(sel0):,}点  的中率{hit0:.1f}%  ROI {roi0:.1f}%")
    log("  年別: " + "  ".join(f"{y}:{v:.0f}%" for y, v in yr0.items()))

    log("\n=== ② 7分前オッズで予測・選択（実際に買える形）===")
    sims, hits, yrs = [], [], []
    for _ in range(N_SIM):
        lr = np.empty(len(d))
        for b in range(5):
            for w in (0, 1):
                idx = np.where((d.band.values == b) & (d.w.values == w))[0]
                if not len(idx):
                    continue
                pool = pools[(b, w)]
                if not len(pool):
                    pool = pools[(b, 0)]
                lr[idx] = rng.choice(pool, len(idx))
        o7 = d.odds.values / np.exp(lr)       # 7分前オッズを逆算
        r, h, y, _ = evaluate(o7)
        sims.append(r)
        hits.append(h)
        yrs.append(y)
    med = float(np.median(sims))
    log(f"  期待値最大の1頭  的中率{np.median(hits):.1f}%  ROI {med:.1f}%"
        f"  （{N_SIM}回模擬の中央値・範囲 {min(sims):.1f}〜{max(sims):.1f}%）")
    ym = {y: np.median([s.get(y, np.nan) for s in yrs]) for y in YEARS}
    log("  年別: " + "  ".join(f"{y}:{v:.0f}%" for y, v in ym.items()))
    log(f"\n  スリッページの影響: {med - roi0:+.1f}pt")
    log(f"  → {'✅ 7分前でも成立する' if med >= 100 else '⚠ 実際に買える形では100%に届かない'}")

    log("\n=== ③ 確定オッズで選び、確定オッズで買えたとしたら（参考・実現不可）===")
    log(f"  {roi0:.1f}%  ← ①と同じ。この差が「知り得ない情報」の価値")


if __name__ == "__main__":
    main()
