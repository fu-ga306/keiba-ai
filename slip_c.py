# -*- coding: utf-8 -*-
"""C案（複勝）がスリッページを通過するか（2026-08-13）

C案: 長距離2100m+ × MF複勝1位 × 20倍以下 × 4番人気以下 → 複勝
     5年124点・的中55本(44.4%)・累計110.2%・95%区間[87.3,134.0]

なぜC案だけ検証する価値があるか
  A案は 確定117.0% → 7分前88.4%（-28.7pt）で落ちた。
  B案は 10-20倍というオッズ指定が中核なので同じ罠にはまる（相関-0.13で先読み不能）。
  C案は選択条件の大半（距離・モデル順位・人気）がオッズに依存せず、
  オッズ条件は「20倍以下」だけ。しかも複勝は配当が着順で決まるので
  オッズ変動の影響が単勝より小さいはず。

⚠ 前回の誤り
  確定オッズで絞った集合から7分前を逆算すると、負け馬だけが範囲外に落ちて
  勝ち馬が残る選択バイアスが入る（344.7%という嘘の数字が出た）。
  必ず**オッズで絞る前の母集団**から逆算し、7分前の値で選び直すこと。

やり方
  母集団 = 長距離2100+ × MF複勝1位 × 4番人気以下（オッズ無制限）
  各馬の7分前オッズを、実測ドリフト（odds_history）の経験分布から逆算
  7分前オッズが20倍以下の馬だけを買い、複勝の実払戻で精算
  これを3000回繰り返して分布を出す

実行: python slip_c.py → slip_c_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
rng = np.random.default_rng(20260813)


def log(m):
    print(m, flush=True)


def drift_pools():
    """実測の 7分前→確定 対数比を、人気帯×3着内かで層別して返す。"""
    o = pd.read_csv("odds_history.csv", dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    o = o[o["単勝オッズ"] > 0].sort_values("t")
    last = (o.groupby(["race_id", "馬名"]).tail(1)[["race_id", "馬名", "単勝オッズ"]]
            .rename(columns={"単勝オッズ": "pre"}))
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"]) \
        .rename(columns={"単勝オッズ": "fin", "人気": "ninki", "着順_num": "着"})
    m = last.merge(rf, on=["race_id", "馬名"], how="inner").dropna(
        subset=["pre", "fin", "ninki", "着"])
    m = m[(m.pre > 0) & (m.fin > 0)].copy()
    m["lr"] = np.log(m.fin / m.pre)
    m["in3"] = (m["着"] <= 3).astype(int)
    pools = {}
    for band, sel in (("4-7", (m.ninki >= 4) & (m.ninki <= 7)),
                      ("8+", m.ninki >= 8)):
        for k in (0, 1):
            v = m[sel & (m.in3 == k)].lr.values
            pools[(band, k)] = v
    log("ドリフト標本（7分前→確定の対数比）")
    for k, v in pools.items():
        log(f"  人気{k[0]}帯 × {'3着内' if k[1] else '着外'}: {len(v):>4}頭"
            f"  中央倍率 {np.exp(np.median(v)):.3f}" if len(v) else f"  {k}: 0頭")
    return pools


def main():
    pools = drift_pools()
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    FUKU = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "複勝"].itertuples()}
    D["fuku"] = [FUKU.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]
    D["mr"] = D.groupby("race_id")["c_top3"].rank(ascending=False)

    # 母集団: オッズ条件を外した状態
    P = D[(D["距離"] >= 2100) & (D.mr == 1) & (D.pr >= 4)].copy()
    base = P[P.odds <= 20]
    log(f"\n母集団（オッズ無制限）: {len(P)}頭")
    log(f"  うち確定20倍以下: {len(base)}頭  複勝ROI {base.fuku.mean():.1f}%"
        f"  的中{int((base.fuku>0).sum())}本")

    fin = P.odds.values
    fk = P.fuku.values
    in3 = (P.fuku.values > 0).astype(int)
    band = np.where(P.pr.values <= 7, "4-7", "8+")

    sims, ns, hits = [], [], []
    for _ in range(3000):
        lr = np.empty(len(P))
        for b in ("4-7", "8+"):
            for k in (0, 1):
                idx = np.where((band == b) & (in3 == k))[0]
                if not len(idx):
                    continue
                pool = pools.get((b, k))
                if pool is None or not len(pool):
                    pool = np.concatenate([v for v in pools.values() if len(v)])
                lr[idx] = rng.choice(pool, len(idx))
        pre = fin / np.exp(lr)              # 確定 → 7分前 を逆算
        sel = pre <= 20                     # 7分前オッズで選び直す
        if sel.sum() < 5:
            continue
        sims.append(fk[sel].mean())
        ns.append(int(sel.sum()))
        hits.append(int((fk[sel] > 0).sum()))
    s = np.array(sims)
    lo, hi = np.percentile(s, [2.5, 97.5])
    log(f"\n=== 7分前オッズで選び直した場合（実運用の姿）===")
    log(f"  選ばれる点数 中央 {int(np.median(ns))}点（確定基準では {len(base)}点）")
    log(f"  的中 中央 {int(np.median(hits))}本")
    log(f"  複勝ROI 中央値 {np.median(s):.1f}%   95%[{lo:.1f}, {hi:.1f}]")
    log(f"  100%を超える確率 {np.mean(s > 100) * 100:.1f}%")
    log(f"  → スリッページの影響 {np.median(s) - base.fuku.mean():+.1f}pt")
    pd.DataFrame([{"案": "C 複勝案", "確定基準": round(base.fuku.mean(), 1),
                   "確定点数": len(base),
                   "7分前_中央": round(float(np.median(s)), 1),
                   "点数中央": int(np.median(ns)), "的中中央": int(np.median(hits)),
                   "CI下": round(float(lo), 1), "CI上": round(float(hi), 1),
                   "P100": round(float(np.mean(s > 100) * 100), 1),
                   "影響pt": round(float(np.median(s) - base.fuku.mean()), 1)}]) \
        .to_csv("slip_c_result.csv", index=False, encoding="utf-8-sig")
    log("\n保存 → slip_c_result.csv")


if __name__ == "__main__":
    main()
