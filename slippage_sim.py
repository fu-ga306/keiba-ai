# -*- coding: utf-8 -*-
"""7分前オッズで選び、確定オッズで払い戻される現実をシミュレートする。

なぜ必要か（2026-08-11）
  バックテストは確定オッズで「20倍以下・EV>=1.7」を判定し、確定オッズで払い戻す。
  実運用は**7分前オッズで判定し、確定オッズで払い戻される**。
  この差（スリッページ）が織り込まれていないので、全ての回収率が楽観的な可能性がある。

実測でわかったこと（145レース・1,882頭）
  ・7分前→確定の変化は人気帯で正負が逆転する
      1番人気 -14.3% / 2-3番 -14.9% / 4-5番 +1.3% / 6-7番 +22.4% / 8-10番 +54.5%
  ・**逆選択がある**。1着馬は中央値-11.1%、それ以外は+33.5%（p<0.0001）
    勝つ馬は直前に買われて下がる。当たったときほど配当が渋くなる。
  ・我々が買うのは4-7番人気。その帯の1着馬は +1.1% / +15.9% で**下がっていない**。
    → 逆選択の打撃は上位人気を買う戦略ほど大きく、中穴狙いでは軽い可能性がある。
       これを定量化するのがこのスクリプトの目的。

方式（3層）
  A. 実測リプレイ … 145レースの実際の7分前オッズで選び、実払戻で精算。モデル不要の真値
  B. モンテカルロ … 5年分の確定オッズから7分前オッズを逆算して選び直す。
       変化率は log(確定/7分前) の**経験分布から直接抽出**する（分布は当てはめない）。
       対数比なら確定オッズが負にならず、-100%の下限が自動的に守られる。
       層は「人気帯 × 1着/非1着」。1着側は検体が薄いので4番人気以上をまとめる。
       セルが薄いのでガウス核で平滑化する（同じ値の反復を防ぐ）。
  C. 二重ブートストラップ … 外側で**ドリフト標本145レースそのもの**を再抽出する。
       「145レースしか観測していない」という限界を信頼区間に反映させるため。

⚠ 145レース・1着馬145頭は薄い。出せるのは幅であって確定値ではない。

実行: python slippage_sim.py → slippage_sim_result.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
KDE_BW = 0.12          # 対数比に足すガウス核の幅。薄いセルの反復を和らげる
N_OUTER = 40           # 外側（ドリフト標本の再抽出）
N_INNER = 25           # 内側（ドリフトの引き直し）
rng = np.random.default_rng(20260811)


# ── ドリフト分布 ──────────────────────────────────────────────
def load_drift():
    """7分前と確定のオッズが両方ある馬を集め、対数比を返す。"""
    o = pd.read_csv(os.path.join(BASE_DIR, "odds_history.csv"), dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    last = (o.sort_values("t").groupby(["race_id", "馬名"]).tail(1)
            [["race_id", "馬名", "単勝オッズ"]]
            .rename(columns={"単勝オッズ": "odds_pre"}))
    # ⚠ usecols は元ファイルの列順で返る。必ず rename で対応付けること
    rf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                     dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"]) \
        .rename(columns={"単勝オッズ": "odds_fin", "人気": "pop", "着順_num": "着"})
    m = last.merge(rf, on=["race_id", "馬名"], how="inner")
    m = m.dropna(subset=["odds_pre", "odds_fin", "pop"])
    m = m[(m.odds_pre > 0) & (m.odds_fin > 0)]
    m["lr"] = np.log(m.odds_fin / m.odds_pre)
    m["win"] = (m["着"] == 1).astype(int)
    return m


def band(pop):
    """人気帯。1着側は検体が薄いので4番人気以上をまとめる前提の区分。"""
    p = np.asarray(pop, dtype=float)
    return np.select([p <= 1, p <= 3, p <= 5, p <= 7, p <= 10], [0, 1, 2, 3, 4], 5)


def build_pools(d, pooled_win=True):
    """{(帯, 勝敗): 対数比の配列} を作る。"""
    d = d.copy()
    d["b"] = band(d["pop"])
    pools = {}
    for w in (0, 1):
        for b in range(6):
            s = d[(d.win == w) & (d.b == b)].lr.values
            pools[(b, w)] = s
    if pooled_win:
        # 1着側の中穴以降はまとめる（4-5/6-7/8-10/11- を1つに）
        merged = np.concatenate([pools[(b, 1)] for b in (2, 3, 4, 5)])
        for b in (2, 3, 4, 5):
            pools[(b, 1)] = merged
    return pools


def sample_lr(pools, bands, wins, gen):
    """層ごとに経験分布から抽出し、ガウス核で平滑化する。"""
    out = np.empty(len(bands))
    for b in range(6):
        for w in (0, 1):
            idx = np.where((bands == b) & (wins == w))[0]
            if len(idx) == 0:
                continue
            pool = pools.get((b, w))
            if pool is None or len(pool) == 0:
                pool = np.concatenate([v for v in pools.values() if len(v)])
            pick = gen.choice(pool, len(idx), replace=True)
            out[idx] = pick + gen.normal(0, KDE_BW, len(idx))
    return out


# ── 買い方 ────────────────────────────────────────────────────
def bet_roi(g, pay, odds_col):
    """1レース分。odds_col のオッズで選び、実払戻で精算する。"""
    ev = g["c_win_n"] * g[odds_col]
    c = g[(g["乖離"] >= GAP_MIN) & (g[odds_col] <= ODDS_MAX) &
          (((g["mr"] == 1) & (ev >= EV_TOP)) |
           (g["mr"].between(2, 5) & (ev >= EV_SUB)))]
    if not len(c):
        return None
    ax = c.assign(_e=ev[c.index]).sort_values("_e", ascending=False).bn.iloc[0]
    rid = g["race_id"].iloc[0]
    cost = 1000.0
    ret = pay.get((rid, "単勝", ax), 0.0) * 10
    pr = g["人気"].rank(method="first")
    for m in g[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= 3)].bn:
        if m == ax:
            continue
        cost += 500
        ret += pay.get((rid, "馬単", f"{ax}-{m}"), 0.0) * 5
    return cost, ret


def load_pay():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}


def main():
    print("読み込み中...", flush=True)
    drift = load_drift()
    pay = load_pay()
    print(f"  ドリフト標本: {len(drift)}頭 / {drift.race_id.nunique()}レース", flush=True)

    turf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                       dtype={"race_id": str},
                       usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    E = []
    for y in YEARS:
        d = pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                        dtype={"race_id": str, "bn": str})
        E.append(d.merge(turf, on="race_id", how="left"))
    E = pd.concat(E, ignore_index=True)
    print(f"  5年の買い目候補: {E.race_id.nunique()}レース", flush=True)

    rows = []
    for lbl, sub in (("全体", E), ("芝のみ", E[E.is_turf == 1])):
        races = [g for _, g in sub.groupby("race_id", sort=False)]

        # ── 基準: 確定オッズで選ぶ（従来のバックテスト）──
        base = [r for r in (bet_roi(g, pay, "odds") for g in races) if r]
        a = np.array(base, float)
        base_roi = a[:, 1].sum() / a[:, 0].sum() * 100
        print(f"\n=== {lbl} ===", flush=True)
        print(f"  従来（確定オッズで選ぶ）: {len(a)}レース  回収率 {base_roi:.1f}%",
              flush=True)

        # ── B+C: 7分前オッズを逆算して選び直す ──
        res = []
        for oi in range(N_OUTER):
            # 外側: ドリフト標本そのものを再抽出（推定誤差を伝播させる）
            rid = drift.race_id.unique()
            boot = drift[drift.race_id.isin(rng.choice(rid, len(rid), replace=True))]
            pools = build_pools(boot)
            for ii in range(N_INNER):
                gen = np.random.default_rng(rng.integers(1 << 30))
                tot_c = tot_r = 0.0
                for g in races:
                    b = band(g["人気"].values)
                    w = (g["着"] == 1).astype(int).values if "着" in g else \
                        (g["win"] == 1).astype(int).values
                    lr = sample_lr(pools, b, w, gen)
                    gg = g.assign(odds_pre=g["odds"].values / np.exp(lr))
                    r = bet_roi(gg, pay, "odds_pre")
                    if r:
                        tot_c += r[0]
                        tot_r += r[1]
                if tot_c > 0:
                    res.append(tot_r / tot_c * 100)
            if oi % 10 == 9:
                print(f"    外側 {oi+1}/{N_OUTER} … 中央値 {np.median(res):.1f}%",
                      flush=True)
        res = np.array(res)
        lo, hi = np.percentile(res, [2.5, 97.5])
        print(f"  7分前で選ぶ: 中央値 {np.median(res):.1f}%  "
              f"95%区間[{lo:.1f}, {hi:.1f}]  100%超の確率 {np.mean(res>100)*100:.1f}%",
              flush=True)
        print(f"  → スリッページの影響 {np.median(res)-base_roi:+.1f}pt", flush=True)
        rows.append({"区分": lbl, "従来": round(base_roi, 1),
                     "7分前_中央値": round(float(np.median(res)), 1),
                     "95%下限": round(float(lo), 1), "95%上限": round(float(hi), 1),
                     "P100": round(float(np.mean(res > 100) * 100), 1),
                     "影響pt": round(float(np.median(res) - base_roi), 1)})
        pd.DataFrame(rows).to_csv(
            os.path.join(BASE_DIR, "slippage_sim_result.csv"),
            index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n※ ドリフト標本は145レース。出せるのは幅であって確定値ではない。")


if __name__ == "__main__":
    sys.exit(main())
