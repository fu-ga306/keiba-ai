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

方式（3層）
  A. 従来 … 確定オッズで選び確定オッズで払う（これまでのバックテスト）
  B. モンテカルロ … 確定オッズから7分前オッズを逆算して選び直し、実払戻で精算。
       変化率は log(確定/7分前) の**経験分布から直接抽出**（分布は当てはめない）。
       対数比なので確定オッズが負にならず -100% の下限が自動的に守られる。
       層は「人気帯 × 1着/非1着」。1着側は薄いので4番人気以上をまとめる。
       セルが薄いのでガウス核で平滑化する。
  C. 二重ブートストラップ … 外側で**ドリフト標本145レースそのもの**を再抽出し、
       「145レースしか観測していない」という限界を信頼区間に反映させる。

速度について（2026-08-11に作り直し）
  最初はレースごとにpandasで回して1,500万回の評価になり数日かかる見込みだった。
  レースを可変長配列に一度だけ展開し、以降はnumpyだけで回すよう書き直した。

⚠ ドリフト標本は145レース・1着馬145頭。出せるのは幅であって確定値ではない。
⚠ 7分前の「人気順位」は確定時のものを使っている。オッズが動けば人気も動くが、
   乖離の計算に使う順位まで作り直すと仮定が増えるため、そこは固定した。

実行: python slippage_sim.py → slippage_sim_result.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
KDE_BW = 0.12
N_OUTER, N_INNER = 30, 20
rng = np.random.default_rng(20260811)


def load_drift():
    o = pd.read_csv(os.path.join(BASE_DIR, "odds_history.csv"), dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    last = (o.sort_values("t").groupby(["race_id", "馬名"]).tail(1)
            [["race_id", "馬名", "単勝オッズ"]].rename(columns={"単勝オッズ": "odds_pre"}))
    # ⚠ usecols は元ファイルの列順で返る。必ず rename で対応付ける
    rf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                     dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"]) \
        .rename(columns={"単勝オッズ": "odds_fin", "人気": "pop", "着順_num": "着"})
    m = last.merge(rf, on=["race_id", "馬名"], how="inner")
    m = m.dropna(subset=["odds_pre", "odds_fin", "pop"])
    m = m[(m.odds_pre > 0) & (m.odds_fin > 0)].copy()
    m["lr"] = np.log(m.odds_fin / m.odds_pre)
    m["win"] = (m["着"] == 1).astype(int)
    return m


def band(pop):
    p = np.asarray(pop, dtype=float)
    return np.select([p <= 1, p <= 3, p <= 5, p <= 7, p <= 10], [0, 1, 2, 3, 4], 5)


def build_pools(d):
    """{(帯, 勝敗): 対数比} を作る。1着側の4番人気以上はまとめる。"""
    b = band(d["pop"].values)
    w = d["win"].values
    lr = d["lr"].values
    pools = {(bb, ww): lr[(b == bb) & (w == ww)] for bb in range(6) for ww in (0, 1)}
    merged = np.concatenate([pools[(bb, 1)] for bb in (2, 3, 4, 5)])
    for bb in (2, 3, 4, 5):
        pools[(bb, 1)] = merged
    return pools


def load_pay():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}


def pack(sub, pay):
    """レースを配列に一度だけ展開する。以降はnumpyだけで回せるようにする。

    各レースについて必要なのは
      odds_fin / c_win_n / mr / 乖離 / 人気 / 着 / 単勝払戻 / 馬単払戻(相手ごと)
    馬単の相手は「MF複勝1〜5位かつ人気3位以内」で、ドリフトに依存しないので
    ここで確定させておける。
    """
    races = []
    for rid, g in sub.groupby("race_id", sort=False):
        gap = g["乖離"].values.astype(float)
        if not (gap >= GAP_MIN).any():
            continue                      # 乖離条件は不変なので先に落とす
        pr = g["人気"].rank(method="first").values
        mate = g.bn.values[np.isin(g.mr.values, [1, 2, 3, 4, 5]) & (pr <= 3)]
        races.append({
            "odds": g["odds"].values.astype(float),
            "p": g["c_win_n"].values.astype(float),
            "mr": g["mr"].values.astype(float),
            "gap": gap,
            "band": band(g["人気"].values),
            "win": (g["着"].values == 1).astype(int),
            "bn": g.bn.values,
            "tan": np.array([pay.get((rid, "単勝", b), 0.0) for b in g.bn.values]),
            "uma": {b: np.array([pay.get((rid, "馬単", f"{a}-{b2}"), 0.0)
                                 for b2 in mate]) for a, b in [(a, a) for a in g.bn.values]},
            "mate": mate,
        })
    return races


def run_once(races, pools, gen, use_drift=True):
    """1試行。全レースをnumpyで評価して回収率を返す。"""
    cost = ret = 0.0
    for r in races:
        if use_drift:
            lr = np.empty(len(r["odds"]))
            for bb in range(6):
                for ww in (0, 1):
                    idx = np.where((r["band"] == bb) & (r["win"] == ww))[0]
                    if not len(idx):
                        continue
                    pool = pools[(bb, ww)]
                    lr[idx] = gen.choice(pool, len(idx)) + gen.normal(0, KDE_BW, len(idx))
            odds = r["odds"] / np.exp(lr)      # 確定 → 7分前 を逆算
        else:
            odds = r["odds"]
        ev = r["p"] * odds
        ok = ((r["gap"] >= GAP_MIN) & (odds <= ODDS_MAX) &
              (((r["mr"] == 1) & (ev >= EV_TOP)) |
               ((r["mr"] >= 2) & (r["mr"] <= 5) & (ev >= EV_SUB))))
        if not ok.any():
            continue
        i = np.where(ok)[0][np.argmax(ev[ok])]
        cost += 1000.0
        ret += r["tan"][i] * 10                 # 払戻は確定オッズのまま
        mates = r["uma"][r["bn"][i]]
        sel = r["mate"] != r["bn"][i]
        cost += 500.0 * sel.sum()
        ret += mates[sel].sum() * 5
    return ret / cost * 100 if cost else np.nan


def main():
    t0 = time.time()
    print("読み込み中...", flush=True)
    drift = load_drift()
    pay = load_pay()
    print(f"  ドリフト標本 {len(drift)}頭 / {drift.race_id.nunique()}レース", flush=True)

    turf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                       dtype={"race_id": str},
                       usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    E = pd.concat([pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                               dtype={"race_id": str, "bn": str}).merge(
                       turf, on="race_id", how="left") for y in YEARS],
                  ignore_index=True)

    rows = []
    for lbl, sub in (("全体", E), ("芝のみ", E[E.is_turf == 1])):
        races = pack(sub, pay)
        base = run_once(races, None, None, use_drift=False)
        print(f"\n=== {lbl} ===", flush=True)
        print(f"  従来（確定オッズで選ぶ）: 評価対象{len(races)}レース  "
              f"回収率 {base:.1f}%", flush=True)

        res = []
        for oi in range(N_OUTER):
            rid = drift.race_id.unique()
            boot = drift[drift.race_id.isin(rng.choice(rid, len(rid), replace=True))]
            pools = build_pools(boot)
            for _ in range(N_INNER):
                gen = np.random.default_rng(rng.integers(1 << 30))
                res.append(run_once(races, pools, gen))
            if (oi + 1) % 10 == 0:
                print(f"    外側 {oi+1}/{N_OUTER}  中央値 {np.median(res):.1f}%  "
                      f"({time.time()-t0:.0f}秒)", flush=True)
        res = np.array(res)
        lo, hi = np.percentile(res, [2.5, 97.5])
        print(f"  7分前で選ぶ: 中央値 {np.median(res):.1f}%  95%区間[{lo:.1f}, {hi:.1f}]"
              f"  100%超の確率 {np.mean(res>100)*100:.1f}%", flush=True)
        print(f"  → スリッページの影響 {np.median(res)-base:+.1f}pt", flush=True)
        rows.append({"区分": lbl, "従来": round(base, 1),
                     "7分前_中央値": round(float(np.median(res)), 1),
                     "95%下限": round(float(lo), 1), "95%上限": round(float(hi), 1),
                     "P100": round(float(np.mean(res > 100) * 100), 1),
                     "影響pt": round(float(np.median(res) - base), 1)})
        pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "slippage_sim_result.csv"),
                                  index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n所要 {time.time()-t0:.0f}秒")
    print("※ ドリフト標本は145レース。出せるのは幅であって確定値ではない。")


if __name__ == "__main__":
    sys.exit(main())
