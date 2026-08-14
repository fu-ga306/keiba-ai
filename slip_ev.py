# -*- coding: utf-8 -*-
"""EV方式がスリッページを通るか（2026-08-14・リーク修正後のクリーンデータ）

背景
  バックテストは確定オッズで買い目を決めるが、実運用は7分前で決める。
  リーク修正前は 117.0% → 88.4%（-28.7pt）だった。
  修正後のクリーンデータでは出発点が 107.0% なので、同じ幅なら78%程度に落ちる。
  それを実測する。

⚠ 前回踏んだ誤り
  確定オッズで絞った集合から7分前を逆算すると、負け馬だけが範囲外に落ちて
  勝ち馬が残る選択バイアスが入る（344.7%という嘘の数字が出た）。
  必ず**オッズで絞る前の全馬**から逆算し、7分前の値で選び直す。

EV方式の条件は「乖離≥3・20倍以下・MF複勝1位はEV≥1.7／2-5位はEV≥2.2」。
  乖離 = 人気順位 − MF複勝順位
  EV   = 較正済み勝率 × オッズ
オッズが動けば**人気順位もEVも両方変わる**ので、両方を7分前基準で作り直す。
これをやらないと「オッズだけ動いて人気は固定」という現実にない状態を測ることになる。

実行: python slip_ev.py → slip_ev_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2
N_SIM = 400
rng = np.random.default_rng(20260814)


def log(m):
    print(m, flush=True)


def drift_pools():
    """実測の 7分前→確定 対数比を 人気帯×勝敗 で層別。"""
    o = pd.read_csv("odds_history.csv", dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    o = o[o["単勝オッズ"] > 0].sort_values("t")
    last = (o.groupby(["race_id", "馬名"]).tail(1)[["race_id", "馬名", "単勝オッズ"]]
            .rename(columns={"単勝オッズ": "pre"}))
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num"]) \
        .rename(columns={"単勝オッズ": "fin", "人気": "ninki", "着順_num": "着"})
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
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
    # 勝ち側は薄いので 4番人気以上をまとめる
    merged = np.concatenate([pools[(b, 1)] for b in (2, 3, 4) if len(pools[(b, 1)])])
    for b in (2, 3, 4):
        pools[(b, 1)] = merged
    log("ドリフト標本（7分前→確定）")
    for b in range(5):
        log(f"  帯{b}: 着外{len(pools[(b,0)]):>4}頭 中央{np.exp(np.median(pools[(b,0)])):.3f}"
            f" / 1着{len(pools[(b,1)]):>4}頭")
    return pools


def main():
    pools = drift_pools()
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    TAN = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "単勝"].itertuples()}
    UMA = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "馬単"].itertuples()}
    D["mr"] = D.groupby("race_id")["c_top3"].rank(ascending=False)
    D["band"] = np.select([D.pr <= 1, D.pr <= 3, D.pr <= 5, D.pr <= 7], [0, 1, 2, 3], 4)
    log(f"\n検体 {len(D):,}頭 / {D.race_id.nunique():,}レース")

    races = []
    for rid, g in D.groupby("race_id", sort=False):
        races.append({
            "rid": rid, "odds": g.odds.values.astype(float),
            "p": g.c_win_n.values.astype(float), "mr": g.mr.values.astype(float),
            "band": g.band.values.astype(int), "win": g.win.values.astype(int),
            "bn": g.bn.values,
            "tan": np.array([TAN.get((rid, b), 0.0) for b in g.bn.values]),
            "uma": {a: {b: UMA.get((rid, f"{a}-{b}"), 0.0) for b in g.bn.values}
                    for a in g.bn.values},
        })
    log(f"  展開完了 {len(races):,}レース")

    def evaluate(use_drift):
        cost = ret = 0.0
        hits = n = 0
        for r in races:
            if use_drift:
                lr = np.empty(len(r["odds"]))
                for b in range(5):
                    for w in (0, 1):
                        idx = np.where((r["band"] == b) & (r["win"] == w))[0]
                        if not len(idx):
                            continue
                        pool = pools[(b, w)]
                        if not len(pool):
                            pool = pools[(b, 0)]
                        lr[idx] = rng.choice(pool, len(idx))
                odds = r["odds"] / np.exp(lr)         # 確定 → 7分前
            else:
                odds = r["odds"]
            # 人気順位もオッズから作り直す（ここを固定すると現実と違う）
            pr = odds.argsort().argsort() + 1.0
            gap = pr - r["mr"]
            ev = r["p"] * odds
            ok = ((gap >= GAP_MIN) & (odds <= ODDS_MAX) &
                  (((r["mr"] == 1) & (ev >= EV_TOP)) |
                   ((r["mr"] >= 2) & (r["mr"] <= 5) & (ev >= EV_SUB))))
            if not ok.any():
                continue
            i = np.where(ok)[0][np.argmax(ev[ok])]
            n += 1
            cost += 1000.0
            ret += r["tan"][i] * 10                  # 払戻は確定オッズ
            hits += int(r["win"][i] == 1)
            mate = r["bn"][(np.isin(r["mr"], [1, 2, 3, 4, 5])) & (pr <= 3)]
            for b in mate:
                if b == r["bn"][i]:
                    continue
                cost += 500.0
                ret += r["uma"][r["bn"][i]][b] * 5
        return (ret / cost * 100 if cost else np.nan), n, hits

    base, nb, hb = evaluate(False)
    log(f"\n=== 確定オッズで選ぶ（従来のバックテスト）===")
    log(f"  {nb:,}レース  的中{hb}本  回収率 {base:.1f}%")

    log(f"\n=== 7分前オッズで選び直す（実運用の姿・{N_SIM}試行）===")
    sims, ns, hs = [], [], []
    for k in range(N_SIM):
        v, n, h = evaluate(True)
        sims.append(v); ns.append(n); hs.append(h)
        if (k + 1) % 100 == 0:
            log(f"  {k+1}/{N_SIM}  中央値 {np.median(sims):.1f}%")
    s = np.array(sims)
    lo, hi = np.percentile(s, [2.5, 97.5])
    log(f"\n  選ばれるレース 中央 {int(np.median(ns)):,}（確定基準 {nb:,}）")
    log(f"  的中 中央 {int(np.median(hs))}本（確定基準 {hb}本）")
    log(f"  回収率 中央値 {np.median(s):.1f}%  95%[{lo:.1f}, {hi:.1f}]")
    log(f"  100%を超える確率 {np.mean(s > 100) * 100:.1f}%")
    log(f"  → スリッページの影響 {np.median(s) - base:+.1f}pt")
    pd.DataFrame([{"方式": "EV方式(単勝+馬単)", "確定基準": round(base, 1),
                   "確定R数": nb, "7分前_中央": round(float(np.median(s)), 1),
                   "R数中央": int(np.median(ns)), "CI下": round(float(lo), 1),
                   "CI上": round(float(hi), 1),
                   "P100": round(float(np.mean(s > 100) * 100), 1),
                   "影響pt": round(float(np.median(s) - base), 1)}]) \
        .to_csv("slip_ev_result.csv", index=False, encoding="utf-8-sig")
    log("\n保存 → slip_ev_result.csv")


if __name__ == "__main__":
    main()
