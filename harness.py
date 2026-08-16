# -*- coding: utf-8 -*-
"""買い方を評価する唯一の窓口（2026-08-16）

なぜ作るか
  検証スクリプトを書くたびに、本番の実装との間にズレが出た。2026-08-16だけで
  3回。集計ミス、条件の欠落、馬番のゼロ埋め漏れ。どれも「検証で見た数字」と
  「本番が実際に買うもの」が違うという同じ事故だった。

  そこで評価の入口を1つに固定する。買い方は下の Strategy で表し、
  買い目の作り方も payout の引き方もここにしか書かない。
  本番(keiba_predict)へは、ここで検証した Strategy をそのまま移す。

このハーネスが必ずやること
  ① 5年通算・直近2年・年ごとのROIを出す
  ② ブートストラップで95%区間を出す（点推定を信用しない）
  ③ スリッページ模擬（7分前オッズで選び直す）
  ④ 的中数を必ず併記する（少ないと何も言えない）
  ⑤ 馬番は2桁ゼロ埋めで払戻表と照合する

⚠ 探索には使わない。探索して選ぶと、その行為自体が結果を歪める
  （2,509構成から選ぶと翌年 -70.5pt 落ちることを実測済み）。
  事前に決めた少数の仮説を、1回だけ測るために使う。

使い方:
    from harness import Strategy, evaluate, report
    s = Strategy(name="...", kind="馬単裏", basis="r1", ax_rank=1, mate_rank=2,
                 odds=(1, 20), cond="are")
    report(evaluate(s))
"""
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]
_rng = np.random.default_rng(20260816)
_CACHE = {}


@dataclass
class Strategy:
    """買い方の定義。ここに書いたものだけが評価できる。

    kind    : 単勝 / 複勝 / 馬連 / ワイド / 馬単表(軸→相手) / 馬単裏(相手→軸)
    basis   : r1=MF勝率順位 / r2=MF連対順位 / r3=MF複勝順位
    ax_rank : 軸にする順位の上限（この中でオッズ条件を満たす最上位1頭が軸）
    mate_rank: 相手にする順位の上限。単系では使わない
    odds    : 軸に課すオッズ範囲 [下限, 上限)。相手には課さない
    cond    : レース条件を返す関数。None なら全レース
    """
    name: str
    kind: str
    basis: str = "r3"
    ax_rank: int = 1
    mate_rank: int = 3
    odds: Tuple[float, float] = (1.0, 9999.0)
    cond: Optional[Callable] = None
    mate_odds: Optional[Tuple[float, float]] = None   # 相手にもオッズ条件を課す場合


# ── よく使うレース条件 ────────────────────────────────────────────
def are_race(g):
    """荒れR: 1番人気のMF複勝順位が4位以下。モデルが1番人気を信用していない。"""
    f = g.fav_mr.iloc[0]
    return pd.notna(f) and f >= 4


def solid_race(g):
    """堅いR: 1番人気がMF複勝1位。"""
    f = g.fav_mr.iloc[0]
    return pd.notna(f) and f == 1


def long_race(g):
    return g["距離"].iloc[0] >= 1900


def load():
    if "D" in _CACHE:
        return _CACHE["D"], _CACHE["PAY"], _CACHE["races"]
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False, method="first")
    D["r2"] = g["c_top2"].rank(ascending=False, method="first")
    D["r3"] = g["c_top3"].rank(ascending=False, method="first")
    fav = D[D.pr == 1][["race_id", "r3"]].rename(columns={"r3": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {}
    for r in jv.itertuples():
        PAY[(r.race_id, r.券種, r.組み合わせ)] = r.払戻金
    races = {r: gg for r, gg in D.groupby("race_id", sort=False)}
    _CACHE.update(D=D, PAY=PAY, races=races)
    return D, PAY, races


def _bets(g, rid, s: Strategy, PAY, odds_override=None):
    """1レースの買い目を {(券種,組み合わせ): 払戻} で返す。

    馬番は必ず2桁ゼロ埋め。jv_payouts が "09-14" 形式なので、
    ここを揃えないと1桁馬番の買い目が照合できない（2026-08-16の事故）。
    """
    if s.cond is not None and not s.cond(g):
        return {}
    odds = odds_override if odds_override is not None else g.odds.values
    r = g[s.basis].values
    bn = g.bn.values                     # bet_cache の bn は既に2桁ゼロ埋め
    lo, hi = s.odds
    m = (r <= s.ax_rank) & (odds >= lo) & (odds < hi)
    if not m.any():
        return {}
    a = bn[m][np.argmin(r[m])]
    out = {}
    if s.kind in ("単勝", "複勝"):
        out[(s.kind, a)] = PAY.get((rid, s.kind, a), 0.0)
        return out
    mm = (r <= s.mate_rank) & (bn != a)
    if s.mate_odds is not None:
        mm &= (odds >= s.mate_odds[0]) & (odds < s.mate_odds[1])
    for b in bn[mm]:
        if s.kind == "馬単裏":
            out[("馬単", f"{b}-{a}")] = PAY.get((rid, "馬単", f"{b}-{a}"), 0.0)
        elif s.kind == "馬単表":
            out[("馬単", f"{a}-{b}")] = PAY.get((rid, "馬単", f"{a}-{b}"), 0.0)
        elif s.kind == "馬連":
            c = f"{min(a,b)}-{max(a,b)}"
            out[("馬連", c)] = PAY.get((rid, "馬連", c), 0.0)
        elif s.kind == "ワイド":
            c = f"{min(a,b)}-{max(a,b)}"
            out[("ワイド", c)] = PAY.get((rid, "ワイド", c), 0.0)
    return out


def _drift_pools():
    if "pools" in _CACHE:
        return _CACHE["pools"]
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
    pools = {(b, w): m[(m.band == b) & (m.w == w)].lr.values
             for b in range(5) for w in (0, 1)}
    merged = np.concatenate([pools[(b, 1)] for b in (2, 3, 4) if len(pools[(b, 1)])])
    for b in (2, 3, 4):
        pools[(b, 1)] = merged
    _CACHE["pools"] = pools
    return pools


def evaluate(s: Strategy, n_sim=40):
    D, PAY, races = load()
    acc = {y: [0.0, 0.0, 0] for y in YEARS}
    nrace = 0
    recent = []
    for rid, g in races.items():
        d = _bets(g, rid, s, PAY)
        if not d:
            continue
        nrace += 1
        y = int(rid[:4])
        for p in d.values():
            acc[y][0] += 100
            acc[y][1] += p
            acc[y][2] += 1 if p > 0 else 0
            if y in RECENT:
                recent.append(p)
    tc = sum(acc[y][0] for y in YEARS)
    if tc == 0:
        return {"name": s.name, "点数": 0}
    tr = sum(acc[y][1] for y in YEARS)
    th = sum(acc[y][2] for y in YEARS)
    rc = sum(acc[y][0] for y in RECENT)
    rr = sum(acc[y][1] for y in RECENT)
    rh = sum(acc[y][2] for y in RECENT)
    v = np.array(recent)
    if len(v) > 20:
        b = np.array([_rng.choice(v, len(v)).mean() for _ in range(2000)])
        lo, hi = np.percentile(b, [2.5, 97.5])
    else:
        lo = hi = np.nan
    # スリッページ模擬
    pools = _drift_pools()
    # 対象レースだけ、band/winを1度だけ計算しておく（毎回作ると遅い）
    if "recent_prep" not in _CACHE:
        prep = []
        for rid, g in races.items():
            if int(rid[:4]) in RECENT:
                band = np.select([g.pr <= 1, g.pr <= 3, g.pr <= 5, g.pr <= 7],
                                 [0, 1, 2, 3], 4).astype(int)
                prep.append((rid, g, band, g.win.values.astype(int), g.odds.values))
        _CACHE["recent_prep"] = prep
    prep = _CACHE["recent_prep"]
    sims = []
    for _ in range(n_sim):
        c = r_ = 0.0
        for rid, g, band, w, od in prep:
            lr = np.empty(len(band))
            for bb in range(5):
                for ww in (0, 1):
                    idx = np.where((band == bb) & (w == ww))[0]
                    if not len(idx):
                        continue
                    pool = pools[(bb, ww)]
                    lr[idx] = _rng.choice(pool if len(pool) else pools[(bb, 0)], len(idx))
            d = _bets(g, rid, s, PAY, odds_override=od / np.exp(lr))
            for p in d.values():
                c += 100
                r_ += p
        if c:
            sims.append(r_ / c * 100)
    yr = {y: (acc[y][1] / acc[y][0] * 100 if acc[y][0] else np.nan) for y in YEARS}
    return {"name": s.name, "点数": int(tc / 100), "的中": th,
            "5年ROI": round(tr / tc * 100, 1),
            "直近2年点数": int(rc / 100), "直近2年的中": rh,
            "直近2年ROI": round(rr / rc * 100, 1) if rc else np.nan,
            "CI下": round(lo, 1), "CI上": round(hi, 1),
            "7分前ROI": round(float(np.median(sims)), 1) if sims else np.nan,
            "影響pt": round(float(np.median(sims)) - rr / rc * 100, 1) if sims and rc else np.nan,
            "買うR数": nrace, "1R点数": round(tc / 100 / max(nrace, 1), 1),
            **{f"y{y}": round(yr[y], 1) for y in YEARS},
            "100超年": sum(1 for y in YEARS if yr[y] >= 100)}


def report(rows):
    if isinstance(rows, dict):
        rows = [rows]
    R = pd.DataFrame([r for r in rows if r.get("点数")])
    if R.empty:
        print("  評価できる構成なし")
        return R
    print(f"{'名前':<30}{'点数':>7}{'的中':>6}{'5年':>8}{'直近2年':>9}"
          f"{'95%区間':>18}{'7分前':>9}{'影響':>8}{'1R点':>7}{'100超年':>8}")
    for _, r in R.iterrows():
        print(f"{r['name'][:29]:<30}{r['点数']:>7,}{r['的中']:>6}{r['5年ROI']:>7.1f}%"
              f"{r['直近2年ROI']:>8.1f}%  [{r['CI下']:>5.1f},{r['CI上']:>6.1f}]"
              f"{r['7分前ROI']:>8.1f}%{r['影響pt']:>+7.1f}{r['1R点数']:>7.1f}{r['100超年']:>7}")
    return R
