# -*- coding: utf-8 -*-
"""層A: 実際の7分前オッズで買い目を選び直し、確定オッズ基準との差分を出す。

なぜ必要か（2026-08-11）
  シミュレータ(slippage_sim.py)は、確定オッズから7分前オッズを逆算する推定だった。
  結果は 全体 117.0%→88.4%（-28.7pt）／芝 148.9%→95.0%（-53.9pt）。
  ただし推定なので、**実際の7分前オッズで選び直した真値**で答え合わせが要る。

  odds_history.csv には7分前の実オッズがある（217レース・2026/07/23〜08/09）。
  ここでは一切の仮定を置かず、当時の入力信号だけで選び直す。

シミュレータより厳密にする点
  ・オッズだけでなく**人気順位も7分前のものに作り直す**。
    乖離 = 人気順位 - MF複勝順位 なので、人気が動けば選ばれる馬も変わる。
    シミュレータは人気を確定時のまま固定しており、そこが唯一残った仮定だった。

出力
  ・7分前基準と確定基準それぞれの回収率
  ・**買い目の差分**（どの馬が混入し、どの馬が消えたか）を1頭ずつ列挙
    混入 = 7分前では条件を満たすが確定では満たさない（本来買うべきでない馬）
    消失 = 確定では満たすが7分前では満たさない（取り逃がした馬）

モデル出力の入手先
  2026年のOOS予測が要る。本番モデルは2026年まで学習しているので使えない。
  train_mf_v2 backtest（KEIBA_TEST_YEAR=2026）の出力 model_mf_result.csv を使う。
  それが2026年になっていない間は、history_marks.csv（8/9の36レース）で
  差分の仕組みだけ検証する（※こちらは本番モデル＝in-sampleなので回収率は参考値）。

実行: python replay_a.py [--src auto|bt|hist]  → replay_a_result.csv / replay_a_diff.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2


def load_pre():
    """7分前（＝各レースの最終記録）のオッズと、そこから作った人気順位。"""
    o = pd.read_csv(os.path.join(BASE_DIR, "odds_history.csv"), dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    last = o.sort_values("t").groupby(["race_id", "馬名"]).tail(1).copy()
    last = last[["race_id", "馬名", "単勝オッズ"]].rename(columns={"単勝オッズ": "odds_pre"})
    last = last[last.odds_pre > 0]
    # 7分前の人気順位を作り直す（ここがシミュレータとの違い）
    last["pop_pre"] = last.groupby("race_id")["odds_pre"].rank(method="first")
    return last


def load_model(src):
    """モデルの確率と着順。src で入手先を切り替える。"""
    bt = os.path.join(BASE_DIR, "model_mf_result.csv")
    if src in ("auto", "bt") and os.path.exists(bt):
        d = pd.read_csv(bt, dtype={"race_id": str})
        if (d.race_id.str[:4] == "2026").any():
            d = d[d.race_id.str[:4] == "2026"].copy()
            d = d.rename(columns={"MF勝率": "p_raw", "MF複勝率": "f_raw",
                                  "着順_num": "着", "単勝オッズ": "odds_fin",
                                  "人気": "pop_fin"})
            return d[["race_id", "馬名", "p_raw", "f_raw", "着", "odds_fin",
                      "pop_fin"]], "backtest2026(OOS)"
        if src == "bt":
            return None, None
    if src in ("auto", "hist"):
        h = os.path.join(BASE_DIR, "history_marks.csv")
        if os.path.exists(h):
            d = pd.read_csv(h, dtype={"race_id": str})
            d = d.rename(columns={"MF勝ち確率": "p_raw", "MF複勝率": "f_raw",
                                  "着順": "着", "単勝オッズ": "odds_fin",
                                  "人気": "pop_fin"})
            need = ["race_id", "馬名", "p_raw", "f_raw", "着", "odds_fin", "pop_fin"]
            if all(c in d.columns for c in need):
                return d[need], "history_marks(本番モデル・参考値)"
    return None, None


def load_pay():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    return {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}


def pick(g, odds_col, pop_col):
    """1レースの軸を返す。無ければ None。"""
    p = g["p"].values
    odds = g[odds_col].values.astype(float)
    ev = p * odds
    gap = g[pop_col].values.astype(float) - g["mr"].values
    ok = ((gap >= GAP_MIN) & (odds <= ODDS_MAX) &
          (((g["mr"].values == 1) & (ev >= EV_TOP)) |
           ((g["mr"].values >= 2) & (g["mr"].values <= 5) & (ev >= EV_SUB))))
    if not ok.any():
        return None
    i = np.where(ok)[0][np.argmax(ev[ok])]
    return {"idx": g.index[i], "馬名": g["馬名"].iloc[i], "bn": g["bn"].iloc[i],
            "odds": odds[i], "ev": ev[i], "pop": g[pop_col].iloc[i],
            "mr": g["mr"].iloc[i], "着": g["着"].iloc[i]}


def settle(g, ax, pay, rid, pop_col):
    """買い目を実払戻で精算する。払戻は常に確定オッズ基準。"""
    cost = 1000.0
    ret = pay.get((rid, "単勝", ax["bn"]), 0.0) * 10
    pr = g[pop_col].rank(method="first")
    for b in g.bn[g.mr.isin([1, 2, 3, 4, 5]) & (pr <= 3)]:
        if b == ax["bn"]:
            continue
        cost += 500
        ret += pay.get((rid, "馬単", f"{ax['bn']}-{b}"), 0.0) * 5
    return cost, ret


def main():
    src = "auto"
    for i, a in enumerate(sys.argv):
        if a == "--src" and i + 1 < len(sys.argv):
            src = sys.argv[i + 1]

    pre = load_pre()
    model, label = load_model(src)
    if model is None:
        print("モデル出力が見つかりません（2026年のOOSかhistory_marksが要ります）")
        return 1
    pay = load_pay()
    print(f"モデル出力: {label}")

    d = model.merge(pre, on=["race_id", "馬名"], how="inner")
    d = d.dropna(subset=["odds_pre", "odds_fin", "p_raw", "f_raw"])
    d["bn"] = pd.to_numeric(d.get("馬番", np.nan), errors="coerce")
    if d["bn"].isna().all():
        # 馬番が無い場合は結果側から補う
        rf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                         dtype={"race_id": str}, usecols=["race_id", "馬名", "馬番"])
        d = d.drop(columns=["bn"]).merge(rf, on=["race_id", "馬名"], how="left")
        d["bn"] = pd.to_numeric(d["馬番"], errors="coerce")
    d["bn"] = d["bn"].astype("Int64").astype(str).str.zfill(2)
    # レース内で正規化した勝率と、複勝順位
    d["p"] = d.groupby("race_id")["p_raw"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0)
    d["mr"] = d.groupby("race_id")["f_raw"].rank(ascending=False)
    print(f"照合できた: {len(d)}頭 / {d.race_id.nunique()}レース")

    rows, diffs = [], []
    agg = {"pre": [0.0, 0.0, 0], "fin": [0.0, 0.0, 0]}
    for rid, g in d.groupby("race_id", sort=False):
        a_pre = pick(g, "odds_pre", "pop_pre")
        a_fin = pick(g, "odds_fin", "pop_fin")
        for key, ax, pc in (("pre", a_pre, "pop_pre"), ("fin", a_fin, "pop_fin")):
            if ax:
                c, r = settle(g, ax, pay, rid, pc)
                agg[key][0] += c
                agg[key][1] += r
                agg[key][2] += 1
        if (a_pre is None) != (a_fin is None) or \
                (a_pre and a_fin and a_pre["bn"] != a_fin["bn"]):
            def desc(x):
                return (f"{x['馬名']}({x['bn']}番 {x['pop']:.0f}人気 "
                        f"{x['odds']:.1f}倍 EV{x['ev']:.2f} → {x['着']:.0f}着)"
                        if x else "—")
            kind = ("混入" if a_pre and not a_fin else
                    "消失" if a_fin and not a_pre else "入替")
            diffs.append({"race_id": rid, "種別": kind,
                          "7分前で選ぶ馬": desc(a_pre), "確定で選ぶ馬": desc(a_fin),
                          "7分前の的中": (a_pre["着"] == 1) if a_pre else None,
                          "確定の的中": (a_fin["着"] == 1) if a_fin else None})

    for key, lbl in (("fin", "確定オッズで選ぶ（従来）"), ("pre", "7分前オッズで選ぶ（実運用）")):
        c, r, n = agg[key]
        roi = r / c * 100 if c else float("nan")
        rows.append({"基準": lbl, "レース数": n, "投資": int(c), "払戻": int(r),
                     "回収率": round(roi, 1)})
        print(f"  {lbl:26s} {n:>3d}レース  回収率 {roi:6.1f}%")
    if agg["fin"][0] and agg["pre"][0]:
        d1 = agg["pre"][1] / agg["pre"][0] * 100 - agg["fin"][1] / agg["fin"][0] * 100
        print(f"  → 差 {d1:+.1f}pt")

    pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "replay_a_result.csv"),
                              index=False, encoding="utf-8-sig")
    dd = pd.DataFrame(diffs)
    dd.to_csv(os.path.join(BASE_DIR, "replay_a_diff.csv"),
              index=False, encoding="utf-8-sig")

    print(f"\n=== 買い目の差分 {len(dd)}件 ===")
    if len(dd):
        print(dd.種別.value_counts().to_string())
        print()
        for _, r in dd.head(12).iterrows():
            print(f"  [{r.種別}] {r.race_id}")
            print(f"      7分前: {r['7分前で選ぶ馬']}")
            print(f"      確定  : {r['確定で選ぶ馬']}")
        if len(dd) > 12:
            print(f"  … 他{len(dd)-12}件（replay_a_diff.csv に全件）")
    print("\n保存 → replay_a_result.csv / replay_a_diff.csv")


if __name__ == "__main__":
    sys.exit(main())
