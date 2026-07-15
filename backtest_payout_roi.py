# -*- coding: utf-8 -*-
"""
backtest_payout_roi.py
──────────────────────
妙味重視の買い方を「印別」に2025 honest（out-of-sample）で回収率評価する。

印の再現（本番 keiba_predict と同じ役割分離）:
  ◎ = ブレンド(0.6*MF + 0.4*通常)1位（総合スコア1位の近似）
  ○▲△ = 複勝確率(place3)順の上位3頭（◎除く）
  × = 人気薄(人気>=6)で複勝確率が高い1頭（◎○▲△除く）
  ◎妙 = MF勝率1位（◎と別馬のときのみ）＝市場の穴の価値馬

回収率:
  単勝 = 単勝オッズ×100（model_result.csvから正確）… payout不要
  複勝/馬連/ワイド/3連複 = payout_data.csv(2025・馬番キー)と突合
  ※ payout_dataに2025が無ければ単勝のみ出力（スクレイプ完了待ち）

使い方:
  python backtest_payout_roi.py
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
YEAR = "2025"


def _read(name, cols):
    return pd.read_csv(os.path.join(BASE, name), usecols=cols, low_memory=False)


def load_marks():
    """2025全レースに印(◎○▲△×◎妙)と馬番を付与したDataFrameを返す。"""
    w = _read("model_result.csv",
              ["race_id", "馬名", "着順_num", "人気", "単勝オッズ", "勝ち確率"])
    p3 = _read("model_result_place3.csv", ["race_id", "馬名", "予測順位"]) \
        .rename(columns={"予測順位": "r_p3"})
    mf = _read("model_mf_result.csv", ["race_id", "馬名", "MF勝率"])
    rf = _read("race_features.csv", ["race_id", "馬名", "馬番"])
    for d in (w, p3, mf, rf):
        d["race_id"] = d["race_id"].astype(str)
    df = (w.merge(p3, on=["race_id", "馬名"], how="left")
            .merge(mf, on=["race_id", "馬名"], how="left")
            .merge(rf, on=["race_id", "馬名"], how="left"))
    df = df[df["race_id"].str.startswith(YEAR)].dropna(subset=["着順_num"]).copy()
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["人気"] = pd.to_numeric(df["人気"], errors="coerce")

    marks = []
    for rid, g in df.groupby("race_id"):
        g = g.copy()
        g["印"] = ""
        g["妙"] = ""
        g["blend"] = 0.6 * g["MF勝率"].fillna(g["勝ち確率"]) + 0.4 * g["勝ち確率"]
        # ◎ = ブレンド1位
        hon = g["blend"].idxmax()
        g.at[hon, "印"] = "◎"
        assigned = {hon}
        # ○▲△ = place3順（◎除く）上位3頭
        rest = g.loc[g.index.difference(assigned)].sort_values("r_p3")
        for mk, idx in zip(["○", "▲", "△"], rest.index[:3]):
            g.at[idx, "印"] = mk
            assigned.add(idx)
        # × = 人気>=6の残りで複勝(place3)最上位。居なければ残り全体
        rx = g.loc[g.index.difference(assigned)]
        pool = rx[rx["人気"] >= 6]
        if pool.empty:
            pool = rx
        if not pool.empty:
            g.at[pool.sort_values("r_p3").index[0], "印"] = "×"
        # ◎妙 = MF勝率1位（◎と別馬のときのみ）
        if g["MF勝率"].notna().any():
            mfi = g["MF勝率"].idxmax()
            if mfi != hon:
                g.at[mfi, "妙"] = "◎妙"
        marks.append(g)
    return pd.concat(marks, ignore_index=False)


def load_payouts():
    """payout_data.csvの2025分を券種別の突合辞書にする。無ければNone。"""
    path = os.path.join(BASE, "payout_data.csv")
    if not os.path.exists(path):
        return None
    p = pd.read_csv(path, dtype={"組み合わせ": str, "払戻金": str})
    p["race_id"] = p["race_id"].astype(str)
    p = p[p["race_id"].str.startswith(YEAR)]
    if p.empty:
        return None
    p["payout"] = pd.to_numeric(p["払戻金"], errors="coerce")

    def _nums(s):
        return tuple(int(x) for x in str(s).split("-") if x.isdigit())

    fuku, wide, umaren, fuku3 = {}, {}, {}, {}
    for _, r in p.iterrows():
        nums = _nums(r["組み合わせ"])
        if not nums or pd.isna(r["payout"]):
            continue
        key = r["race_id"]
        if r["券種"] == "複勝" and len(nums) == 1:
            fuku[(key, nums[0])] = r["payout"]
        elif r["券種"] == "ワイド" and len(nums) == 2:
            wide[(key, frozenset(nums))] = r["payout"]
        elif r["券種"] == "馬連" and len(nums) == 2:
            umaren[(key, frozenset(nums))] = r["payout"]
        elif r["券種"] == "3連複" and len(nums) == 3:
            fuku3[(key, frozenset(nums))] = r["payout"]
    return {"複勝": fuku, "ワイド": wide, "馬連": umaren, "3連複": fuku3,
            "レース数": p["race_id"].nunique()}


def roi_line(label, n, bets_return, hit):
    cost = n * 100
    roi = bets_return / cost * 100 if cost else 0
    hr = hit / n * 100 if n else 0
    return f"  {label:10} 点数{n:4d}  的中率{hr:5.1f}%  回収率{roi:6.1f}%"


def _fuku_line(sub, label, fuku, races):
    """印の複勝ROI: payoutがあるレースの該当馬だけを対象に集計。"""
    sub = sub[sub["馬番"].notna() & sub["race_id"].isin(races)]
    if sub.empty:
        return
    n = ret = hit = 0
    for _, r in sub.iterrows():
        n += 1
        if (r["race_id"], int(r["馬番"])) in fuku:
            ret += fuku[(r["race_id"], int(r["馬番"]))]
            hit += 1
    print(roi_line(label, n, ret, hit))


def _pair_roi(df, name, d, races, a_mark, b_col_val):
    """◎-相手の2頭券種(ワイド/馬連)ROI。b_col_val=('印','○')や('妙','◎妙')。"""
    n = ret = hit = 0
    for rid, g in df.groupby("race_id"):
        if rid not in races:
            continue
        ga = g[g["印"] == a_mark]
        gb = g[g[b_col_val[0]] == b_col_val[1]]
        if ga.empty or gb.empty:
            continue
        na, nb = ga["馬番"].iloc[0], gb["馬番"].iloc[0]
        if pd.isna(na) or pd.isna(nb) or int(na) == int(nb):
            continue
        n += 1
        if (rid, frozenset((int(na), int(nb)))) in d:
            ret += d[(rid, frozenset((int(na), int(nb))))]
            hit += 1
    if n:
        print(roi_line(name, n, ret, hit))


def _trio_roi(df, name, fuku3, races, third_mark):
    """◎○＋third_markの3連複ROI。"""
    n = ret = hit = 0
    for rid, g in df.groupby("race_id"):
        if rid not in races:
            continue
        picks = [g[g["印"] == m] for m in ("◎", "○", third_mark)]
        if any(v.empty for v in picks):
            continue
        nums = [v["馬番"].iloc[0] for v in picks]
        if any(pd.isna(x) for x in nums) or len({int(x) for x in nums}) < 3:
            continue
        n += 1
        if (rid, frozenset(int(x) for x in nums)) in fuku3:
            ret += fuku3[(rid, frozenset(int(x) for x in nums))]
            hit += 1
    if n:
        print(roi_line(name, n, ret, hit))


def main():
    print("印を再現中（2025 out-of-sample）...")
    df = load_marks()
    print(f"対象レース数: {df['race_id'].nunique()}\n")

    # ── 印別 単勝回収率（payout不要・単勝オッズ×100）──
    print("=" * 60)
    print("【印別 単勝回収率】2025 honest（単勝オッズ実測）")
    print("=" * 60)
    for mk in ["◎", "○", "▲", "△", "×"]:
        sub = df[df["印"] == mk]
        if sub.empty:
            continue
        win = sub["着順_num"] == 1
        print(roi_line(mk, len(sub), (sub.loc[win, "単勝オッズ"] * 100).sum(), win.sum()))
    myo = df[df["妙"] == "◎妙"]
    if not myo.empty:
        win = myo["着順_num"] == 1
        print(roi_line("◎妙", len(myo), (myo.loc[win, "単勝オッズ"] * 100).sum(), win.sum()))

    # ── 印別 複勝＆連系回収率（payout_data必要）──
    pay = load_payouts()
    if pay is None:
        print("\n※ payout_data.csvに2025分がまだありません。")
        print("  → スクレイプ完了後に再実行すると複勝・馬連・ワイド・3連複も出ます。")
        return

    print(f"\n（2025 payout: {pay['レース数']}レース分で照合）")
    print("=" * 60)
    print("【印別 複勝回収率】2025 honest（payout実測）")
    print("=" * 60)
    fuku = pay["複勝"]
    races = {k[0] for k in fuku}
    for mk in ["◎", "○", "▲", "△", "×"]:
        _fuku_line(df[df["印"] == mk], mk, fuku, races)
    _fuku_line(df[df["妙"] == "◎妙"], "◎妙", fuku, races)

    print("=" * 60)
    print("【妙味重視の連系回収率】◎軸")
    print("=" * 60)
    _pair_roi(df, "◎-○ ワイド",   pay["ワイド"], races, "◎", ("印", "○"))
    _pair_roi(df, "◎-◎妙 ワイド", pay["ワイド"], races, "◎", ("妙", "◎妙"))
    _pair_roi(df, "◎-○ 馬連",     pay["馬連"],  races, "◎", ("印", "○"))
    _pair_roi(df, "◎-◎妙 馬連",   pay["馬連"],  races, "◎", ("妙", "◎妙"))
    _trio_roi(df, "◎○▲ 3連複",   pay["3連複"], races, "▲")
    _trio_roi(df, "◎○× 3連複",   pay["3連複"], races, "×")


if __name__ == "__main__":
    main()
