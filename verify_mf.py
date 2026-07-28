# -*- coding: utf-8 -*-
"""MFの勝率/連対率/複勝率が正しく動いているかの健全性チェック → その上で選抜検証。

Part 1 健全性:
  ①単調性  勝率 ≤ 連対率 ≤ 複勝率 になっているか（各馬で）
  ②較正    「30%」と言った馬が実際に30%来ているか（確率の意味が合っているか）
  ③レース内合計 勝率の合計が1に近いか（独立な二値モデルなのでズレる想定・要確認）
  ④順位整合 MF複勝順位が本当に複勝率の降順になっているか
Part 2 選抜:
  MF順位・各確率を基準に、2-5番人気を対象とした選抜のROIを測る（<=2024学習/2025検証）。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    for c in ["着順_num", "単勝オッズ", "人気", "MF勝率", "MF連対率", "MF複勝率",
              "MF勝率順位", "MF連対順位", "MF複勝順位"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "MF勝率"])
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["ren"] = (d["着順_num"] <= 2).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    return d


def part1(d):
    log("\n" + "=" * 72)
    log("【Part 1】MF確率の健全性チェック")
    log("=" * 72)

    log("\n① 単調性: 勝率 ≤ 連対率 ≤ 複勝率 になっているか")
    ok1 = (d["MF勝率"] <= d["MF連対率"] + 1e-9).mean() * 100
    ok2 = (d["MF連対率"] <= d["MF複勝率"] + 1e-9).mean() * 100
    log(f"   勝率≤連対率: {ok1:.1f}%   連対率≤複勝率: {ok2:.1f}%   ※100%でないと矛盾")

    log("\n② 較正: 予測確率と実際の発生率が一致しているか")
    for col, act, nm in [("MF勝率", "win", "勝率"), ("MF連対率", "ren", "連対率"),
                         ("MF複勝率", "fuku", "複勝率")]:
        log(f"   ― {nm} ―")
        q = pd.qcut(d[col], 6, duplicates="drop")
        g = d.groupby(q, observed=True).agg(予測=(col, "mean"), 実際=(act, "mean"),
                                            n=(col, "size"))
        for idx, r in g.iterrows():
            gap = (r["実際"] - r["予測"]) * 100
            flag = "  ←ズレ大" if abs(gap) > 5 else ""
            log(f"     予測{r['予測']*100:5.1f}% → 実際{r['実際']*100:5.1f}% "
                f"(差{gap:+5.1f}pt, n={int(r['n'])}){flag}")

    log("\n③ レース内合計（勝率は1.0、連対率は2.0、複勝率は3.0が理論値）")
    s = d.groupby("race_id")[["MF勝率", "MF連対率", "MF複勝率"]].sum()
    log(f"   勝率の合計   中央値 {s['MF勝率'].median():.2f}（理論 1.00）")
    log(f"   連対率の合計 中央値 {s['MF連対率'].median():.2f}（理論 2.00）")
    log(f"   複勝率の合計 中央値 {s['MF複勝率'].median():.2f}（理論 3.00）")
    log("   ※独立な二値モデルの平均なので合計は合わない＝確率の絶対値は信用せず順位で使うべき")

    log("\n④ 順位整合: 順位が確率の降順になっているか")
    for pcol, rcol, nm in [("MF勝率", "MF勝率順位", "勝率"),
                           ("MF連対率", "MF連対順位", "連対率"),
                           ("MF複勝率", "MF複勝順位", "複勝率")]:
        chk = d.groupby("race_id").apply(
            lambda g: (g[pcol].rank(ascending=False, method="min") == g[rcol]).all(),
            include_groups=False)
        log(f"   {nm}: 整合しているレース {chk.mean()*100:.1f}%")

    log("\n⑤ 実測: 各順位の実際の成績（順位が機能しているか）")
    log(f"   {'順位':<6}{'勝率(MF勝率順)':>16}{'連対(MF連対順)':>16}{'複勝(MF複勝順)':>16}")
    for r in range(1, 7):
        a = d[d["MF勝率順位"] == r]["win"].mean() * 100
        b = d[d["MF連対順位"] == r]["ren"].mean() * 100
        c = d[d["MF複勝順位"] == r]["fuku"].mean() * 100
        log(f"   {r}位  {a:14.1f}%{b:15.1f}%{c:15.1f}%")


def part2(d):
    log("\n" + "=" * 72)
    log("【Part 2】MF順位を軸にした選抜 ― 2-5番人気を対象に")
    log("=" * 72)
    s = d[(d["人気"] >= 2) & (d["人気"] <= 5)].copy()
    base = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
    log(f"  対象 {len(s):,}頭（2-5番人気）  全部買った場合の単勝ROI = {base:.1f}%")
    log(f"\n  {'選抜条件':<30}{'n':>7}{'勝率':>7}{'平均オッズ':>10}{'単勝ROI':>9}{'複勝率':>8}")
    conds = []
    for r in [1, 2, 3]:
        conds.append((f"MF勝率順位 {r}位", s["MF勝率順位"] == r))
        conds.append((f"MF複勝順位 {r}位", s["MF複勝順位"] == r))
    conds += [
        ("MF勝率1位 かつ MF複勝1位", (s["MF勝率順位"] == 1) & (s["MF複勝順位"] == 1)),
        ("MF3系すべて1位", (s["MF勝率順位"] == 1) & (s["MF連対順位"] == 1) & (s["MF複勝順位"] == 1)),
        ("MF勝率1位 かつ 連対1位", (s["MF勝率順位"] == 1) & (s["MF連対順位"] == 1)),
        ("MF勝率1-2位 かつ 複勝1-2位",
         (s["MF勝率順位"] <= 2) & (s["MF複勝順位"] <= 2)),
    ]
    for nm, mask in conds:
        sub = s[mask]
        if len(sub) < 100:
            continue
        roi = (sub["win"] * sub["単勝オッズ"]).sum() / len(sub) * 100
        log(f"  {nm:<30}{len(sub):7d}{sub['win'].mean()*100:6.1f}%"
            f"{sub['単勝オッズ'].mean():9.1f}{roi:8.1f}%{sub['fuku'].mean()*100:7.1f}%")

    log(f"\n  ― 人気別に見る（MF勝率1位 かつ MF複勝1位 の馬）―")
    log(f"  {'人気':<8}{'n':>7}{'勝率':>7}{'単勝ROI':>9}{'複勝率':>8}")
    both = s[(s["MF勝率順位"] == 1) & (s["MF複勝順位"] == 1)]
    for p in [2, 3, 4, 5]:
        sub = both[both["人気"] == p]
        if len(sub) < 50:
            continue
        roi = (sub["win"] * sub["単勝オッズ"]).sum() / len(sub) * 100
        log(f"  {p}番人気{'':<3}{len(sub):7d}{sub['win'].mean()*100:6.1f}%{roi:8.1f}%"
            f"{sub['fuku'].mean()*100:7.1f}%")


def main():
    d = load()
    log(f"検証データ: {d['race_id'].nunique()}レース {len(d):,}頭（2025・<=2024学習）")
    part1(d)
    part2(d)


if __name__ == "__main__":
    main()
