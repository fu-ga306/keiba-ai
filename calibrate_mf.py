# -*- coding: utf-8 -*-
"""MF確率の較正修正 → 較正済み確率で選抜を組み直す。

2026-07-28の健全性チェックで判明した欠陥:
  ・勝率モデルの較正が崩壊（予測29.1%→実際16.4%、レース内合計1.31＝31%の水増し）
  ・連対率も上位でズレ（41.4%→33.1%）
  ・複勝率は良好（合計3.02・ズレ±1.6pt以内）
→ 較正は再学習不要で直せる（予測値の変換のみ）。直してから選抜を再設計する。

手順:
  1. 等調回帰(isotonic)で「予測→実際」の写像を学習し補正
  2. レース内で合計が理論値(1/2/3)になるよう正規化
  3. 較正前後で信頼度(Brier/対数尤度)と選抜ROIを比較
※較正の学習と評価が同じデータだと自己満足になるので、2025を実開催日で
  前半(較正の学習)/後半(評価)に分ける。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    for c in ["着順_num", "単勝オッズ", "人気", "MF勝率", "MF連対率", "MF複勝率",
              "MF勝率順位", "MF複勝順位", "MF連対順位"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "MF勝率"])
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["ren"] = (d["着順_num"] <= 2).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    return d.dropna(subset=["dt"])


TARGETS = [("MF勝率", "win", 1.0, "勝率"), ("MF連対率", "ren", 2.0, "連対率"),
           ("MF複勝率", "fuku", 3.0, "複勝率")]


def main():
    d = load()
    mid = d["dt"].quantile(0.5)
    tr = d[d["dt"] <= mid].copy()
    te = d[d["dt"] > mid].copy()
    log(f"較正の学習(前半) {tr['race_id'].nunique()}R / 評価(後半) {te['race_id'].nunique()}R")

    log("\n" + "=" * 70)
    log("【1】較正の効果（評価は後半データ・学習に使っていない）")
    log("=" * 70)
    for col, act, total, nm in TARGETS:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(tr[col], tr[act])
        cal = iso.predict(te[col])
        te[col + "_cal"] = cal
        # レース内で理論値に正規化
        s = te.groupby("race_id")[col + "_cal"].transform("sum")
        te[col + "_norm"] = te[col + "_cal"] / s.replace(0, np.nan) * total
        b0 = brier_score_loss(te[act], te[col].clip(0, 1))
        b1 = brier_score_loss(te[act], te[col + "_cal"].clip(0, 1))
        b2 = brier_score_loss(te[act], te[col + "_norm"].clip(0, 1))
        l0 = -log_loss(te[act], te[col].clip(1e-6, 1 - 1e-6))
        l2 = -log_loss(te[act], te[col + "_norm"].clip(1e-6, 1 - 1e-6))
        log(f"  {nm}: Brier {b0:.5f} → 較正{b1:.5f} → 正規化{b2:.5f}  "
            f"(LL {l0:.5f}→{l2:.5f})")
        g0 = te.groupby("race_id")[col].sum().median()
        g2 = te.groupby("race_id")[col + "_norm"].sum().median()
        log(f"        レース内合計 中央値 {g0:.2f} → {g2:.2f}（理論{total:.1f}）")

    log("\n" + "=" * 70)
    log("【2】較正後の確率で見た較正曲線（勝率）")
    log("=" * 70)
    for tag, col in [("較正前", "MF勝率"), ("較正後", "MF勝率_norm")]:
        q = pd.qcut(te[col], 5, duplicates="drop")
        g = te.groupby(q, observed=True).agg(予測=(col, "mean"), 実際=("win", "mean"))
        s = "  ".join(f"{r['予測']*100:.0f}→{r['実際']*100:.0f}" for _, r in g.iterrows())
        log(f"  {tag}: {s}   ※予測→実際(%)")

    log("\n" + "=" * 70)
    log("【3】較正済み確率での選抜 ― 2-5番人気（後半データ）")
    log("=" * 70)
    s = te[(te["人気"] >= 2) & (te["人気"] <= 5)].copy()
    base = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
    log(f"  対象{len(s)}頭  全部買った場合 = {base:.1f}%")
    # 較正済み確率 × オッズ = 期待値
    s["EV単"] = s["MF勝率_norm"] * s["単勝オッズ"]
    s["EV複"] = s["MF複勝率_norm"]
    log(f"\n  {'選抜':<30}{'n':>6}{'勝率':>7}{'平均オッズ':>10}{'単勝ROI':>9}")
    for frac in [0.05, 0.10, 0.20, 0.33]:
        th = s["EV単"].quantile(1 - frac)
        p = s[s["EV単"] >= th]
        if len(p) < 40:
            continue
        roi = (p["win"] * p["単勝オッズ"]).sum() / len(p) * 100
        log(f"  {'期待値(較正勝率×オッズ)上位' + f'{frac:.0%}':<30}{len(p):6d}"
            f"{p['win'].mean()*100:6.1f}%{p['単勝オッズ'].mean():9.1f}{roi:8.1f}%")
    for th_ev in [1.0, 1.1, 1.2]:
        p = s[s["EV単"] >= th_ev]
        if len(p) < 40:
            continue
        roi = (p["win"] * p["単勝オッズ"]).sum() / len(p) * 100
        log(f"  {'期待値 ≥ ' + f'{th_ev:.1f}':<30}{len(p):6d}"
            f"{p['win'].mean()*100:6.1f}%{p['単勝オッズ'].mean():9.1f}{roi:8.1f}%")

    log(f"\n  ― 参考: 較正前の期待値で同じ選抜をした場合 ―")
    s["EV単_旧"] = s["MF勝率"] * s["単勝オッズ"]
    for frac in [0.05, 0.10, 0.20]:
        th = s["EV単_旧"].quantile(1 - frac)
        p = s[s["EV単_旧"] >= th]
        roi = (p["win"] * p["単勝オッズ"]).sum() / len(p) * 100
        log(f"  {'（旧）期待値上位' + f'{frac:.0%}':<30}{len(p):6d}"
            f"{p['win'].mean()*100:6.1f}%{p['単勝オッズ'].mean():9.1f}{roi:8.1f}%")


if __name__ == "__main__":
    main()
