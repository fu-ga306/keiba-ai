# -*- coding: utf-8 -*-
"""2025年だけモデルが効かない理由を探す（2026-09-04）

年別（コース脚質バイアス修正後）
  2021:129%  2022:109%  2023:149%  2024:132%  2025:97%
シャッフルとの差  +36 +38 +51 +99 → +7

市場側は変わっていない（シャッフルの水準は72.8%で普通）。
**モデルの上乗せだけが消えている。**

⚠ 2025年はもう見てしまった。ここで分かったことをもとに構成を変えると
  事前登録の枠組みが無効になる。**理解までにとどめ、変更はしない。**

調べること
  ① データの質が2025年だけ落ちていないか（列ごとの欠損率）
  ② 確率の較正が2025年だけ崩れていないか
  ③ 特徴量の分布が2025年でずれていないか
  ④ 場・クラス・月で偏りがないか
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JYO = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
       "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def log(m):
    print(m, flush=True)


def main():
    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]

    # ── ① 欠損率を年ごとに ──────────────────────────────────
    log("  === ① データの質（欠損率）を年ごとに ===")
    BATCH = 40
    acc = {}
    for i in range(0, len(cols), BATCH):
        part = cols[i:i + BATCH]
        for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                              usecols=part + ["race_id"], dtype={"race_id": str},
                              chunksize=100000, low_memory=True):
            ch["年"] = ch["race_id"].str[:4]
            sub = ch.reindex(columns=part)
            for y, idx in ch.groupby("年").groups.items():
                a = acc.setdefault(y, {"n": 0, "na": np.zeros(len(cols))})
                a["n"] += len(idx)
                a["na"][i:i + len(part)] += sub.loc[idx].isna().sum().values
    yrs = sorted(k for k in acc if k >= "2021")
    log("  %-6s %8s %10s" % ("年", "行数", "平均欠損率"))
    for y in yrs:
        a = acc[y]
        log("  %-6s %8d %9.1f%%" % (y, a["n"], (a["na"] / a["n"]).mean() * 100))

    log("")
    log("  2025年だけ欠損が増えた列（上位10）")
    base = np.mean([acc[y]["na"] / acc[y]["n"] for y in yrs if y != "2025"], axis=0)
    cur = acc["2025"]["na"] / acc["2025"]["n"]
    d = pd.DataFrame({"列": cols, "2021-24": base * 100, "2025": cur * 100})
    d["差"] = d["2025"] - d["2021-24"]
    for _, r in d.sort_values("差", ascending=False).head(10).iterrows():
        log("    %-30s %6.1f%% → %6.1f%%  %+6.1f"
            % (r["列"][:30], r["2021-24"], r["2025"], r["差"]))

    # ── ② 較正が年ごとに崩れていないか ──────────────────────
    log("")
    log("  === ② 確率の較正を年ごとに ===")
    p = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred.csv"),
                    dtype={"race_id": str, "bn": str})
    p["着"] = pd.to_numeric(p["着"], errors="coerce")
    p = p[p["着"].notna()]
    p["年"] = p["race_id"].str[:4]
    p["gap"] = p.p1 / p.q
    log("  軸（gap最大かつ1.5以上）について")
    log("  %-6s %8s %12s %12s %8s" % ("年", "頭数", "予測勝率", "実際の勝率", "比"))
    ax = p.loc[p.groupby("race_id")["gap"].idxmax()]
    ax = ax[ax["gap"] >= 1.5]
    for y, g in ax.groupby("年"):
        pr, ac = g["p1"].mean(), (g["着"] == 1).mean()
        log("  %-6s %8d %11.1f%% %11.1f%% %8.2f" % (y, len(g), pr*100, ac*100, ac/max(pr,1e-9)))

    # ── ④ 場・月で偏りがないか ─────────────────────────────
    log("")
    log("  === ④ 2025年の内訳 ===")
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv = jv[jv.券種 == "単勝"].copy()
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    jv["bn"] = pd.to_numeric(jv["組み合わせ"], errors="coerce")
    ax2 = ax.copy()
    ax2["bn2"] = pd.to_numeric(ax2["bn"], errors="coerce")
    ax2 = ax2.merge(jv[["race_id", "bn", "払戻金"]].rename(columns={"bn": "bn2"}),
                    on=["race_id", "bn2"], how="left")
    ax2["払戻"] = ax2["払戻金"].fillna(0.0)
    ax2["場"] = ax2["race_id"].str[4:6].map(JYO)
    log("  場ごとの回収率")
    log("  %-6s" % "場" + "".join("%8s" % y for y in yrs))
    for j, g in ax2.groupby("場"):
        row = []
        for y in yrs:
            s = g[g["年"] == y]
            row.append("%7.0f%%" % s["払戻"].mean() if len(s) >= 40 else "      -")
        log("  %-6s" % j + "".join(row))


if __name__ == "__main__":
    main()
