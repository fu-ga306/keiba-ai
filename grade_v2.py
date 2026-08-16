# -*- coding: utf-8 -*-
"""評価の改善案を詰める（2026-08-16・第2段）

第1段(grade_audit.py)でわかったこと
  現行S     複勝率75.8% 勝率42.0%
  市場＋MF  複勝率80.9% 勝率49.6%   ← 明確に良い
  ただし「市場のみ」が測れなかった（人気が整数で1.7%に切れない）。
  市場のみでも同じ精度が出るなら、モデルは評価に何も足していないことになる。
  ここを先に確かめる。

そのうえで決めること
  「Sは高確率で最低馬券内、あわよくば1着」という要件に対し、
  Sをどれくらい絞るのが良いか。絞るほど複勝率は上がるが、出現しなくなる。
  1レースあたり何頭Sが出るかと複勝率の兼ね合いを見て決める。

実行: python grade_v2.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-6


def log(m):
    print(m, flush=True)


def logit(v):
    v = np.clip(v, EPS, 1 - EPS)
    return np.log(v / (1 - v))


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬番", "着順_num"])
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rf["bn"] = pd.to_numeric(rf["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    D = D.merge(rf[["race_id", "bn", "着順_num"]], on=["race_id", "bn"], how="left")
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D = D[D["着"].notna()].copy()
    D["win"] = (D["着"] == 1).astype(int)
    D["top2"] = (D["着"] <= 2).astype(int)
    D["top3"] = (D["着"] <= 3).astype(int)
    # 市場の推定確率（オッズを控除率で割り戻してレース内で正規化）
    D["imp"] = 1.0 / D.odds.clip(lower=1.01)
    D["imp"] = D.groupby("race_id")["imp"].transform(lambda s: s / s.sum())
    D["lm"] = logit(D.imp)
    return D


def wf(D, feats, target):
    """walk-forward: 各年を、それ以前の年だけで学習した係数で予測する"""
    out = pd.Series(np.nan, index=D.index)
    for y in YEARS[1:]:
        tr = D[D.年 < y]
        te = D.年 == y
        m = LogisticRegression(max_iter=1000).fit(tr[feats], tr[target])
        out[te] = m.predict_proba(D.loc[te, feats])[:, 1]
    return out


def main():
    D = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")

    D["lg3"] = logit(D.c_top3)
    D["lg1"] = logit(D.c_win)
    D["i3"] = D.lm * D.lg3
    D["i1"] = D.lm * D.lg1

    log("=== 市場のみ vs 市場＋MF（同じ人数で比べる）===")
    log("  市場のみで同じ精度なら、モデルは評価に何も足していない。")
    cands = {
        "市場のみ(複勝)": wf(D, ["lm"], "top3"),
        "市場＋MF(複勝)": wf(D, ["lm", "lg3", "i3"], "top3"),
        "市場のみ(勝率)": wf(D, ["lm"], "win"),
        "市場＋MF(勝率)": wf(D, ["lm", "lg1", "i1"], "win"),
        "現行(MFのみ)": D.c_win + D.c_top2 + D.c_top3,
    }
    d = D[cands["市場＋MF(複勝)"].notna()].copy()
    for k, v in cands.items():
        d[k] = v[d.index]

    for pct in (0.5, 1.0, 1.7, 3.0):
        log(f"\n  --- 上位{pct}%を S としたとき ---")
        log(f"  {'案':<22}{'S頭数':>8}{'1R当り':>8}{'複勝率':>9}{'勝率':>8}{'連対率':>8}")
        for k in cands:
            q = d[k].rank(pct=True, ascending=False)
            s = d[q <= pct / 100]
            if s.empty:
                continue
            log(f"  {k:<22}{len(s):>8,}{len(s)/d.race_id.nunique():>7.2f}"
                f"{s.top3.mean()*100:>8.1f}%{s.win.mean()*100:>7.1f}%"
                f"{s.top2.mean()*100:>7.1f}%")

    # ── Sをどこで切るか ──────────────────────────────────
    log("\n=== Sの絞り具合と質のトレードオフ（市場＋MF・複勝＋勝率）===")
    log("  スコア = 複勝確率 + 勝率（1着に寄せるため勝率を足す）")
    d["sc"] = d["市場＋MF(複勝)"] + d["市場＋MF(勝率)"]
    nr = d.race_id.nunique()
    log(f"  {'上位':>6}{'S頭数':>9}{'1R当り':>8}{'何レースに1頭':>13}"
        f"{'複勝率':>9}{'勝率':>8}{'連対率':>8}")
    for pct in (0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 1.7, 2.5, 4.0):
        q = d.sc.rank(pct=True, ascending=False)
        s = d[q <= pct / 100]
        per = len(s) / nr
        log(f"  {pct:>5.1f}%{len(s):>9,}{per:>8.2f}{1/per if per else 0:>12.1f}回"
            f"{s.top3.mean()*100:>8.1f}%{s.win.mean()*100:>7.1f}%{s.top2.mean()*100:>7.1f}%")

    # ── 年ごとにブレないか ────────────────────────────────
    log("\n=== 年ごとの安定性（上位1.0%をSとしたとき）===")
    q = d.sc.rank(pct=True, ascending=False)
    s = d[q <= 0.01]
    log(f"  {'年':<7}{'S頭数':>8}{'複勝率':>9}{'勝率':>8}")
    for y in YEARS[1:]:
        ss = s[s.年 == y]
        if len(ss):
            log(f"  {y:<7}{len(ss):>8,}{ss.top3.mean()*100:>8.1f}%{ss.win.mean()*100:>7.1f}%")

    d.to_csv("grade_v2_result.csv", index=False, encoding="utf-8-sig",
             columns=["race_id", "bn", "年", "pr", "odds", "着", "sc",
                      "市場＋MF(複勝)", "市場＋MF(勝率)", "現行(MFのみ)"])
    log("\n→ grade_v2_result.csv")


if __name__ == "__main__":
    main()
