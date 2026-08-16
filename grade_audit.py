# -*- coding: utf-8 -*-
"""評価（S/A/B/D）の精度を測り、改善案を比べる（2026-08-16）

求めていること（ユーザー指定）
  「Sは高確率で最低馬券内、あわよくば1着じゃないと使えない」
  → Sの複勝率がまず高いこと。そのうえで勝率も高いこと。

今の作り方
  score = 勝ち確率 + 連対確率 + 複勝確率（すべてMFモデル＝市場を見ない）
  S >= 1.555 / A >= 0.821 / B >= 0.395 / それ未満は D

疑っている点
  MFモデルは市場をわざと見ない。だが市場の重みはモデルの9〜32.6倍ある。
  つまり市場を無視したスコアで評価を切ると、市場が「来る」と言っている馬を
  取りこぼし、市場が「来ない」と言っている馬にSを付けてしまう。

比べる案
  案0 現行            : MFのみの合成スコア
  案1 市場のみ         : 人気（オッズ）だけで決める。これに勝てないなら評価は無意味
  案2 市場＋MF        : 2次元ロジスティックで複勝確率を出し直す
  案3 案2＋勝率重視    : 複勝だけでなく1着確率も見て、Sは両方高い馬に限る

実行: python grade_audit.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
GRADE_TH = [("S", 1.555), ("A", 0.821), ("B", 0.395)]


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    # 着順は race_features から。win/top3 は bet_cache にある想定だが念のため確認
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬番", "着順_num", "出走頭数"])
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rf["bn"] = pd.to_numeric(rf["馬番"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
    D = D.merge(rf[["race_id", "bn", "着順_num", "出走頭数"]], on=["race_id", "bn"], how="left")
    D["着"] = pd.to_numeric(D["着順_num"], errors="coerce")
    D["win"] = (D["着"] == 1).astype(int)
    D["top2"] = (D["着"] <= 2).astype(int)
    D["top3"] = (D["着"] <= 3).astype(int)
    D = D[D["着"].notna()].copy()
    return D


def summarize(D, gcol, label):
    log(f"\n=== {label} ===")
    log(f"  {'評価':<5}{'頭数':>9}{'割合':>7}{'複勝率':>9}{'勝率':>8}"
        f"{'連対率':>8}{'単勝ROI':>9}{'複勝ROI':>9}")
    tot = len(D)
    for g in ("S", "A", "B", "C", "D"):
        sub = D[D[gcol] == g]
        if sub.empty:
            continue
        troi = (sub.win * sub.odds * 100).sum() / (len(sub) * 100) * 100
        log(f"  {g:<5}{len(sub):>9,}{len(sub)/tot*100:>6.1f}%"
            f"{sub.top3.mean()*100:>8.1f}%{sub.win.mean()*100:>7.1f}%"
            f"{sub.top2.mean()*100:>7.1f}%{troi:>8.1f}%{'':>9}")


def cut(score, ths):
    """スコアを上から ths の割合で切って S/A/B/D にする（人数割合を揃えて比較する）"""
    q = score.rank(pct=True, ascending=False)
    return np.select([q <= ths[0], q <= ths[1], q <= ths[2]], ["S", "A", "B"], "D")


def main():
    D = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース（2021-2025 walk-forward OOS）")

    # ── 案0 現行 ─────────────────────────────────────────
    D["sc0"] = D.c_win + D.c_top2 + D.c_top3
    D["g0"] = np.select([D.sc0 >= 1.555, D.sc0 >= 0.821, D.sc0 >= 0.395],
                        ["S", "A", "B"], "D")
    summarize(D, "g0", "案0 現行（MFのみ・固定しきい値）")
    share = [(D.g0 == g).mean() for g in ("S", "A", "B")]
    ths = [share[0], share[0] + share[1], share[0] + share[1] + share[2]]
    log(f"\n  現行の人数割合 S{share[0]*100:.1f}% A{share[1]*100:.1f}% B{share[2]*100:.1f}%")
    log("  → 以下の案は、この割合を揃えて比べる（同じ人数でどれだけ当たるか）")

    # ── 案1 市場のみ ───────────────────────────────────────
    D["sc1"] = -D.pr.astype(float)      # 人気が良いほど高スコア
    D["g1"] = cut(D.sc1, ths)
    summarize(D, "g1", "案1 市場のみ（人気順）※これに勝てないと評価の意味がない")

    # ── 案2 市場＋MF（2次元ロジスティック）──────────────────
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    D["imp"] = (1.0 / D.odds.clip(lower=1.01))
    D["imp"] = D.groupby("race_id")["imp"].transform(lambda s: s / s.sum())
    for c, src in (("lm", "imp"), ("lg", "c_top3")):
        v = D[src].clip(eps, 1 - eps)
        D[c] = np.log(v / (1 - v))
    D["ix"] = D.lm * D.lg
    # walk-forward: 各年を、それ以前の年だけで学習した係数で予測する
    D["p2"] = np.nan
    for y in YEARS[1:]:
        tr, te = D[D.年 < y], D.年 == y
        m = LogisticRegression(max_iter=1000).fit(tr[["lm", "lg", "ix"]], tr.top3)
        D.loc[te, "p2"] = m.predict_proba(D.loc[te, ["lm", "lg", "ix"]])[:, 1]
    d2 = D[D.p2.notna()].copy()
    d2["g2"] = cut(d2.p2, ths)
    summarize(d2, "g2", "案2 市場＋MF（複勝確率を2次元で出し直す）")

    # ── 案3 案2＋勝率も見る ────────────────────────────────
    D["pw"] = np.nan
    for y in YEARS[1:]:
        tr, te = D[D.年 < y], D.年 == y
        v = D.c_win.clip(eps, 1 - eps)
        D["lgw"] = np.log(v / (1 - v))
        D["ixw"] = D.lm * D.lgw
        m = LogisticRegression(max_iter=1000).fit(
            D.loc[D.年 < y, ["lm", "lgw", "ixw"]], D.loc[D.年 < y, "win"])
        D.loc[te, "pw"] = m.predict_proba(D.loc[te, ["lm", "lgw", "ixw"]])[:, 1]
    d3 = D[D.p2.notna() & D.pw.notna()].copy()
    # 1着3点・2着以内2点・3着以内1点の期待点に相当させる
    d3["sc3"] = d3.pw * 2 + d3.p2
    d3["g3"] = cut(d3.sc3, ths)
    summarize(d3, "g3", "案3 市場＋MF・勝率を2倍重視（Sを1着に寄せる）")

    log("\n=== まとめ：Sの質だけを並べる ===")
    log(f"  {'案':<32}{'S頭数':>8}{'S複勝率':>9}{'S勝率':>8}")
    for lab, df, c in (("案0 現行（MFのみ）", D, "g0"),
                       ("案1 市場のみ（人気順）", D, "g1"),
                       ("案2 市場＋MF", d2, "g2"),
                       ("案3 市場＋MF・勝率重視", d3, "g3")):
        s = df[df[c] == "S"]
        log(f"  {lab:<32}{len(s):>8,}{s.top3.mean()*100:>8.1f}%{s.win.mean()*100:>7.1f}%")

    d3.to_csv("grade_audit_result.csv", index=False, encoding="utf-8-sig",
              columns=["race_id", "bn", "年", "pr", "odds", "着", "sc0", "g0",
                       "p2", "pw", "sc3", "g3"])


if __name__ == "__main__":
    main()
