# -*- coding: utf-8 -*-
"""評価（S/A/B/D）を作り直して本番用の較正器を書き出す（2026-08-16）

要件（ユーザー指定）
  「Sは高確率で最低馬券内、あわよくば1着じゃないと使えない」

これまでにわかったこと
  ① 現行の評価はMFモデルだけで決めていた。S は複勝率75.8%・勝率41.8%。
  ② 市場（オッズ）を入れると同じ人数で複勝80.8%・勝率49.4%になる。
  ③ ただし「市場のみ」と「市場＋モデル」はほぼ同じ（80.8% vs 80.8%）。
     つまり評価の精度向上はほぼ全部が市場を入れたことによるもので、
     モデルは評価にはあまり足していない。これは正直に書いておく。
  ④ 「市場1位かつモデル1位」のような一致要求は劣る（68.8%）。連続スコアが良い。

作るもの
  複勝確率と勝率を、市場とモデルの2次元ロジスティックで出し直し、
    score = P(複勝) + P(1着)     （0〜2の範囲）
  として固定しきい値で S/A/B/D に切る。
  固定しきい値なのは、少頭数レースで実力のない馬にSが付くのを防ぐため。

しきい値の決め方
  Sは「1日に数頭」かつ「複勝率85%以上」を狙って上位0.8%に置く。
  A/B/D はそこから、各帯の複勝率が明確に分かれるように置く。

⚠ 較正は walk-forward で確かめたうえで、本番用には全年で学習し直す。
  ここで報告する数字は walk-forward（未来を見ていない）のもの。

実行: python build_grade.py → grade_calib.pkl
"""
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-6
S_PCT = 0.008          # Sは上位0.8%（1日36レースなら3〜4頭）


def log(m):
    print(m, flush=True)


def logit(v):
    return np.log(np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS)))


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
    return D


def feats(D):
    """本番でも同じ作り方ができること。使うのはオッズと2つの確率だけ。"""
    imp = 1.0 / D["odds"].clip(lower=1.01)
    imp = imp / D.groupby(D["race_id"])["odds"].transform(
        lambda s: (1.0 / s.clip(lower=1.01)).sum())
    X = pd.DataFrame(index=D.index)
    X["lm"] = logit(imp)
    X["l3"] = logit(D["c_top3"])
    X["l1"] = logit(D["c_win"])
    X["i3"] = X.lm * X.l3
    X["i1"] = X.lm * X.l1
    return X


F3 = ["lm", "l3", "i3"]
F1 = ["lm", "l1", "i1"]


def main():
    D = load()
    X = feats(D)
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")

    # ── walk-forward で成績を測る（未来を見ない）──────────────
    D["p3"] = np.nan
    D["p1"] = np.nan
    for y in YEARS[1:]:
        tr = D.年 < y
        te = D.年 == y
        D.loc[te, "p3"] = LogisticRegression(max_iter=1000).fit(
            X[tr][F3], D.top3[tr]).predict_proba(X[te][F3])[:, 1]
        D.loc[te, "p1"] = LogisticRegression(max_iter=1000).fit(
            X[tr][F1], D.win[tr]).predict_proba(X[te][F1])[:, 1]
    d = D[D.p3.notna()].copy()
    d["score"] = d.p3 + d.p1

    # ── しきい値を決める ────────────────────────────────
    nr = d.race_id.nunique()
    th_s = float(d.score.quantile(1 - S_PCT))
    th_a = float(d.score.quantile(1 - 0.08))
    th_b = float(d.score.quantile(1 - 0.25))
    TH = [("S", round(th_s, 3)), ("A", round(th_a, 3)), ("B", round(th_b, 3))]
    log(f"しきい値: S>={TH[0][1]} / A>={TH[1][1]} / B>={TH[2][1]} / それ未満D\n")

    d["g"] = np.select([d.score >= th_s, d.score >= th_a, d.score >= th_b],
                       ["S", "A", "B"], "D")
    d["g_old"] = np.select([(d.c_win + d.c_top2 + d.c_top3) >= 1.555,
                            (d.c_win + d.c_top2 + d.c_top3) >= 0.821,
                            (d.c_win + d.c_top2 + d.c_top3) >= 0.395], ["S", "A", "B"], "D")

    for col, lab in (("g_old", "現行（MFのみ）"), ("g", "新（市場＋MF）")):
        log(f"=== {lab} ===")
        log(f"  {'評価':<5}{'頭数':>9}{'割合':>7}{'1Rあたり':>9}{'複勝率':>9}"
            f"{'連対率':>8}{'勝率':>8}{'平均オッズ':>10}")
        for gg in ("S", "A", "B", "D"):
            s = d[d[col] == gg]
            if s.empty:
                continue
            log(f"  {gg:<5}{len(s):>9,}{len(s)/len(d)*100:>6.1f}%{len(s)/nr:>9.2f}"
                f"{s.top3.mean()*100:>8.1f}%{s.top2.mean()*100:>7.1f}%"
                f"{s.win.mean()*100:>7.1f}%{s.odds.median():>10.1f}")
        log("")

    log("=== Sの年ごとの安定性（新方式）===")
    log(f"  {'年':<7}{'頭数':>8}{'複勝率':>9}{'連対率':>8}{'勝率':>8}")
    ss = d[d.g == "S"]
    for y in YEARS[1:]:
        s = ss[ss.年 == y]
        if len(s):
            log(f"  {y:<7}{len(s):>8,}{s.top3.mean()*100:>8.1f}%"
                f"{s.top2.mean()*100:>7.1f}%{s.win.mean()*100:>7.1f}%")

    log("\n=== 要件の確認 ===")
    log(f"  「高確率で最低馬券内」 → S の複勝率 {ss.top3.mean()*100:.1f}%"
        f"（現行 {d[d.g_old=='S'].top3.mean()*100:.1f}%）")
    log(f"  「あわよくば1着」     → S の勝率   {ss.win.mean()*100:.1f}%"
        f"（現行 {d[d.g_old=='S'].win.mean()*100:.1f}%）")
    log(f"  出現頻度              → {nr/len(ss):.1f}レースに1頭"
        f"（1日36レースなら {len(ss)/nr*36:.1f}頭）")

    # ── 本番用に全年で学習し直して保存 ──────────────────────
    m3 = LogisticRegression(max_iter=1000).fit(X[F3], D.top3)
    m1 = LogisticRegression(max_iter=1000).fit(X[F1], D.win)
    with open("grade_calib.pkl", "wb") as fh:
        pickle.dump({"m3": m3, "m1": m1, "F3": F3, "F1": F1, "TH": TH,
                     "years": YEARS, "n": len(D),
                     "wf": {"S_top3": float(ss.top3.mean()), "S_win": float(ss.win.mean()),
                            "S_per_race": float(len(ss) / nr)}}, fh)
    log("\n→ grade_calib.pkl（本番用・全年で学習）")
    log(f"  係数(複勝): 市場 {m3.coef_[0][0]:+.3f} / モデル {m3.coef_[0][1]:+.3f}"
        f" / 交互作用 {m3.coef_[0][2]:+.3f}")
    log(f"  係数(勝率): 市場 {m1.coef_[0][0]:+.3f} / モデル {m1.coef_[0][1]:+.3f}"
        f" / 交互作用 {m1.coef_[0][2]:+.3f}")


if __name__ == "__main__":
    main()
