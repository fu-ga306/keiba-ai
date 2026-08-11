# -*- coding: utf-8 -*-
"""市場確率を説明変数に入れた2次元較正。オプティマイザの呪いを直せるか。

なぜ必要か（2026-08-12）
  現在の較正は全馬まとめた周辺分布で行っている。全体のECEは0.0052と良い。
  ところが**買う馬だけ**を取り出すと予測18.0%に対し実勝率6.8%（比0.376）。
  5年OOF 207,518頭で確認したところ、EVと実払戻の順位相関は -0.1554、
  5年すべて負だった。EVが高い馬ほど損をする。

  原因は較正の不備ではなく**選び方**。EV = p × odds なので、
  オッズが大きい側でpが過大だとEVは二重に膨らむ。そこだけを選ぶので
  誤差の上側の裾を集めることになる（オプティマイザの呪い）。

やり方
  バケット分割はサンプルの薄い大穴帯で階段状になるので使わない。
  ロジスティック回帰で連続的に較正する。

    入力: logit(モデル確率), logit(市場確率), その交互作用
    出力: 実勝率

  市場確率を説明変数に入れると「モデルの自信」と「市場の評価」の
  妥協点を滑らかに学習する。市場と割れるほどモデル側を割り引く。

  較正器自体も walk-forward で作る（学習=検証年より前、適用=検証年）。
  較正器の学習に検証年を使うとそこでリークする。

判定
  1. EV帯ごとの「比（実勝率/予測）」が1.0付近に収まるか
  2. 収まった結果、買える馬が何頭残るか（ゼロになる可能性が高い）
  3. 残った馬のROIはどうか

実行: python calib2d.py → calib2d_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    # 市場確率: 1/オッズをレース内で正規化する（控除率を抜く）
    D["m_raw"] = 1.0 / D["odds"]
    D["m"] = D.groupby("race_id")["m_raw"].transform(lambda s: s / s.sum())
    D["pay"] = D["win"] * D["odds"]
    return D


def design(d):
    lm, lp = logit(d["m"].values), logit(d["c_win_n"].values)
    return np.column_stack([lp, lm, lp * lm])


def main():
    D = load()
    print(f"5年OOF {len(D):,}頭 / {D.race_id.nunique():,}レース", flush=True)

    # 較正器も walk-forward（2021は学習材料が無いので2022以降を評価）
    D["p2"] = np.nan
    for y in YEARS[1:]:
        tr, te = D[D.年 < y], D.年 == y
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(design(tr), tr["win"].values)
        D.loc[te, "p2"] = lr.predict_proba(design(D[te]))[:, 1]
        print(f"  {y} 較正 学習{len(tr):,}頭  係数 "
              f"モデル{lr.coef_[0][0]:+.3f} 市場{lr.coef_[0][1]:+.3f} "
              f"交互{lr.coef_[0][2]:+.3f}", flush=True)
    E = D[D.p2.notna()].copy()
    # レース内で正規化してから期待値を作り直す
    E["p2n"] = E.groupby("race_id")["p2"].transform(lambda s: s / s.sum())
    E["EV2"] = E["p2n"] * E["odds"]

    rows = []
    for lbl, pcol, ecol in (("現行(1次元較正)", "c_win_n", "EV_tan"),
                            ("2次元較正", "p2n", "EV2")):
        from scipy import stats
        r = stats.spearmanr(E[ecol], E["pay"]).correlation
        print(f"\n=== {lbl} ===  EVと実払戻の順位相関 {r:+.4f}", flush=True)
        print(f"{'EV帯':>10} {'頭数':>8} {'予測':>7} {'実勝率':>7} {'比':>6} {'ROI':>7}")
        for lo, hi in [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 999)]:
            s = E[(E[ecol] >= lo) & (E[ecol] < hi)]
            if len(s) < 200:
                continue
            ratio = s.win.mean() / s[pcol].mean()
            print(f"{lo:.1f}-{hi:<5.1f} {len(s):>8,} {s[pcol].mean():>7.4f} "
                  f"{s.win.mean():>7.4f} {ratio:>6.2f} {s.pay.mean()*100:>6.1f}%")
            rows.append({"較正": lbl, "EV下": lo, "EV上": hi, "頭数": len(s),
                         "予測": round(s[pcol].mean(), 4), "実勝率": round(s.win.mean(), 4),
                         "比": round(ratio, 2), "ROI": round(s.pay.mean() * 100, 1)})
        # 買える馬が何頭残るか
        for th in (1.0, 1.3, 1.7, 2.2):
            s = E[E[ecol] >= th]
            roi = s.pay.mean() * 100 if len(s) else float("nan")
            print(f"    EV>={th}: {len(s):>7,}頭 ({len(s)/len(E)*100:5.2f}%)  ROI {roi:6.1f}%")
            rows.append({"較正": lbl, "EV下": th, "EV上": None, "頭数": len(s),
                         "予測": None, "実勝率": None, "比": None,
                         "ROI": round(roi, 1) if len(s) else None})
    pd.DataFrame(rows).to_csv("calib2d_result.csv", index=False, encoding="utf-8-sig")
    print("\n保存 → calib2d_result.csv")


if __name__ == "__main__":
    main()
