# -*- coding: utf-8 -*-
"""7分前に決めた場合、バックテストの数字がどれだけ変わるかを測る（2026-08-22）

なぜ要るか
  バックテストは確定オッズで買い目を選んでいる（race_features の単勝オッズ）。
  だが本番は発走7分前に決めるしかない。締切後のオッズは賭けた後の情報なので使えない。
  2026-08-22の実測では、7分前の判定と締切時の判定で23点中4点(17%)が違った。
  つまりBTの120.6%は「確定オッズを見てから選べる」前提の数字で、
  そのまま再現できるとは限らない。ここを測らないと実測と比べる基準が作れない。

測り方（再学習しない）
  resid_pred.csv には確定オッズでの q, p, gap が入っている。
  p ∝ exp(f)×q なので gap = p/q = exp(f)/Z。つまり log(gap) から f が
  レース内定数を除いて復元できる。復元した f に「7分前のオッズ」を組み合わせれば、
  7分前に判定した場合の gap を厳密に作り直せる。学習し直す必要はない。

7分前のオッズの作り方
  odds_history.csv の 直前(-7分) と 締切前(-1分) の実測ペア1,119頭から、
  オッズ帯ごとに「7分前オッズ ÷ 確定オッズ」の分布を作り、そこから抽出する。
  平均のズレ（人気馬は買われ、人気薄は見放される）とバラつきの両方が入る。

払戻は確定オッズで計算する
  選ぶのは7分前、払うのは確定。本番と同じ非対称をそのまま再現する。

実行: python odds_timing_bt.py
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_MIN = 1.5
N_TRIAL = 200                      # 抽出の繰り返し回数
BANDS = [0, 3, 7, 15, 30, 60, 10 ** 9]
rng = np.random.default_rng(20260822)


def log(m):
    print(m, flush=True)


def observed_ratios():
    """実測から「7分前オッズ ÷ 確定オッズ」の分布をオッズ帯別に作る。"""
    p = os.path.join(BASE_DIR, "odds_history.csv")
    d = pd.read_csv(p, dtype={"race_id": str}, low_memory=False)
    d["単勝オッズ"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d = d[d["ジョブ"].isin(["直前", "締切前"])
          & d["単勝オッズ"].notna() & (d["単勝オッズ"] > 1)]
    w = d.pivot_table(index=["race_id", "馬番"], columns="ジョブ",
                      values="単勝オッズ").dropna()
    w = w.rename(columns={"直前": "o7", "締切前": "o1"})
    w["r"] = w.o7 / w.o1                      # 7分前 ÷ 確定
    w["帯"] = pd.cut(w.o1, BANDS, labels=False)
    out = {int(k): g.r.values for k, g in w.groupby("帯", observed=True)}
    log(f"実測ペア {len(w):,}頭 / {w.reset_index().race_id.nunique()}レース")
    log(f"  {'確定オッズ帯':<12}{'頭数':>6}{'7分前÷確定':>12}{'ばらつき':>10}")
    lab = ["〜3倍", "3-7", "7-15", "15-30", "30-60", "60超"]
    for k in sorted(out):
        v = out[k]
        log(f"  {lab[k]:<12}{len(v):>6}{np.median(v):>12.3f}{v.std():>10.3f}")
    log("  ※ 1未満＝7分前の方がオッズが低い（締切に向けて見放された）")
    return out


def simulate(d, ratios):
    """7分前のオッズを1回ぶん作り、その時点で判定した場合の成績を返す。"""
    band = pd.cut(d.odds, BANDS, labels=False).fillna(0).astype(int).values
    r = np.ones(len(d))
    for k, v in ratios.items():
        m = band == k
        if m.any():
            r[m] = rng.choice(v, m.sum())
    o7 = np.clip(d.odds.values * r, 1.01, None)

    x = d.assign(o7=o7)
    inv = 1.0 / x.o7
    x["q7"] = inv / inv.groupby(x.race_id).transform("sum")
    # log(gap) から f を復元（レース内定数は softmax で消える）
    s = np.log(x.gap.values) + np.log(x.q7.values)
    x["s"] = s
    e = np.exp(x.s - x.groupby("race_id").s.transform("max"))
    x["p7"] = e / e.groupby(x.race_id).transform("sum")
    x["gap7"] = x.p7 / x.q7

    sel = x.loc[x.groupby("race_id")["gap7"].idxmax()]
    buy = sel[sel.gap7 >= GAP_MIN]
    if buy.empty:
        return None
    # 払戻は確定オッズ。選ぶのは7分前、払うのは確定。本番と同じ非対称。
    roi = (buy.win * buy.odds * 100).mean()
    return {"点数": len(buy), "的中": int(buy.win.sum()), "ROI": roi,
            "中央人気": buy.人気.median(), "中央オッズ": buy.odds.median()}


def main():
    ratios = observed_ratios()
    fp = os.path.join(BASE_DIR, "resid_pred.csv")
    if not os.path.exists(fp):
        log("\nresid_pred.csv がありません。python train_resid.py backtest で作られます。")
        return
    d = pd.read_csv(fp, dtype={"race_id": str})
    d = d[d.odds.notna() & (d.odds > 1) & d.gap.notna() & (d.gap > 0)]
    log(f"\nバックテスト検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")

    # ── 基準: 確定オッズで選ぶ（いまのBTと同じ）
    sel = d.loc[d.groupby("race_id")["gap"].idxmax()]
    buy = sel[sel.gap >= GAP_MIN]
    base = (buy.win * buy.odds * 100).mean()
    log("\n=== 基準: 確定オッズで選ぶ（現行BT）===")
    log(f"  {len(buy):,}点  的中{int(buy.win.sum())}  ROI {base:.1f}%"
        f"  中央{buy.人気.median():.0f}番人気 {buy.odds.median():.1f}倍")

    # ── 7分前に決める場合
    log(f"\n=== 7分前に決める場合（{N_TRIAL}回の抽出）===")
    rs = [simulate(d, ratios) for _ in range(N_TRIAL)]
    rs = [x for x in rs if x]
    R = pd.DataFrame(rs)
    log(f"  点数     中央 {R.点数.median():.0f}  （基準 {len(buy):,}）")
    log(f"  的中     中央 {R.的中.median():.0f}  （基準 {int(buy.win.sum())}）")
    log(f"  ROI      中央 {R.ROI.median():.1f}%  "
        f"90%範囲 [{R.ROI.quantile(.05):.1f}, {R.ROI.quantile(.95):.1f}]")
    log(f"  中央人気  {R.中央人気.median():.1f}番  （基準 {buy.人気.median():.0f}番）")
    log(f"  中央オッズ {R.中央オッズ.median():.1f}倍  （基準 {buy.odds.median():.1f}倍）")
    log(f"\n  基準からの差  {R.ROI.median()-base:+.1f}pt")
    log(f"  100%を割る確率 {(R.ROI < 100).mean()*100:.0f}%")
    log("\n  ※ 選ぶのは7分前、払戻は確定オッズ。本番と同じ条件。")
    log("  ※ 7分前オッズの分布は8/15〜8/22の84レースから作っている。"
        "少ないので目安。")


if __name__ == "__main__":
    main()
