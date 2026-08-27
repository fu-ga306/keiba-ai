# -*- coding: utf-8 -*-
"""複勝確率を実測に合わせて較正する（2026-08-27）

なぜ必要か
  販売用の評価表を作る過程で、**人気帯ごとに系統的なズレ**が見つかった。

    人気        頭数     予測     実際      差
    1番人気     176   49.2%  63.6%  +14.5  ← 過小評価
    2-3番      352   39.5%  45.7%   +6.2
    4-5番      349   29.1%  30.7%   +1.5
    6-8番      515   21.6%  19.4%   -2.2
    9-12番     578   12.3%   7.3%   -5.0  ← 過大評価
    13番以下    363    6.8%   3.9%   -2.9

  全体で平均すると合っているように見えるが（予測22.3%・実際22.3%）、
  **過小評価と過大評価が打ち消し合っているだけ**だった。
  1.1倍の1番人気が「3着内20%」と出るので、そのままでは商品にならない。

⚠ 既存の列は絶対に書き換えない
  `複勝確率` は 評価ランク(build_grade) と ダッシュボード表示 が使っている。
  ここを動かすと既存の挙動が変わる。**新しい列 `複勝確率_較正` を足すだけ**にする。
  買い判定(resid_io)は自前のgapを使っており複勝確率を見ていないので、
  そもそも影響しない（確認済み）。

やり方
  モデルの予測と市場の情報を2つ入れたロジスティック回帰で出し直す。
    入力: logit(モデルの複勝確率), log(単勝オッズ)
    出力: 3着以内に入る確率
  パラメータは3つだけ（係数2＋切片）。2,333頭に対して十分小さく、過学習しにくい。

  isotonic回帰も試したが、人気帯ごとのズレは「オッズを使えていない」ことが
  原因なので、オッズを説明変数に入れるほうが筋がよい。

実行
  python build_place_calib.py            学習して place_calib.pkl を保存
  python build_place_calib.py --check    期間を分けて検証だけする（保存しない）
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE_DIR, "history_marks.csv")
OUT = os.path.join(BASE_DIR, "place_calib.pkl")
BANDS = [(1, 1, "1番人気"), (2, 3, "2-3番"), (4, 5, "4-5番"),
         (6, 8, "6-8番"), (9, 12, "9-12番"), (13, 99, "13番以下")]


def log(m):
    print(m, flush=True)


def load():
    h = pd.read_csv(HIST, dtype={"race_id": str})
    h["着順"] = pd.to_numeric(h["着順"], errors="coerce")
    h["p"] = pd.to_numeric(h["複勝確率"], errors="coerce")
    h["人気"] = pd.to_numeric(h["人気"], errors="coerce")
    h["odds"] = pd.to_numeric(h["単勝オッズ"], errors="coerce")
    h = h[h["着順"].notna() & h["p"].notna() & h["odds"].notna() & (h["odds"] > 1)]
    h["y"] = (h["着順"] <= 3).astype(int)
    return h


def features(df):
    """logit(モデル確率) と log(単勝オッズ) の2つ。"""
    p = np.clip(df["p"].values, 1e-4, 1 - 1e-4)
    return np.column_stack([np.log(p / (1 - p)), np.log(df["odds"].values)])


def fit(df):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000)
    m.fit(features(df), df["y"].values)
    return m


def report(df, pred, title):
    """人気帯ごとのズレを出す。ここが揃わなければ意味がない。"""
    log(f"\n  {title}")
    log(f"    {'人気':<10}{'頭数':>6}{'予測':>8}{'実際':>8}{'差':>8}")
    worst = 0.0
    for lo, hi, lab in BANDS:
        m = (df["人気"] >= lo) & (df["人気"] <= hi)
        if m.sum() < 20:
            continue
        pr, ac = pred[m.values].mean() * 100, df["y"].values[m.values].mean() * 100
        worst = max(worst, abs(ac - pr))
        flag = "  ⚠" if abs(ac - pr) > 7 else ""
        log(f"    {lab:<10}{int(m.sum()):>6}{pr:>7.1f}%{ac:>7.1f}%{ac-pr:>+7.1f}{flag}")
    log(f"    最大のズレ {worst:.1f}pt")
    return worst


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    if not os.path.exists(HIST):
        log("history_marks.csv がありません。")
        return
    h = load()
    days = sorted(h["日付"].unique())
    log(f"検体 {h.race_id.nunique()}レース / {len(h):,}頭 / {len(days)}開催日")
    log(f"  {days[0]} 〜 {days[-1]}")

    # ── 期間を分けた検証（補正に使ったデータで測ると必ず良く見えるため）──
    #   最後の1日を検証に回す。開催日が5つしかないので粗いが、
    #   同じデータで測るよりはるかにマシ。
    te_day = days[-1]
    tr, te = h[h["日付"] != te_day], h[h["日付"] == te_day]
    log(f"\n  学習 {len(tr):,}頭（〜{days[-2]}）/ 検証 {len(te):,}頭（{te_day}）")

    m = fit(tr)
    pred_te = m.predict_proba(features(te))[:, 1]
    log("\n" + "=" * 60)
    log("  検証日（学習に使っていない日）での比較")
    log("=" * 60)
    w_before = report(te, te["p"].values, "補正前")
    w_after = report(te, pred_te, "補正後")
    log(f"\n  最大のズレ {w_before:.1f}pt → {w_after:.1f}pt")
    ll_b, ll_a = logloss(te["y"].values, te["p"].values), logloss(te["y"].values, pred_te)
    log(f"  対数損失   {ll_b:.4f} → {ll_a:.4f}"
        f"（{'改善' if ll_a < ll_b else '悪化'}）")

    if w_after >= w_before or ll_a >= ll_b:
        log("\n  ⚠ 改善していません。保存しません。")
        log("    データが足りないか、この形の補正が合っていない可能性があります。")
        return

    if "--check" in sys.argv:
        log("\n  --check なので保存しませんでした。")
        return

    # ── 全データで学習し直して保存 ──
    mf = fit(h)
    with open(OUT, "wb") as f:
        pickle.dump({"model": mf, "n": len(h), "races": int(h.race_id.nunique()),
                     "days": days, "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
                     "features": ["logit(複勝確率)", "log(単勝オッズ)"]}, f)
    log(f"\n  ○ {os.path.basename(OUT)} を保存（{len(h):,}頭で学習）")
    log("    使う側は 複勝確率_較正 という**新しい列**に入れること。")
    log("    既存の 複勝確率 は書き換えない（評価ランクとダッシュボードが使っている）。")


if __name__ == "__main__":
    main()
