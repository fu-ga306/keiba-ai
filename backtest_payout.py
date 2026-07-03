# -*- coding: utf-8 -*-
"""
backtest_payout.py
──────────────────
2025年テスト予測（model_result.csv）に対して、払戻データ（payout_data.csv）で
複勝・ワイド・馬連・三連複の実回収率をバックテストする。

analyze_payout_roi.py は「運用記録(prediction_record)」ベースで件数が少ないが、
本スクリプトは「2025年テスト全レース(約3千)」で統計的に信頼できる回収率を出す。

検証する買い方（すべて予測順位ベース・100円固定）:
  ① 予測1位の複勝
  ② 予測1-2位のワイド（1点）
  ③ 予測1-2位の馬連（1点）
  ④ 予測1-3位ボックスのワイド（3点）
  ⑤ 予測1-3位の三連複（1点）

前提データ:
  model_result.csv : race_id, 馬名, 着順_num, 予測順位
  race_data_clean.csv : race_id, 馬名, 馬番（馬名→馬番の紐付け）
  payout_data.csv : race_id, 券種, 組み合わせ, 払戻金（payout_scraperで取得）

使い方:
  python backtest_payout.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_CSV = os.path.join(BASE_DIR, "model_result.csv")
CLEAN_CSV  = os.path.join(BASE_DIR, "race_data_clean.csv")
PAYOUT_CSV = os.path.join(BASE_DIR, "payout_data.csv")

BET = 100  # 1点100円


def _norm_combo(s):
    """組み合わせ文字列を馬番セットに正規化（先頭0除去）。'04-11' → {'4','11'}"""
    return {c.lstrip("0") for c in str(s).replace("-", " ").split() if c.strip()}


def find_payout(payout_idx, rid, kenshu, umabans):
    """指定レース・券種・馬番集合に一致する払戻金を返す（なければNone）。"""
    key = (str(rid), kenshu)
    rows = payout_idx.get(key)
    if not rows:
        return None
    target = {str(int(float(u))) for u in umabans}
    for combo_set, pay in rows:
        if combo_set == target:
            return pay
    return None


def main():
    if not os.path.exists(PAYOUT_CSV):
        print(f"払戻データがありません: {PAYOUT_CSV}")
        print("→ 血統完走後に『python payout_scraper.py 2025』で取得してください")
        return

    # ── 払戻データを (race_id,券種)→[(馬番セット,払戻金)] のインデックスに ──
    pay = pd.read_csv(PAYOUT_CSV, dtype=str)
    pay["払戻金"] = pd.to_numeric(pay["払戻金"], errors="coerce")
    payout_idx = {}
    for _, r in pay.iterrows():
        key = (str(r["race_id"]), r["券種"])
        payout_idx.setdefault(key, []).append((_norm_combo(r["組み合わせ"]), r["払戻金"]))

    # ── 予測結果 × 馬番マッピング ──
    res = pd.read_csv(RESULT_CSV, encoding="utf-8-sig").dropna(subset=["着順_num", "予測順位"])
    clean = pd.read_csv(CLEAN_CSV, low_memory=False, usecols=["race_id", "馬名", "馬番"])
    clean["race_id"] = clean["race_id"].astype(str)
    res["race_id"] = res["race_id"].astype(str)
    res = res.merge(clean, on=["race_id", "馬名"], how="left")

    stats = {k: {"n": 0, "hit": 0, "pay": 0.0} for k in
             ["複勝", "ワイド", "馬連", "ワイドBOX", "三連複"]}

    for rid, g in res.groupby("race_id"):
        g = g.sort_values("予測順位")
        if len(g) < 3 or g["馬番"].isna().any():
            continue
        p1, p2, p3 = g.iloc[0], g.iloc[1], g.iloc[2]
        u1, u2, u3 = p1["馬番"], p2["馬番"], p3["馬番"]
        a1, a2, a3 = p1["着順_num"], p2["着順_num"], p3["着順_num"]

        # ① 複勝（予測1位）
        s = stats["複勝"]; s["n"] += 1
        if a1 <= 3:
            pmt = find_payout(payout_idx, rid, "複勝", [u1])
            if pmt: s["hit"] += 1; s["pay"] += pmt

        # ② ワイド（予測1-2位）
        s = stats["ワイド"]; s["n"] += 1
        if a1 <= 3 and a2 <= 3:
            pmt = find_payout(payout_idx, rid, "ワイド", [u1, u2])
            if pmt: s["hit"] += 1; s["pay"] += pmt

        # ③ 馬連（予測1-2位）
        s = stats["馬連"]; s["n"] += 1
        if a1 <= 2 and a2 <= 2:
            pmt = find_payout(payout_idx, rid, "馬連", [u1, u2])
            if pmt: s["hit"] += 1; s["pay"] += pmt

        # ④ ワイドBOX（予測1-3位・3点買い＝コスト300円）
        s = stats["ワイドBOX"]; s["n"] += 1
        got = 0.0
        for pair in [(u1, u2), (u1, u3), (u2, u3)]:
            aa = [g.set_index("馬番").loc[u, "着順_num"] for u in pair]
            if all(x <= 3 for x in aa):
                pmt = find_payout(payout_idx, rid, "ワイド", list(pair))
                if pmt: got += pmt
        if got > 0: s["hit"] += 1
        s["pay"] += got
        s["_cost"] = s.get("_cost", 0) + BET * 3  # 3点買い

        # ⑤ 三連複（予測1-3位・1点）
        s = stats["三連複"]; s["n"] += 1
        if a1 <= 3 and a2 <= 3 and a3 <= 3:
            pmt = find_payout(payout_idx, rid, "3連複", [u1, u2, u3])
            if pmt: s["hit"] += 1; s["pay"] += pmt

    print("=" * 62)
    print("2025テスト 券種別バックテスト（予測順位ベース・100円/点）")
    print("=" * 62)
    print(f"{'券種':<10}{'ベット':>7}{'的中':>6}{'的中率':>8}{'回収率':>9}{'収支':>10}")
    print("-" * 62)
    for k, s in stats.items():
        if s["n"] == 0:
            continue
        cost = s.get("_cost", s["n"] * BET)
        roi = s["pay"] / cost * 100 if cost > 0 else 0
        hit_rate = s["hit"] / s["n"] * 100
        pl = int(s["pay"] - cost)
        mark = " 🟢" if roi >= 100 else ""
        print(f"{k:<10}{s['n']:>7}{s['hit']:>6}{hit_rate:>7.1f}%{roi:>8.1f}%{pl:>+10}{mark}")
    print("-" * 62)
    print("※ 回収率100%超（🟢）= 勝てる買い方。note公開・予想販売の核心。")


if __name__ == "__main__":
    main()
