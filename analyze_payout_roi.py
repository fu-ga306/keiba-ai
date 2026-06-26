# -*- coding: utf-8 -*-
"""
analyze_payout_roi.py
─────────────────────
払戻データ(payout_data.csv)と予想記録(prediction_record_v2.csv)を結合し、
◎本命を各券種で買った場合の回収率を検証する。

目的:
  ◎単勝は回収率65.8%で負ける。
  でも◎は2-3着が多い（複勝率59%）→ 複勝なら勝てるか検証する。
  「勝てる買い方」を見つけるのが、予想販売(note公開)の核心。

検証する買い方:
  ① ◎複勝     : ◎が3着以内なら的中、複勝払戻で回収率
  ② ◎-○ワイド : ◎と○が両方3着以内なら的中
  ③ ◎-○馬連   : ◎と○が1-2着なら的中

前提データ:
  payout_data.csv     : race_id, 券種, 組み合わせ, 払戻金, 人気（payout_scraperで取得）
  prediction_record_v2.csv : race_id, honmei, honmei_actual, taiko, ...
  ※ 馬名↔馬番の紐付けに、各レースの着順表(馬番・馬名)が必要
     → result_master.csv があれば使用、なければ払戻の人気/組み合わせから推定

使い方:
  python analyze_payout_roi.py
"""
import os
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PAYOUT_CSV  = os.path.join(BASE_DIR, "payout_data.csv")
RECORD_CSV  = os.path.join(BASE_DIR, "prediction_record_v2.csv")
RESULT_CSV  = os.path.join(BASE_DIR, "result_master.csv")  # 着順表（馬番・馬名）があれば

BET = 100  # 1点100円


def load_payout():
    if not os.path.exists(PAYOUT_CSV):
        print(f"払戻データがありません: {PAYOUT_CSV}")
        print("→ 血統完走後に payout_scraper.py で取得してください")
        return None
    df = pd.read_csv(PAYOUT_CSV, dtype=str)
    df["払戻金"] = pd.to_numeric(df["払戻金"], errors="coerce")
    df["race_id"] = df["race_id"].astype(str)
    return df


def fukusho_payout(payout, race_id, umaban):
    """指定レース・馬番の複勝払戻金を返す（なければNone）。"""
    sub = payout[(payout["race_id"] == str(race_id)) & (payout["券種"] == "複勝")]
    for _, r in sub.iterrows():
        # 組み合わせが馬番と一致（複勝は単一馬番）
        if str(r["組み合わせ"]).strip().lstrip("0") == str(umaban).strip().lstrip("0"):
            return r["払戻金"]
    return None


def wide_payout(payout, race_id, u1, u2):
    """指定レース・2頭のワイド払戻金を返す（順不同、なければNone）。"""
    sub = payout[(payout["race_id"] == str(race_id)) & (payout["券種"] == "ワイド")]
    targets = {str(u1).lstrip("0"), str(u2).lstrip("0")}
    for _, r in sub.iterrows():
        combo = set(str(r["組み合わせ"]).replace("-", " ").split())
        combo = {c.lstrip("0") for c in combo}
        if combo == targets:
            return r["払戻金"]
    return None


def main():
    payout = load_payout()
    if payout is None:
        return

    if not os.path.exists(RECORD_CSV):
        print(f"予想記録がありません: {RECORD_CSV}")
        return
    rec = pd.read_csv(RECORD_CSV)
    rec = rec[rec["hit"].notna()].copy()  # 結果照合済みのみ
    rec["race_id"] = rec["race_id"].astype(str)

    print("=" * 60)
    print("払戻データ回収率検証（◎本命の各券種・100円固定）")
    print("=" * 60)
    print(f"対象レース: {len(rec)}")
    print(f"払戻データ: {payout['race_id'].nunique()}レース分")

    # 馬番は prediction_record に直接記録される（honmei_umaban等）
    # 古い記録には馬番がないので、その分はスキップされる
    has_umaban = "honmei_umaban" in rec.columns
    if not has_umaban:
        print("⚠️ prediction_record に馬番列がありません。")
        print("   result_tracker.py 更新後の新しい予想記録から検証可能になります。")
        return

    # ① ◎複勝
    n_fuku, hit_fuku, pay_fuku = 0, 0, 0
    for _, row in rec.iterrows():
        rid = row["race_id"]
        umaban = row.get("honmei_umaban")
        actual = pd.to_numeric(row.get("honmei_actual"), errors="coerce")
        if pd.isna(umaban) or pd.isna(actual):
            continue
        umaban = str(int(float(umaban)))
        n_fuku += 1
        if actual <= 3:
            p = fukusho_payout(payout, rid, umaban)
            if p is not None:
                hit_fuku += 1
                pay_fuku += p
    if n_fuku > 0:
        roi = pay_fuku / (n_fuku * BET) * 100
        print("\n" + "─" * 60)
        print("① ◎複勝")
        print(f"  ベット{n_fuku}回 的中{hit_fuku}回 "
              f"的中率{hit_fuku/n_fuku*100:.1f}%")
        print(f"  回収率 {roi:.1f}%  収支 {int(pay_fuku - n_fuku*BET):+}円")
        mark = "🟢 単勝(65.8%)より良い" if roi > 65.8 else "🔴"
        print(f"  {mark}")

    # ② ◎-○ワイド
    n_w, hit_w, pay_w = 0, 0, 0
    for _, row in rec.iterrows():
        rid = row["race_id"]
        u1 = row.get("honmei_umaban")
        u2 = row.get("taiko_umaban")
        h_act = pd.to_numeric(row.get("honmei_actual"), errors="coerce")
        t_act = pd.to_numeric(row.get("taiko_actual"), errors="coerce")
        if pd.isna(u1) or pd.isna(u2) or pd.isna(h_act) or pd.isna(t_act):
            continue
        u1 = str(int(float(u1))); u2 = str(int(float(u2)))
        n_w += 1
        if h_act <= 3 and t_act <= 3:
            p = wide_payout(payout, rid, u1, u2)
            if p is not None:
                hit_w += 1
                pay_w += p
    if n_w > 0:
        roi = pay_w / (n_w * BET) * 100
        print("\n" + "─" * 60)
        print("② ◎-○ワイド")
        print(f"  ベット{n_w}回 的中{hit_w}回 的中率{hit_w/n_w*100:.1f}%")
        print(f"  回収率 {roi:.1f}%  収支 {int(pay_w - n_w*BET):+}円")

    print("\n" + "=" * 60)
    print("【判断】回収率100%超の買い方 = 勝てる買い方")
    print("  → これを note 公開・予想販売の核心にする")
    print("=" * 60)


if __name__ == "__main__":
    main()
