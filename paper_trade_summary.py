# -*- coding: utf-8 -*-
"""
paper_trade_summary.py
──────────────────────
ペーパートレード（紙の上での仮想馬券）の成績を自動集計する。
「もし機械的に買っていたら回収率○%」を継続的に把握し、
・自分で実際に賭ける価値があるか
・予想を販売できる実績があるか
を客観的に判断するための土台。

入力: prediction_record_v2.csv（result_tracker.py が結果照合済みのもの）
  列: 日付, race_id, honmei, honmei_odds, honmei_ninki,
      honmei_strat（戦略フラグ）, honmei_kenshu（券種推奨）,
      honmei_actual（◎の実着順）, hit（◎単勝的中）

集計する買い方（単勝ベース・100円固定）:
  ① ◎単勝         : 本命を毎レース単勝で買う
  ② 戦略該当のみ   : 🔥戦略該当の◎だけ単勝で買う
  ③ 券種推奨「軸◎」: 軸◎だけ単勝で買う
  ④ 人気帯別       : ◎を人気帯ごとに見た回収率

  ※ 複勝・ワイド回収率は payout_data.csv 取得後に追加予定

出力: コンソール表示 + paper_trade_log.csv（日次集計を追記）

使い方:
  python paper_trade_summary.py            # 全期間集計
  python paper_trade_summary.py 2026/06    # 特定月で絞る
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RECORD_CSV = os.path.join(BASE_DIR, "prediction_record_v2.csv")
LOG_CSV    = os.path.join(BASE_DIR, "paper_trade_log.csv")

BET = 100  # 1点100円


def roi_block(df, label, odds_col="honmei_odds"):
    """単勝回収率を計算して表示用dictを返す。"""
    df = df.copy()
    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")
    df = df.dropna(subset=[odds_col])
    df = df[df["hit"].notna()]  # 結果照合済みのみ
    n = len(df)
    if n == 0:
        return None
    hits = df[df["hit"] == 1]
    n_hit = len(hits)
    invest = n * BET
    payout = (hits[odds_col] * BET).sum()
    roi = payout / invest * 100 if invest > 0 else 0
    return {
        "買い方": label,
        "ベット": n,
        "的中": n_hit,
        "的中率": f"{n_hit/n*100:.1f}%",
        "投資": invest,
        "払戻": int(payout),
        "回収率": f"{roi:.1f}%",
        "収支": int(payout - invest),
        "_roi": roi,
    }


def main():
    if not os.path.exists(RECORD_CSV):
        print(f"予想記録がありません: {RECORD_CSV}")
        return

    df = pd.read_csv(RECORD_CSV)
    # 結果照合済み（hitが入っている）行のみ対象
    df = df[df["hit"].notna()].copy()

    # 期間フィルタ（引数で月指定）
    period = sys.argv[1] if len(sys.argv) > 1 else None
    if period and "日付" in df.columns:
        df = df[df["日付"].astype(str).str.startswith(period)]

    if len(df) == 0:
        print("対象データがありません（結果照合済みの記録が必要）。")
        return

    print("=" * 60)
    print("ペーパートレード成績集計（単勝・100円固定）")
    print("=" * 60)
    if period:
        print(f"期間: {period}")
    if "日付" in df.columns:
        dates = df["日付"].dropna().astype(str)
        if len(dates) > 0:
            print(f"対象期間: {dates.min()} 〜 {dates.max()}")
    print(f"対象レース数: {len(df)}")

    results = []

    # ① ◎単勝（全レース）
    r = roi_block(df, "① ◎単勝（毎レース）")
    if r:
        results.append(r)

    # ② 戦略該当のみ
    if "honmei_strat" in df.columns:
        strat_df = df[
            df["honmei_strat"].notna()
            & (df["honmei_strat"].astype(str) != "")
            & (df["honmei_strat"].astype(str) != "nan")
        ]
        r = roi_block(strat_df, "② 戦略該当◎のみ")
        if r:
            results.append(r)

    # ③ 券種推奨「軸◎」のみ
    if "honmei_kenshu" in df.columns:
        jiku_df = df[df["honmei_kenshu"].astype(str).str.contains("軸◎", na=False)]
        r = roi_block(jiku_df, "③ 券種推奨「軸◎」")
        if r:
            results.append(r)

    # 結果テーブル表示
    print("\n" + "─" * 60)
    print("【買い方別の回収率】")
    print("─" * 60)
    if results:
        tbl = pd.DataFrame(results).drop(columns=["_roi"])
        print(tbl.to_string(index=False))
        print("\n  ※ 回収率100%超 = 理論上プラス（控除率の壁を越えている）")
        for r in results:
            mark = "🟢" if r["_roi"] >= 100 else "🔴"
            print(f"  {mark} {r['買い方']}: {r['回収率']}")
    else:
        print("  集計可能なデータがありません。")

    # ④ 人気帯別（◎単勝）
    print("\n" + "─" * 60)
    print("【◎単勝の人気帯別 回収率】")
    print("─" * 60)
    if "honmei_ninki" in df.columns:
        d = df.copy()
        d["honmei_ninki"] = pd.to_numeric(d["honmei_ninki"], errors="coerce")
        d["honmei_odds"]  = pd.to_numeric(d["honmei_odds"], errors="coerce")
        bands = [
            ("1番人気",     d["honmei_ninki"] == 1),
            ("2-3番人気",   (d["honmei_ninki"] >= 2) & (d["honmei_ninki"] <= 3)),
            ("4-6番人気",   (d["honmei_ninki"] >= 4) & (d["honmei_ninki"] <= 6)),
            ("7番人気以下", d["honmei_ninki"] >= 7),
        ]
        for label, mask in bands:
            sub = d[mask]
            r = roi_block(sub, label)
            if r:
                mark = "🟢" if r["_roi"] >= 100 else "🔴"
                print(f"  {mark} {label:12}: {r['ベット']:3}回 "
                      f"的中{r['的中率']:>6} 回収率{r['回収率']:>7}")

    # ── 日次ログに追記（推移を追える）──
    if results:
        today = datetime.now().strftime("%Y/%m/%d %H:%M")
        main_roi = results[0]["_roi"]  # ◎単勝の回収率
        log_row = {
            "集計日時": today,
            "対象期間": period or "全期間",
            "レース数": len(df),
            "◎単勝回収率": f"{main_roi:.1f}%",
        }
        for r in results:
            log_row[r["買い方"]] = r["回収率"]
        log_df = pd.DataFrame([log_row])
        if os.path.exists(LOG_CSV):
            log_df.to_csv(LOG_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            log_df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
        print(f"\n  集計結果を記録 → paper_trade_log.csv")

    print("\n" + "=" * 60)
    print("【判断の目安】")
    print("=" * 60)
    print("  ・3ヶ月以上の記録で安定して回収率100%超 → 実戦/販売の価値あり")
    print("  ・サンプルが少ない（数十回）うちは偶然の可能性大、継続が必要")
    print("  ・複勝・ワイド回収率は payout_data.csv 取得後に追加")
    print("\n※ これはあくまで仮想集計。実際の馬券購入は自己責任で。")


if __name__ == "__main__":
    main()
