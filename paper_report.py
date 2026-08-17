# -*- coding: utf-8 -*-
"""残差モデルの前向き検証（買わずに記録）を集計する（2026-08-17）

なぜ前向き検証か
  バックテストでは 1,891点・的中203・5年通算157.1% だった。順列検定も通った
  （p=0.0000）。だが過去8回、バックテストで良く見えたものがすべて崩れている。
  だから今回は**実際に走らせて記録を貯めてから**決める。買わない。

見るもの
  ① 買い率が想定どおりか
     バックテストでは 14,972レース中1,891レース＝12.6%。
     実測がこれと大きく違えば、本番と検証で条件が違っている。
  ② 選ばれる馬の人気・オッズの分布が想定どおりか
     バックテストの中央オッズは19.8倍・中央7番人気。
  ③ 実際の回収率
     ただし的中100本を超えるまでは数字を信用しない。
     バックテストでも年ごとの的中は28〜67本で、その本数だとROIは倍半分に振れる。

判断の目安
  的中100本 … ようやく傾向が見え始める
  的中200本 … バックテスト（203本）と同じ土俵で比べられる

実行: python paper_report.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(BASE_DIR, "paper_resid.csv")
rng = np.random.default_rng(20260817)

# train_resid.py backtest / check_resid.py で確認済みの値
BT = {"買い率": 12.6, "点数": 2926, "的中": 236, "ROI": 163.3,
      "中央オッズ": 19.8, "中央人気": 7, "1R点数": 1.55,
      "年別": "2021:210% 2022:106% 2023:154% 2024:337% 2025:69%",
      "券種": "単勝1,891点(157.0%) / ワイド1,035点(174.9%)"}


def log(m):
    print(m, flush=True)


def main():
    if not os.path.exists(PAPER):
        log("paper_resid.csv がまだありません。")
        log("予想を1回まわすと作られます（購入はしません）。")
        return
    d = pd.read_csv(PAPER, dtype={"race_id": str})
    log(f"記録 {len(d):,}レース（{d.race_id.str[:8].min()} 〜 {d.race_id.str[:8].max()}）\n")

    buy = d[d.判定 == "買い"]
    rate = len(buy) / len(d) * 100
    log("=== ① 買い率 ===")
    log(f"  実測 {len(buy):,}/{len(d):,} = {rate:.1f}%   バックテスト {BT['買い率']}%")
    diff = abs(rate - BT["買い率"])
    log(f"  → {'○ 想定どおり' if diff < 5 else '⚠ ズレている。本番と検証で条件が違う可能性'}"
        f"（差 {diff:.1f}pt）")

    if buy.empty:
        log("\nまだ買い判定のレースがありません。")
        return

    log("\n=== ② 選ばれている馬 ===")
    o = pd.to_numeric(buy["単勝オッズ"], errors="coerce")
    p = pd.to_numeric(buy["人気"], errors="coerce")
    log(f"  中央オッズ {o.median():.1f}倍（BT {BT['中央オッズ']}倍）")
    log(f"  中央人気   {p.median():.0f}番人気（BT {BT['中央人気']}番人気）")
    log(f"  gap 中央値 {pd.to_numeric(buy['gap'], errors='coerce').median():.2f}")
    log(f"  {'人気帯':<12}{'頭数':>7}{'割合':>7}")
    for lo, hi, lab in [(1, 1, "1番人気"), (2, 3, "2-3番"), (4, 5, "4-5番"),
                        (6, 7, "6-7番"), (8, 12, "8-12番"), (13, 99, "13番以下")]:
        n = int(((p >= lo) & (p <= hi)).sum())
        if n:
            log(f"  {lab:<12}{n:>7}{n/len(buy)*100:>6.1f}%")

    # ── 結果との照合 ────────────────────────────────────
    res = os.path.join(BASE_DIR, "jv_payouts.csv")
    if not os.path.exists(res):
        log("\n（払戻データが無いので回収率は出せません）")
        return
    jv = pd.read_csv(res, dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    b = buy.copy()
    # 結果が出たレースだけを対象にする（払戻表にそのレースが載っているか）
    done_races = {r.race_id for r in jv.itertuples()}
    b = b[b.race_id.isin(done_races)]
    b["払戻"] = [PAY.get((r, k, c), 0.0)
                for r, k, c in zip(b.race_id, b["券種"], b["組み合わせ"])]
    done = b
    log(f"\n=== ③ 回収率（結果が出た {len(done):,}レース）===")
    if len(done) < 10:
        log("  まだ照合できるレースが少ないです。")
        return
    hit = int((done.払戻 > 0).sum())
    roi = done.払戻.sum() / (len(done) * 100) * 100
    log(f"  {len(done):,}点  的中{hit}（{hit/len(done)*100:.1f}%）  回収率 {roi:.1f}%")
    if hit >= 20:
        v = done.払戻.values
        bs = np.array([rng.choice(v, len(v)).mean() for _ in range(3000)])
        log(f"  95%区間 [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]")
    log(f"\n  バックテスト: {BT['点数']:,}点 的中{BT['的中']} ROI {BT['ROI']}%")
    log(f"               年別 {BT['年別']}")
    log("\n=== 判断の目安 ===")
    if hit < 100:
        log(f"  的中{hit}本。**まだ数字を信用しない。**"
            f"（目安100本まであと{100-hit}本）")
        log("  バックテストでも年ごとの的中は28〜67本で、その本数だとROIは倍半分に振れる。")
    elif hit < 200:
        log(f"  的中{hit}本。傾向が見え始める段階。まだ確定ではない。")
    else:
        log(f"  的中{hit}本。バックテスト(203本)と同じ土俵。比較できる。")


if __name__ == "__main__":
    main()
