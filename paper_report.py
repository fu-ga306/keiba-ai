# -*- coding: utf-8 -*-
"""残差モデルの前向き検証（買わずに記録）を集計する（2026-08-22 全面改訂）

なぜ前向き検証か
  バックテストでは 軸gap>=1.5 で 10,349点・的中1,178・5年120.6% だった。
  順列検定も通っている（p=0.0000）。だが過去8回、バックテストで良く見えたものが
  すべて崩れている。だから実際に走らせて記録を貯めてから決める。買わない。

記録の形（paper_resid.csv）
  1レースにつき複数行。軸1行＋ダートなら相手が最大3行、見送りなら1行。
  軸gap 列があるので、しきい値1.5と1.7の両方を後から集計できる。

見るもの
  ① 買い率がバックテストと合っているか
     ズレていれば本番と検証で条件が違う。真っ先に気づくべき異常。
  ② 選ばれる軸の人気・オッズの分布が想定どおりか
  ③ 実際の回収率（結果が出たレースのみ）
     ただし的中100本を超えるまで数字を信用しない。5年検証でも60倍以上の帯は
     的中11本・95%区間[59.5, 288.4]で何も言えなかった。同じことが起きる。

⚠ 2026-08-22に全面改訂。以前の版は BT を辞書にした際に買い率の参照を直し忘れ、
  KeyError で落ちていた。また記録が1レース1行の前提だったが、実際は軸＋相手で
  複数行になるため集計もずれていた。

実行: python paper_report.py
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(BASE_DIR, "paper_resid.csv")
rng = np.random.default_rng(20260822)

# 軸のしきい値ごとのバックテスト値（2021-2025 walk-forward・実払戻で照合）
BT = {
    1.5: {"買い率": 47.7, "点数": 10349, "的中": 1178, "ROI": 120.6,
          "年別": "137/116/111/163/79"},
    1.7: {"買い率": 27.5, "点数": 6195, "的中": 584, "ROI": 131.1,
          "年別": "146/119/115/217/77"},
}
BT_ODDS_MED = 10.6      # 軸の中央オッズ（gap>=1.5）
BT_POP_MED = 5          # 軸の中央人気


def log(m):
    print(m, flush=True)


def main():
    if not os.path.exists(PAPER):
        log("paper_resid.csv がまだありません。")
        log("予想を1回まわすと作られます（購入はしません）。")
        return
    d = pd.read_csv(PAPER, dtype={"race_id": str, "組み合わせ": str})
    d["軸gap"] = pd.to_numeric(d.get("軸gap"), errors="coerce")
    nrace = d.race_id.nunique()
    ndays = d.race_id.str[:10].nunique()
    log(f"記録 {nrace:,}レース / {len(d):,}行"
        f"（race_id {d.race_id.min()} 〜 {d.race_id.max()}）\n")

    # ── ① 買い率 ────────────────────────────────────────
    log("=== ① 買い率（しきい値ごと）===")
    log("  ズレていたら本番と検証で条件が違う。真っ先に疑うところ。")
    log(f"  {'軸gap':<9}{'買うレース':>11}{'買い率':>9}{'BT買い率':>10}{'判定':>10}")
    for th, bt in sorted(BT.items()):
        n = d[(d.判定 == "買い") & (d.軸gap >= th)].race_id.nunique()
        r = n / nrace * 100 if nrace else 0
        diff = abs(r - bt["買い率"])
        log(f"  >={th:<7}{n:>11,}{r:>8.1f}%{bt['買い率']:>9.1f}%"
            f"{('○' if diff < 10 else '⚠ ズレ'):>10}")
    log("  ※ 開催日ごとにレースの堅さが違うので、数日ぶんでは揺れます。")
    log("    1番人気が4倍以上のような混戦が多い日は、買い率が上がり人気薄に寄ります。")

    ax = d[(d.判定 == "買い") & (d.役割 == "軸")]
    if ax.empty:
        log("\nまだ買い判定の軸がありません。")
        return

    # ── ② 選ばれている軸 ──────────────────────────────────
    o = pd.to_numeric(ax["単勝オッズ"], errors="coerce")
    p = pd.to_numeric(ax["人気"], errors="coerce")
    log("\n=== ② 選ばれている軸 ===")
    log(f"  中央オッズ {o.median():>6.1f}倍（BT {BT_ODDS_MED}倍）")
    log(f"  中央人気   {p.median():>6.0f}番人気（BT {BT_POP_MED}番人気）")
    log(f"  {'人気帯':<12}{'頭数':>7}{'割合':>8}")
    for lo, hi, lab in [(1, 1, "1番人気"), (2, 3, "2-3番"), (4, 5, "4-5番"),
                        (6, 7, "6-7番"), (8, 9, "8-9番"), (10, 12, "10-12番"),
                        (13, 99, "13番以下")]:
        n = int(((p >= lo) & (p <= hi)).sum())
        log(f"  {lab:<12}{n:>7}{n/len(ax)*100:>7.1f}%")

    # ── ③ 回収率 ───────────────────────────────────────
    # 払戻は netkeiba(payout_data.csv) と JV(jv_payouts.csv) の両方を見る。
    # JRA-VANの契約終了で取得元をnetkeibaへ移したが、過去分はJV側に残っている。
    frames = []
    for path in ("jv_payouts.csv", "payout_data.csv"):
        fp = os.path.join(BASE_DIR, path)
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, dtype=str))
    if not frames:
        log("\n（払戻データが無いので回収率は出せません）")
        return
    jv = pd.concat(frames, ignore_index=True)
    jv["race_id"] = jv["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    # 同じ race_id が両方にあれば後勝ち（＝netkeiba側）
    jv = jv.drop_duplicates(["race_id", "券種", "組み合わせ"], keep="last")
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    done = set(jv.race_id)

    b = d[(d.判定 == "買い") & d.race_id.isin(done)].copy()
    log(f"\n=== ③ 回収率（結果が出た {b.race_id.nunique():,}レース）===")
    if len(b) < 5:
        log("  まだ結果が出たレースがありません。")
        log("  ※ 払戻は週次(Step0.55)でnetkeibaから取得します。反映は次の週次更新後です。")
        log(f"     記録は {nrace:,}レース貯まっています。")
        return
    b["払戻"] = [PAY.get((r, k, c), 0.0)
                for r, k, c in zip(b.race_id, b["券種"], b["組み合わせ"])]

    for th, bt in sorted(BT.items()):
        s = b[b.軸gap >= th]
        if len(s) < 5:
            continue
        hit = int((s.払戻 > 0).sum())
        line = (f"  軸gap>={th}  {len(s):>5,}点  的中{hit:>4}"
                f"（{hit/len(s)*100:>4.1f}%）  回収率 {s.払戻.mean():>6.1f}%")
        if hit >= 20:
            v = s.払戻.values
            bs = np.array([rng.choice(v, len(v)).mean() for _ in range(3000)])
            line += f"  95%区間[{np.percentile(bs,2.5):.0f}, {np.percentile(bs,97.5):.0f}]"
        log(line)
        log(f"           BT: {bt['点数']:,}点 的中{bt['的中']} {bt['ROI']}%"
            f"  年別 {bt['年別']}")

    log("\n  券種別:")
    for k, g in b.groupby("券種"):
        if len(g) >= 5:
            log(f"    {k:<6}{len(g):>5,}点  的中{int((g.払戻>0).sum()):>4}"
                f"  回収率 {g.払戻.mean():>6.1f}%")

    # ── 判断の目安 ──────────────────────────────────────
    hit = int((b.払戻 > 0).sum())
    log("\n=== 判断の目安 ===")
    if hit < 100:
        log(f"  的中{hit}本。**まだ数字を信用しない。**（目安100本まであと{100-hit}本）")
        log("  的中が少ないと1本の大穴で数字がひっくり返る。5年検証でも")
        log("  60倍以上の帯は的中11本・95%区間[59.5, 288.4]で何も言えなかった。")
    elif hit < 300:
        log(f"  的中{hit}本。傾向が見え始める段階。まだ確定ではない。")
    else:
        log(f"  的中{hit}本。バックテスト(1,178本)と比べられる水準に近い。")
    log(f"\n  {ndays}開催日で{nrace:,}レース記録。的中100本には数か月かかります。")


if __name__ == "__main__":
    main()
