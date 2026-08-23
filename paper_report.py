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
    # ⚠ 券種ごとに「結果が出たか」を分ける（2026-08-22）
    #   today_results.csv には単勝・複勝の払戻しか無い。ワイドの払戻は
    #   払戻表(payout_data.csv)を週次で取るまで分からない。
    #   券種をまとめて「結果が出た」と扱うと、**ワイドの的中を外れ(0円)として
    #   数えてしまう**。回収率を実際より低く見せる方向の誤りで、
    #   「やっぱりダメだった」と誤って結論づける事故につながる。
    done_tan = set(jv.race_id)      # 単勝: 払戻表 or today_results
    done_wide = set(jv.race_id)     # ワイド: 払戻表のみ

    # ── 払戻表が来る前でも単勝は照合できる ─────────────────────────────
    #   払戻表(payout_data.csv)は週次でしか取りに行かない。それを待つと
    #   開催日の夜に見ても「結果が出た0レース」としか出ない。
    #   単勝の払戻を持っているファイルが2つあるので、そこから先に拾う。
    #     today_results.csv  … 当日ぶんだけ。翌日には上書きされて消える
    #     history_marks.csv  … 日次アーカイブ(21:10)が積んでいる過去ぶん
    #   ⚠ today_results.csv だけを見ていると前日ぶんが照合できず、
    #     通算の回収率が出せない（2026-08-23に判明）。両方見ること。
    #   ワイドの払戻はどちらにも無いので、そちらは週次を待つ。
    for src, note in (("history_marks.csv", "過去ぶん"),
                      ("today_results.csv", "当日ぶん")):
        fp2 = os.path.join(BASE_DIR, src)
        if not os.path.exists(fp2):
            continue
        try:
            t = pd.read_csv(fp2, dtype={"race_id": str, "馬番": str})
        except Exception as e:
            log(f"  （{src} を読めません: {type(e).__name__}）")
            continue
        if "単勝" not in t.columns or "着順" not in t.columns:
            continue
        t["馬番"] = (t["馬番"].astype(str)
                   .str.replace(r"\.0$", "", regex=True).str.zfill(2))
        t["単勝"] = pd.to_numeric(t["単勝"], errors="coerce").fillna(0)
        for r in t[t["単勝"] > 0].itertuples():
            key = (r.race_id, "単勝", r.馬番)
            if key not in PAY:                 # 払戻表があればそちらを優先
                PAY[key] = float(r.単勝)
        # 着順が入っているレースは「結果が出た」とみなす（外れも0円で数える）
        fin = set(t.loc[pd.to_numeric(t["着順"], errors="coerce").notna(), "race_id"])
        n_add = len(fin - done_tan)
        done_tan |= fin        # 単勝だけ。ワイドは払戻が無いので足さない
        if n_add:
            log(f"  （{note} {n_add}レースを {src} から反映。単勝のみ）")

    _ok = (((d["券種"] == "単勝") & d.race_id.isin(done_tan))
           | ((d["券種"] == "ワイド") & d.race_id.isin(done_wide)))
    b = d[(d.判定 == "買い") & _ok].copy()
    _pend = d[(d.判定 == "買い") & ~_ok & d["券種"].isin(("単勝", "ワイド"))]
    if len(_pend):
        log("  （結果待ち "
            + " / ".join(f"{k}{v}点" for k, v in _pend["券種"].value_counts().items())
            + "。ワイドの払戻は週次取得後）")
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

    # ── ④ 決める時刻による買い目の違い ──────────────────────────
    #   バックテストは確定オッズで買い目を選んでいる(bet_cache の odds)。
    #   本番は発走7分前に決めるしかない。同じ買い目にならないので、
    #   BTの数字がそのまま出るとは限らない。その差をここで測る。
    close = os.path.join(BASE_DIR, "paper_resid_close.csv")
    if os.path.exists(close):
        c = pd.read_csv(close, dtype={"race_id": str, "組み合わせ": str})
        A = {(r.race_id, r.券種, str(r.組み合わせ))
             for r in d[d.判定 == "買い"].itertuples()}
        B = {(r.race_id, r.券種, str(r.組み合わせ))
             for r in c[c.判定 == "買い"].itertuples()}
        both = len(A & B)
        # ⚠ 必ず「両方に記録があるレース」だけで比べる（2026-08-23）
        #   締切時の記録は7分前より遅れて貯まるので、母集団が揃わない。
        #   全体で突き合わせると、まだ締切記録が無いレースの買い目が
        #   全部「7分前のみ」に数えられ、食い違い93%のような嘘の数字が出る。
        #   比較は同じレースどうしでしかできない。
        common = {r for r in d.race_id.unique()} & {r for r in c.race_id.unique()}
        A = {x for x in A if x[0] in common}
        B = {x for x in B if x[0] in common}
        both = len(A & B)
        log("\n=== ④ いつ決めるかで買い目がどれだけ変わるか ===")
        log(f"  両方に記録があるレース {len(common)}件で比較")
        log(f"  7分前(実際に賭ける) {len(A):>5}点")
        log(f"  締切時(BTに近い)   {len(B):>5}点")
        uni = len(A | B)
        if uni:
            log(f"  一致 {both}点  7分前のみ {len(A-B)}点  締切時のみ {len(B-A)}点")
            log(f"  → 選び方の食い違い {uni-both}/{uni} ({(uni-both)/uni*100:.0f}%)")
            log("  BTは確定オッズで選べるが、本番は7分前に決めるしかない。")
            log("  ただし odds_timing_bt.py の検証では、この食い違いがあっても")
            log("  ROIの目減りは-0.8ptだった（順位はオッズに依存しないため）。")
        else:
            log("  まだ買い判定が両方に無いので比べられません。")
        if len(common) < 20:
            log(f"  ※ {len(common)}レースでは揺れます。数開催ぶん貯めてから見ること。")
    else:
        log("\n=== ④ いつ決めるかで買い目がどれだけ変わるか ===")
        log("  paper_resid_close.csv がまだありません（次の開催日から貯まります）。")

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
