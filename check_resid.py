# -*- coding: utf-8 -*-
"""残差モデルの本番実装が検証と一致するかを機械的に確かめる（2026-08-17）

なぜ必要か
  2026-08-16〜17に「検証で見た数字」と「本番が実際に買うもの」がズレる事故を
  5回起こした。うち1つは的中54本が11本に見えていた（馬番のゼロ埋め漏れ）。
  数字を出す前に、まず実装が検証どおりかを確かめる。

確かめる買い方（resid_io.pick_bets が唯一の実装）
  軸  : 残差モデルの gap が最大の1頭・gap>=2.0 → 単勝1点
  ダートなら、相手（軸以外で gap>=1.3・最大3頭）にワイドを追加
  芝は単勝のみ

やること
  ① 検証データ(resid_kinds_pred.csv)を本番の関数に通し、買い目を作る
  ② その買い目を実払戻(jv_payouts)で照合し、EXPECT と一致するか見る
  ③ 本番で起こりうる欠け（列なし・全欠損・1頭）に耐えるか見る

⚠ 買い方を変えたら EXPECT を更新すること。更新せずに数字が変わったら実装ズレ。

実行: python check_resid.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import resid_io

# resid_gate.py で測った値。ここと一致しなければ実装がズレている。
EXPECT = {"買い方": "軸gap>=2.0 単勝 ＋ ダートならワイド(相手gap>=1.3・最大3頭)",
          "点数": 2926, "的中": 236, "ROI": 163.3}


def log(m):
    print(m, flush=True)


def main():
    try:
        d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    except FileNotFoundError:
        log("resid_kinds_pred.csv がありません。先に python resid_kinds.py")
        return
    d["gap"] = d.p1 / d.q
    d["馬番"] = pd.to_numeric(d["bn"], errors="coerce")
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    log(f"検証データ {len(d):,}頭 / {d.race_id.nunique():,}レース")
    log(f"買い方: {EXPECT['買い方']}\n")

    # ── ① 本番の関数で買い目を作り、実払戻で照合 ─────────────────
    m = {"gap_min": resid_io.AX_GAP}
    rows, nrace, nwide = [], 0, 0
    for rid, g in d.groupby("race_id", sort=False):
        bets = resid_io.pick_bets(g, model=m)
        if not bets:
            continue
        nrace += 1
        for b in bets:
            if b["券種"] == "ワイド":
                nwide += 1
            rows.append({"年": int(rid[:4]), "券種": b["券種"],
                         "払戻": PAY.get((rid, b["券種"], b["組み合わせ"]), 0.0)})
    R = pd.DataFrame(rows)
    if R.empty:
        log("⚠ 買い目が1つも出ませんでした")
        return
    roi = R.払戻.sum() / (len(R) * 100) * 100
    hit = int((R.払戻 > 0).sum())
    log("=== ① 本番の関数(resid_io.pick_bets)が作った買い目 ===")
    log(f"  買うレース {nrace:,}  点数 {len(R):,}（単勝{len(R)-nwide:,} / ワイド{nwide:,}）")
    log(f"  的中 {hit}（{hit/len(R)*100:.1f}%）  ROI {roi:.1f}%")
    log("  年別: " + "  ".join(
        f"{y}:{g.払戻.sum()/(len(g)*100)*100:.0f}%" for y, g in R.groupby("年")))
    log("  券種別: " + "  ".join(
        f"{k} {len(g):,}点 的中{int((g.払戻>0).sum())} {g.払戻.sum()/(len(g)*100)*100:.1f}%"
        for k, g in R.groupby("券種")))

    # ── ② EXPECT と照合 ─────────────────────────────────
    log("\n=== ② 検証(resid_gate.py)で測った値 ===")
    log(f"  {EXPECT['点数']:,}点  的中{EXPECT['的中']}  ROI {EXPECT['ROI']}%")
    ok = (abs(len(R) - EXPECT["点数"]) <= 5 and abs(hit - EXPECT["的中"]) <= 3
          and abs(roi - EXPECT["ROI"]) < 1.0)
    log(f"\n  点数 {len(R):,} vs {EXPECT['点数']:,}  {'○' if abs(len(R)-EXPECT['点数'])<=5 else '×'}")
    log(f"  的中 {hit} vs {EXPECT['的中']}  {'○' if abs(hit-EXPECT['的中'])<=3 else '×'}")
    log(f"  ROI {roi:.1f}% vs {EXPECT['ROI']}%  {'○' if abs(roi-EXPECT['ROI'])<1.0 else '×'}")

    # ── ③ 欠けへの耐性 ───────────────────────────────────
    log("\n=== ③ 本番で起こりうる欠けへの耐性 ===")
    g0 = d[d.race_id == d.race_id.iloc[0]].copy()
    cases = [("空のDataFrame", g0.iloc[0:0]), ("gap列が無い", g0.drop(columns=["gap"])),
             ("gapが全部欠損", g0.assign(gap=np.nan)), ("1頭だけ", g0.iloc[:1]),
             ("馬番が欠損", g0.assign(馬番=np.nan)),
             ("is_turfが欠損", g0.assign(is_turf=np.nan))]
    safe = True
    for lab, x in cases:
        try:
            r = resid_io.pick_bets(x, model=m)
            log(f"  {lab:<18} → {len(r)}点（例外なし）")
        except Exception as e:
            safe = False
            log(f"  {lab:<18} → ⚠ 例外 {type(e).__name__}: {e}")

    log("\n" + "=" * 58)
    log("✅ 実装は検証どおり。この数字は本番の成績を表す" if ok and safe
        else "⚠ 実装がズレている。この数字を成績として使ってはいけない")


if __name__ == "__main__":
    main()
