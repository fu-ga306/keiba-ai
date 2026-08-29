# -*- coding: utf-8 -*-
"""残差モデルの本番実装が検証と一致するかを機械的に確かめる（2026-08-17）

なぜ必要か
  2026-08-16〜17に「検証で見た数字」と「本番が実際に買うもの」がズレる事故を
  5回起こした。うち1つは的中54本が11本に見えていた（馬番のゼロ埋め漏れ）。
  数字を出す前に、まず実装が検証どおりかを確かめる。

確かめる買い方（resid_io.pick_bets が唯一の実装）
  軸  : 残差モデルの gap が最大の1頭・gap>=1.5 → 単勝1点
  ダートなら、相手（軸以外で gap>=1.3・最大3頭）にワイドを追加
  芝は単勝のみ

やること
  ① 検証データ(resid_kinds_pred.csv)を本番の関数に通し、買い目を作る
  ② その買い目を実払戻(jv_payouts)で照合し、EXPECT と一致するか見る
  ③ 本番で起こりうる欠け（列なし・全欠損・1頭）に耐えるか見る

⚠ 買い方を変えたら EXPECT を必ず更新すること。忘れると誤警告が出続け、
  本当にズレたときに気づけなくなる（2026-08-17〜25に実際に起きた）。

実行: python check_resid.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import resid_io

# 採用時に固定した基準値。ここと一致しなければ実装が変わったということ。
#   ⚠ 2026-08-25まで gap>=2.0 時代の値(2,926点/236/163.3%)が残っていた。
#     8/17にしきい値を 2.0→1.5 に緩めたとき EXPECT を更新し忘れたため、
#     以後ずっと「⚠ 実装がズレている。この数字を成績として使ってはいけない」と
#     誤った警告を出し続けていた。オオカミ少年になっていて、本当にズレたときに
#     気づけない状態だった。
#   これは独立検証ではなく回帰テスト（golden値）。実装が意図せず変わったことを
#   検出するのが役目で、買い方の良し悪しを保証するものではない。
EXPECT = {"買い方": "軸gap>=1.5 単勝 ＋ ダートならワイド(相手gap>=1.3・最大3頭)",
          "点数": 10349, "的中": 1178, "ROI": 120.6}


def log(m):
    print(m, flush=True)



def _vintage_ok():
    """検証データが、本番のモデル・特徴量より新しいかを見る（2026-08-29）

    なぜ要るか
      このスクリプトは「選び方(resid_io.pick_bets)が検証どおりか」しか見ていない。
      **選び方が同じでも、モデルが違えば選ぶ馬が変わる。**
      実際に次の状態で「✅ 実装は検証どおり」と出していた。

        08/17  resid_kinds_pred.csv  ← 検証データ。120.6%の根拠
        08/25  race_features.csv     ← 8日後に作り直された
        08/25  model_resid.pkl       ← 8日後に学習し直された。本番はこちら

      本番の芝の軸は平均8.7番人気、検証は5.9番人気で、95%区間の外に出ていた。
      gapの分布も頭数もほぼ同じなのに選ぶ馬だけが違う＝モデルが違う。
      これは6原因の④「検証と本番で同じものを計算していない」そのもの。
    """
    import os
    base = "resid_kinds_pred.csv"
    if not os.path.exists(base):
        return True, []
    t0 = os.path.getmtime(base)
    # ⚠ 見るのは「①で検証できないもの」だけにする。
    #   resid_io.py と features.py は①が実際に走らせて数字の一致を見ているので
    #   ここで日付だけを見て警告すると、二重に鳴る＝オオカミ少年になる。
    #   ①で見られないのは、本番だけが使うモデルと特徴量。
    ng = []
    for f in ("race_features.csv", "model_resid.pkl"):
        if os.path.exists(f) and os.path.getmtime(f) > t0:
            d = (os.path.getmtime(f) - t0) / 86400
            ng.append(f"{f} が {d:.1f}日 新しい")
    return (not ng), ng


def main():
    try:
        d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    except FileNotFoundError:
        log("resid_kinds_pred.csv がありません。先に python resid_kinds.py")
        return
    d["gap"] = d.p1 / d.q
    d["馬番"] = pd.to_numeric(d["bn"], errors="coerce")
    # ⚠ 分割して読む（2026-08-29）
    #   一括で読むと予想システムがMFモデル3GBを抱えている時間帯にメモリ不足で落ちる。
    #   実際に日次点検が「check_resid.py が異常終了」と報告したが、
    #   単体で走らせると通る＝**実装ではなく点検側が落ちていた**。
    #   これを放置すると重要度高の警告が出続けてオオカミ少年になる。
    _p = []
    for _ch in pd.read_csv("race_features.csv", usecols=["race_id", "is_turf"],
                           dtype={"race_id": str}, chunksize=200000):
        _p.append(_ch.drop_duplicates("race_id"))
    rf = pd.concat(_p).drop_duplicates("race_id")
    del _p
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

    # ── ②' 検証データが古くないか ───────────────────────────
    #   ここが×なら、上の○は「古いモデルに対して選び方が合っている」という
    #   意味しか持たない。本番の成績を表さない。
    v_ok, v_ng = _vintage_ok()
    log("")
    log("=== 検証データと本番の世代が揃っているか ===")
    if v_ok:
        log("  ○ resid_kinds_pred.csv は本番のモデル・特徴量より新しい")
    else:
        log("  ⚠ 検証データのほうが古い。**この数字は本番の成績を表さない**")
        for m in v_ng:
            log(f"     - {m}")
        log("     → python resid_kinds.py で作り直す（開催日は避ける。メモリを食う）")
    ok = ok and v_ok

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
