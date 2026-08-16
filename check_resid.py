# -*- coding: utf-8 -*-
"""残差モデルの本番実装が検証と一致するかを機械的に確かめる（2026-08-17）

なぜ必要か
  2026-08-16〜17に「検証で見た数字」と「本番が実際に買うもの」がズレる事故を
  5回起こした。うち1つは的中54本が11本に見えていた（馬番のゼロ埋め漏れ）。
  数字を出す前に、まず実装が検証どおりかを確かめる。

やること
  ① 検証で作った予測（resid_pred.csv）を、本番の窓口（resid_io.pick_bet）に
     通し、同じ買い目が出るかを見る
  ② その買い目で回収率を計算し、EXPECT と一致するかを見る
  ③ 特徴量の欠損・オッズ未取得など、本番で起こりうる欠けに耐えるかを見る

⚠ 買い方を変えたら EXPECT を更新すること。更新せずに数字が変わったら、
  それは実装がズレたということ。

実行: python check_resid.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import resid_io

# train_resid.py backtest で測った値。ここと一致しなければ実装がズレている。
EXPECT = {"買い方": "gap>=2.0 の1頭を単勝1点（3シード平均・600回）",
          "点数": 1891, "的中": 203, "ROI": 157.1}


def log(m):
    print(m, flush=True)


def main():
    try:
        d = pd.read_csv("resid_pred.csv", dtype={"race_id": str})
    except FileNotFoundError:
        log("resid_pred.csv がありません。先に python train_resid.py backtest")
        return
    log(f"検証データ {len(d):,}頭 / {d.race_id.nunique():,}レース\n")

    # ── ① 本番の窓口で買い目を作る ────────────────────────────
    m = {"gap_min": 2.0}
    picks = []
    for rid, g in d.groupby("race_id", sort=False):
        b = resid_io.pick_bet(g, model=m)
        if b is not None:
            picks.append(b)
    if not picks:
        log("⚠ 買い目が1つも出ませんでした")
        return
    P = pd.concat(picks)
    roi = (P.win * P.odds).sum() / len(P) * 100
    log("=== ① 本番の関数(resid_io.pick_bet)が作った買い目 ===")
    log(f"  {len(P):,}点  的中{int(P.win.sum())}  ROI {roi:.1f}%")
    log("  年別: " + "  ".join(
        f"{y}:{(x.win*x.odds).sum()/len(x)*100:.0f}%" for y, x in P.groupby("年")))

    # ── ② 検証側と同じ選び方を素朴に書いて突き合わせる ───────────
    sel = d.loc[d.groupby("race_id")["gap"].idxmax()]
    ref = sel[sel.gap >= 2.0]
    roi_ref = (ref.win * ref.odds).sum() / len(ref) * 100
    log("\n=== ② 検証側の素朴な実装 ===")
    log(f"  {len(ref):,}点  的中{int(ref.win.sum())}  ROI {roi_ref:.1f}%")

    same_n = len(P) == len(ref)
    key_p = set(zip(P.race_id.astype(str), P.get("馬名", P.index).astype(str)))
    key_r = set(zip(ref.race_id.astype(str), ref.get("馬名", ref.index).astype(str)))
    same_h = key_p == key_r
    log(f"\n  点数一致 {'○' if same_n else '×'}"
        f"（{len(P):,} vs {len(ref):,}）")
    log(f"  買う馬が完全一致 {'○' if same_h else '×'}"
        f"（違い {len(key_p ^ key_r)}頭）")
    log(f"  回収率一致 {'○' if abs(roi-roi_ref) < 0.1 else '×'}"
        f"（{roi:.1f}% vs {roi_ref:.1f}%）")

    # ── ③ 欠けに耐えるか ───────────────────────────────────
    log("\n=== ③ 本番で起こりうる欠けへの耐性 ===")
    g0 = d[d.race_id == d.race_id.iloc[0]].copy()
    cases = [
        ("空のDataFrame", g0.iloc[0:0]),
        ("gap列が無い", g0.drop(columns=["gap"])),
        ("gapが全部欠損", g0.assign(gap=np.nan)),
        ("1頭だけ", g0.iloc[:1]),
    ]
    ok = True
    for lab, x in cases:
        try:
            r = resid_io.pick_bet(x, model=m)
            log(f"  {lab:<18} → {'買い目なし' if r is None else f'{len(r)}点'}（例外なし）")
        except Exception as e:
            ok = False
            log(f"  {lab:<18} → ⚠ 例外 {type(e).__name__}: {e}")
    log(f"  {'○ どの欠けでも落ちない' if ok else '× 例外が出る。修正が必要'}")

    # ── 判定 ──────────────────────────────────────────
    log("\n" + "=" * 56)
    good = same_n and same_h and abs(roi - roi_ref) < 0.1 and ok
    log("✅ 実装は検証どおり。この数字は本番の成績を表す" if good
        else "⚠ 実装がズレている。この数字を成績として使ってはいけない")
    if EXPECT["点数"] is None:
        log(f"\n  EXPECT を更新してください:")
        log(f'    EXPECT = {{"買い方": "gap>=2.0 の1頭を単勝1点",')
        log(f'              "点数": {len(P)}, "的中": {int(P.win.sum())},'
            f' "ROI": {roi:.1f}}}')
    else:
        hit = int(P.win.sum())
        match = (abs(len(P) - EXPECT["点数"]) <= 5 and abs(hit - EXPECT["的中"]) <= 2
                 and abs(roi - EXPECT["ROI"]) < 1.0)
        log(f"\n  EXPECT との照合: {'✅ 一致' if match else '⚠ 不一致'}"
            f"（{len(P)}点/{hit}的中/{roi:.1f}% vs "
            f"{EXPECT['点数']}点/{EXPECT['的中']}的中/{EXPECT['ROI']}%）")


if __name__ == "__main__":
    main()
