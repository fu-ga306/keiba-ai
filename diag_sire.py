# -*- coding: utf-8 -*-
"""2025年の弱さは「新種牡馬の情報が無いこと」で説明できるか（2026-08-18）

経緯（私の誤診の訂正）
  2025年の血統特徴量に欠損が多いのを見て「データ不備」と診断し、
  features.py を再生成した。だが結果は同じだった。

  正しくは**意図した設計**だった。features.py は学習・検証時に
  sire_stats_father_train.csv（≤2024集計・626頭）を使う。全期間版は708頭なので、
  2025年以降に初めて産駒が走った82頭の種牡馬は含まれない。
  全期間版を2025年のレースに使うと、その馬自身の2025年の結果が種牡馬成績に
  入ってしまうため、**リーク防止として正しい**。

  つまりバックテストは本番より情報が少ない状態で測っている。
  本番（use_train_snapshot=False）は全期間版を使うので血統情報が全部ある。

この検証で確かめること
  2025年の馬を「父が train版にいる馬／いない馬」に分けて成績を比べる。

    父がいない馬だけ悪い → 本番では改善する見込みがある
    どちらも同じく悪い   → 血統は原因ではない。別を探す

  ⚠ 「父がいない馬」は新種牡馬の産駒＝若い馬に偏るので、
    年齢や出走数の違いも一緒に出る。そこも併せて見る。

実行: python diag_sire.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAP, MATE_GAP, MATE_MAX = 1.5, 1.3, 3
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["年"] = d.race_id.str[:4].astype(int)

    # 各馬の父を引く
    hm = pd.read_csv("horse_master.csv")
    hm["horse_id"] = hm["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    rc = pd.read_csv("race_data_clean.csv", usecols=["race_id", "馬名", "horse_id"],
                     dtype=str, low_memory=False)
    rc["race_id"] = rc["race_id"].str.replace(r"\.0$", "", regex=True)
    rc["horse_id"] = rc["horse_id"].astype(str).str.replace(".0", "", regex=False).str.strip()
    rc = rc.merge(hm[["horse_id", "父馬"]], on="horse_id", how="left")
    d = d.merge(rc[["race_id", "馬名", "父馬"]].drop_duplicates(["race_id", "馬名"]),
                on=["race_id", "馬名"], how="left")

    tr = set(pd.read_csv("sire_stats_father_train.csv")["父名"].dropna())
    al = set(pd.read_csv("sire_stats_father.csv")["父名"].dropna())
    d["父あり"] = d["父馬"].isin(tr)
    log(f"検体 {len(d):,}頭  train版の種牡馬 {len(tr)}頭 / 全期間版 {len(al)}頭")
    log(f"train版に無く全期間版にいる父: {len(al - tr)}頭\n")

    log("=== 年ごと: 父が train版にいる割合 ===")
    log(f"  {'年':<8}{'頭数':>9}{'父あり':>9}{'父なし':>9}{'父なし率':>10}")
    for y in YEARS:
        s = d[d.年 == y]
        log(f"  {y:<8}{len(s):>9,}{s.父あり.sum():>9,}{(~s.父あり).sum():>9,}"
            f"{(~s.父あり).mean()*100:>9.1f}%")

    # 買い目を作って、軸の父の有無で分ける
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")

    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        a = g.bn.values[k]
        y = int(rid[:4])
        ok = bool(g.父あり.values[k])
        rows.append({"年": y, "父あり": ok, "券種": "単勝",
                     "払戻": PAY.get((rid, "単勝", a), 0.0),
                     "win": g.win.values[k], "gap": gv[k]})
        if pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0:
            for j in [x for x in np.argsort(-gv) if x != k and gv[x] >= MATE_GAP][:MATE_MAX]:
                b = g.bn.values[j]
                rows.append({"年": y, "父あり": ok, "券種": "ワイド",
                             "払戻": PAY.get((rid, "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0),
                             "win": np.nan, "gap": gv[j]})
    R = pd.DataFrame(rows)

    log("\n=== 軸の父が train版にいるかで分けた成績 ===")
    log(f"  {'年':<8}{'父あり点数':>10}{'父ありROI':>11}{'父なし点数':>11}{'父なしROI':>11}")
    for y in YEARS:
        a = R[(R.年 == y) & R.父あり]
        b = R[(R.年 == y) & ~R.父あり]
        la = f"{a.払戻.mean():.1f}%" if len(a) > 50 else "--"
        lb = f"{b.払戻.mean():.1f}%" if len(b) > 50 else "--"
        log(f"  {y:<8}{len(a):>10,}{la:>11}{len(b):>11,}{lb:>11}")

    log("\n=== 2025年だけ詳しく ===")
    s = R[R.年 == 2025]
    for lab, f in (("父あり", s.父あり), ("父なし", ~s.父あり)):
        x = s[f]
        if len(x) < 30:
            log(f"  {lab}: {len(x)}点（少なすぎる）")
            continue
        bs = np.array([rng.choice(x.払戻.values, len(x)).mean() for _ in range(2000)])
        log(f"  {lab:<8}{len(x):>6,}点  的中{int((x.払戻>0).sum()):>4}"
            f"（{(x.払戻>0).mean()*100:>4.1f}%）  ROI {x.払戻.mean():>6.1f}%"
            f"  95%区間[{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]")

    log("\n=== 他の年（2021-2024）の同じ切り分け ===")
    o = R[R.年 != 2025]
    for lab, f in (("父あり", o.父あり), ("父なし", ~o.父あり)):
        x = o[f]
        if len(x) < 30:
            continue
        log(f"  {lab:<8}{len(x):>6,}点  的中{int((x.払戻>0).sum()):>4}"
            f"（{(x.払戻>0).mean()*100:>4.1f}%）  ROI {x.払戻.mean():>6.1f}%")

    log("\n=== 判定 ===")
    a25 = R[(R.年 == 2025) & R.父あり]
    b25 = R[(R.年 == 2025) & ~R.父あり]
    if len(a25) > 50:
        log(f"  2025年の『父あり』だけでも ROI {a25.払戻.mean():.1f}%")
        if a25.払戻.mean() >= 100:
            log("  → 父の有無が原因。本番（全期間版）なら改善する見込み")
        else:
            log("  → 父ありでも100%未満。血統は主因ではない。別の原因を探す必要がある")


if __name__ == "__main__":
    main()
