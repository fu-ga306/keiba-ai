# -*- coding: utf-8 -*-
"""★軸に選ばれる馬の「怪しい型」を5年分で検証する（2026-08-22）

きっかけ
  本日8/22の★軸22頭を見たところ、次のような馬が混じっていた。

    ショーマンフリート  138.6倍  gap1.80  14走  通常モデル1.09%
    ビューティドリーム   54.3倍  gap2.01   1走  通常モデル0.00%
    ライヴマティーニ    23.9倍  gap1.87   3走  通常モデル0.00%
    レットミーテイク    12.7倍  gap3.05   2走  通常モデル6.1%（市場6.2%と同じ）

  共通するのは「データが薄い」「通常モデルは全く推していない」こと。
  残差モデルだけが強く推している。これが妥当なのかを確かめる。

事前登録（ROIを見る前に固定。あとから条件を足さない）
  対象は確定した買い方（軸gap>=1.5 単勝＋ダートならワイド）の**軸の単勝**だけ。
  次の切り口で成績を比べる。

    A 出走数     1走以下 / 2-3走 / 4-9走 / 10走以上
    B 通常モデルの勝率  1%未満 / 1-3% / 3-10% / 10%以上
    C 軸のオッズ  10倍未満 / 10-30 / 30-60 / 60倍以上
    D クラス     新馬・未勝利 / 1-2勝 / 3勝以上

  判定
    その区分のROIが明確に100%を割り、かつ的中が100本以上あれば「除外候補」。
    的中が100本未満なら「測れていない」として何もしない。

  ⚠ ここで見つかった区分をそのまま除外すると、それは後付けの条件になる。
    除外するかどうかは順列検定を通してから決める（この検証では判定しない）。

実行: python diag_axis.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
AX_GAP = 1.5
MIN_HIT = 100
rng = np.random.default_rng(20260822)


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["年"] = d.race_id.str[:4].astype(int)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "馬名", "過去出走数", "クラス_num", "is_turf"])
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    bc = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str},
                                usecols=["race_id", "bn", "c_win"]) for y in YEARS])
    d = d.merge(bc, on=["race_id", "bn"], how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    TAN = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv.券種 == "単勝"].itertuples()}
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース")

    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        r = g.iloc[k]
        rows.append({"年": int(rid[:4]), "払戻": TAN.get((rid, r.bn), 0.0),
                     "gap": gv[k], "odds": r.odds, "n": r["過去出走数"],
                     "cw": r.c_win, "cls": r["クラス_num"]})
    R = pd.DataFrame(rows)
    log(f"★軸 {len(R):,}レース  ROI {R.払戻.mean():.1f}%  的中率 {(R.払戻>0).mean()*100:.1f}%\n")

    def show(title, groups):
        log(f"=== {title} ===")
        log(f"  {'区分':<20}{'点数':>8}{'的中':>7}{'的中率':>8}{'ROI':>8}{'判定':>16}")
        for lab, m in groups:
            s = R[m]
            if len(s) < 30:
                log(f"  {lab:<20}{len(s):>8,}{'':>7}{'':>8}{'':>8}{'標本不足':>16}")
                continue
            h = int((s.払戻 > 0).sum())
            roi = s.払戻.mean()
            if h < MIN_HIT:
                v = "測れていない"
            elif roi < 100:
                v = "⚠ 除外候補"
            else:
                v = "○"
            log(f"  {lab:<20}{len(s):>8,}{h:>7}{h/len(s)*100:>7.1f}%{roi:>7.1f}%{v:>16}")
        log("")

    n = pd.to_numeric(R.n, errors="coerce")
    cw = pd.to_numeric(R.cw, errors="coerce")
    o = pd.to_numeric(R.odds, errors="coerce")
    cl = pd.to_numeric(R.cls, errors="coerce")

    show("A 出走数（キャリアの浅さ）", [
        ("1走以下", n <= 1), ("2-3走", (n >= 2) & (n <= 3)),
        ("4-9走", (n >= 4) & (n <= 9)), ("10走以上", n >= 10)])
    show("B 通常モデルの勝率（他のモデルも推しているか）", [
        ("1%未満", cw < 0.01), ("1-3%", (cw >= 0.01) & (cw < 0.03)),
        ("3-10%", (cw >= 0.03) & (cw < 0.10)), ("10%以上", cw >= 0.10)])
    show("C 軸のオッズ", [
        ("10倍未満", o < 10), ("10-30倍", (o >= 10) & (o < 30)),
        ("30-60倍", (o >= 30) & (o < 60)), ("60倍以上", o >= 60)])
    show("D クラス", [
        ("新馬・未勝利", cl <= 1), ("1-2勝", (cl >= 2) & (cl <= 3)),
        ("3勝以上", cl >= 4)])

    log("=== 除外候補を全部外したらどうなるか（参考・順列検定は未実施）===")
    bad = pd.Series(False, index=R.index)
    for lab, m in [("出走数1走以下", n <= 1), ("通常モデル1%未満", cw < 0.01),
                   ("60倍以上", o >= 60)]:
        s = R[m]
        if len(s) >= 30 and int((s.払戻 > 0).sum()) >= MIN_HIT and s.払戻.mean() < 100:
            bad |= m
            log(f"  除外: {lab}")
    if bad.any():
        k = R[~bad]
        log(f"  除外前 {len(R):,}点 ROI {R.払戻.mean():.1f}%")
        log(f"  除外後 {len(k):,}点 ROI {k.払戻.mean():.1f}%"
            f"（{len(R)-len(k):,}点を除外）")
        log("  年別: " + "  ".join(
            f"{y}:{k[k.年==y].払戻.mean():.0f}%" for y in YEARS))
    else:
        log("  的中100本以上で100%を割る区分は無かった → 除外する理由が無い")


if __name__ == "__main__":
    main()
