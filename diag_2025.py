# -*- coding: utf-8 -*-
"""2025年だけ弱いのはなぜか（2026-08-18）

なぜ調べるか
  どの買い方でも2025年だけ成績が落ちる。
    軸gap>=1.5   137 / 116 / 111 / 163 / **79**
    軸gap>=2.0   210 / 106 / 154 / 337 / **69**
  「モデルが古くなったから」は否定済み（2024年末で固定しても毎年学習しても
  2025年は79%前後で同じ）。ならば別の原因がある。

  2026年（未見データ）は115.6%だったので、2025年が例外だった可能性が高い。
  だが原因が分からないままだと、また起きたときに対応できない。

調べること（いずれも回収率とは独立に測れるもの）
  ① 市場そのものが効率的になったか … 市場だけのR²が上がっていないか
  ② モデルの実力が落ちたか        … ΔR²が下がっていないか
  ③ 買っている馬の性質が変わったか  … 人気・オッズ・頭数の分布
  ④ 特定の場・クラス・距離に偏っているか
  ⑤ 運の問題か                  … 的中率は保たれているのに配当だけ低い、など

  ⑤が本命の仮説。的中率が変わらず配当だけ落ちているなら、それは
  「当てているが配当に恵まれなかった」＝運。

実行: python diag_2025.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import model_diag as M

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
AX_GAP, MATE_GAP, MATE_MAX = 1.5, 1.3, 3
rng = np.random.default_rng(20260818)


def log(m):
    print(m, flush=True)


def main():
    d = pd.read_csv("resid_kinds_pred.csv", dtype={"race_id": str, "bn": str})
    d["gap"] = d.p1 / d.q
    d["年"] = d.race_id.str[:4].astype(int)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf", "距離", "出走頭数",
                              "クラス_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    PAY = {(r.race_id, r.券種, r.組み合わせ): r.払戻金
           for r in jv[jv.券種.isin(("単勝", "ワイド"))].itertuples()}
    log(f"検体 {len(d):,}頭 / {d.race_id.nunique():,}レース\n")

    # ── ① 市場そのものが効率的になったか ────────────────────
    log("=== ① 市場そのものの精度（市場だけで1着を当てたときのR²）===")
    log(f"  {'年':<8}{'レース':>8}{'市場R2':>9}{'ΔR2(モデルの上乗せ)':>22}")
    for y in YEARS:
        s = d[d.年 == y].copy()
        s["_rc"] = pd.factorize(s.race_id)[0]
        s["lq"] = np.log(s.q.clip(EPS))
        s["lp"] = np.log((s.p1 / s.groupby("race_id").p1.transform("sum")).clip(EPS))
        l0 = M.null_ll(s)
        _, lm = M.clogit(s, ["lq"])
        _, lb = M.clogit(s, ["lq", "lp"])
        log(f"  {y:<8}{s.race_id.nunique():>8,}{1-lm/l0:>9.4f}"
            f"{(1-lb/l0)-(1-lm/l0):>21.4f}")

    # ── ③⑤ 買っている馬と結果 ──────────────────────────
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        gv = g.gap.values
        k = int(np.argmax(gv))
        if gv[k] < AX_GAP:
            continue
        a = g.bn.values[k]
        y = int(rid[:4])
        rows.append({"年": y, "券種": "単勝", "払戻": PAY.get((rid, "単勝", a), 0.0),
                     "オッズ": g.odds.values[k], "人気": g["人気"].values[k],
                     "gap": gv[k], "頭数": g["出走頭数"].iloc[0],
                     "is_turf": g["is_turf"].iloc[0], "cls": g["クラス_num"].iloc[0]})
        if pd.to_numeric(g["is_turf"], errors="coerce").iloc[0] == 0:
            for j in [x for x in np.argsort(-gv) if x != k and gv[x] >= MATE_GAP][:MATE_MAX]:
                b = g.bn.values[j]
                rows.append({"年": y, "券種": "ワイド",
                             "払戻": PAY.get((rid, "ワイド", f"{min(a,b)}-{max(a,b)}"), 0.0),
                             "オッズ": g.odds.values[j], "人気": g["人気"].values[j],
                             "gap": gv[j], "頭数": g["出走頭数"].iloc[0],
                             "is_turf": 0, "cls": g["クラス_num"].iloc[0]})
    R = pd.DataFrame(rows)

    log("\n=== ② 買っている馬の性質は変わったか ===")
    log(f"  {'年':<8}{'点数':>7}{'的中率':>8}{'中央オッズ':>10}{'中央人気':>9}"
        f"{'平均gap':>9}{'ROI':>8}")
    for y in YEARS:
        s = R[R.年 == y]
        log(f"  {y:<8}{len(s):>7,}{(s.払戻>0).mean()*100:>7.1f}%"
            f"{s.オッズ.median():>10.1f}{s.人気.median():>9.0f}"
            f"{s.gap.mean():>9.2f}{s.払戻.mean():>7.1f}%")

    log("\n=== ⑤ 的中率は保たれているのに配当だけ落ちたのか ===")
    log("  当たったときの平均払戻を見る。的中率が同じで払戻だけ低いなら『運』")
    log(f"  {'年':<8}{'的中率':>8}{'当たりの平均払戻':>16}{'ROI':>8}")
    for y in YEARS:
        s = R[R.年 == y]
        h = s[s.払戻 > 0]
        log(f"  {y:<8}{(s.払戻>0).mean()*100:>7.1f}%{h.払戻.mean():>15.0f}円"
            f"{s.払戻.mean():>7.1f}%")

    log("\n=== ④ 2025年の内訳（どこで落ちたか）===")
    s25 = R[R.年 == 2025]
    oth = R[R.年 != 2025]
    log(f"  {'区分':<16}{'2025点数':>9}{'2025 ROI':>10}{'他年 ROI':>10}{'差':>8}")
    segs = [("単勝", R.券種 == "単勝"), ("ワイド", R.券種 == "ワイド"),
            ("芝", R.is_turf == 1), ("ダート", R.is_turf == 0),
            ("少頭数(<=12)", R.頭数 <= 12), ("多頭数(13+)", R.頭数 >= 13),
            ("下級(cls<=2)", R.cls <= 2), ("上級(cls>=4)", R.cls >= 4),
            ("人気1-3", R.人気 <= 3), ("人気4-9", (R.人気 >= 4) & (R.人気 <= 9)),
            ("人気10+", R.人気 >= 10)]
    for lab, f in segs:
        a = R[(R.年 == 2025) & f]
        b = R[(R.年 != 2025) & f]
        if len(a) < 50 or len(b) < 200:
            continue
        log(f"  {lab:<16}{len(a):>9,}{a.払戻.mean():>9.1f}%{b.払戻.mean():>9.1f}%"
            f"{a.払戻.mean()-b.払戻.mean():>+8.1f}")

    log("\n=== 2025年の成績は偶然の範囲か ===")
    v_oth = oth.払戻.values
    n25 = len(s25)
    sim = np.array([rng.choice(v_oth, n25).mean() for _ in range(4000)])
    p = float((sim <= s25.払戻.mean()).mean())
    log(f"  他の4年の払戻から{n25:,}点を無作為に取ると、2025年({s25.払戻.mean():.1f}%)"
        f"以下になる確率")
    log(f"  p = {p:.4f}")
    log(f"  → {'⚠ 偶然では説明しにくい。2025年に何かあった' if p < 0.05 else '○ 偶然の範囲。運の悪い年だったで説明がつく'}")
    log(f"  （参考）他4年の分布: 中央{np.median(sim):.1f}% "
        f"5%点{np.percentile(sim,5):.1f}% 95%点{np.percentile(sim,95):.1f}%")


if __name__ == "__main__":
    main()
