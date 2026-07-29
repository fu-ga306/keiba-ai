# -*- coding: utf-8 -*-
"""判定帯で絞らず「全レース」で評価し、配分最適化で利益が出せるかを検証する。

ユーザー提案:
  ①判定帯ではなく全体で評価する → サンプルが161R→3144Rに増え判定精度が上がる
  ②掛け金の配分を変えれば利益が出せるのでは

配分の数学: ポートフォリオのROIは各戦略のROIの加重平均。
  ROI_total = Σ w_i * ROI_i  (Σw_i = 1)
  → すべての ROI_i < 100% なら、どう配分しても 100% を超えられない。
  → 100%超が存在する場合のみ、そこへ寄せることで全体を上げられる。
したがって「100%超の戦略が存在するか」を全サンプルで厳密に判定するのが先。

各候補について:
  ・全レース(2025 3,144R)での実測ROI
  ・レース単位ブートストラップの95%信頼区間（下限>100%なら本物）
  ・的中率と必要サンプル数
最後に、100%超の候補だけで組んだ最適配分のROIを示す。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

UNORD = {"馬連", "ワイド", "3連複", "枠連"}
RNG = np.random.default_rng(42)


def log(m):
    print(m, flush=True)


def load(year="2025"):
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF複勝率", "MF勝率順位", "MF複勝順位"]]
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "馬番", "人気", "単勝オッズ", "クラス_num", "着順_num"]]
    d = rf.merge(p3, on=["race_id", "馬名"], how="inner").merge(
        mf, on=["race_id", "馬名"], how="inner")
    for c in ["馬番", "人気", "単勝オッズ", "クラス_num", "着順_num",
              "MF勝率", "MF複勝率", "MF勝率順位", "MF複勝順位", "place3順"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["馬番", "人気", "単勝オッズ", "着順_num"])
    d = d[d["race_id"].str.startswith(year)]
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(year)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORD:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def candidates(d, pay):
    """全レースを対象に、候補戦略ごとの (レース, 投資, 払戻) を作る。"""
    out = {}

    def add(name, rid, inv, ret):
        out.setdefault(name, []).append((rid, inv, ret))

    for rid, g in d.groupby("race_id"):
        if len(g) < 4:
            continue
        fav = g.loc[g["人気"].idxmin()]
        hon_i = fav.name if fav["単勝オッズ"] <= 2.0 else g["place3順"].idxmin()
        hon = g.loc[hon_i]
        w1 = g[g["MF勝率順位"] == 1]
        f1 = g[g["MF複勝順位"] == 1]
        myo = w1.iloc[0] if len(w1) else None
        fm = f1.iloc[0] if len(f1) else None
        part = g.sort_values("MF複勝順位")

        def P(kind, combo):
            c = "-".join(sorted(combo.split("-"))) if kind in UNORD else combo
            return pay.get((rid, kind, c), 0)

        # 単体（軸候補ごと）
        for nm, h in [("◎", hon), ("複妙", fm), ("妙", myo)]:
            if h is None:
                continue
            add(f"{nm}の複勝", rid, 100, P("複勝", h["bn"]))
            add(f"{nm}の単勝", rid, 100, P("単勝", h["bn"]))
        # 人気1位（対照）
        add("1番人気の複勝", rid, 100, P("複勝", fav["bn"]))
        # ワイド: 複妙-複勝上位2,3
        if fm is not None:
            inv = ret = 0
            for _, p in part[part["MF複勝順位"].between(2, 3)].iterrows():
                inv += 100
                ret += P("ワイド", f"{fm['bn']}-{p['bn']}")
            if inv:
                add("複妙-複勝2,3のワイド", rid, inv, ret)
        # 馬連: 複妙-複勝2,3
        if fm is not None:
            inv = ret = 0
            for _, p in part[part["MF複勝順位"].between(2, 3)].iterrows():
                inv += 100
                ret += P("馬連", f"{fm['bn']}-{p['bn']}")
            if inv:
                add("複妙-複勝2,3の馬連", rid, inv, ret)
        # 馬単: 妙→複勝上位5
        if myo is not None:
            inv = ret = 0
            for _, p in part[part["馬名"] != myo["馬名"]].head(5).iterrows():
                inv += 100
                ret += P("馬単", f"{myo['bn']}-{p['bn']}")
            if inv:
                add("妙→複勝5の馬単", rid, inv, ret)
        # 3連複: 複勝上位3
        top3 = part.head(3)
        if len(top3) == 3:
            add("複勝上位3の3連複", rid, 100,
                P("3連複", "-".join(sorted(top3["bn"]))))
    return out


def boot_ci(rows, n_boot=4000):
    arr = np.array([[r[1], r[2]] for r in rows], dtype=float)
    n = len(arr)
    if n < 30:
        return None
    idx = RNG.integers(0, n, size=(n_boot, n))
    rois = arr[:, 1][idx].sum(axis=1) / arr[:, 0][idx].sum(axis=1) * 100
    return (arr[:, 1].sum() / arr[:, 0].sum() * 100,
            (arr[:, 1] > 0).mean() * 100, n,
            np.percentile(rois, 2.5), np.percentile(rois, 97.5))


def main():
    for year in ("2025",):
        d, pay = load(year)
        cands = candidates(d, pay)
        log("=" * 94)
        log(f"【全レース評価】{year} {d['race_id'].nunique():,}レース"
            f"（判定帯で絞らない＝サンプル最大）")
        log("=" * 94)
        log(f"  {'戦略':<24}{'R数':>7}{'的中率':>8}{'ROI':>8}"
            f"{'95%信頼区間':>24}{'判定':>6}")
        res = {}
        for nm, rows in cands.items():
            b = boot_ci(rows)
            if not b:
                continue
            roi, hit, n, lo, hi = b
            res[nm] = b
            mark = "◎黒字" if lo > 100 else ("△不明" if hi > 100 else "×赤字")
            log(f"  {nm:<24}{n:7,}{hit:7.1f}%{roi:7.1f}%"
                f"   [{lo:6.1f}% 〜 {hi:7.1f}%]{mark:>6}")

        log("\n" + "=" * 94)
        log("【配分を変えれば利益が出るか】")
        log("=" * 94)
        best = max(res.items(), key=lambda x: x[1][0])
        over = [k for k, v in res.items() if v[3] > 100]
        log(f"  最良の戦略: {best[0]} = {best[1][0]:.1f}% "
            f"[{best[1][3]:.1f}〜{best[1][4]:.1f}]")
        log(f"  信頼区間の下限が100%を超える戦略: {over if over else 'なし'}")
        log("")
        log("  ポートフォリオROI = Σ(配分比 × 各戦略ROI) ＝ 加重平均。")
        log(f"  → どう配分しても最良戦略の {best[1][0]:.1f}% を超えることはできない。")
        log("  → 配分で変えられるのは『変動の大きさ』と『どれだけ最良に寄せるか』だけ。")
        log("  → 100%超の戦略が存在しない限り、配分最適化では利益は出ない。")


if __name__ == "__main__":
    main()
