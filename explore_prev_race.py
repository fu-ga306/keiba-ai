# -*- coding: utf-8 -*-
"""前走の「人気×着順」の組み合わせ別に回収率を測る。

発見の発端: 次走の複勝率と次走人気に非対称があった（2025）。
  人気で凡走(3人気以内→8着以下) → 次走複勝27.2% / 次走平均5.5番人気
  人気薄で健闘(8人気以下→5着以内) → 次走複勝23.1% / 次走平均7.0番人気
市場が「前走人気だった馬」を引きずって買い、「前走人気薄で好走した馬」を
軽視しているなら、そこに歪みがある。

ただし複勝率の比較だけでは足りない（人気が違えば期待値も違う）。
回収率で直接測る。判定は:
  ・全馬購入の理論値80%(控除率20%)を明確に超えるか
  ・レース単位ブートストラップの95%信頼区間の下限が100%を超えるか
  ・2024と2025の両方で再現するか
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def log(m):
    print(m, flush=True)


def load():
    import features as F
    d = pd.read_csv("race_features.csv", dtype={"race_id": str},
                    usecols=["race_id", "馬名", "馬番", "着順_num", "単勝オッズ",
                             "人気", "クラス_num", "出走頭数"])
    for c in ["馬番", "着順_num", "単勝オッズ", "人気", "クラス_num", "出走頭数"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["馬番", "着順_num", "単勝オッズ", "人気"])
    d = F.sort_by_horse_time(F.attach_race_date(d))
    # 前走情報（実開催日順のshift＝リークなし）
    g = d.groupby("馬名")
    d["前走人気"] = g["人気"].shift(1)
    d["前走着順"] = g["着順_num"].shift(1)
    d["前走頭数"] = g["出走頭数"].shift(1)
    d["年"] = d["race_id"].str[:4].astype(int)
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}
    return d.dropna(subset=["前走人気", "前走着順"]), tan, fuk


def roi_ci(s, table, n_boot=3000):
    """レース単位ではなく馬単位（1レース1頭が基本なので実質同じ）でブートストラップ。"""
    if len(s) < 100:
        return None
    ret = np.array([table.get((r.race_id, r.bn), 0) for r in s.itertuples()], dtype=float)
    n = len(ret)
    idx = RNG.integers(0, n, size=(n_boot, n))
    rois = ret[idx].mean(axis=1)
    return (ret.mean(), np.percentile(rois, 2.5), np.percentile(rois, 97.5),
            (ret > 0).mean() * 100, n)


def cell_table(d, tan, fuk, year_label):
    log("\n" + "=" * 100)
    log(f"【前走 人気×着順 別の回収率】{year_label}  ※単勝100円あたりの払戻(=ROI%)")
    log("=" * 100)
    pops = [(1, 1, "前走1人気"), (2, 3, "前走2-3"), (4, 6, "前走4-6"),
            (7, 9, "前走7-9"), (10, 99, "前走10-")]
    chas = [(1, 1, "前走1着"), (2, 3, "前走2-3着"), (4, 6, "前走4-6着"),
            (7, 9, "前走7-9着"), (10, 99, "前走10着-")]
    log(f"  {'':<12}" + "".join(f"{c[2]:>13}" for c in chas))
    for plo, phi, pnm in pops:
        row = f"  {pnm:<12}"
        for clo, chi, cnm in chas:
            s = d[(d["前走人気"] >= plo) & (d["前走人気"] <= phi)
                  & (d["前走着順"] >= clo) & (d["前走着順"] <= chi)]
            r = roi_ci(s, tan)
            row += f"{r[0]:12.0f}%" if r else f"{'-':>13}"
        log(row)
    log("  ※理論値80%(控除率20%)。90%超なら市場の歪みの可能性")


def detail(d, tan, fuk, label):
    log("\n" + "=" * 100)
    log(f"【注目セルの詳細】{label}")
    log("=" * 100)
    log(f"  {'条件':<34}{'n':>7}{'単勝ROI':>9}{'95%CI':>20}{'複勝ROI':>9}{'今走人気':>9}")
    cands = [
        ("前走人気薄(7-)×好走(1-3着)",
         (d["前走人気"] >= 7) & (d["前走着順"] <= 3)),
        ("前走人気薄(10-)×好走(1-3着)",
         (d["前走人気"] >= 10) & (d["前走着順"] <= 3)),
        ("前走人気薄(7-)×1着",
         (d["前走人気"] >= 7) & (d["前走着順"] == 1)),
        ("前走人気(1-3)×大敗(10着-)",
         (d["前走人気"] <= 3) & (d["前走着順"] >= 10)),
        ("前走人気(1)×大敗(10着-)",
         (d["前走人気"] == 1) & (d["前走着順"] >= 10)),
        ("前走人気(1-3)×凡走(6-9着)",
         (d["前走人気"] <= 3) & (d["前走着順"].between(6, 9))),
        ("前走1人気×1着(王道)",
         (d["前走人気"] == 1) & (d["前走着順"] == 1)),
        ("全体(基準)", pd.Series(True, index=d.index)),
    ]
    for nm, mask in cands:
        s = d[mask]
        rt = roi_ci(s, tan)
        rf = roi_ci(s, fuk)
        if not rt:
            continue
        mark = " ◎" if rt[1] > 100 else (" △" if rt[2] > 100 else "")
        log(f"  {nm:<34}{rt[4]:7,}{rt[0]:8.1f}%  [{rt[1]:6.1f}〜{rt[2]:6.1f}]"
            f"{rf[0]:8.1f}%{s['人気'].mean():8.1f}{mark}")


def main():
    d, tan, fuk = load()
    log(f"前走情報のある走: {len(d):,}件 ({d['年'].min()}-{d['年'].max()})")
    for yr, lab in [(2025, "2025"), (2024, "2024")]:
        s = d[d["年"] == yr]
        cell_table(s, tan, fuk, lab)
        detail(s, tan, fuk, lab)


if __name__ == "__main__":
    main()
