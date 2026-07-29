# -*- coding: utf-8 -*-
"""血統特徴をリークなしで作り直す（レース日時点までの累積）。

問題: 従来の sire_stats は「max_year以前」という年単位のカットオフで集計していた。
      そのため学習データ(2019-2024)の各行にとって、同じ年の自分自身の結果が
      集計に含まれる自己リークが起きていた。学習中は最重要特徴(gain16.9%)に
      見えるが、検証年では性能を毀損する（2026-07-28に確認、除去して +5.7〜9.1pt）。

対策: 各レース日の時点までの累積成績で父系・母父系の指標を作る。
      その馬自身の当該レースは当然含まれない（shift相当）。
出力: blood_expanding.csv (race_id, 馬名, 父系_*, 母父系_*)
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def log(m):
    print(m, flush=True)


def main():
    import features as F
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "horse_id", "着順", "距離",
                              "馬場種別"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rc["horse_id"] = rc["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    hm = pd.read_csv("horse_master.csv")
    hm["horse_id"] = hm["horse_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    d = rc.merge(hm[["horse_id", "父馬", "母父馬"]], on="horse_id", how="left")
    d["着"] = pd.to_numeric(d["着順"], errors="coerce")
    d["距離"] = pd.to_numeric(d["距離"], errors="coerce")
    d = d.dropna(subset=["着"])
    d = F.attach_race_date(d)
    d = d.sort_values(["_race_dt", "race_id"]).reset_index(drop=True)
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["is_turf"] = d["馬場種別"].astype(str).str.contains("芝").astype(float)
    d["長距離"] = (d["距離"] >= 2000).astype(float)
    log(f"{len(d):,}行 / 父馬{d['父馬'].nunique():,}頭 / 母父馬{d['母父馬'].nunique():,}頭")

    out = pd.DataFrame({"race_id": d["race_id"], "馬名": d["馬名"]})
    for sire_col, pre in [("父馬", "父系"), ("母父馬", "母父系")]:
        g = d.groupby(sire_col, sort=False)
        # レース日時点までの累積（自分の行を含まないよう shift(1)）
        for src, name in [("fuku", "複勝率"), ("win", "勝率")]:
            s = g[src].apply(lambda x: x.shift(1).expanding().mean())
            out[f"{pre}_{name}"] = s.reset_index(level=0, drop=True)
        cnt = g.cumcount()
        out[f"{pre}_出走数"] = cnt
        # 距離適性: 今回と同じ距離帯での累積複勝率
        d["_band"] = (d["距離"] / 400).round()
        g2 = d.groupby([sire_col, "_band"], sort=False)
        s = g2["fuku"].apply(lambda x: x.shift(1).expanding().mean())
        out[f"{pre}_今回距離適性"] = s.reset_index(level=[0, 1], drop=True)
        # 芝ダ適性
        g3 = d.groupby([sire_col, "is_turf"], sort=False)
        s = g3["fuku"].apply(lambda x: x.shift(1).expanding().mean())
        out[f"{pre}_芝ダ適性"] = s.reset_index(level=[0, 1], drop=True)
        # 長距離適性
        g4 = d.groupby([sire_col, "長距離"], sort=False)
        s = g4["win"].apply(lambda x: x.shift(1).expanding().mean())
        out[f"{pre}_長距離勝率"] = s.reset_index(level=[0, 1], drop=True)
        # サンプルが少ないうちは信用しない
        for c in [f"{pre}_複勝率", f"{pre}_勝率", f"{pre}_今回距離適性",
                  f"{pre}_芝ダ適性", f"{pre}_長距離勝率"]:
            out.loc[cnt < 30, c] = np.nan

    out.to_csv("blood_expanding.csv", index=False, encoding="utf-8-sig")
    log(f"保存 → blood_expanding.csv ({len(out):,}行 × {len(out.columns)}列)")
    cols = [c for c in out.columns if c not in ("race_id", "馬名")]
    log("\n各列の非欠損率:")
    for c in cols:
        log(f"  {c:<24}{out[c].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
