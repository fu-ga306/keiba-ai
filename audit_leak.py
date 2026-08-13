# -*- coding: utf-8 -*-
"""累積系の特徴量に未来が混ざっていないかを直接検算する（2026-08-14）

背景
  騎手・調教師・馬主の累積勝率が race_id 順で集計されていた。race_id は
  「場コード順 → 各場の中は日付順」なので、場が切り替わるたびに日付が巻き戻る。
  修正して再構築したところ、騎手勝率は98.17%の行で値が変わり、
  複勝方式の的中率は44.8%→34.8%、ROIは105.2%→84.7%に落ちた。

  「直った」と主張するには、値を作り直しただけでは足りない。
  各行の値が本当に「その行より前の実績だけ」で説明できるかを検算する。

やり方
  騎手勝率の場合、行 i の値は「その騎手の、i より前の日付のレースの勝率」に
  一致するはず。実開催日でソートし直して自分で累積を作り、突き合わせる。
  一致しなければ、未来が混ざっているか定義が違う。

実行: python audit_leak.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import features as F


def log(m):
    print(m, flush=True)


def main():
    log("読み込み中...")
    raw = pd.read_csv("race_data_clean.csv", low_memory=False,
                      usecols=["race_id", "馬名", "騎手", "着順"], dtype={"race_id": str})
    raw["race_id"] = raw["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw["着"] = pd.to_numeric(raw["着順"], errors="coerce")
    raw = F.attach_race_date(raw)
    log(f"  元データ {len(raw):,}行  日付を引けた率 {raw._race_dt.notna().mean()*100:.1f}%")

    feat = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                       usecols=["race_id", "馬名", "騎手勝率", "過去勝率"])
    feat["race_id"] = feat["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    d = raw.merge(feat, on=["race_id", "馬名"], how="inner")
    d = d.dropna(subset=["_race_dt", "騎手"])
    log(f"  照合できた {len(d):,}行")

    # 自分で「その行より前の日付だけ」で騎手勝率を作り直す
    d = d.sort_values(["騎手", "_race_dt", "race_id"]).reset_index(drop=True)
    g = d.groupby("騎手", sort=False)
    d["_win"] = (d["着"] == 1).astype(float)
    d.loc[d["着"].isna(), "_win"] = np.nan
    # shift(1) してから累積 → その行を含まない過去だけの平均
    s = g["_win"].shift(1)
    d["_expect"] = s.groupby(d["騎手"], sort=False).expanding().mean() \
        .reset_index(level=0, drop=True)

    m = d.dropna(subset=["騎手勝率", "_expect"])
    diff = (m["騎手勝率"] - m["_expect"]).abs()
    log(f"\n=== 騎手勝率の検算（{len(m):,}行）===")
    log(f"  完全一致(差<1e-9)   {(diff < 1e-9).mean()*100:6.2f}%")
    log(f"  ほぼ一致(差<1e-4)   {(diff < 1e-4).mean()*100:6.2f}%")
    log(f"  平均差 {diff.mean():.8f}   最大差 {diff.max():.6f}")

    # 未来混入の直接検定: 特徴量が「その行の着順」と関係してはならない。
    # 同じ騎手の中で、当該行の勝敗と特徴量の相関を見る（本来ほぼ0のはず）。
    log(f"\n=== 未来混入の検定（特徴量 vs その行の勝敗）===")
    log("  ※ 累積特徴は当該レースの結果を含んではいけない。含むと相関が出る。")
    for col in ("騎手勝率", "過去勝率"):
        t = d.dropna(subset=[col, "着"])
        if len(t) < 1000:
            continue
        # 騎手の平均を引いて「その騎手の中での上下」にしてから相関を見る
        resid = t[col] - t.groupby("騎手")[col].transform("mean")
        r = np.corrcoef(resid, (t["着"] == 1).astype(float))[0, 1]
        log(f"  {col:<10} 騎手内での相関 {r:+.5f}"
            + ("  ← 要調査" if abs(r) > 0.02 else "  OK"))

    # 巻き戻りの確認
    u = d.drop_duplicates("race_id")[["race_id", "_race_dt"]].sort_values("race_id")
    back = (u["_race_dt"].diff() < pd.Timedelta(0)).sum()
    log(f"\n=== 参考: race_id順に並べたときの日付の巻き戻り ===")
    log(f"  {back:,}回 / {len(u):,}レース（修正前はこの順で集計していた）")


if __name__ == "__main__":
    main()
