# -*- coding: utf-8 -*-
"""予想（推奨1〜5番手）と実際の着順がどれだけズレるかを可視化する。

ユーザー要望: 予想と実際の着順にどのくらい差があるのか（1〜5番手推奨でいい）。

推奨順位は主モデルのplace3（複勝確率）順＝本番の◎○▲△×に対応:
  1番手=◎ / 2番手=○ / 3番手=▲ / 4番手=△ / 5番手=×
各推奨順位について:
  ・実際の着順の分布（1着/2着/3着/4-5着/6着以下）
  ・平均着順と中央値、着順のばらつき
  ・馬券内(3着以内)率、連対率、勝率
  ・「推奨順位と実着順のズレ」の大きさ
  ・市場(人気順)との比較 ← 同じ土俵で見る
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def log(m):
    print(m, flush=True)


def load():
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位", "着順_num", "人気", "単勝オッズ"]]
    for c in ["予測順位", "着順_num", "人気", "単勝オッズ"]:
        p3[c] = pd.to_numeric(p3[c], errors="coerce")
    p3 = p3.dropna(subset=["予測順位", "着順_num", "人気"])
    p3["頭数"] = p3.groupby("race_id")["馬名"].transform("size")
    return p3


def dist_table(d, rank_col, label):
    log("\n" + "=" * 86)
    log(f"【{label}】推奨順位ごとの実際の着順（2025・{d['race_id'].nunique():,}レース）")
    log("=" * 86)
    log(f"  {'推奨':<6}{'n':>6}{'1着':>7}{'2着':>7}{'3着':>7}{'4-5着':>8}"
        f"{'6着以下':>9}{'平均着順':>9}{'中央値':>7}")
    for r in range(1, 6):
        s = d[d[rank_col] == r]
        if len(s) < 50:
            continue
        c = s["着順_num"]
        log(f"  {r}番手{'':<2}{len(s):6,}"
            f"{(c == 1).mean()*100:6.1f}%{(c == 2).mean()*100:6.1f}%"
            f"{(c == 3).mean()*100:6.1f}%{c.between(4, 5).mean()*100:7.1f}%"
            f"{(c >= 6).mean()*100:8.1f}%{c.mean():8.1f}{c.median():7.0f}")


def gap_table(d, rank_col, label):
    log(f"\n  ― {label}: 予想と実着順のズレ ―")
    log(f"  {'推奨':<6}{'ピタリ':>8}{'±1着以内':>10}{'±2着以内':>10}"
        f"{'平均ズレ':>9}{'馬券内率':>9}{'連対率':>8}{'勝率':>7}")
    for r in range(1, 6):
        s = d[d[rank_col] == r]
        if len(s) < 50:
            continue
        gap = (s["着順_num"] - r).abs()
        c = s["着順_num"]
        log(f"  {r}番手{'':<2}{(gap == 0).mean()*100:7.1f}%{(gap <= 1).mean()*100:9.1f}%"
            f"{(gap <= 2).mean()*100:9.1f}%{gap.mean():8.2f}"
            f"{(c <= 3).mean()*100:8.1f}%{(c <= 2).mean()*100:7.1f}%{(c == 1).mean()*100:6.1f}%")


def main():
    d = load()
    d["人気順"] = d["人気"]
    log(f"対象: 2025年 {d['race_id'].nunique():,}レース / {len(d):,}頭")
    log("推奨順位 = 主モデルの複勝確率順（1番手=◎ 2番手=○ 3番手=▲ 4番手=△ 5番手=×）")

    dist_table(d, "予測順位", "AIの推奨")
    gap_table(d, "予測順位", "AIの推奨")
    dist_table(d, "人気順", "市場の人気順（比較用）")
    gap_table(d, "人気順", "市場の人気順（比較用）")

    log("\n" + "=" * 86)
    log("【まとめ】推奨1〜5番手が実際にどこに来るか")
    log("=" * 86)
    log(f"  {'':<10}{'AI 馬券内率':>13}{'市場 馬券内率':>14}{'差':>8}")
    for r in range(1, 6):
        a = d[d["予測順位"] == r]["着順_num"]
        m = d[d["人気順"] == r]["着順_num"]
        if len(a) < 50 or len(m) < 50:
            continue
        ra, rm = (a <= 3).mean() * 100, (m <= 3).mean() * 100
        log(f"  {r}番手{'':<5}{ra:12.1f}%{rm:13.1f}%{ra-rm:+7.1f}")
    log("")
    a5 = d[d["予測順位"] <= 5].groupby("race_id")["着順_num"].apply(lambda s: (s <= 3).sum())
    m5 = d[d["人気順"] <= 5].groupby("race_id")["着順_num"].apply(lambda s: (s <= 3).sum())
    log(f"  推奨1-5番手に馬券内(3着以内)が何頭入るか:")
    log(f"    AI  : 平均{a5.mean():.2f}頭 / 3頭とも{(a5 == 3).mean()*100:.1f}% / "
        f"2頭以上{(a5 >= 2).mean()*100:.1f}% / 0頭{(a5 == 0).mean()*100:.1f}%")
    log(f"    市場: 平均{m5.mean():.2f}頭 / 3頭とも{(m5 == 3).mean()*100:.1f}% / "
        f"2頭以上{(m5 >= 2).mean()*100:.1f}% / 0頭{(m5 == 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
