# -*- coding: utf-8 -*-
"""3モデルの合議で単勝を狙う（2026-08-13）

発想（利用者の指摘）
  我々は3つの独立したモデルを持っている。
    勝率モデル(p_win) / 連対モデル(p_top2) / 複勝モデル(p_top3)
  これまでは複勝モデルの順位(mr)だけを使い、他は表示に使うだけだった。
  3つが揃って推す馬は、1つだけが推す馬より信頼できるはず。

  さらに逆の使い方もある。「勝率は1位だが複勝は低い」＝勝ち切るか飛ぶかの
  極端な馬。単勝ならむしろそちらが向く可能性がある。

やり方
  レース内で3モデルそれぞれの順位を作り、合議の度合いで層別する。
    完全合議 : 3モデルすべてで1位
    多数決   : 3モデル中2つで1位
    分裂     : 勝率1位だが複勝順位が離れている
  単勝の累計回収率・的中数・年別で評価する。

⚠ 複勝は1点の標準偏差78%で測りやすいが、単勝は579%ある。
  同じ点数でも単勝の結論は7倍振れる。的中数を必ず併記すること。

検体: bet_cache_2021〜2025（207,518頭・14,972レース）
実行: python consensus.py → consensus_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
rng = np.random.default_rng(20260813)


def log(m):
    print(m, flush=True)


def main():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    D["tan"] = D.win * D.odds * 100

    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False)      # 勝率モデルの順位
    D["r2"] = g["c_top2"].rank(ascending=False)     # 連対モデルの順位
    D["r3"] = g["c_top3"].rank(ascending=False)     # 複勝モデルの順位
    D["一致数"] = ((D.r1 == 1).astype(int) + (D.r2 == 1).astype(int)
                 + (D.r3 == 1).astype(int))
    D["ばらつき"] = D[["r1", "r2", "r3"]].max(axis=1) - D[["r1", "r2", "r3"]].min(axis=1)
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")

    def ev(s, name, angle):
        if len(s) < 150:
            return None
        v = s.tan.values
        yr = [s[s.年 == y].tan.mean() for y in YEARS]
        if any(np.isnan(x) for x in yr):
            return None
        b = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
        return {"角度": angle, "構成": name, "点数": len(s),
                "的中": int(s.win.sum()), "的中率": round(s.win.mean() * 100, 1),
                "累計ROI": round(v.mean(), 1), "CI下": round(np.percentile(b, 2.5), 1),
                "CI上": round(np.percentile(b, 97.5), 1),
                "最悪年": round(min(yr), 1),
                "100超年": sum(1 for x in yr if x >= 100)}

    rows = []
    log("① 合議の度合い（単勝）")
    for k in (3, 2, 1, 0):
        r = ev(D[D.一致数 == k], f"3モデル中{k}つが1位", "1.合議")
        if r:
            rows.append(r)
            log(f"  {r['構成']:<22}{r['点数']:>7,}点 的中{r['的中']:>5} "
                f"({r['的中率']:>4.1f}%) ROI{r['累計ROI']:>6.1f}% [{r['CI下']},{r['CI上']}]")

    log("\n② 完全合議 × オッズ帯")
    C = D[D.一致数 == 3]
    for lo, hi in ((1, 3), (3, 5), (5, 10), (10, 20), (20, 50), (1, 10), (5, 20)):
        r = ev(C[(C.odds >= lo) & (C.odds < hi)], f"完全合議 × {lo}-{hi}倍", "2.合議×オッズ")
        if r:
            rows.append(r)
            log(f"  {r['構成']:<22}{r['点数']:>7,}点 的中{r['的中']:>5} "
                f"({r['的中率']:>4.1f}%) ROI{r['累計ROI']:>6.1f}% [{r['CI下']},{r['CI上']}]")

    log("\n③ 分裂（勝率1位だが複勝順位が離れている）＝勝ち切るか飛ぶか")
    for lo in (2, 3, 5):
        s = D[(D.r1 == 1) & (D.r3 >= lo)]
        r = ev(s, f"勝率1位 かつ 複勝{lo}位以下", "3.分裂")
        if r:
            rows.append(r)
            log(f"  {r['構成']:<22}{r['点数']:>7,}点 的中{r['的中']:>5} "
                f"({r['的中率']:>4.1f}%) ROI{r['累計ROI']:>6.1f}% [{r['CI下']},{r['CI上']}]")
    for th in (0, 1, 3, 5):
        s = D[(D.r1 == 1) & (D.ばらつき >= th)]
        r = ev(s, f"勝率1位 かつ 3モデルの開き{th}以上", "3.分裂")
        if r:
            rows.append(r)

    log("\n④ 合議 × レース条件")
    CO = {"長距離2100+": D["距離"] >= 2100, "芝": D.is_turf == 1, "ダート": D.is_turf == 0,
          "重賞級": D["クラス_num"] >= 5, "道悪": D["馬場状態_num"] >= 3,
          "多頭数16+": D["出走頭数"] >= 16, "少頭数-12": D["出走頭数"] <= 12}
    for cl, m in CO.items():
        for k, kl in ((3, "完全合議"), (2, "2モデル一致")):
            for lo, hi in ((1, 20), (5, 20), (10, 30), (1, 99)):
                s = D[m & (D.一致数 == k) & (D.odds >= lo) & (D.odds < hi)]
                r = ev(s, f"{cl} × {kl} × {lo}-{hi}倍", "4.合議×条件")
                if r:
                    rows.append(r)

    R = pd.DataFrame(rows).sort_values("累計ROI", ascending=False)
    R.to_csv("consensus_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n\n構成 {len(R)}件  累計100%超 {int((R.累計ROI>=100).sum())}件")
    log(f"うち的中50本以上 {int(((R.累計ROI>=100)&(R.的中>=50)).sum())}件\n")
    log("=== 累計ROI 上位15 ===")
    log(R.head(15).to_string(index=False))
    log("\n=== 的中50本以上に限った上位10 ===")
    log(R[R.的中 >= 50].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
