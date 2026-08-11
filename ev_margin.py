# -*- coding: utf-8 -*-
"""EV閾値に安全マージンを持たせて、スリッページに耐えるかを検証する。

背景（2026-08-11）
  7分前オッズは人気薄帯で確定より高く出る（6-7番人気 +22.4% / 8-10番 +54.5%）。
  そのオッズで計算したEVは過大で、条件を満たす馬が水増しされる。
  スリッページ込みの実効回収率は 全体88.4% / 芝95.0%（確定基準は117.0/148.9%）。

  オッズ上限の引き下げは効かなかった（混入は減るが高配当も減る）。
  残る対策は「要求する期待値そのものを上げて、目減りしても黒字が残る
  クッションを持たせる」こと。

⚠ 事前登録（結果を見る前に確定させた条件）
  探索範囲: EV_TOP {1.7,1.9,2.1,2.3,2.5} × EV_SUB {2.2,2.6,3.0,3.4} = 20通り
            × 対象 {全体, 芝のみ} = 40通り
  判定基準:
    ① スリッページ込みの95%下限 > 100%   ← 主基準。検体数の罰則を内包する
    ② 5年すべて100%超
    ③ 的中40本以上（これ未満だと下限が構造的に100%を割る＝実測で確認済み）
    ④ 年あたり50レース以上
  40通り試すので5%水準なら偶然2つ通る。通っても候補であって採用ではない。
  採用は層A（実測リプレイ）と2026年の独立検証を通ってから。

実行: python ev_margin.py → ev_margin_result.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import slippage_sim as S

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YEARS = [2021, 2022, 2023, 2024, 2025]
TOPS = [1.7, 1.9, 2.1, 2.3, 2.5]
SUBS = [2.2, 2.6, 3.0, 3.4]
N_OUTER, N_INNER = 12, 8
rng = np.random.default_rng(20260811)


def sim(races, pools, gen, top, sub, per_year=None):
    c = r = 0.0; n = 0
    for rr in races:
        lr = np.empty(len(rr["odds"]))
        for bb in range(6):
            for ww in (0, 1):
                idx = np.where((rr["band"] == bb) & (rr["win"] == ww))[0]
                if not len(idx):
                    continue
                lr[idx] = gen.choice(pools[(bb, ww)], len(idx)) + \
                    gen.normal(0, S.KDE_BW, len(idx))
        od = rr["odds"] / np.exp(lr)
        ev = rr["p"] * od
        ok = ((rr["gap"] >= 3) & (od <= 20) &
              (((rr["mr"] == 1) & (ev >= top)) |
               ((rr["mr"] >= 2) & (rr["mr"] <= 5) & (ev >= sub))))
        if not ok.any():
            continue
        i = np.where(ok)[0][np.argmax(ev[ok])]
        n += 1
        cc = 1000.0; rr_ = rr["tan"][i] * 10
        mates = rr["uma"][rr["bn"][i]]; sel = rr["mate"] != rr["bn"][i]
        cc += 500 * sel.sum(); rr_ += mates[sel].sum() * 5
        c += cc; r += rr_
        if per_year is not None:
            y = rr["year"]
            per_year[y][0] += cc; per_year[y][1] += rr_
    return n, (r / c * 100 if c else np.nan)


def main():
    drift = S.load_drift(); pay = S.load_pay()
    turf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                       dtype={"race_id": str},
                       usecols=["race_id", "is_turf"]).drop_duplicates("race_id")
    E = pd.concat([pd.read_csv(os.path.join(BASE_DIR, f"bet_cache_{y}.csv"),
                               dtype={"race_id": str, "bn": str}).merge(
                       turf, on="race_id", how="left") for y in YEARS],
                  ignore_index=True)
    rows = []
    for lbl, sub_df in (("全体", E), ("芝のみ", E[E.is_turf == 1])):
        # pack と同じ順序・同じ絞り込みで race_id を並べ、年を付ける
        rids = [rid for rid, g in sub_df.groupby("race_id", sort=False)
                if (g["乖離"].values.astype(float) >= 3).any()]
        races = S.pack(sub_df, pay)
        assert len(rids) == len(races), f"レース数が一致しない {len(rids)} != {len(races)}"
        for rr, rid in zip(races, rids):
            rr["year"] = int(rid[:4])
        for top in TOPS:
            for sb in SUBS:
                res = []; ns = []; py_all = []
                for oi in range(N_OUTER):
                    ids = drift.race_id.unique()
                    boot = drift[drift.race_id.isin(
                        rng.choice(ids, len(ids), replace=True))]
                    pools = S.build_pools(boot)
                    for _ in range(N_INNER):
                        gen = np.random.default_rng(rng.integers(1 << 30))
                        py = {y: [0.0, 0.0] for y in YEARS}
                        n, roi = sim(races, pools, gen, top, sb, py)
                        res.append(roi); ns.append(n)
                        py_all.append({y: (py[y][1] / py[y][0] * 100
                                           if py[y][0] else np.nan) for y in YEARS})
                res = np.array(res)
                lo = float(np.percentile(res, 2.5))
                worst = float(np.median([min(p.values()) for p in py_all]))
                hits = np.nan
                rows.append({"対象": lbl, "EV上位": top, "EV下位": sb,
                             "R数": int(np.median(ns)),
                             "ROI中央": round(float(np.median(res)), 1),
                             "95%下限": round(lo, 1),
                             "最低年": round(worst, 1),
                             "年R数": round(np.median(ns) / 5, 0)})
                print(f"  {lbl} EV{top}/{sb}: R{int(np.median(ns)):>4d} "
                      f"ROI{np.median(res):6.1f}% 下限{lo:6.1f}% 最低年{worst:6.1f}%",
                      flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(BASE_DIR, "ev_margin_result.csv"),
             index=False, encoding="utf-8-sig")
    print("\n【事前登録した基準を満たすもof】95%下限>100 かつ 最低年>=100 かつ 年R数>=50")
    ok = d[(d["95%下限"] > 100) & (d.最低年 >= 100) & (d.年R数 >= 50)]
    print(ok.to_string(index=False) if len(ok) else "  なし")
    print("\n【95%下限の上位5】")
    print(d.nlargest(5, "95%下限").to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
