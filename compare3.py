# -*- coding: utf-8 -*-
"""現行ルール vs 単勝案 vs 複勝案 を同一条件で比較する（2026-08-13）

比較する3案
  A 現行（EV方式）  乖離≥3・20倍以下・MF複勝1位はEV≥1.7/2-5位はEV≥2.2
                    期待値最大の1頭を単勝1,000円＋馬単500円（複勝1-5位かつ人気3位内へ）
  B 単勝案          10-20倍 × 3モデル中2つ以上が1位  → 単勝
  C 複勝案          長距離2100m+ × MF複勝1位 × 20倍以下 × 4番人気以下 → 複勝

同じ検体（bet_cache 2021-2025・jv_payouts）で、同じ指標で並べる。
金額の重み付けが違うと比べられないので、**全案とも1点100円**に揃える。
現行だけは実運用の金額（単勝1,000/馬単500）版も併記する。

見る指標
  累計ROI / 的中数 / 95%区間 / 年別 / 年あたり点数
  そして「下限が100%を超えるのに必要な的中数」＝あと何年かかるか

実行: python compare3.py → compare3_result.csv
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
                     usecols=["race_id", "距離"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    FUKU = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "複勝"].itertuples()}
    UMA = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "馬単"].itertuples()}
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False)
    D["r2"] = g["c_top2"].rank(ascending=False)
    D["r3"] = g["c_top3"].rank(ascending=False)
    D["一致"] = (D.r1 == 1).astype(int) + (D.r2 == 1).astype(int) + (D.r3 == 1).astype(int)
    D["tan"] = D.win * D.odds * 100
    D["fuku"] = [FUKU.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")

    def summarize(cost, ret, hits, yrs, name, note=""):
        roi = ret.sum() / cost.sum() * 100
        per = ret / cost.mean()          # 1点あたり収益率（ブートストラップ用）
        b = np.array([rng.choice(per, len(per)).mean() * 100 for _ in range(3000)])
        lo, hi = np.percentile(b, [2.5, 97.5])
        ys = [yrs[y][1] / yrs[y][0] * 100 if yrs[y][0] else np.nan for y in YEARS]
        # 下限が100を超えるのに必要な倍率
        sd = per.std(ddof=1) * 100
        mu = per.mean() * 100
        need = int(np.ceil((1.96 * sd / (mu - 100)) ** 2)) if mu > 100 else None
        return {"案": name, "内容": note, "点数": len(per), "的中": int(hits),
                "的中率": round(hits / len(per) * 100, 2),
                "累計ROI": round(roi, 1), "CI下": round(lo, 1), "CI上": round(hi, 1),
                "年別": " ".join(f"{v:.0f}" for v in ys),
                "年100超": sum(1 for v in ys if not np.isnan(v) and v >= 100),
                "年間点数": round(len(per) / 5),
                "必要点数": f"{need:,}" if need else "—",
                "必要年数": round(need / (len(per) / 5)) if need else "∞"}

    rows = []

    # ── A 現行（EV方式）
    ev = D.c_win_n * D.odds
    hit = ((D.乖離 >= 3) & (D.odds <= 20) &
           (((D.r3 == 1) & (ev >= 1.7)) | (D.r3.between(2, 5) & (ev >= 2.2))))
    T = D[hit.fillna(False)].copy()
    T["_ev"] = ev[T.index]
    ax = T.sort_values("_ev", ascending=False).groupby("race_id").head(1)
    c = np.full(len(ax), 100.0); r = ax.tan.values
    yrs = {y: [0.0, 0.0] for y in YEARS}
    for y, v in zip(ax.年, ax.tan):
        yrs[y][0] += 100; yrs[y][1] += v
    rows.append(summarize(c, r, (ax.win == 1).sum(), yrs, "A 現行(単勝のみ)",
                          "乖離≥3・20倍以下・EV条件・最大EVの1頭"))
    # 現行の馬単込み（実運用の金額）
    cost = ret = 0.0
    yrs2 = {y: [0.0, 0.0] for y in YEARS}
    for rid, a in zip(ax.race_id, ax.itertuples()):
        y = int(rid[:4])
        cost += 1000; v = a.tan / 100 * 1000; ret += v
        yrs2[y][0] += 1000; yrs2[y][1] += v
        gg = D[(D.race_id == rid) & (D.r3 <= 5) & (D.pr <= 3) & (D.bn != a.bn)]
        for b in gg.bn:
            cost += 500; v2 = UMA.get((rid, f"{a.bn}-{b}"), 0.0) * 5
            ret += v2; yrs2[y][0] += 500; yrs2[y][1] += v2
    ys2 = [yrs2[y][1] / yrs2[y][0] * 100 if yrs2[y][0] else np.nan for y in YEARS]
    log(f"A 現行(単勝＋馬単・実運用の金額)  投資{int(cost):,}円  回収率 {ret/cost*100:.1f}%"
        f"  年別 {' '.join(f'{v:.0f}' for v in ys2)}")

    # ── B 単勝案
    S = D[(D.odds >= 10) & (D.odds < 20) & (D.一致 >= 2)]
    yrs = {y: [0.0, 0.0] for y in YEARS}
    for y, v in zip(S.年, S.tan):
        yrs[y][0] += 100; yrs[y][1] += v
    rows.append(summarize(np.full(len(S), 100.0), S.tan.values, S.win.sum(), yrs,
                          "B 単勝案", "10-20倍 × 3モデル中2つ以上が1位"))

    # ── C 複勝案
    F = D[(D["距離"] >= 2100) & (D.r3 == 1) & (D.odds <= 20) & (D.pr >= 4)]
    yrs = {y: [0.0, 0.0] for y in YEARS}
    for y, v in zip(F.年, F.fuku):
        yrs[y][0] += 100; yrs[y][1] += v
    rows.append(summarize(np.full(len(F), 100.0), F.fuku.values, (F.fuku > 0).sum(), yrs,
                          "C 複勝案", "長距離2100+ × MF複勝1位 × 20倍以下 × 4番人気以下"))

    R = pd.DataFrame(rows)
    R.to_csv("compare3_result.csv", index=False, encoding="utf-8-sig")
    log("")
    for _, x in R.iterrows():
        log(f"── {x['案']} ─────────────────────────────────")
        log(f"   {x['内容']}")
        log(f"   {x['点数']:,}点（年{x['年間点数']}点） 的中{x['的中']}本({x['的中率']}%)")
        log(f"   累計ROI {x['累計ROI']}%  95%区間[{x['CI下']}, {x['CI上']}]")
        log(f"   年別 {x['年別']}  （100%超 {x['年100超']}/5年）")
        log(f"   下限が100%を超えるのに 的中{x['必要点数']}点相当 → 約{x['必要年数']}年")
        log("")


if __name__ == "__main__":
    main()
