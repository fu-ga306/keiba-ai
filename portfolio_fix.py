# -*- coding: utf-8 -*-
"""採用候補を「検証時とまったく同じロジック」で合算する（2026-08-16・修正版）

前版(portfolio_roi.py)の誤り
  構成名の「1-20倍」「1-10倍」は**軸のオッズ条件**なのに、実装で丸ごと落としていた。
  そのため検証時とは別の買い方を測ってしまい、A構成が148.5%→96.5%に見えていた。
  keiba_predict.py の実装にも同じ欠落がある（要修正）。

  正しい軸の選び方（verify34.py と同じ）:
    ax = 順位<=av かつ オッズが[olo,ohi)の馬 → その中で順位が最小の1頭
    相手 = 順位<=mn の馬（軸を除く）※相手にオッズ条件は無い

やること
  ① 各構成を単独で、検証時と同じロジックで再計算（検証値と一致するか確かめる）
  ② 構成を組み合わせたときの実際の回収率（本番と同じ重複除去つき）
  ③ どの組み合わせが最良かを出す

実行: python portfolio_fix.py → portfolio_fix_result.csv
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT = [2024, 2025]
ARE_FAV_MR_MIN = 4
rng = np.random.default_rng(20260816)

# (記号, 名前, 券種, 基準, 軸順位, 相手順位, オッズ下限, オッズ上限, レース条件)
PLANS = {
    "A": ("荒れR勝率1位x2 馬単裏", "馬単裏", "r1", 1, 2, 1, 20, "are"),
    "B": ("クラス4+勝率1位x2 馬単裏", "馬単裏", "r1", 1, 2, 1, 10, "cls4"),
    "C": ("荒れR勝率1位x2 馬連", "馬連", "r1", 1, 2, 1, 20, "are"),
    "D": ("荒れR連対1位x4 馬単裏", "馬単裏", "r2", 1, 4, 1, 20, "are"),
    "E": ("荒れR勝率1位x3 馬単裏", "馬単裏", "r1", 1, 3, 1, 10, "are"),
    "F": ("荒れR連対1位x5 馬単裏", "馬単裏", "r2", 1, 5, 1, 20, "are"),
    "G": ("荒れR勝率1位x4 馬単表", "馬単表", "r1", 1, 4, 1, 20, "are"),
}


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "クラス_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    g = D.groupby("race_id")
    D["r1"] = g["c_win"].rank(ascending=False, method="first")
    D["r2"] = g["c_top2"].rank(ascending=False, method="first")
    D["r3"] = g["c_top3"].rank(ascending=False, method="first")
    fav = D[D.pr == 1][["race_id", "r3"]].rename(columns={"r3": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    UMA, UREN = {}, {}
    for r in jv[jv.券種 == "馬単"].itertuples():
        UMA[(r.race_id, r.組み合わせ)] = r.払戻金
    for r in jv[jv.券種 == "馬連"].itertuples():
        UREN[(r.race_id, r.組み合わせ)] = r.払戻金
    return D, UMA, UREN


def race_bets(g, rid, keys, UMA, UREN):
    """1レースぶんの買い目を {(券種,組み合わせ): 払戻} で返す。検証時と同じ選び方。"""
    fav = g.fav_mr.iloc[0]
    cls = g["クラス_num"].iloc[0]
    is_are = pd.notna(fav) and fav >= ARE_FAV_MR_MIN
    is_c4 = pd.notna(cls) and cls >= 4
    odds = g.odds.values
    bn = g.bn.values
    out = {}
    for k in keys:
        nm, kind, bas, axr, mtr, olo, ohi, cond = PLANS[k]
        if cond == "are" and not is_are:
            continue
        if cond == "cls4" and not is_c4:
            continue
        r = g[bas].values
        # 軸: 順位<=axr かつ オッズが範囲内。その中で順位最小の1頭
        m = (r <= axr) & (odds >= olo) & (odds < ohi)
        if not m.any():
            continue
        a = bn[m][np.argmin(r[m])]
        for b in bn[(r <= mtr) & (bn != a)]:
            if kind == "馬単裏":
                key = ("馬単", f"{b}-{a}")
                out[key] = UMA.get((rid, f"{b}-{a}"), 0.0)
            elif kind == "馬単表":
                key = ("馬単", f"{a}-{b}")
                out[key] = UMA.get((rid, f"{a}-{b}"), 0.0)
            else:
                c2 = f"{min(a,b)}-{max(a,b)}"
                out[("馬連", c2)] = UREN.get((rid, c2), 0.0)
    return out


def evaluate(races, keys, UMA, UREN):
    acc = {y: [0.0, 0.0, 0] for y in YEARS}
    nr = 0
    recent = []
    for rid, g in races.items():
        y = int(rid[:4])
        d = race_bets(g, rid, keys, UMA, UREN)
        if not d:
            continue
        nr += 1
        for p in d.values():
            acc[y][0] += 100
            acc[y][1] += p
            acc[y][2] += 1 if p > 0 else 0
            if y in RECENT:
                recent.append(p)
    tc = sum(acc[y][0] for y in YEARS)
    if tc == 0:
        return None
    tr = sum(acc[y][1] for y in YEARS)
    th = sum(acc[y][2] for y in YEARS)
    rc = sum(acc[y][0] for y in RECENT)
    rr = sum(acc[y][1] for y in RECENT)
    rh = sum(acc[y][2] for y in RECENT)
    yr = {y: (acc[y][1] / acc[y][0] * 100 if acc[y][0] else np.nan) for y in YEARS}
    v = np.array(recent)
    if len(v) > 10:
        b = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
        lo, hi = np.percentile(b, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return {"構成": "".join(keys), "点数": int(tc / 100), "的中": th,
            "5年ROI": round(tr / tc * 100, 1),
            "直近2年点数": int(rc / 100), "直近2年的中": rh,
            "直近2年ROI": round(rr / rc * 100, 1) if rc else np.nan,
            "CI下": round(lo, 1), "CI上": round(hi, 1),
            "買うR数": nr, "1R点数": round(tc / 100 / nr, 1),
            **{f"y{y}": round(yr[y], 1) for y in YEARS},
            "100超年": sum(1 for y in YEARS if yr[y] >= 100)}


def main():
    D, UMA, UREN = load()
    races = {r: g for r, g in D.groupby("race_id", sort=False)}
    log(f"検体 {len(D):,}頭 / {len(races):,}レース\n")

    log("=== ① 各構成を単独で（検証値と一致するか確認）===")
    log(f"{'':<3}{'名前':<24}{'点数':>7}{'的中':>6}{'5年':>8}{'直近2年':>9}"
        f"{'1R点数':>8}{'100超年':>8}")
    rows = []
    for k in PLANS:
        r = evaluate(races, [k], UMA, UREN)
        if r:
            rows.append(r)
            log(f"{k:<3}{PLANS[k][0]:<24}{r['点数']:>7,}{r['的中']:>6}"
                f"{r['5年ROI']:>7.1f}%{r['直近2年ROI']:>8.1f}%"
                f"{r['1R点数']:>7.1f}{r['100超年']:>7}")

    log("\n=== ② 組み合わせ（本番と同じ重複除去つき）===")
    log(f"{'構成':<10}{'点数':>7}{'的中':>6}{'5年':>8}{'直近2年':>9}{'95%区間':>18}"
        f"{'1R点数':>8}{'100超年':>8}")
    combos = []
    ks = list(PLANS)
    for n in range(1, len(ks) + 1):
        for c in itertools.combinations(ks, n):
            if n > 3 and n < len(ks):
                continue          # 4〜6個の組み合わせは数が多いので飛ばす
            combos.append(c)
    res = []
    for c in combos:
        r = evaluate(races, list(c), UMA, UREN)
        if r:
            res.append(r)
    R = pd.DataFrame(res).sort_values("直近2年ROI", ascending=False)
    R.to_csv("portfolio_fix_result.csv", index=False, encoding="utf-8-sig")
    for _, r in R.head(20).iterrows():
        log(f"{r['構成']:<10}{r['点数']:>7,}{r['的中']:>6}{r['5年ROI']:>7.1f}%"
            f"{r['直近2年ROI']:>8.1f}%  [{r['CI下']:>5.1f},{r['CI上']:>6.1f}]"
            f"{r['1R点数']:>7.1f}{r['100超年']:>7}")

    log("\n=== ③ 直近2年のCI下限が100%を超えるもの ===")
    ok = R[R["CI下"] >= 100]
    log(ok.to_string(index=False) if len(ok) else "  なし")
    log("\n=== ④ 5年通算も直近2年も100%超のもの ===")
    ok2 = R[(R["5年ROI"] >= 100) & (R["直近2年ROI"] >= 100)]
    log(ok2[["構成", "点数", "的中", "5年ROI", "直近2年ROI", "CI下", "1R点数",
             "100超年"]].to_string(index=False) if len(ok2) else "  なし")


if __name__ == "__main__":
    main()
