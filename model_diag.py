# -*- coding: utf-8 -*-
"""モデルが市場に何を足しているかを、Benterと同じ物差しで測る（2026-08-17）

なぜこの物差しか
  買い方をいくら工夫しても100%に届かないことが分かった。単勝の上限は88%で、
  控除率20%に対しモデルの寄与は+8pt。足りないのはモデルの精度そのもの。

  Benterは「市場のオッズだけで説明したときの当たり具合」と「そこにモデルを
  足したときの当たり具合」の差（ΔR²）で測り、0.0178あれば黒字になるとした。
  我々の従来値は0.0010で、必要な精度の約1/18。

  ただしこの0.0010はリーク混入時代の値で、正しい入力で測り直していない。
  まずそこから。

測り方
  レースごとに「どの馬が1着か」を条件付きロジット（多項ロジット）で当てる。
    式A: log(市場確率) だけ
    式B: log(市場確率) + log(モデル確率)
  この2つの当たり具合の差が、モデルが市場に足した分。
  当たり具合は McFadden の擬似R²（対数尤度の改善率）で測る。

  ⚠ 1頭ずつ独立に「勝つ/勝たない」を当てる今の作り方と違い、
    条件付きロジットは「1レースにつき1着は1頭」という制約を使う。
    これはBenterの枠組みそのもので、比較の基準として正しい。

どこが弱いかも見る
  芝/ダート・距離・クラス・頭数で分けて、どの区分でモデルが効いていないかを出す。
  実測では単勝ROIが 芝79.6% / ダート88.2% と8.6pt違う。原因を特定する。

実行: python model_diag.py
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
EPS = 1e-9
BENTER = 0.0178          # Benterが黒字化に必要とした水準


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "is_turf", "距離", "クラス_num",
                              "馬場状態_num"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    D["win"] = pd.to_numeric(D["win"], errors="coerce")
    D = D[D.odds > 0].copy()
    # 市場確率: オッズの逆数をレース内で正規化（控除率を割り戻したのと同じ）
    inv = 1.0 / D.odds
    D["q"] = inv / D.groupby("race_id")["odds"].transform(lambda s: (1.0 / s).sum())
    # モデル確率: 較正済み勝率をレース内で正規化
    D["p"] = D.c_win / D.groupby("race_id")["c_win"].transform("sum")
    D["lq"] = np.log(D.q.clip(EPS))
    D["lp"] = np.log(D.p.clip(EPS))
    return D


def clogit(df, cols, n_iter=60):
    """条件付きロジット。レース内で softmax して1着を当てる。

    パラメータは少数（1〜2個）なのでニュートン法で十分収束する。
    """
    X = df[cols].values
    y = df["win"].values
    rid = df["_rc"].values           # レース番号（0始まりの連番）
    nr = rid.max() + 1
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = X @ beta
        m = np.zeros(nr)
        np.maximum.at(m, rid, eta)
        e = np.exp(eta - m[rid])
        s = np.zeros(nr)
        np.add.at(s, rid, e)
        p = e / s[rid]
        g = X.T @ (y - p)
        # ヘッセ行列（レース内の共分散）
        H = np.zeros((X.shape[1], X.shape[1]))
        for a in range(X.shape[1]):
            for b in range(X.shape[1]):
                wa = np.zeros(nr)
                np.add.at(wa, rid, p * X[:, a])
                wb = np.zeros(nr)
                np.add.at(wb, rid, p * X[:, b])
                cross = np.zeros(nr)
                np.add.at(cross, rid, p * X[:, a] * X[:, b])
                H[a, b] = -(cross - wa * wb).sum()
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    eta = X @ beta
    m = np.zeros(nr)
    np.maximum.at(m, rid, eta)
    e = np.exp(eta - m[rid])
    s = np.zeros(nr)
    np.add.at(s, rid, e)
    ll = float((y * np.log((e / s[rid]).clip(EPS))).sum())
    return beta, ll


def null_ll(df):
    """全馬が等確率のときの対数尤度（McFadden の分母）"""
    n = df.groupby("_rc").size()
    return float(-np.log(n).sum())


def measure(df, label, min_race=200):
    if df["_rc"].nunique() < min_race:
        return None
    d = df.copy()
    d["_rc"] = pd.factorize(d["_rc"])[0]
    l0 = null_ll(d)
    _, l_mkt = clogit(d, ["lq"])
    _, l_both = clogit(d, ["lq", "lp"])
    r2_mkt = 1 - l_mkt / l0
    r2_both = 1 - l_both / l0
    return {"区分": label, "レース": d["_rc"].nunique(), "頭数": len(d),
            "市場R2": r2_mkt, "市場+モデルR2": r2_both,
            "ΔR2": r2_both - r2_mkt, "Benter比": (r2_both - r2_mkt) / BENTER}


def main():
    D = load()
    D["_rc"] = pd.factorize(D["race_id"])[0]
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース（5年OOS・修正後キャッシュ）")
    log(f"Benterが黒字化に必要とした ΔR² = {BENTER}\n")

    rows = [measure(D, "全体")]
    log("=== 全体 ===")
    r = rows[0]
    log(f"  市場だけ        R² = {r['市場R2']:.4f}")
    log(f"  市場＋モデル    R² = {r['市場+モデルR2']:.4f}")
    log(f"  モデルの寄与    ΔR² = {r['ΔR2']:.4f}  （Benter基準の {r['Benter比']*100:.1f}%）")

    log("\n=== どこで効いていて、どこで効いていないか ===")
    log(f"  {'区分':<20}{'レース':>8}{'市場R2':>9}{'ΔR2':>9}{'Benter比':>10}")
    segs = [("芝", D.is_turf == 1), ("ダート", D.is_turf == 0),
            ("短距離(<1400)", D["距離"] < 1400),
            ("マイル(1400-1799)", (D["距離"] >= 1400) & (D["距離"] < 1800)),
            ("中距離(1800-2199)", (D["距離"] >= 1800) & (D["距離"] < 2200)),
            ("長距離(2200+)", D["距離"] >= 2200),
            ("新馬・未勝利(<=1)", D["クラス_num"] <= 1),
            ("1-2勝(2-3)", (D["クラス_num"] >= 2) & (D["クラス_num"] <= 3)),
            ("3勝-OP(4+)", D["クラス_num"] >= 4),
            ("少頭数(<=12)", D["頭数"] <= 12),
            ("多頭数(13+)", D["頭数"] >= 13),
            ("良馬場", D["馬場状態_num"] == 0),
            ("道悪", D["馬場状態_num"] >= 1)]
    out = []
    for lab, f in segs:
        m = measure(D[f], lab)
        if m:
            out.append(m)
            log(f"  {lab:<20}{m['レース']:>8,}{m['市場R2']:>9.4f}{m['ΔR2']:>9.4f}"
                f"{m['Benter比']*100:>9.1f}%")

    log("\n=== 年ごと（モデルは良くなっているか）===")
    log(f"  {'年':<8}{'レース':>8}{'市場R2':>9}{'ΔR2':>9}{'Benter比':>10}")
    for y in YEARS:
        m = measure(D[D.年 == y], str(y))
        if m:
            log(f"  {y:<8}{m['レース']:>8,}{m['市場R2']:>9.4f}{m['ΔR2']:>9.4f}"
                f"{m['Benter比']*100:>9.1f}%")

    pd.DataFrame(rows + out).to_csv("model_diag_result.csv", index=False,
                                    encoding="utf-8-sig")
    log("\n→ model_diag_result.csv")


if __name__ == "__main__":
    main()
