# -*- coding: utf-8 -*-
"""市場の「誤差」そのものを特徴量にする。

ユーザー着想: モデルが市場と同じ予想をできるなら、市場予想と実結果のズレを
              特徴量にして予想すればいいのでは。

考え方: 各レースについて「市場の期待（人気/オッズ）」と「実際の着順」の差＝残差を取る。
        その残差を馬・騎手・調教師・馬主ごとに過去分だけ累積すれば、
        「人気より走る／人気ほど走らない」という系統的な偏りを捉えられる。
        これは結果の予測ではなく“市場の間違い方”の予測であり、既存特徴とは別物。

リーク対策: 残差は必ず shift(1) の expanding（そのレースより前だけ）で集計する。
検証: 市場(オッズ)・主モデルに対する対数尤度の増分と、実際のROI。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

RNG = np.random.default_rng(42)


def log(m):
    print(m, flush=True)


def load():
    import features as F
    d = pd.read_csv("race_features.csv", dtype={"race_id": str},
                    usecols=["race_id", "馬名", "馬番", "着順_num", "単勝オッズ",
                             "人気", "出走頭数"])
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "騎手", "調教師", "馬主"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    for c in ["馬番", "着順_num", "単勝オッズ", "人気", "出走頭数"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["馬番", "着順_num", "単勝オッズ", "人気"])
    d = F.sort_by_horse_time(F.attach_race_date(d))
    d = d.sort_values(["_race_dt", "race_id"]).reset_index(drop=True)
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["年"] = d["race_id"].str[:4].astype(int)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    n = d["出走頭数"].replace(0, np.nan)
    # 残差の定義
    #  ①順位残差: 人気順位より上位に来たか（正=人気より走った）。頭数で正規化。
    d["res_rank"] = (d["人気"] - d["着順_num"]) / n
    #  ②確率残差: 実際に勝ったか − 市場の想定勝率
    d["res_prob"] = d["win"] - d["q"]
    #  ③複勝残差: 実際に3着内か − 市場の想定(qの3倍で近似)
    d["res_fuku"] = d["fuku"] - (3 * d["q"]).clip(upper=1)
    return d


def add_expanding(d, key, cols, prefix, min_n=3):
    """keyごとに、そのレースより前だけの平均を付ける（リークなし）。"""
    g = d.groupby(key, sort=False)
    for c in cols:
        s = g[c].apply(lambda x: x.shift(1).expanding().mean())
        d[f"{prefix}_{c}"] = s.reset_index(level=0, drop=True)
    cnt = g.cumcount()
    d[f"{prefix}_n"] = cnt
    for c in cols:
        d.loc[cnt < min_n, f"{prefix}_{c}"] = np.nan
    return d


def main():
    d = load()
    log(f"データ {len(d):,}行 / {d['race_id'].nunique():,}レース")

    log("残差の累積を計算中（馬・騎手・調教師・馬主）...")
    res_cols = ["res_rank", "res_prob", "res_fuku"]
    d = add_expanding(d, "馬名", res_cols, "h", min_n=2)
    d = add_expanding(d, "騎手", res_cols, "j", min_n=30)
    d = add_expanding(d, "調教師", res_cols, "t", min_n=30)
    d = add_expanding(d, "馬主", res_cols, "o", min_n=30)

    te = d[d["年"] == 2025].copy()
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測スコア"]].rename(columns={"予測スコア": "p3"})
    te = te.merge(p3, on=["race_id", "馬名"], how="left")
    te["p3"] = pd.to_numeric(te["p3"], errors="coerce")

    log("\n" + "=" * 86)
    log("【1】残差は次のレースでも続くか（＝系統的な偏りか、ただの偶然か）")
    log("=" * 86)
    log(f"  {'残差の主体':<16}{'指標':<12}{'n':>8}{'今回の残差との相関':>20}")
    for pre, nm in [("h", "馬"), ("j", "騎手"), ("t", "調教師"), ("o", "馬主")]:
        for c in res_cols:
            col = f"{pre}_{c}"
            s = te.dropna(subset=[col, c])
            if len(s) < 500:
                continue
            r = s[col].corr(s[c])
            mark = " ←継続性あり" if abs(r) > 0.03 else ""
            log(f"  {nm:<16}{c:<12}{len(s):8,}{r:19.4f}{mark}")

    log("\n" + "=" * 86)
    log("【2】市場・主モデルに情報を足せるか（2025・複勝を予測）")
    log("=" * 86)
    feats = [f"{p}_{c}" for p in ("h", "j", "t", "o") for c in res_cols]
    s = te.dropna(subset=["q", "p3"]).copy()
    for f in feats:
        s[f] = s[f].fillna(0.0)
    y = s["fuku"].values
    eps = 1e-6

    def fit(cols, label, base=None):
        X = np.nan_to_num(s[cols].values)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        ll = -log_loss(y, np.clip(p, eps, 1 - eps))
        auc = roc_auc_score(y, p)
        g = ""
        if base is not None:
            inc = ll - base
            g = f"  LL増分{inc:+.5f}" + (" ◎有効" if inc >= 0.001 else " ×効果なし")
        log(f"  {label:<32} LL={ll:.5f} AUC={auc:.4f}{g}")
        return ll

    b1 = fit(["q"], "市場のみ")
    fit(["q"] + [f"h_{c}" for c in res_cols], "市場 ＋ 馬の残差", b1)
    fit(["q"] + [f"j_{c}" for c in res_cols], "市場 ＋ 騎手の残差", b1)
    fit(["q"] + [f"t_{c}" for c in res_cols], "市場 ＋ 調教師の残差", b1)
    fit(["q"] + [f"o_{c}" for c in res_cols], "市場 ＋ 馬主の残差", b1)
    fit(["q"] + feats, "市場 ＋ 全残差", b1)
    b2 = fit(["q", "p3"], "市場＋主モデル")
    fit(["q", "p3"] + feats, "市場＋主モデル ＋ 全残差", b2)

    log("\n" + "=" * 86)
    log("【3】残差が大きい層を買うと儲かるか（2025・単勝/複勝ROI）")
    log("=" * 86)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}

    def roi(x, t):
        if not len(x):
            return float("nan")
        return sum(t.get((r.race_id, r.bn), 0) for r in x.itertuples()) / len(x)

    log(f"  {'条件':<34}{'n':>8}{'単勝ROI':>9}{'複勝ROI':>9}{'平均人気':>9}")
    log(f"  {'全体(基準)':<34}{len(te):8,}{roi(te, tan):8.1f}%{roi(te, fuk):8.1f}%"
        f"{te['人気'].mean():8.1f}")
    for pre, nm in [("h", "馬"), ("j", "騎手"), ("t", "調教師")]:
        col = f"{pre}_res_rank"
        x = te.dropna(subset=[col])
        if len(x) < 1000:
            continue
        hi = x[x[col] >= x[col].quantile(0.9)]
        lo = x[x[col] <= x[col].quantile(0.1)]
        log(f"  {nm + 'の残差 上位10%(人気より走る)':<34}{len(hi):8,}"
            f"{roi(hi, tan):8.1f}%{roi(hi, fuk):8.1f}%{hi['人気'].mean():8.1f}")
        log(f"  {nm + 'の残差 下位10%(人気ほど走らない)':<34}{len(lo):8,}"
            f"{roi(lo, tan):8.1f}%{roi(lo, fuk):8.1f}%{lo['人気'].mean():8.1f}")


if __name__ == "__main__":
    main()
