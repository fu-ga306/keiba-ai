# -*- coding: utf-8 -*-
"""非1番人気専用モデルを本気でチューニングする（血統・残差・市場情報を全部使う）。

前提: 非1番人気に絞った専用モデルが、単勝ROI 91.4%(前92.2/後90.7)と
      今日で唯一の再現性ある成果になった。ここを伸ばす。

投入する情報:
  ・能力特徴(既存123)
  ・市場情報(人気/オッズ/市場想定確率/オッズ比 など)
  ・血統(blood_expanding.csv ＝ レース日時点までの累積で作り直したリークなし版)
  ・残差(馬・騎手・調教師・馬主の「人気より走る」傾向・過去分のみ)
試す軸:
  A. 特徴セット（能力のみ / +市場 / +血統 / +残差 / 全部）
  B. 目的変数（勝つか / 1番人気に先着するか / 3着以内か）
  C. ハイパーパラメータ（深さ・学習率・正則化）
判定は 2025を前後半に割った単勝/複勝ROIの安定性。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from market_free_model import FEATURE_COLS_MF


def log(m):
    print(m, flush=True)


def load():
    import features as F
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着"])
    for c in ["単勝オッズ", "人気", "馬番", "出走頭数"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["単勝オッズ", "人気", "馬番"])
    # リークなし血統
    bl = pd.read_csv("blood_expanding.csv", dtype={"race_id": str})
    d = d.drop(columns=[c for c in d.columns if c.startswith(("父系_", "母父系_"))],
               errors="ignore")
    d = d.merge(bl, on=["race_id", "馬名"], how="left")
    # 残差（騎手・調教師・馬主・馬）
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "騎手", "調教師", "馬主"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    d = F.sort_by_horse_time(F.attach_race_date(d))
    d = d.sort_values(["_race_dt", "race_id"]).reset_index(drop=True)
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["年"] = d["race_id"].str[:4].astype(int)
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["fav_odds"] = d.groupby("race_id")["単勝オッズ"].transform("min")
    d["odds比"] = d["単勝オッズ"] / d["fav_odds"]
    d["q_top"] = d.groupby("race_id")["q"].transform("max")
    d["人気率"] = d["人気"] / d["出走頭数"]
    d["log_odds"] = np.log(d["単勝オッズ"])
    fav = d[d["人気"] == 1].groupby("race_id")["着"].min()
    d["fav着"] = d["race_id"].map(fav)
    d["先着"] = (d["着"] < d["fav着"]).astype(float)
    d["res_rank"] = (d["人気"] - d["着"]) / d["出走頭数"].replace(0, np.nan)
    for key, pre, mn in [("馬名", "h", 2), ("騎手", "j", 30),
                         ("調教師", "t", 30), ("馬主", "o", 30)]:
        g = d.groupby(key, sort=False)
        s = g["res_rank"].apply(lambda x: x.shift(1).expanding().mean())
        d[f"{pre}_res"] = s.reset_index(level=0, drop=True)
        d.loc[g.cumcount() < mn, f"{pre}_res"] = np.nan
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    return d, tan, fuk


ABILITY = None
MARKET = ["人気", "単勝オッズ", "log_odds", "q", "odds比", "q_top", "人気率"]
BLOOD = ["父系_複勝率", "父系_勝率", "父系_今回距離適性", "父系_芝ダ適性",
         "父系_長距離勝率", "母父系_複勝率", "母父系_今回距離適性", "母父系_芝ダ適性"]
RESID = ["h_res", "j_res", "t_res", "o_res"]


def run(tr, te, feats, target, params, tan, fuk, label, mid, frac=0.05):
    feats = [f for f in dict.fromkeys(feats) if f in tr.columns]
    m = lgb.LGBMClassifier(**params).fit(tr[feats], tr[target])
    p = m.predict_proba(te[feats])[:, 1]
    t = te.copy()
    t["p"] = p
    th = t["p"].quantile(1 - frac)
    s = t[t["p"] >= th]
    if len(s) < 100:
        return None

    def roi(x, tbl):
        return sum(tbl.get((r.race_id, r.bn), 0) for r in x.itertuples()) / len(x)

    a = roi(s, tan)
    h1 = roi(s[s["dt"] <= mid], tan)
    h2 = roi(s[s["dt"] > mid], tan)
    f = roi(s, fuk)
    mk = "◎" if min(h1, h2) >= 100 else ("○" if min(h1, h2) >= 90 else "")
    log(f"  {label:<40}{len(s):5,} 勝率{s['win'].mean()*100:5.1f}% "
        f"単勝{a:6.1f}%(前{h1:5.1f}/後{h2:5.1f}) 複勝{f:6.1f}% "
        f"AUC{roc_auc_score(te[target], p):.4f}{mk}")
    return a, min(h1, h2), f


def main():
    global ABILITY
    d, tan, fuk = load()
    ABILITY = [c for c in FEATURE_COLS_MF if c in d.columns]
    tr = d[(d["年"] <= 2024) & (d["人気"] >= 2)]
    te = d[(d["年"] == 2025) & (d["人気"] >= 2)].copy()
    mid = te["dt"].quantile(0.5)
    log(f"学習 {len(tr):,} / 検証 {len(te):,}（非1番人気のみ）")
    log(f"能力{len(ABILITY)} 市場{len(MARKET)} 血統{len(BLOOD)} 残差{len(RESID)}\n")
    P = dict(objective="binary", learning_rate=0.03, num_leaves=31, n_estimators=600,
             min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=42)

    log("=" * 116)
    log("【A】特徴セット比較（目的=勝つか・上位5%を購入）")
    log("=" * 116)
    sets = [(ABILITY, "能力のみ"), (ABILITY + MARKET, "能力+市場"),
            (ABILITY + MARKET + BLOOD, "能力+市場+血統"),
            (ABILITY + MARKET + RESID, "能力+市場+残差"),
            (ABILITY + MARKET + BLOOD + RESID, "全部(能力+市場+血統+残差)")]
    for f, nm in sets:
        run(tr, te, f, "win", P, tan, fuk, nm, mid)

    log("\n" + "=" * 116)
    log("【B】目的変数の比較（特徴=全部）")
    log("=" * 116)
    full = ABILITY + MARKET + BLOOD + RESID
    for tgt, nm in [("win", "勝つか"), ("先着", "1番人気に先着するか"), ("fuku", "3着以内か")]:
        run(tr, te, full, tgt, P, tan, fuk, f"目的={nm}", mid)

    log("\n" + "=" * 116)
    log("【C】ハイパーパラメータ（特徴=全部・目的=勝つか）")
    log("=" * 116)
    for nm, over in [("既定(leaves31,lr.03)", {}),
                     ("浅い(leaves15)", dict(num_leaves=15)),
                     ("深い(leaves63)", dict(num_leaves=63)),
                     ("低学習率(lr.01,n1500)", dict(learning_rate=0.01, n_estimators=1500)),
                     ("正則化強(l1=.5,l2=.5)", dict(reg_alpha=0.5, reg_lambda=0.5)),
                     ("min_child100", dict(min_child_samples=100))]:
        run(tr, te, full, "win", dict(P, **over), tan, fuk, nm, mid)

    log("\n" + "=" * 116)
    log("【D】購入割合（最良構成）")
    log("=" * 116)
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        run(tr, te, full, "win", P, tan, fuk, f"上位{frac*100:.0f}%を購入",
            mid, frac=frac)


if __name__ == "__main__":
    main()
