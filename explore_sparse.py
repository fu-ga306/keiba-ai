# -*- coding: utf-8 -*-
"""特徴量を絞ったモデルは市場と違う視点を持てるか。

ユーザー仮説: 129特徴は多すぎて穴が見つけられない。絞れば違う答えが出るのでは。
背景: 現行モデルは市場とほぼ同一の予測をする（◎の複勝率が1番人気と同じ64%、
      条件別に切っても差は±1pt以内）。特徴量が多いほど「あらゆる情報を平均的に
      使う」＝市場の集合知の再現に近づくのかもしれない。

検証:
  ・特徴量数を 3/5/8/12/20/40/129 と変えたモデルを作る
  ・選び方も複数試す（重要度上位 / 相関の低いものを選ぶ / 手作業の少数精鋭）
  ・各モデルについて
      - 市場との一致度（1位馬が1番人気と同じ割合・順位相関）← 独自性の指標
      - 予測性能（複勝AUC・軸の複勝率）
      - 単勝/複勝ROI（軸を買った場合）と、市場への情報増分(LL)
  ・「市場と違い、かつ当たる」領域があるかを探す
学習<=2024 / 検証2025。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from market_free_model import FEATURE_COLS_MF

LGBP = dict(objective="binary", learning_rate=0.05, num_leaves=31, n_estimators=500,
            min_child_samples=30, feature_fraction=0.9, bagging_fraction=0.8,
            bagging_freq=1, verbose=-1, seed=42)


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着"])
    for c in ["単勝オッズ", "人気", "馬番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["単勝オッズ", "人気", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["win"] = (d["着"] == 1).astype(float)
    d["年"] = d["race_id"].str[:4].astype(int)
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}
    return d, tan, fuk


def roi(s, table):
    if not len(s):
        return float("nan")
    return sum(table.get((r.race_id, r.bn), 0) for r in s.itertuples()) / len(s)


def evaluate(tr, te, cols, label, tan, fuk, base_ll=None):
    m = lgb.LGBMClassifier(**LGBP).fit(tr[cols], tr["fuku"])
    p = m.predict_proba(te[cols])[:, 1]
    t = te.copy()
    t["p"] = p
    t["rk"] = t.groupby("race_id")["p"].rank(ascending=False, method="min")
    top = t[t["rk"] == 1]
    agree = (top["人気"] == 1).mean() * 100
    rho = spearmanr(t["p"], -t["人気"]).statistic
    auc = roc_auc_score(t["fuku"], t["p"])
    # 市場への情報増分
    eps = 1e-6
    s = t.dropna(subset=["q"])
    X0 = np.log(np.clip(s[["q"]], eps, 1)).values
    X1 = np.hstack([X0, np.log(np.clip(s[["p"]], eps, 1)).values])
    y = s["fuku"].values
    ll0 = -log_loss(y, LogisticRegression(C=1e6).fit(X0, y).predict_proba(X0)[:, 1])
    ll1 = -log_loss(y, LogisticRegression(C=1e6).fit(X1, y).predict_proba(X1)[:, 1])
    log(f"  {label:<26}{len(cols):4d}特徴  AUC{auc:.4f}  軸複勝{top['fuku'].mean()*100:5.1f}%"
        f"  1人気一致{agree:5.1f}%  人気相関{rho:5.2f}"
        f"  単勝ROI{roi(top, tan):6.1f}%  複勝ROI{roi(top, fuk):6.1f}%"
        f"  LL増分{ll1-ll0:+.5f}")
    return dict(auc=auc, agree=agree, rho=rho, ll=ll1 - ll0,
                troi=roi(top, tan), froi=roi(top, fuk))


def main():
    d, tan, fuk = load()
    use = [c for c in FEATURE_COLS_MF if c in d.columns]
    tr = d[d["年"] <= 2024]
    te = d[d["年"] == 2025].copy()
    log(f"学習 {len(tr):,} / 検証 {len(te):,} / 全特徴 {len(use)}")
    log("  ※1人気一致率が低いほど『市場と違う視点』。それで当たるかが焦点\n")

    # 重要度で順位付け（<=2023で算出しリークを避ける）
    dev = d[d["年"] <= 2023]
    imp = None
    for sd in (42, 7):
        mm = lgb.LGBMClassifier(**dict(LGBP, seed=sd, n_estimators=300)).fit(
            dev[use], dev["fuku"])
        s = pd.Series(mm.booster_.feature_importance("gain"), index=use)
        imp = s if imp is None else imp + s
    imp = imp.sort_values(ascending=False)

    log("=" * 118)
    log("【A】重要度の上位N個だけを使う")
    log("=" * 118)
    for n in (3, 5, 8, 12, 20, 40, len(use)):
        evaluate(tr, te, list(imp.head(n).index), f"上位{n}特徴", tan, fuk)

    log("\n" + "=" * 118)
    log("【B】相関の低い特徴を選ぶ（情報の重複を避ける）")
    log("=" * 118)
    corr = tr[use].corr().abs()
    picked = [imp.index[0]]
    for c in imp.index[1:]:
        if len(picked) >= 12:
            break
        if corr.loc[c, picked].max() < 0.5:
            picked.append(c)
    for n in (5, 8, 12):
        evaluate(tr, te, picked[:n], f"低相関{n}特徴", tan, fuk)
    log(f"  選ばれた特徴: {picked[:12]}")

    log("\n" + "=" * 118)
    log("【C】手作業の少数精鋭（意味の違う軸を1つずつ）")
    log("=" * 118)
    sets = {
        "能力3": ["過去複勝率", "直近3走平均着順", "レース内_直近3走平均着順ランク"],
        "能力+騎手4": ["過去複勝率", "直近3走平均着順",
                    "レース内_直近3走平均着順ランク", "騎手複勝率"],
        "能力+適性6": ["過去複勝率", "直近3走平均着順", "レース内_直近3走平均着順ランク",
                    "騎手複勝率", "同距離過去勝率", "距離別過去平均着順"],
        "時計系4": ["過去平均上り", "上り偏差", "過去最速タイム秒", "平均タイム差"],
        "前走系4": ["前走着順", "前走上り", "前走着差_秒", "前走間隔"],
    }
    for nm, cols in sets.items():
        cc = [c for c in cols if c in d.columns]
        if len(cc) >= 3:
            evaluate(tr, te, cc, nm, tan, fuk)

    log("\n  判定: 1人気一致率が下がっても単勝/複勝ROIが上がらなければ、"
        "『市場と違う＝ただ精度が落ちただけ』")


if __name__ == "__main__":
    main()
