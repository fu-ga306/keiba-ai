# -*- coding: utf-8 -*-
"""特徴量の階層化検証: 「重い材料」と「軽い材料」を分け、明示的な重みで合成する。

ユーザー指摘: 全ての判定材料をフラットに見過ぎ。重要なものに重みを持たせるべき。
現状: 129特徴をフラットに投入。重要度は上位2特徴で30%、残り127個で70%という
      ロングテール構造で、軽い特徴のノイズがコア判断を薄めている可能性がある。

設計:
  Tier1(重い): 重要度の累積が上位を占めるコア特徴（能力・調子・血統距離適性の骨格）
  Tier2(軽い): 中間層（補正要素）
  Tier3(死に): 重要度ほぼゼロ → 捨てる
  合成: logit(p) = α·logit(p_heavy) + β·logit(p_light) + c
        α, β をロジスティック回帰で学習 ＝「重い/軽い」の重みをデータで確定させる

評価: <=2023で重要度算出→<=2024で学習→2025評価（place3=複勝予測）。
  フラット vs コアのみ vs 階層合成 を AUC・軸の複勝率・上位3的中率で比較。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from market_free_model import FEATURE_COLS_MF


def log(m):
    print(m, flush=True)


LGBP = dict(objective="binary", learning_rate=0.05, num_leaves=63, n_estimators=700,
            min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, verbose=-1, seed=42)


def axis_quality(te, p, label):
    t = te[["race_id", "fuku"]].copy()
    t["p"] = p
    t["rk"] = t.groupby("race_id")["p"].rank(ascending=False, method="min")
    a1 = t[t["rk"] == 1]["fuku"].mean() * 100
    cap = t[t["rk"] <= 5].groupby("race_id")["fuku"].sum()
    auc = roc_auc_score(t["fuku"], t["p"])
    log(f"  {label:<28} AUC {auc:.4f}  軸(1位)複勝率 {a1:5.2f}%  捕捉@5 {cap.mean():.3f}"
        f"  2頭以上 {(cap >= 2).mean()*100:.1f}%")
    return auc, a1


def main():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着"])
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["年"] = d["race_id"].str[:4].astype(int)
    use = [c for c in FEATURE_COLS_MF if c in d.columns]
    dev = d[d["年"] <= 2023]          # 重要度算出用
    tr = d[d["年"] <= 2024]           # 学習
    te = d[d["年"] == 2025].copy()    # 評価
    log(f"重要度算出 {len(dev):,} / 学習 {len(tr):,} / 評価 {len(te):,} / 特徴量 {len(use)}")

    # ── 重要度で階層を決める（2シードの平均で安定化）──
    imp = None
    for sd in (42, 7):
        p = dict(LGBP, seed=sd, n_estimators=400)
        mm = lgb.LGBMClassifier(**p).fit(dev[use], dev["fuku"])
        s = pd.Series(mm.booster_.feature_importance("gain"), index=use)
        imp = s if imp is None else imp + s
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    cum = imp.cumsum()
    heavy = list(cum[cum <= 65].index)          # 累積65%までのコア
    dead = list(imp[imp < 0.05].index)          # ほぼゼロ
    light = [c for c in use if c not in heavy and c not in dead]
    log(f"\nTier1(重い) {len(heavy)}個 (重要度の65%): {heavy[:8]} …")
    log(f"Tier2(軽い) {len(light)}個 / Tier3(捨てる) {len(dead)}個: {dead[:6]} …")

    # ── モデル群 ──
    y_tr, y_te = tr["fuku"], te["fuku"]
    log("\n【place3(複勝)モデルの比較】")
    m_flat = lgb.LGBMClassifier(**LGBP).fit(tr[use], y_tr)
    p_flat = m_flat.predict_proba(te[use])[:, 1]
    axis_quality(te, p_flat, "① フラット(現行=129特徴)")

    m_h = lgb.LGBMClassifier(**LGBP).fit(tr[heavy], y_tr)
    p_h = m_h.predict_proba(te[heavy])[:, 1]
    axis_quality(te, p_h, f"② コアのみ({len(heavy)}特徴)")

    m_l = lgb.LGBMClassifier(**LGBP).fit(tr[light], y_tr)
    p_l = m_l.predict_proba(te[light])[:, 1]
    axis_quality(te, p_l, f"③ 軽い層のみ({len(light)}特徴)")

    m_nd = lgb.LGBMClassifier(**LGBP).fit(tr[heavy + light], y_tr)
    p_nd = m_nd.predict_proba(te[heavy + light])[:, 1]
    axis_quality(te, p_nd, "④ 死に特徴だけ除去")

    # ── ⑤ 階層合成: 重い/軽いの重みをデータに決めさせる ──
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    # 学習側の out-of-fold っぽく: <=2023で学習した予測を2024に当て、2024で係数を学習
    va = d[d["年"] == 2024].copy()
    mh2 = lgb.LGBMClassifier(**LGBP).fit(dev[heavy], dev["fuku"])
    ml2 = lgb.LGBMClassifier(**LGBP).fit(dev[light], dev["fuku"])
    Xv = np.column_stack([logit(mh2.predict_proba(va[heavy])[:, 1]),
                          logit(ml2.predict_proba(va[light])[:, 1])])
    lr = LogisticRegression(C=1e6).fit(Xv, va["fuku"])
    a, b = lr.coef_[0]
    log(f"\n  学習された重み: 重い層 α={a:.3f} / 軽い層 β={b:.3f} "
        f"(比率 {a/(a+b)*100:.0f}% : {b/(a+b)*100:.0f}%)")
    Xt = np.column_stack([logit(p_h), logit(p_l)])
    p_tier = lr.predict_proba(Xt)[:, 1]
    axis_quality(te, p_tier, "⑤ 階層合成(α重い+β軽い)")

    # ── ⑥ コア強化: 重い特徴に木の本数を集中させる版 ──
    p6 = dict(LGBP, feature_fraction=1.0, n_estimators=900)
    m6 = lgb.LGBMClassifier(**p6).fit(tr[heavy], y_tr,
                                      sample_weight=None)
    p_core = 0.75 * m6.predict_proba(te[heavy])[:, 1] + 0.25 * p_l
    axis_quality(te, p_core, "⑥ コア75%+軽い25%(固定重み)")

    log("\n※比較基準: 現行フラットの軸複勝率と捕捉@5。上回る構成があれば採用候補")


if __name__ == "__main__":
    main()
