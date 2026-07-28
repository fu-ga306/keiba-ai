# -*- coding: utf-8 -*-
"""メタモデル検証: 「MFが当たる場面」を学習で見分けられるか。

ユーザーの仮説: MFは市場と乖離していて、当たる時もある。当たる場面を選べれば利益になる。
              市場が優秀でも穴はあるはずで、そこを拾えれば市場を超えられる。

検証: 市場確率・MF確率・主モデル確率＋文脈(人気/オッズ/頭数/クラス/場/距離/MFの確信度/
      乖離の大きさ)を入力に、勝敗を学習するメタモデルを作る。市場だけのモデルと
      out-of-sampleで比較し、
        ・対数尤度/AUCが上回るか（市場に無い情報を拾えているか）
        ・メタモデルが高く評価した馬の単勝ROIが100%を超えるか
      を見る。2025を実開催日で前半＝学習 / 後半＝検証 に分割（時系列を跨がない）。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def log(m):
    print(m, flush=True)


def load():
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})
    m = pd.read_csv("model_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "勝ち確率"]]
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "クラス_num", "出走頭数", "距離", "枠番"])
    d = (mf.merge(m, on=["race_id", "馬名"], how="inner")
           .merge(p3, on=["race_id", "馬名"], how="left")
           .merge(rf.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left"))
    for c in ["着順_num", "単勝オッズ", "人気", "勝ち確率", "MF勝率", "MF複勝率",
              "MF勝率順位", "MF複勝順位", "place3順", "クラス_num", "出走頭数", "距離", "枠番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "勝ち確率", "MF勝率"])
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["場"] = d["race_id"].str[4:6].astype(int)
    # 各確率をレース内で正規化
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["p_mf"] = d["MF勝率"] / d.groupby("race_id")["MF勝率"].transform("sum")
    d["p_main"] = d["勝ち確率"] / d.groupby("race_id")["勝ち確率"].transform("sum")
    # 文脈: MFの確信度（1位と2位の差）・乖離の大きさ・市場の集中度
    d["mf_gap"] = d.groupby("race_id")["p_mf"].transform(
        lambda s: s.max() - (s.nlargest(2).iloc[-1] if len(s) > 1 else 0))
    d["q_top"] = d.groupby("race_id")["q"].transform("max")
    d["div"] = d["p_mf"] - d["q"]
    d["div_main"] = d["p_main"] - d["q"]
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    return d.dropna(subset=["dt"])


def main():
    d = load()
    mid = d["dt"].quantile(0.5)
    tr = d[d["dt"] <= mid].copy()
    te = d[d["dt"] > mid].copy()
    log(f"学習(前半) {tr['race_id'].nunique()}R {len(tr)}頭 / "
        f"検証(後半) {te['race_id'].nunique()}R {len(te)}頭")

    eps = 1e-6
    # 基準: 市場だけ
    X0tr = np.log(np.clip(tr[["q"]], eps, 1)).values
    X0te = np.log(np.clip(te[["q"]], eps, 1)).values
    b = LogisticRegression(C=1e6).fit(X0tr, tr["win"])
    pb = b.predict_proba(X0te)[:, 1]
    ll0, auc0 = -log_loss(te["win"], pb), roc_auc_score(te["win"], pb)
    log(f"\n[基準] 市場のみ            LL={ll0:.5f}  AUC={auc0:.4f}")

    # メタモデル: 市場＋MF＋主＋文脈（非線形・条件付きの構造も拾える）
    feats = ["q", "p_mf", "p_main", "div", "div_main", "mf_gap", "q_top",
             "人気", "単勝オッズ", "MF勝率順位", "MF複勝順位", "place3順",
             "出走頭数", "クラス_num", "距離", "枠番", "場"]
    mdl = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                             n_estimators=400, min_child_samples=50,
                             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                             verbose=-1, seed=42)
    mdl.fit(tr[feats], tr["win"])
    pm = mdl.predict_proba(te[feats])[:, 1]
    ll1, auc1 = -log_loss(te["win"], pm), roc_auc_score(te["win"], pm)
    log(f"[検証] メタモデル          LL={ll1:.5f}({ll1-ll0:+.5f})  AUC={auc1:.4f}({auc1-auc0:+.4f})")
    log("       ※LL増分が+0.001未満なら「MFが当たる場面は学習でも見分けられない」")

    te = te.copy()
    te["p_meta"] = pm
    te["p_meta"] = te["p_meta"] / te.groupby("race_id")["p_meta"].transform("sum")
    te["edge_meta"] = te["p_meta"] - te["q"]

    log("\n[実利] メタモデルの評価が高い馬を買った時の単勝ROI")
    log(f"  {'選び方':<28}{'n':>7}{'的中率':>8}{'ROI':>8}")
    for frac, nm in [(0.02, "上位2%"), (0.05, "上位5%"), (0.10, "上位10%"), (0.20, "上位20%")]:
        th = te["edge_meta"].quantile(1 - frac)
        s = te[te["edge_meta"] >= th]
        if len(s) < 50:
            continue
        roi = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
        log(f"  {'乖離(メタ-市場) ' + nm:<28}{len(s):7d}{s['win'].mean()*100:7.1f}%{roi:7.1f}%")
    for frac, nm in [(0.05, "上位5%"), (0.10, "上位10%")]:
        th = te["p_meta"].quantile(1 - frac)
        s = te[te["p_meta"] >= th]
        roi = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
        log(f"  {'メタ確率そのもの ' + nm:<28}{len(s):7d}{s['win'].mean()*100:7.1f}%{roi:7.1f}%")

    log("\n[参考] メタモデルが重視した入力 上位8")
    imp = pd.Series(mdl.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp = imp / imp.sum() * 100
    for k, v in imp.head(8).items():
        log(f"   {k:<14}{v:5.1f}%")

    log("\n[追加] MFが当たった場面に後から色をつけられるか（MF1位馬だけを対象）")
    s = te[te["MF勝率順位"] == 1].copy()
    s["hit"] = s["win"]
    log(f"  MF1位馬 {len(s)}頭中 的中{int(s['hit'].sum())}頭 "
        f"({s['hit'].mean()*100:.1f}%) 単勝ROI {(s['win']*s['単勝オッズ']).sum()/len(s)*100:.1f}%")
    for nm, sub in [("メタ評価が高い上位1/3", s[s["p_meta"] >= s["p_meta"].quantile(2/3)]),
                    ("MF確信度が高い上位1/3", s[s["mf_gap"] >= s["mf_gap"].quantile(2/3)]),
                    ("市場も支持(3番人気内)", s[s["人気"] <= 3])]:
        if len(sub) < 40:
            continue
        roi = (sub["win"] * sub["単勝オッズ"]).sum() / len(sub) * 100
        log(f"   {nm:<24}{len(sub):5d}頭 的中{sub['win'].mean()*100:5.1f}% ROI{roi:7.1f}%")


if __name__ == "__main__":
    main()
