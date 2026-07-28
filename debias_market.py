# -*- coding: utf-8 -*-
"""市場の歪み（本命-大穴バイアス）をレース文脈込みで学習し、過剰人気馬を動的に除外する。

背景: 2025全馬の単勝ROIは72.25%だが、100倍超を除くと81.92%まで回復する。
      100-300倍=43.3% / 300倍超=17.8% と、大穴が極端に過剰人気になっている。
      ただし「100倍」は固定線であり、実際の過剰度はレースの状況（頭数・クラス・
      オッズの分布）で変わるはず。そこで市場確率そのものを補正する。

手法: 能力特徴は使わず「市場情報＋レース文脈」だけで勝率を学習する。
      これは予測モデルではなく“オッズの歪みの地図”になる。
      補正後の確率 × オッズ が 1 を超える馬＝市場が過小評価している馬。
検証: <=2024学習 / 2025検証。さらに2025を実開催日で前後半に割って双方向確認。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str},
                    usecols=["race_id", "馬名", "馬番", "着順_num", "単勝オッズ", "人気",
                             "出走頭数", "クラス_num", "距離", "is_turf", "枠番"])
    for c in ["馬番", "着順_num", "単勝オッズ", "人気", "出走頭数", "クラス_num",
              "距離", "is_turf", "枠番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["馬番", "着順_num", "単勝オッズ", "人気"])
    d["年"] = d["race_id"].str[:4].astype(int)
    d["場"] = d["race_id"].str[4:6].astype(int)
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["win"] = (d["着順_num"] == 1).astype(float)
    # 市場情報のみから作る特徴（能力データは一切使わない＝“歪みの地図”）
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["log_odds"] = np.log(d["単勝オッズ"])
    d["fav_odds"] = d.groupby("race_id")["単勝オッズ"].transform("min")
    d["odds比"] = d["単勝オッズ"] / d["fav_odds"]
    d["q_top"] = d.groupby("race_id")["q"].transform("max")
    d["q_std"] = d.groupby("race_id")["q"].transform("std")
    d["人気率"] = d["人気"] / d["出走頭数"]
    dates = pd.read_csv("race_dates.csv", dtype={"kaisai_key": str})
    dm = dict(zip(dates["kaisai_key"], pd.to_datetime(dates["date"])))
    d["dt"] = d["race_id"].str[:10].map(dm)
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith("2025")]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    tan = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "単勝"].itertuples()}
    fuk = {(r.race_id, r.組み合わせ): r.払戻金
           for r in jv[jv["券種"] == "複勝"].itertuples()}
    return d, tan, fuk


FEATS = ["log_odds", "q", "人気", "人気率", "odds比", "q_top", "q_std",
         "出走頭数", "クラス_num", "距離", "is_turf", "枠番", "場"]


def roi(s, table):
    if not len(s):
        return float("nan")
    ret = sum(table.get((r.race_id, r.bn), 0) for r in s.itertuples())
    return ret / (len(s) * 100) * 100


def main():
    d, tan, fuk = load()
    tr = d[d["年"] <= 2024]
    te = d[d["年"] == 2025].copy()
    log(f"学習(<=2024) {len(tr):,}頭 / 検証(2025) {len(te):,}頭  ※能力特徴は不使用")

    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                           n_estimators=700, min_child_samples=100,
                           feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                           verbose=-1, seed=42)
    m.fit(tr[FEATS], tr["win"])
    te["p_adj"] = m.predict_proba(te[FEATS])[:, 1]
    te["p_adj"] = te["p_adj"] / te.groupby("race_id")["p_adj"].transform("sum")
    te["EV"] = te["p_adj"] * te["単勝オッズ"]
    te["bias"] = te["p_adj"] / te["q"].clip(lower=1e-9)   # <1なら過剰人気

    log("\n" + "=" * 72)
    log("【1】補正モデルが捉えた歪み（オッズ帯別の 補正確率÷市場確率）")
    log("=" * 72)
    te["帯"] = pd.cut(te["単勝オッズ"], [0, 5, 10, 30, 60, 100, 300, 99999],
                     labels=["〜5", "5-10", "10-30", "30-60", "60-100", "100-300", "300+"])
    g = te.groupby("帯", observed=True).agg(n=("bias", "size"), bias=("bias", "mean"),
                                            実勝率=("win", "mean"), 市場=("q", "mean"))
    log(f"  {'帯':<10}{'n':>7}{'市場想定':>9}{'実勝率':>8}{'補正係数':>9}")
    for k, r in g.iterrows():
        log(f"  {str(k):<10}{int(r['n']):7d}{r['市場']*100:8.2f}%{r['実勝率']*100:7.2f}%"
            f"{r['bias']:8.2f}")
    log("  ※補正係数<1＝過剰人気（買ってはいけない）/ >1＝過小評価")

    log("\n" + "=" * 72)
    log("【2】動的除外の効果 ― 補正係数の下位を切る")
    log("=" * 72)
    log(f"  {'除外':<26}{'残n':>7}{'単勝ROI':>9}{'複勝ROI':>9}")
    log(f"  {'除外なし(全馬)':<26}{len(te):7d}{roi(te, tan):8.2f}%{roi(te, fuk):8.2f}%")
    for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
        th = te["bias"].quantile(q)
        s = te[te["bias"] > th]
        log(f"  {'補正係数 下位' + f'{q:.0%}' + 'を除外':<26}{len(s):7d}"
            f"{roi(s, tan):8.2f}%{roi(s, fuk):8.2f}%")
    log(f"  {'（比較）100倍超を除外':<26}{len(te[te['単勝オッズ'] < 100]):7d}"
        f"{roi(te[te['単勝オッズ'] < 100], tan):8.2f}%"
        f"{roi(te[te['単勝オッズ'] < 100], fuk):8.2f}%")

    log("\n" + "=" * 72)
    log("【3】過小評価馬を買う ― 期待値(補正確率×オッズ)で選抜")
    log("=" * 72)
    mid = te["dt"].quantile(0.5)
    log(f"  {'選抜':<24}{'n':>7}{'勝率':>7}{'単勝ROI':>9}{'前半':>8}{'後半':>8}{'判定':>5}")
    for lo in [1.0, 1.05, 1.1, 1.2, 1.3]:
        s = te[te["EV"] >= lo]
        if len(s) < 100:
            continue
        r_all = roi(s, tan)
        r1 = roi(s[s["dt"] <= mid], tan)
        r2 = roi(s[s["dt"] > mid], tan)
        ok = "◎" if min(r1, r2) >= 100 else ("○" if min(r1, r2) >= 90 else "")
        log(f"  {'期待値 ≥ ' + f'{lo:.2f}':<24}{len(s):7d}{s['win'].mean()*100:6.1f}%"
            f"{r_all:8.2f}%{r1:7.1f}%{r2:7.1f}%{ok:>5}")
    for frac in [0.02, 0.05, 0.10]:
        th = te["EV"].quantile(1 - frac)
        s = te[te["EV"] >= th]
        r1 = roi(s[s["dt"] <= mid], tan)
        r2 = roi(s[s["dt"] > mid], tan)
        ok = "◎" if min(r1, r2) >= 100 else ("○" if min(r1, r2) >= 90 else "")
        log(f"  {'期待値 上位' + f'{frac:.0%}':<24}{len(s):7d}{s['win'].mean()*100:6.1f}%"
            f"{roi(s, tan):8.2f}%{r1:7.1f}%{r2:7.1f}%{ok:>5}")


if __name__ == "__main__":
    main()
