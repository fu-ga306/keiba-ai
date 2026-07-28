# -*- coding: utf-8 -*-
"""回収率向上の最終候補を厳密検証し、新しい買い方ポートフォリオを設計する。

候補（これまでの検証から生き残っているもの）:
  C1: 較正複勝率≥0.6 の複勝 … 双方向で 90.2/85.2%（的中64%・的中率重視枠）
  C2: 30-100倍 × bias(市場歪み補正)≥0.90 の単勝 … 先の検証で100.0%(n=4278)。
      ただし分割検証が未実施なのでここで厳密に行う（半期×場×月次）。
  C3: 同じ選抜の複勝版
  C4: 現行メニューの剪定 … 券種別ROIから足を引っ張るものを特定

検証原則: 学習は<=2024のみ / 2025で評価 / 期間・場の分割で再現しなければ不採用。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from train_debias import build_market_features, FEATS


def log(m):
    print(m, flush=True)


def load():
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着", "単勝オッズ", "人気"])
    d = build_market_features(d)
    d["win"] = (d["着"] == 1).astype(float)
    d["fuku"] = (d["着"] <= 3).astype(float)
    d["年"] = d["race_id"].str[:4].astype(int)
    d["bn"] = pd.to_numeric(d["馬番"], errors="coerce").astype("Int64").map(
        lambda x: f"{int(x):02d}" if pd.notna(x) else None)
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF複勝率", "MF複勝順位"]]
    d = d.merge(mf, on=["race_id", "馬名"], how="left")
    for c in ["MF複勝率", "MF複勝順位"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
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


def roi(s, table, amt=100):
    if not len(s):
        return float("nan")
    ret = sum(table.get((r.race_id, r.bn), 0) for r in s.itertuples())
    return ret / (len(s) * amt) * amt / 100 * 100 if False else ret / (len(s) * 100) * 100


def splits(s, tan, fuk, label):
    """半期・場・両券種のROIを1行で出す。"""
    mid = s["dt"].median()
    h1, h2 = s[s["dt"] <= mid], s[s["dt"] > mid]
    a = s[s["race_id"].str[4:6] < "06"]
    b = s[s["race_id"].str[4:6] >= "06"]
    t = roi(s, tan)
    t1, t2 = roi(h1, tan), roi(h2, tan)
    ta, tb = roi(a, tan), roi(b, tan)
    f = roi(s, fuk)
    f1, f2 = roi(h1, fuk), roi(h2, fuk)
    ok_t = "◎" if min(t1, t2, ta, tb) >= 95 else ("○" if min(t1, t2, ta, tb) >= 85 else "×")
    ok_f = "◎" if min(f1, f2) >= 95 else ("○" if min(f1, f2) >= 85 else "×")
    log(f"  {label:<30}{len(s):6d} 単勝{t:6.1f}% (半{t1:.0f}/{t2:.0f} 場{ta:.0f}/{tb:.0f}){ok_t}"
        f"  複勝{f:6.1f}% (半{f1:.0f}/{f2:.0f}){ok_f}")


def main():
    d, tan, fuk = load()
    tr = d[d["年"] <= 2024]
    te = d[d["年"] == 2025].copy()
    log(f"学習 {len(tr):,} / 検証 {len(te):,}（<=2024学習・2025評価）")

    # ── bias（市場歪み補正・<=2024学習）──
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=31,
                           n_estimators=700, min_child_samples=100, feature_fraction=0.9,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
    m.fit(tr[FEATS], tr["win"])
    te["p_adj"] = m.predict_proba(te[FEATS])[:, 1]
    te["p_adj"] = te["p_adj"] / te.groupby("race_id")["p_adj"].transform("sum")
    te["bias"] = te["p_adj"] / te["q"].clip(lower=1e-9)

    log("\n" + "=" * 96)
    log("【C2/C3】バリュー帯の厳密検証 ― オッズ帯 × bias≥0.90 で残した馬")
    log("=" * 96)
    for lo, hi in [(10, 30), (30, 60), (60, 100), (30, 100), (20, 100)]:
        band = te[(te["単勝オッズ"] >= lo) & (te["単勝オッズ"] < hi)]
        keep = band[band["bias"] >= 0.90]
        splits(keep, tan, fuk, f"{lo}-{hi}倍 bias≥0.90")
    log("\n  ― biasしきい値の感度（30-100倍）―")
    band = te[(te["単勝オッズ"] >= 30) & (te["単勝オッズ"] < 100)]
    for th in [0.85, 0.90, 0.95, 1.00, 1.05]:
        splits(band[band["bias"] >= th], tan, fuk, f"30-100倍 bias≥{th:.2f}")

    log("\n" + "=" * 96)
    log("【C1】較正複勝率≥0.6 の複勝（半期の相互較正で再確認）")
    log("=" * 96)
    mid = te["dt"].quantile(0.5)
    h1, h2 = te[te["dt"] <= mid], te[te["dt"] > mid]
    picks = []
    for fit, ap in [(h1, h2), (h2, h1)]:
        s = fit.dropna(subset=["MF複勝率"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            s["MF複勝率"], s["fuku"])
        t = ap.dropna(subset=["MF複勝率"]).copy()
        c = iso.predict(t["MF複勝率"])
        t["cal"] = c
        t["cal"] = t["cal"] / t.groupby("race_id")["cal"].transform("sum") * 3.0
        picks.append(t[t["cal"] >= 0.6])
    p = pd.concat(picks)
    splits(p, tan, fuk, "較正複勝率≥0.6（相互較正）")
    log(f"    的中率(複勝): {p['fuku'].mean()*100:.1f}%  平均人気 {p['人気'].mean():.1f}")

    log("\n" + "=" * 96)
    log("【C4】現行メニューの寄与分解（券種別・投資加重）→ 剪定候補")
    log("=" * 96)
    log("  ※現行BT(2025リークなし)の券種別: 単勝78.1 複勝83.9 ワイド80.6 馬連56.4"
        " 馬単85.9 3連複87.0 3連単102.6(n小)")
    log("  剪定シミュレーション（その券種をやめて残りに等配分した場合の全体ROI）:")
    kinds = {"単勝": (849800, 663500), "複勝": (87500, 73370), "ワイド": (145200, 117000),
             "馬連": (207600, 117160), "馬単": (642300, 551770), "3連複": (106700, 92790),
             "3連単": (100500, 103160)}
    tot_i = sum(v[0] for v in kinds.values())
    tot_r = sum(v[1] for v in kinds.values())
    log(f"  {'剪定':<18}{'全体ROI':>9}")
    log(f"  {'剪定なし(現行)':<18}{tot_r/tot_i*100:8.1f}%")
    for drop in [["馬連"], ["馬連", "単勝"], ["馬連", "ワイド"], ["馬連", "単勝", "ワイド"]]:
        i = tot_i - sum(kinds[k][0] for k in drop)
        r = tot_r - sum(kinds[k][1] for k in drop)
        log(f"  {'−' + '·'.join(drop):<18}{r/i*100:8.1f}%")


if __name__ == "__main__":
    main()
