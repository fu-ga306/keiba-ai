# -*- coding: utf-8 -*-
"""新しい角度: 対戦ベースのレーティング(Elo)を作り、既存モデルに情報を足せるか検証。

現行モデルの限界: 各馬の特徴量(過去勝率・平均着順など)から着順を予測している。
これは「誰と走って勝ったか」を捨てている。強い相手に善戦した馬と、弱い相手に
楽勝した馬が、同じ「1着」として扱われる。

Eloは対戦の連鎖を通じて強さを伝播させる:
  ・レース内の全ペアで勝敗を取り、期待勝率との差で更新
  ・強い相手に勝てば大きく上がり、弱い相手に負ければ大きく下がる
  ・「メンバーレベル」を平均値ではなく対戦ネットワークとして捉える

検証:
  1. Eloを時系列順に計算（各レース時点の値のみ使用＝リークなし）
  2. Elo単体の予測力（AUC・軸の複勝率）
  3. 既存モデルの予測にEloを足して情報が増えるか（対数尤度の増分）
  4. 市場(オッズ)に対しても情報を足せるか ← ここが本丸
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

K = 24.0          # 更新幅
BASE = 1500.0
SCALE = 400.0


def log(m):
    print(m, flush=True)


def build_elo(df):
    """時系列順にEloを更新し、各行に「そのレース直前のレーティング」を付ける。
    レース内の全ペア比較で更新する（着順が上＝勝ち）。"""
    rating = {}
    n_run = {}
    pre, pre_max, pre_mean = [], [], []
    for rid, g in df.groupby("race_id", sort=False):
        names = g["馬名"].tolist()
        chaku = pd.to_numeric(g["着順_num"], errors="coerce").tolist()
        cur = np.array([rating.get(n, BASE) for n in names], dtype=float)
        # このレース時点の値を記録（更新前＝リークなし）
        for i, n in enumerate(names):
            pre.append(cur[i])
            others = np.delete(cur, i)
            pre_max.append(others.max() if len(others) else BASE)
            pre_mean.append(others.mean() if len(others) else BASE)
        # 全ペアで更新
        m = len(names)
        delta = np.zeros(m)
        for i in range(m):
            if not np.isfinite(chaku[i]):
                continue
            for j in range(i + 1, m):
                if not np.isfinite(chaku[j]):
                    continue
                exp_i = 1.0 / (1.0 + 10 ** ((cur[j] - cur[i]) / SCALE))
                s_i = 1.0 if chaku[i] < chaku[j] else (0.0 if chaku[i] > chaku[j] else 0.5)
                d = K * (s_i - exp_i) / max(m - 1, 1)
                delta[i] += d
                delta[j] -= d
        for i, n in enumerate(names):
            rating[n] = cur[i] + delta[i]
            n_run[n] = n_run.get(n, 0) + 1
    out = df.copy()
    out["elo"] = pre
    out["elo_max_rival"] = pre_max
    out["elo_mean_rival"] = pre_mean
    out["elo_diff"] = out["elo"] - out["elo_mean_rival"]
    out["elo_rank"] = out.groupby("race_id")["elo"].rank(ascending=False, method="min")
    return out


def main():
    log("データ読み込み・時系列整列...")
    import features as F
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着順_num"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着順_num"])
    d = F.sort_by_horse_time(F.attach_race_date(d))   # 実開催日順
    d = d.sort_values(["_race_dt", "race_id"]).reset_index(drop=True)
    log(f"  {len(d):,}行 / {d['race_id'].nunique():,}レース")

    log("Elo計算中（全ペア対戦・時系列順）...")
    d = build_elo(d)
    d["年"] = d["race_id"].str[:4].astype(int)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    d["人気"] = pd.to_numeric(d["人気"], errors="coerce")
    te = d[(d["年"] == 2025) & d["odds"].notna() & d["人気"].notna()].copy()
    te["raw"] = 1 / te["odds"]
    te["q"] = te["raw"] / te.groupby("race_id")["raw"].transform("sum")
    log(f"  検証(2025) {len(te):,}頭 / Elo範囲 {d['elo'].min():.0f}〜{d['elo'].max():.0f}")

    log("\n" + "=" * 78)
    log("【1】Elo単体の予測力（2025）")
    log("=" * 78)
    a = te[te["elo_rank"] == 1]
    log(f"  Elo1位の複勝率 {a['fuku'].mean()*100:.2f}% / 勝率 {a['win'].mean()*100:.2f}%"
        f" / 平均人気 {a['人気'].mean():.2f}")
    log(f"  複勝AUC(Elo) {roc_auc_score(te['fuku'], te['elo']):.4f}"
        f"  (参考: 主モデルplace3のAUCは約0.75)")
    for lo, hi, nm in [(1, 1, "1番人気"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 99, "7-")]:
        s = te[(te["人気"] >= lo) & (te["人気"] <= hi)]
        if len(s) < 200:
            continue
        log(f"    {nm:<8} Elo複勝AUC {roc_auc_score(s['fuku'], s['elo']):.4f}"
            f"  (人気帯内でEloが効くか)")

    log("\n" + "=" * 78)
    log("【2】既存モデル・市場に情報を足せるか（対数尤度の増分・2025）")
    log("=" * 78)
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測スコア"]].rename(columns={"予測スコア": "p3"})
    te = te.merge(p3, on=["race_id", "馬名"], how="left")
    te["p3"] = pd.to_numeric(te["p3"], errors="coerce")
    s = te.dropna(subset=["p3", "q", "elo", "elo_diff"])
    y = s["fuku"].values
    eps = 1e-6

    def fit(cols, label, base_ll=None):
        X = s[cols].values
        X = np.nan_to_num(X)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        ll = -log_loss(y, np.clip(p, eps, 1 - eps))
        auc = roc_auc_score(y, p)
        gain = f"  LL増分{ll-base_ll:+.5f}" if base_ll is not None else ""
        log(f"  {label:<34} LL={ll:.5f} AUC={auc:.4f}{gain}")
        return ll

    ll_mkt = fit(["q"], "市場(オッズ)のみ")
    fit(["q", "elo", "elo_diff"], "市場 ＋ Elo", ll_mkt)
    ll_p3 = fit(["p3"], "主モデルplace3のみ")
    fit(["p3", "elo", "elo_diff"], "主モデル ＋ Elo", ll_p3)
    ll_both = fit(["q", "p3"], "市場 ＋ 主モデル")
    fit(["q", "p3", "elo", "elo_diff"], "市場 ＋ 主モデル ＋ Elo", ll_both)
    log("\n  ※LL増分が+0.001未満なら『新しい情報を足せていない』")

    d[["race_id", "馬名", "elo", "elo_diff", "elo_max_rival", "elo_rank"]].to_csv(
        "elo_features.csv", index=False, encoding="utf-8-sig")
    log("\n  → elo_features.csv に保存（有効なら特徴量として組み込む）")


if __name__ == "__main__":
    main()
