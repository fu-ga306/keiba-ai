# -*- coding: utf-8 -*-
"""Elo改良版: 騎手Elo / 条件別Elo / 着差考慮Elo を検証する。

前回(explore_elo.py)の結果:
  馬Eloは市場に+0.00252の情報を足したが、主モデルに足すと+0.00035で、
  持っている情報の大半は既存モデルと重複。人気帯内のAUCは0.48-0.53とほぼ無力。
  原因: 馬は年5-10走しかなく、条件も毎回違うため対戦レーティングが安定しない。

改良案:
  ①騎手Elo … 騎手は年数百回騎乗するのでレーティングが安定する。
              現在の「騎手勝率」は相手の強さを無視した粗い指標なので伸びしろがある。
  ②条件別Elo … 芝/ダート別にレーティングを分ける（適性の分離）
  ③着差考慮Elo … 勝敗だけでなく着差の大きさで更新幅を変える（ハナ差と大差を区別）
評価は前回と同じ「市場・主モデルに対する対数尤度の増分」。
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

BASE = 1500.0
SCALE = 400.0


def log(m):
    print(m, flush=True)


def elo_generic(df, key_col, k=24.0, split_col=None, use_margin=False):
    """汎用Elo。key_col(馬名/騎手)ごとのレーティングを時系列順に更新し、
    各行に「そのレース直前の値」を返す。
    split_col指定時はその値ごとに別レーティング（例: 芝/ダート別）。
    use_margin=Trueなら着差(秒)で更新幅を調整する。"""
    rating = {}
    pre = np.empty(len(df))
    pre_rel = np.empty(len(df))
    pos = 0
    idx_map = {}
    for i, rid in enumerate(df["race_id"].values):
        idx_map.setdefault(rid, []).append(i)
    keys = df[key_col].values
    chakus = pd.to_numeric(df["着順_num"], errors="coerce").values
    splits = df[split_col].values if split_col else np.zeros(len(df))
    margins = (pd.to_numeric(df.get("着差_秒"), errors="coerce").values
               if use_margin and "着差_秒" in df.columns else np.zeros(len(df)))
    for rid, idxs in idx_map.items():
        m = len(idxs)
        cur = np.array([rating.get((keys[i], splits[i]), BASE) for i in idxs])
        for t, i in enumerate(idxs):
            pre[i] = cur[t]
            others = np.delete(cur, t)
            pre_rel[i] = cur[t] - (others.mean() if len(others) else BASE)
        delta = np.zeros(m)
        for a in range(m):
            ca = chakus[idxs[a]]
            if not np.isfinite(ca):
                continue
            for b in range(a + 1, m):
                cb = chakus[idxs[b]]
                if not np.isfinite(cb):
                    continue
                exp_a = 1.0 / (1.0 + 10 ** ((cur[b] - cur[a]) / SCALE))
                s_a = 1.0 if ca < cb else (0.0 if ca > cb else 0.5)
                w = 1.0
                if use_margin:
                    ma, mb = margins[idxs[a]], margins[idxs[b]]
                    if np.isfinite(ma) and np.isfinite(mb):
                        w = 1.0 + min(abs(ma - mb), 3.0) / 3.0   # 着差が大きいほど強く更新
                d = k * w * (s_a - exp_a) / max(m - 1, 1)
                delta[a] += d
                delta[b] -= d
        for t, i in enumerate(idxs):
            rating[(keys[i], splits[i])] = cur[t] + delta[t]
        pos += m
    return pre, pre_rel


def main():
    import features as F
    d = pd.read_csv("race_features.csv", dtype={"race_id": str})
    d["着順_num"] = pd.to_numeric(d["着順_num"], errors="coerce")
    d = d.dropna(subset=["着順_num"])
    d = F.sort_by_horse_time(F.attach_race_date(d))
    d = d.sort_values(["_race_dt", "race_id"]).reset_index(drop=True)
    # 騎手はrace_data_cleanから
    rc = pd.read_csv("race_data_clean.csv", low_memory=False,
                     usecols=["race_id", "馬名", "騎手"])
    rc["race_id"] = rc["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    d = d.merge(rc.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left")
    d["騎手"] = d["騎手"].fillna("不明")
    log(f"{len(d):,}行 / 騎手{d['騎手'].nunique():,}人 / 馬{d['馬名'].nunique():,}頭")
    log(f"1騎手あたり平均騎乗数 {len(d)/d['騎手'].nunique():.0f}回"
        f" vs 1頭あたり平均出走数 {len(d)/d['馬名'].nunique():.1f}回")

    log("\nElo計算中...")
    d["elo_h"], d["elo_h_rel"] = elo_generic(d, "馬名")
    log("  ①馬Elo(基準) 完了")
    d["elo_j"], d["elo_j_rel"] = elo_generic(d, "騎手", k=8.0)
    log("  ②騎手Elo 完了")
    d["_turf"] = pd.to_numeric(d.get("is_turf"), errors="coerce").fillna(0).astype(int)
    d["elo_c"], d["elo_c_rel"] = elo_generic(d, "馬名", split_col="_turf")
    log("  ③条件別Elo(芝ダ) 完了")
    d["elo_m"], d["elo_m_rel"] = elo_generic(d, "馬名", use_margin=True)
    log("  ④着差考慮Elo 完了")

    d["年"] = d["race_id"].str[:4].astype(int)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["odds"] = pd.to_numeric(d["単勝オッズ"], errors="coerce")
    te = d[(d["年"] == 2025) & d["odds"].notna()].copy()
    te["raw"] = 1 / te["odds"]
    te["q"] = te["raw"] / te.groupby("race_id")["raw"].transform("sum")
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測スコア"]].rename(columns={"予測スコア": "p3"})
    te = te.merge(p3, on=["race_id", "馬名"], how="left")
    te["p3"] = pd.to_numeric(te["p3"], errors="coerce")
    s = te.dropna(subset=["p3", "q"]).copy()
    y = s["fuku"].values
    eps = 1e-6

    def fit(cols, label, base_ll=None):
        X = np.nan_to_num(s[cols].values)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        p = m.predict_proba(X)[:, 1]
        ll = -log_loss(y, np.clip(p, eps, 1 - eps))
        auc = roc_auc_score(y, p)
        g = f"  LL増分{ll-base_ll:+.5f}" if base_ll is not None else ""
        mark = ""
        if base_ll is not None:
            mark = " ◎有効" if ll - base_ll >= 0.001 else " ×効果なし"
        log(f"  {label:<32} LL={ll:.5f} AUC={auc:.4f}{g}{mark}")
        return ll

    log("\n" + "=" * 80)
    log("【市場に対して情報を足せるか】(2025)")
    log("=" * 80)
    b = fit(["q"], "市場のみ")
    for cols, nm in [(["elo_h", "elo_h_rel"], "＋馬Elo(基準)"),
                     (["elo_j", "elo_j_rel"], "＋騎手Elo"),
                     (["elo_c", "elo_c_rel"], "＋条件別Elo"),
                     (["elo_m", "elo_m_rel"], "＋着差考慮Elo"),
                     (["elo_h", "elo_j", "elo_c", "elo_m",
                       "elo_h_rel", "elo_j_rel", "elo_c_rel", "elo_m_rel"], "＋全Elo")]:
        fit(["q"] + cols, "市場 " + nm, b)

    log("\n" + "=" * 80)
    log("【主モデル＋市場に対して情報を足せるか】← 本命の判定")
    log("=" * 80)
    b2 = fit(["q", "p3"], "市場＋主モデル")
    for cols, nm in [(["elo_h", "elo_h_rel"], "＋馬Elo(基準)"),
                     (["elo_j", "elo_j_rel"], "＋騎手Elo"),
                     (["elo_c", "elo_c_rel"], "＋条件別Elo"),
                     (["elo_m", "elo_m_rel"], "＋着差考慮Elo"),
                     (["elo_h", "elo_j", "elo_c", "elo_m",
                       "elo_h_rel", "elo_j_rel", "elo_c_rel", "elo_m_rel"], "＋全Elo")]:
        fit(["q", "p3"] + cols, "市場＋主モデル " + nm, b2)

    log("\n" + "=" * 80)
    log("【各Eloの単体性能】")
    log("=" * 80)
    for c, nm in [("elo_h", "馬Elo"), ("elo_j", "騎手Elo"),
                  ("elo_c", "条件別Elo"), ("elo_m", "着差考慮Elo")]:
        r1 = s.groupby("race_id")[c].rank(ascending=False, method="min")
        top = s[r1 == 1]
        log(f"  {nm:<12} 複勝AUC {roc_auc_score(s['fuku'], s[c]):.4f}"
            f"  1位馬の複勝率 {top['fuku'].mean()*100:5.2f}%")

    cols = ["race_id", "馬名", "elo_h", "elo_h_rel", "elo_j", "elo_j_rel",
            "elo_c", "elo_c_rel", "elo_m", "elo_m_rel"]
    d[cols].to_csv("elo_features.csv", index=False, encoding="utf-8-sig")
    log("\n→ elo_features.csv に保存")


if __name__ == "__main__":
    main()
