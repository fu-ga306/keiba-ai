# -*- coding: utf-8 -*-
"""回収率を上げる方法の多角的探索（リーク修正後・2025・JV実払戻）。

単勝の勝率市場は効率的だった(市場の想定勝率=実勝率)。残る可能性を総当たり:
  A. モデルは市場に「上乗せ」情報を持つか（対数尤度・AUCの増分検定）
  B. 本命-大穴バイアス（オッズ帯別の単勝/複勝ROI・複勝は別プール＝別市場）
  C. オッズ帯×モデル順位のセル単位スキャン（単勝/複勝）→100%超セルは場2分割で再現確認
  D. 構造ベット（人気順の組み合わせで機械的に買う連系）＝モデル不要の市場歪み
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

TY = "2025"
UNORD = {"馬連", "ワイド", "3連複", "枠連"}


def log(m):
    print(m, flush=True)


def load():
    m = pd.read_csv("model_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "着順_num", "単勝オッズ", "人気", "勝ち確率"]]
    mf = pd.read_csv("model_mf_result.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "MF勝率", "MF複勝率", "MF複勝順位"]]
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "馬番"])
    d = (m.merge(mf, on=["race_id", "馬名"], how="inner")
          .merge(p3, on=["race_id", "馬名"], how="left")
          .merge(rf.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left"))
    for c in ["着順_num", "単勝オッズ", "人気", "勝ち確率", "MF勝率", "MF複勝率", "馬番"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "馬番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["raw"] = 1 / d["単勝オッズ"]
    d["q"] = d["raw"] / d.groupby("race_id")["raw"].transform("sum")
    d["p_main"] = d["勝ち確率"] / d.groupby("race_id")["勝ち確率"].transform("sum")
    d["p_mf"] = d["MF勝率"] / d.groupby("race_id")["MF勝率"].transform("sum")
    d["主順"] = d.groupby("race_id")["p_main"].rank(ascending=False, method="min")
    d["h"] = np.where(d["race_id"].str[4:6] < "06", "A", "B")   # 場コードで2分割

    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv = jv[jv["race_id"].str.startswith(TY)]
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {}
    for r in jv.itertuples():
        c = str(r.組み合わせ)
        if r.券種 in UNORD:
            c = "-".join(sorted(c.split("-")))
        pay[(r.race_id, r.券種, c)] = r.払戻金
    return d, pay


def part_a(d):
    log("\n" + "=" * 74)
    log("【A】モデルは市場(オッズ)に上乗せの情報を持つか ― 対数尤度の増分")
    log("=" * 74)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    s = d.dropna(subset=["q", "p_main", "p_mf"]).copy()
    eps = 1e-6
    X0 = np.log(np.clip(s[["q"]], eps, 1)).values
    y = s["win"].values
    r0 = LogisticRegression(C=1e6).fit(X0, y)
    ll0 = -log_loss(y, r0.predict_proba(X0)[:, 1])
    a0 = roc_auc_score(y, r0.predict_proba(X0)[:, 1])
    for name, cols in [("＋主モデル", ["p_main"]), ("＋MF", ["p_mf"]),
                       ("＋両方", ["p_main", "p_mf"])]:
        X = np.hstack([X0, np.log(np.clip(s[cols], eps, 1)).values])
        r = LogisticRegression(C=1e6).fit(X, y)
        ll = -log_loss(y, r.predict_proba(X)[:, 1])
        a = roc_auc_score(y, r.predict_proba(X)[:, 1])
        log(f"  市場のみ LL={ll0:.5f} AUC={a0:.4f} → {name:<8} LL={ll:.5f}({ll-ll0:+.5f}) "
            f"AUC={a:.4f}({a-a0:+.4f})")
    log("  ※LLの増分が+0.001未満なら「モデルは市場に何も足せていない」")


def part_b(d, pay):
    log("\n" + "=" * 74)
    log("【B】本命-大穴バイアス ― オッズ帯別の単勝/複勝ROI（複勝は別プール）")
    log("=" * 74)
    bins = [1.0, 1.3, 1.6, 2.0, 2.5, 3.5, 5, 8, 15, 30, 60, 9999]
    lab = ["1.0-1.3", "1.3-1.6", "1.6-2.0", "2.0-2.5", "2.5-3.5", "3.5-5",
           "5-8", "8-15", "15-30", "30-60", "60+"]
    d["_b"] = pd.cut(d["単勝オッズ"], bins=bins, labels=lab)
    log(f"  {'帯':<10}{'n':>7}{'単勝ROI':>9}{'複勝ROI':>9}{'複勝的中':>9}")
    for b in lab:
        s = d[d["_b"] == b]
        if len(s) < 150:
            continue
        troi = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
        fi = fr = 0
        for _, r in s.iterrows():
            fi += 100
            fr += pay.get((r["race_id"], "複勝", r["bn"]), 0)
        log(f"  {b:<10}{len(s):7d}{troi:8.1f}%{fr/fi*100:8.1f}%{s['fuku'].mean()*100:8.1f}%")


def part_c(d, pay):
    log("\n" + "=" * 74)
    log("【C】オッズ帯×モデル順位のセル・スキャン ― 100%超セルを2分割で再現確認")
    log("=" * 74)
    bins = [1.0, 2, 3.5, 6, 10, 20, 50, 9999]
    lab = ["-2", "2-3.5", "3.5-6", "6-10", "10-20", "20-50", "50+"]
    d["_b"] = pd.cut(d["単勝オッズ"], bins=bins, labels=lab)
    found = []
    for rk_col, rk_name in [("主順", "主モデル順位"), ("place3順", "place3順位"),
                            ("MF複勝順位", "MF複勝順位")]:
        for b in lab:
            for rlo, rhi, rnm in [(1, 1, "1位"), (2, 3, "2-3位"), (4, 6, "4-6位")]:
                s = d[(d["_b"] == b) & (d[rk_col] >= rlo) & (d[rk_col] <= rhi)]
                if len(s) < 150:
                    continue
                troi = (s["win"] * s["単勝オッズ"]).sum() / len(s) * 100
                fi = fr = 0
                for _, r in s.iterrows():
                    fi += 100
                    fr += pay.get((r["race_id"], "複勝", r["bn"]), 0)
                froi = fr / fi * 100
                for kind, roi in [("単勝", troi), ("複勝", froi)]:
                    if roi > 100:
                        a = s[s["h"] == "A"]
                        bb = s[s["h"] == "B"]
                        if kind == "単勝":
                            ra = (a["win"] * a["単勝オッズ"]).sum() / max(len(a), 1) * 100
                            rb = (bb["win"] * bb["単勝オッズ"]).sum() / max(len(bb), 1) * 100
                        else:
                            def fk(x):
                                i = rr = 0
                                for _, r2 in x.iterrows():
                                    i += 100
                                    rr += pay.get((r2["race_id"], "複勝", r2["bn"]), 0)
                                return rr / i * 100 if i else 0
                            ra, rb = fk(a), fk(bb)
                        found.append((rk_name, b, rnm, kind, roi, len(s), ra, rb))
    if not found:
        log("  100%を超えるセルなし")
    else:
        log(f"  {'順位定義':<12}{'オッズ帯':<8}{'順位':<7}{'券種':<5}{'ROI':>8}{'n':>6}"
            f"{'分割A':>8}{'分割B':>8}{'再現':>5}")
        for f in sorted(found, key=lambda x: -x[4]):
            ok = "○" if (f[6] > 100 and f[7] > 100) else "×"
            log(f"  {f[0]:<12}{f[1]:<8}{f[2]:<7}{f[3]:<5}{f[4]:7.1f}%{f[5]:6d}"
                f"{f[6]:7.1f}%{f[7]:7.1f}%{ok:>5}")


def part_d(d, pay):
    log("\n" + "=" * 74)
    log("【D】構造ベット ― 人気順/モデル順の機械的な連系（モデル不要の市場歪み探し）")
    log("=" * 74)
    strategies = []
    for rk_col, rk_name in [("人気", "人気"), ("主順", "主モデル"), ("place3順", "place3")]:
        strategies += [
            (f"馬連 {rk_name}1-2位", "馬連", [(1, 2)], rk_col),
            (f"馬連 {rk_name}1-3位BOX", "馬連", [(1, 2), (1, 3), (2, 3)], rk_col),
            (f"ワイド {rk_name}1-2位", "ワイド", [(1, 2)], rk_col),
            (f"ワイド {rk_name}2-4位BOX", "ワイド", [(2, 3), (2, 4), (3, 4)], rk_col),
            (f"馬単 {rk_name}1→2位", "馬単", [(1, 2)], rk_col),
            (f"馬単 {rk_name}2→1位", "馬単", [(2, 1)], rk_col),
            (f"3連複 {rk_name}1-2-3位", "3連複", [(1, 2, 3)], rk_col),
        ]
    log(f"  {'戦略':<24}{'点数':>7}{'的中率':>8}{'ROI':>8}{'分割A':>8}{'分割B':>8}{'再現':>5}")
    for name, kind, combos, rk_col in strategies:
        recs = []
        for rid, g in d.groupby("race_id"):
            rk = g.set_index(g[rk_col].astype(int))["bn"].to_dict() if \
                g[rk_col].notna().all() else None
            if not rk:
                continue
            for cb in combos:
                if any(x not in rk for x in cb):
                    continue
                bns = [rk[x] for x in cb]
                key = "-".join(sorted(bns)) if kind in UNORD else "-".join(bns)
                v = pay.get((rid, kind, key), 0)
                recs.append((rid, v, g["h"].iloc[0]))
        if len(recs) < 300:
            continue
        rr = pd.DataFrame(recs, columns=["rid", "ret", "h"])
        roi = rr["ret"].sum() / (len(rr) * 100) * 100
        hit = (rr["ret"] > 0).mean() * 100
        ra = rr[rr["h"] == "A"]["ret"].sum() / max(len(rr[rr["h"] == "A"]) * 100, 1) * 100
        rb = rr[rr["h"] == "B"]["ret"].sum() / max(len(rr[rr["h"] == "B"]) * 100, 1) * 100
        mark = "○" if roi > 100 and ra > 100 and rb > 100 else ("△" if roi > 100 else "")
        log(f"  {name:<24}{len(rr):7d}{hit:7.1f}%{roi:7.1f}%{ra:7.1f}%{rb:7.1f}%{mark:>5}")


def main():
    d, pay = load()
    log(f"検証 {TY}: {d['race_id'].nunique()}レース {len(d)}頭（リーク修正後モデル）")
    part_a(d)
    part_b(d, pay)
    part_c(d, pay)
    part_d(d, pay)


if __name__ == "__main__":
    main()
