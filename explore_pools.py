# -*- coding: utf-8 -*-
"""未検証の切り口を探索: 枠連プール / コンセンサス狙い / 的中率×回収率のトレードオフ。

これまで潰した仮説: 特徴量追加・位置取り予測・乖離狙い・セル総当たり・構造ベット21種・場別モデル。
残る未検証:
  A. 的中率重視と回収率重視の「配合比率」… 期待値は線形なので混ぜても平均以上にはならない
     ことを実データで確認しつつ、変動(分散)がどれだけ下がるかを測る。
  B. 枠連プール … 日本の枠連は購入者が少なく歪みが残りやすいと言われる。馬連と直接比較。
  C. コンセンサス狙い … 乖離では市場が正しかった。逆に「モデルと市場が一致」を買うとどうか。
  D. odds_history に複勝オッズがあるか … プール間裁定(単勝プール vs 複勝プール)の可否確認。
"""
import os
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
    p3 = pd.read_csv("model_result_place3.csv", dtype={"race_id": str})[
        ["race_id", "馬名", "予測順位"]].rename(columns={"予測順位": "place3順"})
    rf = pd.read_csv("race_features.csv", dtype={"race_id": str},
                     usecols=["race_id", "馬名", "馬番", "枠番"])
    d = (m.merge(p3, on=["race_id", "馬名"], how="left")
          .merge(rf.drop_duplicates(["race_id", "馬名"]), on=["race_id", "馬名"], how="left"))
    for c in ["着順_num", "単勝オッズ", "人気", "勝ち確率", "馬番", "枠番", "place3順"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["着順_num", "単勝オッズ", "人気", "馬番", "枠番"])
    d["bn"] = d["馬番"].astype(int).map(lambda x: f"{x:02d}")
    d["wk"] = d["枠番"].astype(int).map(lambda x: f"{x:02d}")
    d["win"] = (d["着順_num"] == 1).astype(float)
    d["fuku"] = (d["着順_num"] <= 3).astype(float)
    d["主順"] = d.groupby("race_id")["勝ち確率"].rank(ascending=False, method="min")
    d["h"] = np.where(d["race_id"].str[4:6] < "06", "A", "B")
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


def bet_stats(recs):
    """[(rid, ret, h)] → ROI・的中率・1点収支の標準偏差・前後半ROI"""
    if len(recs) < 200:
        return None
    r = pd.DataFrame(recs, columns=["rid", "ret", "h"])
    roi = r["ret"].sum() / (len(r) * 100) * 100
    hit = (r["ret"] > 0).mean() * 100
    sd = (r["ret"] - 100).std()
    ra = r[r["h"] == "A"]["ret"].sum() / max(len(r[r["h"] == "A"]) * 100, 1) * 100
    rb = r[r["h"] == "B"]["ret"].sum() / max(len(r[r["h"] == "B"]) * 100, 1) * 100
    return roi, hit, sd, ra, rb, len(r)


def part_a(d, pay):
    log("\n" + "=" * 76)
    log("【A】的中率重視 vs 回収率重視 ― 混ぜると何が変わるか")
    log("=" * 76)
    cands = []
    # 的中率重視の代表: 主モデル1位の複勝 / 1番人気の複勝
    for nm, mask in [("主モデル1位の複勝", d["主順"] == 1), ("1番人気の複勝", d["人気"] == 1),
                     ("主モデル1位の単勝", d["主順"] == 1), ("1番人気の単勝", d["人気"] == 1),
                     ("主モデル4-6位の単勝", (d["主順"] >= 4) & (d["主順"] <= 6))]:
        kind = "複勝" if "複勝" in nm else "単勝"
        s = d[mask]
        recs = [(r["race_id"], pay.get((r["race_id"], kind, r["bn"]), 0), r["h"])
                for _, r in s.iterrows()]
        st = bet_stats(recs)
        if st:
            cands.append((nm, st))
    log(f"  {'戦略':<22}{'的中率':>8}{'ROI':>8}{'1点SD':>8}{'n':>7}")
    for nm, st in cands:
        log(f"  {nm:<22}{st[1]:7.1f}%{st[0]:7.1f}%{st[2]:8.0f}{st[5]:7d}")
    log("\n  ― 配合比率を変えるとどうなるか（的中率重視=主1位複勝 / 回収率重視=主4-6位単勝）―")
    hi = dict(cands)["主モデル1位の複勝"]
    lo = dict(cands)["主モデル4-6位の単勝"]
    log(f"  {'比率(的中:回収)':<18}{'期待ROI':>9}{'合成SD(概算)':>14}")
    for w in [1.0, 0.75, 0.5, 0.25, 0.0]:
        roi = hi[0] * w + lo[0] * (1 - w)
        sd = (w ** 2 * hi[2] ** 2 + (1 - w) ** 2 * lo[2] ** 2) ** 0.5
        log(f"  {f'{w:.0%} : {1-w:.0%}':<18}{roi:8.1f}%{sd:13.0f}")
    log("  ※ROIは線形＝混ぜても両者の間にしかならない。変わるのは変動の大きさだけ。")


def part_b(d, pay):
    log("\n" + "=" * 76)
    log("【B】枠連プール（購入者が少なく歪みが残りやすいと言われる）― 馬連と直接比較")
    log("=" * 76)
    log(f"  {'戦略':<26}{'的中率':>8}{'ROI':>8}{'前半':>8}{'後半':>8}{'n':>7}")
    for rk_col, rk_nm in [("人気", "人気"), ("主順", "主モデル"), ("place3順", "place3")]:
        for kind, key in [("枠連", "wk"), ("馬連", "bn")]:
            recs = []
            for rid, g in d.groupby("race_id"):
                gg = g.dropna(subset=[rk_col])
                if len(gg) < 2:
                    continue
                top = gg.nsmallest(2, rk_col)
                if len(top) < 2:
                    continue
                a, b = top.iloc[0][key], top.iloc[1][key]
                if kind == "枠連" and a == b:
                    continue      # 同枠は枠連の対象外(ゾロ目は別扱い)
                k = "-".join(sorted([a, b]))
                recs.append((rid, pay.get((rid, kind, k), 0), g["h"].iloc[0]))
            st = bet_stats(recs)
            if st:
                log(f"  {kind+' '+rk_nm+'上位2':<26}{st[1]:7.1f}%{st[0]:7.1f}%"
                    f"{st[3]:7.1f}%{st[4]:7.1f}%{st[5]:7d}")
        log("")


def part_c(d, pay):
    log("\n" + "=" * 76)
    log("【C】コンセンサス狙い ― 乖離では市場が正しかった。逆に「一致」を買う")
    log("=" * 76)
    log(f"  {'条件':<34}{'的中率':>8}{'複勝ROI':>9}{'単勝ROI':>9}{'n':>7}")
    conds = [
        ("主1位 かつ 1番人気", (d["主順"] == 1) & (d["人気"] == 1)),
        ("主1位 かつ 1-2番人気", (d["主順"] == 1) & (d["人気"] <= 2)),
        ("主1位 かつ place3-1位 かつ 1番人気",
         (d["主順"] == 1) & (d["place3順"] == 1) & (d["人気"] == 1)),
        ("主1位 かつ 1番人気 かつ オッズ2倍以下",
         (d["主順"] == 1) & (d["人気"] == 1) & (d["単勝オッズ"] <= 2.0)),
        ("主1位 かつ 1番人気 かつ オッズ1.5倍以下",
         (d["主順"] == 1) & (d["人気"] == 1) & (d["単勝オッズ"] <= 1.5)),
        ("place3-1位 かつ 1番人気", (d["place3順"] == 1) & (d["人気"] == 1)),
    ]
    for nm, mask in conds:
        s = d[mask]
        if len(s) < 150:
            continue
        fr = sum(pay.get((r["race_id"], "複勝", r["bn"]), 0) for _, r in s.iterrows())
        tr = sum(pay.get((r["race_id"], "単勝", r["bn"]), 0) for _, r in s.iterrows())
        log(f"  {nm:<34}{s['fuku'].mean()*100:7.1f}%{fr/(len(s)*100)*100:8.1f}%"
            f"{tr/(len(s)*100)*100:8.1f}%{len(s):7d}")


def part_d():
    log("\n" + "=" * 76)
    log("【D】odds_history の中身 ― プール間裁定(単勝プールvs複勝プール)が可能か")
    log("=" * 76)
    p = "odds_history.csv"
    if not os.path.exists(p):
        log("  odds_history.csv なし")
        return
    o = pd.read_csv(p, dtype={"race_id": str})
    log(f"  列: {list(o.columns)}")
    log(f"  行数 {len(o):,} / レース {o['race_id'].nunique()} / "
        f"記録時刻の種類 {o['取得時刻'].nunique() if '取得時刻' in o.columns else '?'}")
    if "race_id" in o.columns:
        cnt = o.groupby("race_id").size()
        log(f"  1レースあたりのスナップショット数: 中央値{cnt.median():.0f} 最大{cnt.max()}")
        multi = (o.groupby("race_id")["取得時刻"].nunique() >= 2).sum() \
            if "取得時刻" in o.columns else 0
        log(f"  2時点以上あるレース: {multi}（オッズ変動の検証にはこれが必要）")


def main():
    d, pay = load()
    log(f"検証 {TY}: {d['race_id'].nunique()}レース {len(d)}頭")
    part_a(d, pay)
    part_b(d, pay)
    part_c(d, pay)
    part_d()


if __name__ == "__main__":
    main()
