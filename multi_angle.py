# -*- coding: utf-8 -*-
"""儲けるための候補を、5つの異なる角度から作る（2026-08-13・第1段階）

これまでの探索は「どの馬を買うか」を変えるだけで、角度が同じだった。
同じグリッドを再走査しても同じ幻が出る（芝限定148.9%、穴馬124.8%はどちらも
family-wise検定で消えた）。そこで角度そのものを変える。

角度1: レースを選ぶ（馬ではなく、どのレースに参加するか）
        市場の割れ方（エントロピー）、モデルと市場の一致度、1番人気の強さ
角度2: 条件の掛け合わせ（レース条件 × 馬条件の交互作用）
角度3: レース内での突出度（絶対確率ではなく、2位との差）
角度4: 弱い1番人気レースの中で、さらに馬を選ぶ
角度5: 複数点買い（ワイド・複勝の組み合わせ）

順位付けは**最悪年の回収率**。5年で一番悪い年を基準にする。
点推定が高くても最悪年が低いものは、運用すると必ず途中で心が折れる。

⚠ ここで出るのは候補であって結論ではない。第2段階で順列検定にかける。

検体: bet_cache_2021〜2025（207,518頭・14,972レース）＋ jv_payouts
実行: python multi_angle.py → multi_angle_result.csv
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

YEARS = [2021, 2022, 2023, 2024, 2025]
MIN_N = 200


def log(m):
    print(m, flush=True)


def load():
    D = pd.concat([pd.read_csv(f"bet_cache_{y}.csv", dtype={"race_id": str, "bn": str})
                   .assign(年=y) for y in YEARS], ignore_index=True)
    rf = pd.read_csv("race_features.csv", low_memory=False, dtype={"race_id": str},
                     usecols=["race_id", "距離", "is_turf", "クラス_num",
                              "馬場状態_num", "出走頭数"]).drop_duplicates("race_id")
    rf["race_id"] = rf["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    D = D.merge(rf, on="race_id", how="left")
    jv = pd.read_csv("jv_payouts.csv", dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    fk = {(r.race_id, r.組み合わせ): r.払戻金 for r in jv[jv.券種 == "複勝"].itertuples()}
    D["tan"] = D.win * D.odds * 100
    D["fuku"] = [fk.get((r, b), 0.0) for r, b in zip(D.race_id, D.bn)]

    # 市場確率とレース単位の指標
    D["m"] = D.groupby("race_id")["odds"].transform(lambda s: (1 / s) / (1 / s).sum())
    g = D.groupby("race_id")
    # 市場のエントロピー: 大きいほど「混戦」＝市場が決めきれていない
    D["ent"] = g["m"].transform(lambda s: -(s * np.log(s + 1e-12)).sum())
    # モデルと市場の一致度（レース内の順位相関）
    D["agree"] = g.apply(
        lambda x: x["mr"].corr(x["pr"], method="spearman")).reindex(D.race_id).values
    # モデル1位の突出度: 1位と2位の確率比
    def _gap(s):
        v = np.sort(s.values)[::-1]
        return v[0] / v[1] if len(v) > 1 and v[1] > 0 else np.nan
    D["edge"] = g["c_win_n"].transform(_gap)
    # 1番人気に対するモデル評価
    fav = D[D.pr == 1][["race_id", "mr"]].rename(columns={"mr": "fav_mr"})
    D = D.merge(fav, on="race_id", how="left")
    return D


def score(s, col):
    """最悪年・平均・年数を返す。"""
    yr = [s[s.年 == y][col].mean() for y in YEARS]
    if any(np.isnan(v) for v in yr):
        return None
    return {"点数": len(s), "的中": int((s[col] > 0).sum()),
            "回収率": round(s[col].mean(), 1), "最悪年": round(min(yr), 1),
            "最良年": round(max(yr), 1), "年数": sum(1 for v in yr if v >= 100)}


def main():
    D = load()
    log(f"検体 {len(D):,}頭 / {D.race_id.nunique():,}レース\n")
    rows = []

    def add(angle, name, sel, col="tan"):
        s = D[sel]
        if len(s) < MIN_N:
            return
        r = score(s, col)
        if r:
            rows.append({"角度": angle, "構成": name,
                         "券種": "単勝" if col == "tan" else "複勝", **r})

    log("角度1: レースを選ぶ")
    eq = D.ent.quantile([.25, .5, .75]).values
    for lbl, m in (("混戦（エントロピー上位25%)", D.ent >= eq[2]),
                   ("堅い（エントロピー下位25%)", D.ent <= eq[0])):
        for mr in (1, 3):
            add("1.レース選択", f"{lbl} × MF{mr}位以内", m & (D.mr <= mr))
            add("1.レース選択", f"{lbl} × MF{mr}位以内", m & (D.mr <= mr), "fuku")
    for lbl, m in (("モデルと市場が一致(相関0.7+)", D.agree >= 0.7),
                   ("モデルと市場が不一致(相関0.3-)", D.agree <= 0.3)):
        for mr in (1, 3):
            add("1.レース選択", f"{lbl} × MF{mr}位以内", m & (D.mr <= mr))
            add("1.レース選択", f"{lbl} × MF{mr}位以内", m & (D.mr <= mr), "fuku")

    log("角度2: 条件の掛け合わせ")
    C = {"長距離2100+": D["距離"] >= 2100, "重賞級": D["クラス_num"] >= 5,
         "道悪": D["馬場状態_num"] >= 3, "多頭数16+": D["出走頭数"] >= 16}
    for c1, m1 in C.items():
        for c2, m2 in C.items():
            if c1 >= c2:
                continue
            for mr in (1, 3, 5):
                add("2.条件の掛合せ", f"{c1}×{c2} × MF{mr}位以内", m1 & m2 & (D.mr <= mr))
                add("2.条件の掛合せ", f"{c1}×{c2} × MF{mr}位以内", m1 & m2 & (D.mr <= mr), "fuku")

    log("角度3: 突出度")
    for th in (1.3, 1.6, 2.0, 3.0):
        for pr in (99, 6, 3):
            m = (D.edge >= th) & (D.mr == 1) & (D.pr <= pr)
            add("3.突出度", f"1位が2位の{th}倍以上 × 人気{pr if pr<99 else '不問'}", m)
            add("3.突出度", f"1位が2位の{th}倍以上 × 人気{pr if pr<99 else '不問'}", m, "fuku")

    log("角度4: 弱い1番人気レースの中で選ぶ")
    for fth in (4, 6):
        w = D.fav_mr >= fth
        for mr in (1, 2, 3):
            for olo, ohi in ((1, 10), (5, 20), (10, 30), (1, 50)):
                m = w & (D.mr <= mr) & (D.pr != 1) & (D.odds >= olo) & (D.odds < ohi)
                add("4.弱い1人気", f"1人気MF{fth}位以下 × MF{mr}位以内 × {olo}-{ohi}倍", m)
                add("4.弱い1人気", f"1人気MF{fth}位以下 × MF{mr}位以内 × {olo}-{ohi}倍", m, "fuku")

    log("角度5: 複勝の多点")
    for mr in (2, 3, 4):
        for olo, ohi in ((3, 15), (5, 20), (10, 30)):
            m = (D.mr <= mr) & (D.odds >= olo) & (D.odds < ohi)
            add("5.複勝多点", f"MF{mr}位以内 × {olo}-{ohi}倍", m, "fuku")

    R = pd.DataFrame(rows).drop_duplicates(subset=["構成", "券種"])
    R = R.sort_values("最悪年", ascending=False)
    R.to_csv("multi_angle_result.csv", index=False, encoding="utf-8-sig")
    log(f"\n候補 {len(R)}件を評価\n")
    log("=== 最悪年が100%を超えた構成 ===")
    ok = R[R.最悪年 >= 100]
    log(ok.to_string(index=False) if len(ok) else "  なし")
    log(f"\n=== 最悪年 上位15（角度別に見る）===")
    log(R.head(15).to_string(index=False))
    log("\n保存 → multi_angle_result.csv")


if __name__ == "__main__":
    main()
