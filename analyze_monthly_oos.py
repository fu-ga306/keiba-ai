# -*- coding: utf-8 -*-
"""月次レポート。蓄積データから「実運用との乖離」を毎月定量化する。

なぜ必要か（2026-08-11）
  紙の上の優位（確定オッズ基準）は実運用では消えることが分かった。
      5年・確定基準 全体117.0% / 芝148.9%
      スリッページ模擬 全体88.4% / 芝95.0%
      2026年の独立検証 全体98.7% / 芝93.2%
  1年後の再挑戦まで待つのではなく、**毎月この乖離が安定しているかを観測する**。
  傾向が変われば早く気づけるし、変わらなければ結論が強化される。

出すもの（初版）
  ① 7分前→確定のオッズ下落率  … 人気帯別 × 着順別
  ② 候補案の月次回収率        … 7分前で買った場合と確定で買った場合の差
  ③ 乖離スコアと着順の相関    … スピアマンの順位相関

前提
  ・7分前オッズは odds_history.csv（2026-08-11から「分前」が入る）
  ・確定オッズと着順は race_features.csv
  ・払戻は jv_payouts.csv（過去分はローカルに永久に残る）
  ・JRA-VANの契約は更新しないので、票数(H1)への拡張は行わない。
    複勝オッズも実時間では取れない（7分前ジョブが取得していないため）。
    過去分の複勝は fuku_odds.csv にあるので、必要になったら後付けする。

実行: python analyze_monthly_oos.py [YYYY-MM]  → monthly_oos_YYYYMM.md
      月1回、行数確認のついでに走らせる想定。
"""
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAP_MIN, ODDS_MAX, EV_TOP, EV_SUB = 3.0, 20.0, 1.7, 2.2


def load_and_merge_jv_fuku(df, jv_fuku_path=None):
    """複勝オッズがあれば後付けし、単複乖離(SMI)を計算する。

    SMI = log(単勝_後/単勝_前) − log(複勝_後/複勝_前)
      比ではなく対数比の差にするのは、複勝オッズが1.0〜1.2倍に張り付きやすく、
      Δ複勝を分母に置くと発散して外れ値がモデルを支配するため。
      対数比の差なら発散せず、符号の意味（単勝だけ売れている＝正）も保たれる。

    現状は複勝の時系列が無いので NaN を入れて素通りする。
    JRA-VANは更新しないため、過去分(fuku_odds.csv)を使うときだけ有効になる。
    """
    if not jv_fuku_path or not os.path.exists(jv_fuku_path):
        df["smi"] = np.nan
        return df
    # 将来ここで race_id + 馬番 で結合し、上式で smi を作る
    df["smi"] = np.nan
    return df


def load_drift(month=None):
    """7分前と確定のオッズが両方ある馬。month='2026-08' で絞れる。"""
    o = pd.read_csv(os.path.join(BASE_DIR, "odds_history.csv"), dtype={"race_id": str})
    o["t"] = pd.to_datetime(o["記録時刻"], errors="coerce")
    if "ジョブ" in o.columns:
        # 「分前」が入っている行があれば直前の記録だけを使う（2026-08-11以降）
        pre = o[o["ジョブ"].astype(str) == "直前"]
        o = pre if len(pre) else o
    last = (o.sort_values("t").groupby(["race_id", "馬名"]).tail(1)
            [["race_id", "馬名", "単勝オッズ", "t"]]
            .rename(columns={"単勝オッズ": "odds_pre"}))
    # ⚠ usecols は元ファイルの列順で返る。必ず rename で対応付ける
    rf = pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"), low_memory=False,
                     dtype={"race_id": str},
                     usecols=["race_id", "馬名", "単勝オッズ", "人気", "着順_num",
                              "is_turf"]) \
        .rename(columns={"単勝オッズ": "odds_fin", "人気": "pop", "着順_num": "着"})
    m = last.merge(rf, on=["race_id", "馬名"], how="inner")
    m = m.dropna(subset=["odds_pre", "odds_fin", "pop"])
    m = m[(m.odds_pre > 0) & (m.odds_fin > 0)].copy()
    if month:
        m = m[m["t"].dt.strftime("%Y-%m") == month]
    m["drift"] = (m.odds_fin - m.odds_pre) / m.odds_pre * 100
    m["lr"] = np.log(m.odds_fin / m.odds_pre)
    return load_and_merge_jv_fuku(m, os.path.join(BASE_DIR, "fuku_odds.csv"))


def sec1_drift(m):
    """① 下落率を人気帯別・着順別に。"""
    out = ["## ① 7分前 → 確定オッズの変化", ""]
    if m.empty:
        return out + ["対象データなし", ""]
    out += [f"対象 {len(m)}頭 / {m.race_id.nunique()}レース", "",
            "| 人気 | 頭数 | 中央値 | 平均 | 標準偏差 |", "|---|---|---|---|---|"]
    for lo, hi, lbl in [(1, 3, "1-3番人気"), (4, 6, "4-6番人気"), (7, 99, "7番人気以下")]:
        s = m[(m["pop"] >= lo) & (m["pop"] <= hi)]
        if len(s) < 10:
            continue
        out.append(f"| {lbl} | {len(s)} | {s.drift.median():+.1f}% | "
                   f"{s.drift.mean():+.1f}% | {s.drift.std():.1f}% |")
    out += ["", "| 着順 | 頭数 | 中央値 | 平均 |", "|---|---|---|---|"]
    for cond, lbl in [((m["着"] == 1), "1着"),
                      ((m["着"].between(2, 3)), "2-3着"),
                      ((m["着"] > 3), "着外")]:
        s = m[cond]
        if len(s) < 10:
            continue
        out.append(f"| {lbl} | {len(s)} | {s.drift.median():+.1f}% | {s.drift.mean():+.1f}% |")
    w, l = m[m["着"] == 1], m[m["着"] != 1]
    if len(w) >= 10 and len(l) >= 10:
        p = stats.mannwhitneyu(w.drift, l.drift, alternative="less").pvalue
        out += ["", f"**逆選択の検定**: 1着馬のほうが下がる p={p:.4f}"
                    f"（{'有意' if p < 0.05 else '有意でない'}）",
                "", "勝つ馬ほど直前に買われて下がる＝当たったときほど配当が渋くなる。"]
    return out + [""]


def sec2_roi(month):
    """② 候補案の月次回収率（7分前 vs 確定）。"""
    out = ["## ② 買い方別の回収率（7分前で買う場合と確定で買う場合）", ""]
    hp = os.path.join(BASE_DIR, "history_marks.csv")
    if not os.path.exists(hp):
        return out + ["history_marks.csv がまだありません", ""]
    h = pd.read_csv(hp, dtype={"race_id": str})
    if month:
        h = h[h["日付"].astype(str).str[:7] == month]
    if h.empty:
        return out + [f"{month} のデータがありません", ""]

    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    pay = {(r.race_id, r.券種, r.組み合わせ): r.払戻金 for r in jv.itertuples()}

    h["bn"] = pd.to_numeric(h["馬番"], errors="coerce").astype("Int64") \
        .astype(str).str.zfill(2)
    h["odds_fin"] = pd.to_numeric(h.get("確定単勝オッズ"), errors="coerce")
    h["odds_pre"] = pd.to_numeric(h.get("単勝オッズ"), errors="coerce")
    h["mr"] = pd.to_numeric(h.get("MF複勝順位"), errors="coerce")
    h["p"] = pd.to_numeric(h.get("MF勝ち確率"), errors="coerce")
    if h["odds_fin"].isna().all():
        out.append("※ 確定オッズの記録が無い期間です（2026-08-10以前）。7分前基準のみ表示")

    def roi(col, pop_col, umatan_pop=3):
        pts = []
        for rid, g in h.groupby("race_id", sort=False):
            od = g[col]
            if od.isna().all():
                continue
            ev = g["p"] / g["p"].sum() * od if g["p"].sum() > 0 else od * 0
            gap = pd.to_numeric(g[pop_col], errors="coerce") - g["mr"]
            ok = ((gap >= GAP_MIN) & (od <= ODDS_MAX) &
                  (((g["mr"] == 1) & (ev >= EV_TOP)) |
                   (g["mr"].between(2, 5) & (ev >= EV_SUB))))
            c = g[ok.fillna(False)]
            if not len(c):
                continue
            ax = c.assign(_e=ev[c.index]).sort_values("_e", ascending=False).bn.iloc[0]
            cost, ret = 1000.0, pay.get((rid, "単勝", ax), 0.0) * 10
            pr = pd.to_numeric(g[pop_col], errors="coerce").rank(method="first")
            for b in g.bn[g["mr"].isin([1, 2, 3, 4, 5]) & (pr <= umatan_pop)]:
                if b == ax:
                    continue
                cost += 500
                ret += pay.get((rid, "馬単", f"{ax}-{b}"), 0.0) * 5
            pts.append((cost, ret))
        if not pts:
            return None
        a = np.array(pts, float)
        return len(a), a[:, 1].sum() / a[:, 0].sum() * 100, int((a[:, 1] > 0).sum())

    out += ["| 買い方 | 基準 | R数 | 的中 | 回収率 |", "|---|---|---|---|---|"]
    for name, pop_kw in [("現行(相手3位以内)", 3), ("候補: 相手2位以内", 2)]:
        for col, pc, lbl in [("odds_pre", "人気", "7分前"),
                             ("odds_fin", "確定人気" if "確定人気" in h.columns else "人気",
                              "確定")]:
            r = roi(col, pc, pop_kw)
            if r:
                out.append(f"| {name} | {lbl} | {r[0]} | {r[2]} | {r[1]:.1f}% |")
    out += ["", "※ 月単位では的中数が一桁になる。**単月の数字で判断しないこと。**", ""]
    return out


def sec3_gap(month):
    """③ 乖離スコアと着順の相関。"""
    out = ["## ③ 乖離と着順の関係", ""]
    hp = os.path.join(BASE_DIR, "history_marks.csv")
    if not os.path.exists(hp):
        return out + ["history_marks.csv がまだありません", ""]
    h = pd.read_csv(hp, dtype={"race_id": str})
    if month:
        h = h[h["日付"].astype(str).str[:7] == month]
    g = pd.to_numeric(h.get("乖離"), errors="coerce")
    c = pd.to_numeric(h.get("着順"), errors="coerce")
    d = pd.DataFrame({"gap": g, "chaku": c}).dropna()
    if len(d) < 50:
        return out + ["検体が足りません", ""]
    rho, p = stats.spearmanr(d.gap, d.chaku)
    out += [f"対象 {len(d)}頭", "",
            f"**スピアマン順位相関 ρ = {rho:+.3f}**（p={p:.4f}）", "",
            "乖離が大きいほど着順が良いなら負の相関になる。", "",
            "| 乖離 | 頭数 | 勝率 | 複勝率 |", "|---|---|---|---|"]
    for lo, hi, lbl in [(-99, 0, "0以下"), (0, 2, "1-2"), (3, 4, "3-4"), (5, 99, "5以上")]:
        s = h[(g >= lo) & (g <= hi)]
        if len(s) < 20:
            continue
        out.append(f"| {lbl} | {len(s)} | {s['1着'].mean()*100:.1f}% | "
                   f"{s['3着内'].mean()*100:.1f}% |")
    return out + [""]


def main():
    month = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m")
    print(f"対象月: {month}")
    lines = [f"# 月次レポート {month}", "",
             f"生成 {datetime.now():%Y-%m-%d %H:%M}", "",
             "実運用（7分前オッズで買う）と検証（確定オッズで買う）の乖離を追う。",
             "", "---", ""]
    lines += sec1_drift(load_drift(month))
    lines += ["---", ""] + sec2_roi(month)
    lines += ["---", ""] + sec3_gap(month)
    lines += ["---", "",
              "## 読み方", "",
              "- 単月では的中が一桁になる。**傾向を見るもので、判断材料ではない**",
              "- ①の逆選択が消えたら市場構造が変わった合図。注意して見る",
              "- ②の7分前と確定の差が、そのままスリッページ損失",
              ""]
    out = os.path.join(BASE_DIR, f"monthly_oos_{month.replace('-', '')}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n保存 → {out}")


if __name__ == "__main__":
    sys.exit(main())
