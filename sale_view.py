# -*- coding: utf-8 -*-
"""販売用の評価表を生成する（2026-08-27）

何を売るか
  **「当たる予想」ではなく「確率が正しい表」を売る。**

  実測180レース・2,333頭で、複勝確率の較正を確認している。
    予測 5.4% → 実際 5.6%
    予測24.5% → 実際23.9%
    予測44.6% → 実際46.6%
  「30%と出た馬は実際に30%来る」が検証できている。これが商品の芯。

なぜ「儲かる」と書かないか
  ① 事実として儲からない。AIの複勝1位は平均2.3番人気で市場とほぼ同じ。
     的中率を追うと人気馬に寄り、控除率20%のぶん負ける
  ② 景品表示法。実測29点・的中1本しかない段階で回収率を謳えば優良誤認になる
  ③ 用途が違う。買う人が自分で馬券を組むための材料であって、買い目の指示ではない

無料と有料の切り分け
  無料 : 印・上位3頭の複勝確率・オッズ  → 集客。これだけでも使える
  有料 : 全頭の複勝確率・能力6軸・市場との乖離 → 本気で組む人はこれが要る

実行
  python sale_view.py                    今日の全レースを生成
  python sale_view.py <race_id>          1レースだけ生成（確認用）
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "sale_view")
PRED = os.path.join(BASE_DIR, "today_predictions.csv")
HIST = os.path.join(BASE_DIR, "history_marks.csv")

FREE_TOP = 3          # 無料で見せる頭数
ABILITY = ["能力_勝負", "能力_安定", "能力_末脚", "能力_先行", "能力_距離", "能力_実績"]
ABI_LABEL = ["勝負", "安定", "末脚", "先行", "距離", "実績"]


def log(m):
    print(m, flush=True)


def calibration():
    """較正後の確率が、**学習に使っていない日**でどれだけ合ったかを出す。

    ⚠ 補正に使ったデータで測ると必ず良く見える。それは信用の担保にならない。
      最後の1開催日を学習から外し、その日だけで測る。
      売り物の根拠になる数字なので、ここは甘くしない。
    """
    if not os.path.exists(HIST):
        return None, 0, 0, ""
    try:
        import pickle
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None, 0, 0, ""
    h = pd.read_csv(HIST, dtype={"race_id": str})
    h["着順"] = pd.to_numeric(h["着順"], errors="coerce")
    h["p"] = pd.to_numeric(h.get("複勝確率"), errors="coerce")
    h["odds"] = pd.to_numeric(h.get("単勝オッズ"), errors="coerce")
    h["人気"] = pd.to_numeric(h.get("人気"), errors="coerce")
    h = h[h["着順"].notna() & h["p"].notna() & h["odds"].notna() & (h["odds"] > 1)]
    if len(h) < 400 or h["日付"].nunique() < 2:
        return None, 0, 0, ""
    h["y"] = (h["着順"] <= 3).astype(int)

    def feat(d):
        q = np.clip(d["p"].values, 1e-4, 1 - 1e-4)
        return np.column_stack([np.log(q / (1 - q)), np.log(d["odds"].values)])

    days = sorted(h["日付"].unique())
    tr, te = h[h["日付"] != days[-1]], h[h["日付"] == days[-1]]
    m = LogisticRegression(max_iter=1000).fit(feat(tr), tr["y"].values)
    te = te.copy()
    te["pc"] = m.predict_proba(feat(te))[:, 1]

    rows = []
    for lo, hi, lab in [(1, 1, "1番人気"), (2, 3, "2-3番人気"), (4, 5, "4-5番人気"),
                        (6, 8, "6-8番人気"), (9, 12, "9-12番人気"), (13, 99, "13番人気以下")]:
        g = te[(te["人気"] >= lo) & (te["人気"] <= hi)]
        if len(g) < 15:
            continue
        rows.append({"帯": lab, "頭数": len(g),
                     "予測": g["pc"].mean() * 100, "実際": g["y"].mean() * 100})
    return pd.DataFrame(rows), te.race_id.nunique(), len(te), days[-1]


def apply_calib(g):
    """複勝確率_較正 を足す。**既存の 複勝確率 は書き換えない。**

    なぜ新しい列にするか
      複勝確率は 評価ランク(build_grade) と ダッシュボード が使っている。
      上書きすると既存の挙動が変わる。販売用の表示だけが較正値を使う。
      買い判定(resid_io)は自前のgapを使うので、そもそも影響しない。

    較正が無ければ元の値のまま返す（欠けても止めない）。
    """
    import pickle
    fp = os.path.join(BASE_DIR, "place_calib.pkl")
    g = g.copy()
    if not os.path.exists(fp):
        g["複勝確率_較正"] = g["複勝確率"]
        g.attrs["calibrated"] = False
        return g
    try:
        with open(fp, "rb") as f:
            pk = pickle.load(f)
        p = np.clip(pd.to_numeric(g["複勝確率"], errors="coerce").values, 1e-4, 1 - 1e-4)
        o = pd.to_numeric(g["単勝オッズ"], errors="coerce").values
        X = np.column_stack([np.log(p / (1 - p)), np.log(np.clip(o, 1.01, None))])
        ok = np.isfinite(X).all(axis=1)
        out = np.where(ok, np.nan, np.nan)
        out[ok] = pk["model"].predict_proba(X[ok])[:, 1]
        g["複勝確率_較正"] = np.where(np.isnan(out), g["複勝確率"], out)
        g.attrs["calibrated"] = True
        g.attrs["calib_info"] = pk
    except Exception as e:
        print(f"  較正に失敗（元の値を使います）: {type(e).__name__}: {e}")
        g["複勝確率_較正"] = g["複勝確率"]
        g.attrs["calibrated"] = False
    return g


def _bar(v, w=5):
    """能力値を簡易バーにする。0-100 → ■の数。"""
    if pd.isna(v):
        return "―"
    n = int(round(float(v) / 100 * w))
    return "■" * n + "□" * (w - n)


def build_race(g, cal, ncal_r, ncal_h, cal_day=""):
    """1レース分の表示（無料版・有料版）を文字列で返す。"""
    g = g.copy()
    g["複勝確率"] = pd.to_numeric(g["複勝確率"], errors="coerce")
    g["人気"] = pd.to_numeric(g["人気"], errors="coerce")
    g["単勝オッズ"] = pd.to_numeric(g["単勝オッズ"], errors="coerce")
    g = apply_calib(g)                      # 販売用は較正した確率を使う
    g = g.sort_values("複勝確率_較正", ascending=False)

    jyo = g["jyo"].iloc[0] if "jyo" in g.columns else ""
    rno = g["race_no"].iloc[0] if "race_no" in g.columns else ""
    head = f"{jyo}{int(rno) if pd.notna(rno) else ''}R"

    L = []
    L.append(f"# {head}　AI評価表")
    L.append("")
    L.append("**3着以内に入る確率**の高い順に並べています。")
    L.append("")
    L.append("表には**2つの別々の見方**が入っています。混同しないでください。")
    L.append("")
    L.append("| | 見ているもの | 高い馬の特徴 |")
    L.append("|---|---|---|")
    L.append("| **3着内確率** | 堅さ。来やすさ | 人気馬に寄ります |")
    L.append("| **印（◎○▲△×）** | 妙味。市場との差 | 人気薄が混じります |")
    L.append("")
    L.append("**確率1位に印が付いていないことがあります。**")
    L.append("それは「堅いが、オッズ相応で妙味は無い」という意味です。故障ではありません。")
    L.append("")

    # ── 無料部分 ──
    L.append("## 上位3頭（無料）")
    L.append("")
    L.append("| 印 | 馬番 | 馬名 | 3着内確率 | 人気 | 単勝 |")
    L.append("|---|---|---|---|---|---|")
    for r in g.head(FREE_TOP).itertuples():
        mark = r.推奨ランク if isinstance(getattr(r, "推奨ランク", None), str) else "―"
        od = f"{r.単勝オッズ:.1f}倍" if pd.notna(r.単勝オッズ) else "―"
        pop = f"{int(r.人気)}番人気" if pd.notna(r.人気) else "―"
        L.append(f"| {mark} | {r.馬番} | {r.馬名} | **{r.複勝確率_較正*100:.1f}%** | {pop} | {od} |")
    L.append("")

    # ── 有料部分 ──
    L.append("<!-- ===== ここから有料 ===== -->")
    L.append("")
    L.append("## 全頭の評価")
    L.append("")
    L.append("| 印 | 馬番 | 馬名 | 3着内確率 | 人気 | 単勝 | 市場との差 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in g.itertuples():
        mark = r.推奨ランク if isinstance(getattr(r, "推奨ランク", None), str) else "―"
        od = f"{r.単勝オッズ:.1f}" if pd.notna(r.単勝オッズ) else "―"
        pop = f"{int(r.人気)}" if pd.notna(r.人気) else "―"
        gap = getattr(r, "resid_gap", np.nan)
        # gap = AIの見立て ÷ 市場の見立て。1.0が市場と同じ
        gs = "―"
        if pd.notna(gap):
            gs = f"{gap:.2f}" + ("　高く見ている" if gap >= 1.3 else
                                 "　低く見ている" if gap <= 0.8 else "")
        L.append(f"| {mark} | {r.馬番} | {r.馬名} | {r.複勝確率_較正*100:.1f}% | {pop} | {od} | {gs} |")
    L.append("")

    L.append("## 能力（オッズを一切見ずに付けた点数）")
    L.append("")
    L.append("市場の評価が入っていない、実力だけの点数です。")
    L.append("人気とズレている馬を探す材料になります。")
    L.append("")
    L.append("| 馬番 | 馬名 | " + " | ".join(ABI_LABEL) + " |")
    L.append("|---|---|" + "---|" * len(ABI_LABEL))
    for r in g.itertuples():
        cells = []
        for c in ABILITY:
            v = getattr(r, c, np.nan)
            cells.append(f"{_bar(v)} {int(v) if pd.notna(v) else '―'}")
        L.append(f"| {r.馬番} | {r.馬名} | " + " | ".join(cells) + " |")
    L.append("")

    # ── 較正の実測（毎回載せる。信用の担保）──
    if cal is not None:
        L.append("## この確率はどれくらい当たっているか")
        L.append("")
        L.append(f"**補正に使っていない日**（{cal_day}・{ncal_r}レース・{ncal_h:,}頭）で")
        L.append("検証した結果です。学習に使ったデータで測ると必ず良く見えるので、")
        L.append("わざと外した日で測っています。")
        L.append("")
        L.append("| 人気 | 頭数 | 予測した確率 | 実際に3着以内 | 差 |")
        L.append("|---|---|---|---|---|")
        w = 0.0
        for r in cal.itertuples():
            d = r.実際 - r.予測
            w = max(w, abs(d))
            L.append(f"| {r.帯} | {r.頭数} | {r.予測:.1f}% | **{r.実際:.1f}%** | {d:+.1f} |")
        L.append("")
        L.append(f"最大のズレは {w:.1f}ポイントでした。")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**この表について**")
    L.append("")
    L.append("- 3着以内に入る確率を予測したものです。**馬券の的中・利益を保証するものではありません**")
    L.append("- 確率が高い馬は人気馬に寄ります。的中率と回収率は別物です")
    L.append("- どう買うかはご自身でご判断ください。買い目の指示は含みません")
    L.append("- 20歳未満の方は勝馬投票券を購入できません")
    return "\n".join(L)


def main():
    if not os.path.exists(PRED):
        log(f"{os.path.basename(PRED)} がありません。予想を1回まわしてください。")
        return
    d = pd.read_csv(PRED, dtype={"race_id": str})
    if "複勝確率" not in d.columns:
        log("複勝確率の列がありません。")
        return
    cal, ncal_r, ncal_h, cal_day = calibration()
    if cal is None:
        log("⚠ 較正の実測が出せません（history_marks.csv が不足）。")
        log("  この表を載せられないなら、販売はまだ早いです。信用の担保が無くなります。")

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = sys.argv[1:] or list(d.race_id.unique())
    n = 0
    for rid in targets:
        g = d[d.race_id == str(rid)]
        if g.empty:
            log(f"  {rid}: データなし")
            continue
        body = build_race(g, cal, ncal_r, ncal_h, cal_day)
        fp = os.path.join(OUT_DIR, f"{rid}.md")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(body)
        n += 1
    log(f"  {n}レース分を {OUT_DIR} に生成しました")
    if n and len(targets) <= 3:
        log("\n" + "=" * 60)
        log(body)


if __name__ == "__main__":
    main()
