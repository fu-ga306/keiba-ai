# -*- coding: utf-8 -*-
"""週次の販売note下書きを生成する（2026-08-27）

考え方：煽らずに、具体性で売る
  この発信の唯一の資産は「都合の悪い数字も出す人」という信頼です。
  煽った瞬間にそれが消えて、他の予想サイトと区別がつかなくなります。

  代わりに使うのは**検証可能な具体性**です。
    ・「確率50%以上と出した馬は、先週57頭中32頭（56%）が3着以内」
    ・較正の実測表（補正に使っていない日で検証）
    ・◎の馬券内率51.4% / ×は14.8%
  これは全部あとから確かめられる数字で、嘘がありません。
  そして**他の予想サイトはこれを出せません。**

  さらに「儲かるとは言いません」と先に書きます。逆説的ですが、
  これが一番効きます。読む人は誇大広告に慣れていて、警戒しています。

書いてはいけないこと（景表法）
  「必ず」「絶対」「儲かる」「回収率◯%」
  実測29点・的中1本で回収率を謳えば優良誤認です。事実としても儲かりません。

実行
  python note_weekly.py            今週の下書きを生成
  python note_weekly.py --print    標準出力に出すだけ
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE_DIR, "history_marks.csv")
CALIB = os.path.join(BASE_DIR, "place_calib.pkl")
OUT_DIR = os.path.join(BASE_DIR, "note_weekly")


def log(m):
    print(m, flush=True)


def _calibrated(g):
    """較正後の確率を付ける。較正器が無ければ元の値。"""
    q = np.clip(pd.to_numeric(g["複勝確率"], errors="coerce").values, 1e-4, 1 - 1e-4)
    o = np.clip(pd.to_numeric(g["単勝オッズ"], errors="coerce").values, 1.01, None)
    if not os.path.exists(CALIB):
        return q
    with open(CALIB, "rb") as f:
        pk = pickle.load(f)
    X = np.column_stack([np.log(q / (1 - q)), np.log(o)])
    ok = np.isfinite(X).all(axis=1)
    out = q.copy()
    out[ok] = pk["model"].predict_proba(X[ok])[:, 1]
    return out


def gather():
    """記事に使う数字を実データから集める。**作文はしない。**"""
    if not os.path.exists(HIST):
        return None
    h = pd.read_csv(HIST, dtype={"race_id": str})
    h["着順"] = pd.to_numeric(h["着順"], errors="coerce")
    h["複勝確率"] = pd.to_numeric(h["複勝確率"], errors="coerce")
    h["単勝オッズ"] = pd.to_numeric(h["単勝オッズ"], errors="coerce")
    h["人気"] = pd.to_numeric(h["人気"], errors="coerce")
    h = h[h["着順"].notna() & h["複勝確率"].notna() & h["単勝オッズ"].notna()]
    if len(h) < 300:
        return None
    h["pc"] = _calibrated(h)
    h["in3"] = (h["着順"] <= 3).astype(int)

    last = sorted(h["日付"].unique())[-1]
    g = h[h["日付"] == last]

    bands = []
    for lo in (0.5, 0.4, 0.3):
        s = g[g["pc"] >= lo]
        if len(s) >= 5:
            bands.append({"th": int(lo * 100), "n": len(s),
                          "hit": int(s["in3"].sum()),
                          "rate": s["in3"].mean() * 100})

    def mark(m):
        x = h[h["推奨ランク"].astype(str) == m]
        return {"n": len(x), "in3": x["in3"].mean() * 100,
                "win": (x["着順"] == 1).mean() * 100} if len(x) >= 30 else None

    # 較正（補正に使っていない日で検証）
    days = sorted(h["日付"].unique())
    cal = []
    cal_day, cal_worst = "", 0.0
    if len(days) >= 2:
        try:
            from sklearn.linear_model import LogisticRegression

            def feat(d):
                q = np.clip(d["複勝確率"].values, 1e-4, 1 - 1e-4)
                return np.column_stack([np.log(q / (1 - q)),
                                        np.log(np.clip(d["単勝オッズ"].values, 1.01, None))])
            tr, te = h[h["日付"] != days[-1]], h[h["日付"] == days[-1]].copy()
            m = LogisticRegression(max_iter=1000).fit(feat(tr), tr["in3"].values)
            te["q"] = m.predict_proba(feat(te))[:, 1]
            cal_day = days[-1]
            for lo, hi, lab in [(1, 1, "1番人気"), (2, 3, "2-3番人気"),
                                (4, 5, "4-5番人気"), (6, 8, "6-8番人気"),
                                (9, 12, "9-12番人気"), (13, 99, "13番人気以下")]:
                s = te[(te["人気"] >= lo) & (te["人気"] <= hi)]
                if len(s) < 15:
                    continue
                pr, ac = s["q"].mean() * 100, s["in3"].mean() * 100
                cal_worst = max(cal_worst, abs(ac - pr))
                cal.append({"band": lab, "n": len(s), "pred": pr, "act": ac})
        except Exception:
            pass

    return {"last": last, "races": int(g.race_id.nunique()), "horses": len(g),
            "bands": bands, "maru": mark("◎"), "batsu": mark("×"),
            "cal": cal, "cal_day": cal_day, "cal_worst": cal_worst,
            "total_races": int(h.race_id.nunique()), "total_horses": len(h)}


def build(d, url, phrase, label):
    L = []
    A = L.append

    A(f"# 【{label}】AIが全出走馬の「3着以内に入る確率」を出します")
    A("")
    # ── フック：先週の具体的な数字から入る ──
    if d["bands"]:
        b = d["bands"][0]
        A(f"先週（{d['last']}・{d['races']}レース）、"
          f"AIが「3着以内に入る確率{b['th']}%以上」と出した馬は **{b['n']}頭**。")
        A(f"そのうち **{b['hit']}頭が実際に3着以内**に入りました。**{b['rate']:.0f}%** です。")
    A("")
    A("この記事で売っているのは「当たる予想」ではありません。")
    A("**確率が正しい表**です。")
    A("")
    A("---")
    A("")

    # ── 何が見られるか ──
    A("## 何が見られるか")
    A("")
    A("開催日の全レース・全出走馬について、次が表示されます。")
    A("")
    A("| | 中身 |")
    A("|---|---|")
    A("| **3着内確率** | その馬が3着以内に入る確率。較正済み |")
    A("| **印 ◎○▲△×** | 市場との差（妙味）から付けた評価 |")
    A("| **能力6軸** | 勝負・安定・末脚・先行・距離・実績。**オッズを一切見ずに付けた点数** |")
    A("| **市場との差** | AIの見立て ÷ 市場の見立て |")
    A("")
    A("**上位3頭は無料で見られます。**購読すると全頭ぶんが表示されます。")
    A("")
    A("能力6軸は市場の評価が入っていないので、**人気とズレている馬**を探せます。")
    A("「人気は無いが末脚だけ突出している」といった馬が見つかります。")
    A("")

    # ── 信用の担保 ──
    if d["cal"]:
        A("## 「確率が正しい」とはどういうことか")
        A("")
        A(f"**補正に使っていない日**（{d['cal_day']}）で検証した結果です。")
        A("学習に使ったデータで測ると必ず良く見えるので、わざと外した日で測っています。")
        A("")
        A("| 人気 | 頭数 | AIの予測 | 実際に3着以内 | 差 |")
        A("|---|---|---|---|---|")
        for r in d["cal"]:
            A(f"| {r['band']} | {r['n']} | {r['pred']:.1f}% | "
              f"**{r['act']:.1f}%** | {r['act']-r['pred']:+.1f} |")
        A("")
        A(f"最大のズレは **{d['cal_worst']:.1f}ポイント**。")
        A("「40%と出た馬は、だいたい40%来る」という状態です。")
        A("")

    if d["maru"] and d["batsu"]:
        A(f"通算 {d['total_races']}レース・{d['total_horses']:,}頭での印の成績です。")
        A("")
        A("| 印 | 頭数 | 馬券内率 | 勝率 |")
        A("|---|---|---|---|")
        A(f"| ◎ | {d['maru']['n']} | **{d['maru']['in3']:.1f}%** | {d['maru']['win']:.1f}% |")
        A(f"| × | {d['batsu']['n']} | {d['batsu']['in3']:.1f}% | ― |")
        A("")

    # ── 正直な限界。ここが差別化になる ──
    A("---")
    A("")
    A("## 先に、正直に書いておきます")
    A("")
    A("**これを買っても儲かるとは言いません。**")
    A("")
    A("理由は単純で、確率が高い馬は人気馬に寄るからです。")
    A("当たりやすい馬を買えば、その情報はすでにオッズに入っています。")
    A("控除率のぶん、的中率だけを追えば負けます。")
    A("")
    A("では何のために使うのか。**自分で馬券を組むための材料**です。")
    A("")
    A("- どの馬が堅いか（3着内確率）")
    A("- どの馬に妙味があるか（印・市場との差）")
    A("- 人気とAIの評価がズレているのはどこか（能力6軸）")
    A("")
    A("この3つを見て、**買い方はご自身で決めてください。**")
    A("買い目の指示は含みません。")
    A("")
    A("私自身、このAIの買い目でまだ1円も賭けていません。")
    A("記録だけを取り続けていて、その記録も公開しています。")
    A("")

    # ── 購入導線 ──
    A("---")
    A("")
    A("## ご購入後のご案内")
    A("")
    A("以下のリンクを開き、合言葉を入力すると全頭の評価が表示されます。")
    A("")
    A(f"**閲覧ページ**")
    A(f"{url}")
    A("")
    A(f"**合言葉**： `{phrase}`")
    A("")
    A("※ 合言葉は毎週変わります。前の週のものも1週間は有効です。")
    A("※ レースごとにページが分かれています。開催日の朝に更新されます。")
    A("")
    A("---")
    A("")
    A("**この表について**")
    A("")
    A("- 3着以内に入る確率を予測したものです。**馬券の的中・利益を保証するものではありません**")
    A("- 確率が高い馬は人気馬に寄ります。的中率と回収率は別物です")
    A("- どう買うかはご自身でご判断ください。買い目の指示は含みません")
    A("- 20歳未満の方は勝馬投票券を購入できません")
    return "\n".join(L)


def main():
    import sale_gate
    d = gather()
    if d is None:
        log("  実績データが足りません。記事を作りません。")
        log("  （数字の裏付けが無い記事は出さない、という方針です）")
        return
    url = os.environ.get("SALE_URL", "")
    if not url:
        p = os.path.join(BASE_DIR, ".env")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8", errors="ignore"):
                if line.strip().startswith("SALE_URL="):
                    url = line.split("=", 1)[1].split("#", 1)[0].strip().strip('"')
    url = url or "（ダッシュボードのURL。.env に SALE_URL= を設定してください）"

    now = datetime.now()
    body = build(d, url, sale_gate.passphrase(now), sale_gate.period_label(now))
    if "--print" in sys.argv:
        log(body)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    y, w, _ = now.isocalendar()
    fp = os.path.join(OUT_DIR, f"{y}-W{w:02d}.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(body)
    log(f"  {os.path.relpath(fp, BASE_DIR)} を生成しました（{len(body)}文字）")
    log("\n" + body)


if __name__ == "__main__":
    main()
