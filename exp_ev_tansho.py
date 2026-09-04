# -*- coding: utf-8 -*-
"""単勝を、較正済みの1着確率による期待値で選ぶ（2026-09-04）

■ 事前登録（数字を見る前に固定する）

なぜ単勝に絞るか
  2頭の同時確率は較正しても 実際÷推定 = 0.876 までしか合わなかった。
  1頭ぶん（p1）は 9.0pt → 3.4pt まで合っている。
  **合っている材料だけを使う。**

何が変わるか
  いま   gap = p1/q ≥ 1.5
         しきい値1.5に意味がない。経験的に決めた数字
  案    EV = p1_cal × 単勝オッズ ≥ 1.0
         「賭け金を上回る」という意味を持つ

  ⚠ gap と EV はほぼ同じもの（EV ≈ gap ÷ 控除率の逆数）。
    違いは①較正で確率が実測に合うこと ②しきい値に意味があること。

試すこと（この4つだけ。あとから足さない）
  A 現行         gap >= 1.5
  B EV >= 1.0    賭け金を上回ると見込めるものを買う
  C EV >= 1.1    余裕を持たせる
  D EV >= 1.2    さらに絞る

  ⚠ B/C/D のしきい値は**理屈で決めた値**であって、
    結果を見て一番良いものを選ぶのではない。全部載せる。

枠組み
  開発  2022-2024（p1_calが作れる年）
  評価  2025      最後に1回だけ

採用の規則
  1. 95%区間が100%を上回ること
  2. 区間が重なるなら、証明までの月数が最短のもの
  3. 現行（A）が候補に残るなら現行を優先。同等なら変えない
  4. 選んだものを2025で1回だけ測る。下がっても選び直さない
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEV = [2022, 2023, 2024]
HOLD = 2025
RNG = np.random.default_rng(20260904)


def log(m):
    print(m, flush=True)


def payouts():
    jv = pd.read_csv(os.path.join(BASE_DIR, "jv_payouts.csv"), dtype=str)
    jv = jv[jv.券種 == "単勝"].copy()
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    jv["bn"] = pd.to_numeric(jv["組み合わせ"], errors="coerce")
    return jv[["race_id", "bn", "払戻金"]]


def stat(ret, months):
    n = len(ret)
    if n < 100:
        return None
    s = RNG.choice(ret, size=(3000, n)).mean(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])

    def need(hi_n=60000):
        if ret.mean() <= 100:
            return None
        def bl(k):
            return np.percentile(RNG.choice(ret, size=(1200, k)).mean(axis=1), 2.5)
        if bl(hi_n) <= 100:
            return None
        a, b = 100, hi_n
        while a < b:
            mid = (a + b) // 2
            if bl(mid) > 100:
                b = mid
            else:
                a = mid + 1
        return a

    nd = need()
    return dict(n=n, roi=ret.mean(), hit=(ret > 0).mean() * 100, lo=lo, hi=hi,
                need=nd, months=(nd / (n / months) if nd else None))


def select(d, PAY, mode, th):
    """各レースで1頭選び、単勝の払戻を返す。軸はいずれも gap 最大の1頭。"""
    g = d.sort_values(["race_id", "gap"], ascending=[True, False])
    ax = g.groupby("race_id").head(1).copy()
    if mode == "gap":
        ax = ax[ax["gap"] >= th]
    else:
        ax = ax[ax["EV"] >= th]
    ax["bn"] = pd.to_numeric(ax["bn"], errors="coerce")
    m = ax.merge(PAY, on=["race_id", "bn"], how="left")
    return m["払戻金"].fillna(0.0).to_numpy()


def main():
    d = pd.read_csv(os.path.join(BASE_DIR, "resid_kinds_pred_cal.csv"),
                    dtype={"race_id": str, "bn": str})
    d["着"] = pd.to_numeric(d["着"], errors="coerce")
    d = d[d["着"].notna()].copy()
    d["gap"] = d["p1"] / d["q"]
    d["EV"] = d["p1_cal"] * d["odds"]
    PAY = payouts()

    log("  ■ 較正の効きを確認（開発期間）")
    dv = d[d["年"].isin(DEV)]
    ax = dv.sort_values(["race_id", "gap"], ascending=[True, False]).groupby("race_id").head(1)
    log(f"    軸 {len(ax):,}頭  p1_cal平均 {ax['p1_cal'].mean()*100:.1f}%  "
        f"実際の勝率 {(ax['着']==1).mean()*100:.1f}%")
    log(f"    EV平均 {ax['EV'].mean():.3f}（1.0で収支トントンの見込み）")
    log("")

    CASES = [("A 現行 gap>=1.5", "gap", 1.5),
             ("B EV>=1.0", "ev", 1.0),
             ("C EV>=1.1", "ev", 1.1),
             ("D EV>=1.2", "ev", 1.2)]

    log(f"  === 開発 {DEV}（2025は見ていない） ===")
    log("  %-18s %7s %7s %8s %17s %10s" %
        ("案", "点数", "的中率", "回収率", "95%区間", "証明まで"))
    log("  " + "-" * 74)
    rows = []
    for lab, mode, th in CASES:
        ret = select(dv, PAY, mode, th)
        st = stat(ret, 36.0)
        if st is None:
            log(f"  {lab:<18} 点数が少なすぎます")
            continue
        mo = f"{st['months']:.0f}か月" if st["months"] else "届かない"
        log("  %-18s %7d %6.1f%% %7.1f%% [%5.1f, %6.1f] %10s %s"
            % (lab, st["n"], st["hit"], st["roi"], st["lo"], st["hi"], mo,
               "○" if st["lo"] > 100 else ""))
        rows.append({"案": lab, "mode": mode, "th": th, **st})

    R = pd.DataFrame(rows)
    ok = R[R["lo"] > 100]
    log("")
    log(f"  95%区間が100%を上回った案: {len(ok)} / {len(R)}")
    if len(ok) == 0:
        log("    無し。規則3により何も変えない。")
        return
    cand = ok.dropna(subset=["months"]).sort_values("months")
    log("  規則2: 証明までが最短の順")
    for r in cand.itertuples():
        log(f"    {r.案:<18} {r.months:.0f}か月")
    if "A 現行 gap>=1.5" in set(ok["案"]):
        log("  規則3: 現行も候補に残っているので**現行を優先**")
        pick = R[R["案"] == "A 現行 gap>=1.5"].iloc[0]
    else:
        pick = cand.iloc[0]
        log(f"  → 現行は候補外。候補は {pick['案']}")

    log("")
    log(f"  === 評価 {HOLD}（一度きり） ===")
    hd = d[d["年"] == HOLD]
    for lab, mode, th in CASES:
        ret = select(hd, PAY, mode, th)
        st = stat(ret, 12.0)
        if st is None:
            continue
        log("  %-18s %7d %6.1f%% %7.1f%% [%5.1f, %6.1f] %s"
            % (lab, st["n"], st["hit"], st["roi"], st["lo"], st["hi"],
               "← 採用案" if lab == pick["案"] else ""))


if __name__ == "__main__":
    main()
