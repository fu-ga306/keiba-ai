# -*- coding: utf-8 -*-
"""列のズレが「予想の結果」を変えるかを測る（2026-09-04）

列の一致率だけ見ても、実際に買う馬が変わらなければ実害はない。
逆に1列でも軸が変われば実害がある。**そこを直接測る。**

測ること
  同じレースについて
    ① BTの特徴量（race_features.csv）でモデルを回した結果
    ② 本番の経路で作った特徴量でモデルを回した結果
  を比べ、gapの順位と軸馬が一致するかを見る。

  上り基準の修正（2026-09-04）を入れた状態で測る。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK = ["着順", "着順_num", "タイム", "上り", "通過", "着差", "賞金"]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F
    import resid_io

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    ids = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))
    meet = ids[-1][:10]
    targets = sorted([r for r in ids if r[:10] == meet])
    log(f"  同じ開催日の {len(targets)}レース")

    race_df = clean[clean.race_id.isin(targets)].copy()
    race_df = race_df.drop(columns=[c for c in MASK if c in race_df.columns])
    hist = clean[~clean.race_id.isin(targets)].copy()

    F._MAX_PRED_HISTORY_OVERRIDE = None
    t0 = time.time()
    live = F.build_features_for_prediction(race_df, hist)
    log(f"  本番経路で計算 {time.time()-t0:.0f}秒")

    need = list(set(cols) | {"race_id", "馬番", "単勝オッズ"})
    parts = []
    for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                          usecols=lambda c: c in need, dtype={"race_id": str},
                          chunksize=200000, low_memory=False):
        ch["race_id"] = ch["race_id"].str.replace(r"\.0$", "", regex=True)
        x = ch[ch.race_id.isin(targets)]
        if len(x):
            parts.append(x)
    BT = pd.concat(parts)
    log(f"  BT側 {len(BT)}頭")

    # 同じオッズを使う（オッズ差の影響を除くため、BT側のオッズで揃える）
    live = live.copy()
    live["bn"] = pd.to_numeric(live["馬番"], errors="coerce")
    BT = BT.copy()
    BT["bn"] = pd.to_numeric(BT["馬番"], errors="coerce")
    od = BT[["race_id", "bn", "単勝オッズ"]].rename(columns={"単勝オッズ": "_odds"})
    live = live.merge(od, on=["race_id", "bn"], how="left")
    live["単勝オッズ"] = live["_odds"]

    res = {}
    for lab, d in (("BT", BT), ("本番", live)):
        g = resid_io.predict_gap(m, d)
        if g is None:
            log(f"  {lab}: gapを計算できませんでした")
            return
        res[lab] = g[["race_id", "bn", "gap"]].copy()

    a = res["BT"].rename(columns={"gap": "gap_BT"})
    b = res["本番"].rename(columns={"gap": "gap_本番"})
    mg = a.merge(b, on=["race_id", "bn"])
    log(f"  突き合わせ {len(mg)}頭")

    log("")
    log("  === gapの一致 ===")
    d = (mg["gap_BT"] - mg["gap_本番"]).abs()
    log(f"    相関           {mg['gap_BT'].corr(mg['gap_本番']):.4f}")
    log(f"    平均差         {d.mean():.4f}")
    log(f"    差が0.05未満    {(d < 0.05).mean()*100:.1f}%")

    log("")
    log("  === 軸馬（gap最大）が一致するか ===")
    same = diff = 0
    ex = []
    for rid, g in mg.groupby("race_id"):
        ax_bt = g.loc[g["gap_BT"].idxmax()]
        ax_lv = g.loc[g["gap_本番"].idxmax()]
        if int(ax_bt["bn"]) == int(ax_lv["bn"]):
            same += 1
        else:
            diff += 1
            ex.append((rid, int(ax_bt["bn"]), ax_bt["gap_BT"],
                       int(ax_lv["bn"]), ax_lv["gap_本番"]))
    log(f"    同じ馬 {same} / {same+diff} レース（{same/(same+diff)*100:.1f}%）")
    for rid, b1, g1, b2, g2 in ex[:8]:
        log(f"      {rid}  BT={b1}番(gap{g1:.2f})  本番={b2}番(gap{g2:.2f})")

    log("")
    log("  === 買うかどうか（gap>=1.5）が一致するか ===")
    s2 = d2 = 0
    for rid, g in mg.groupby("race_id"):
        buy_bt = g["gap_BT"].max() >= 1.5
        buy_lv = g["gap_本番"].max() >= 1.5
        if buy_bt == buy_lv:
            s2 += 1
        else:
            d2 += 1
    log(f"    同じ判断 {s2} / {s2+d2} レース（{s2/(s2+d2)*100:.1f}%）")


if __name__ == "__main__":
    main()
