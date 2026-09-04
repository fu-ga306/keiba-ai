# -*- coding: utf-8 -*-
"""本番とBTの一致を、繰り返し測れる形にする（2026-09-04）

なぜ要るか
  列の一致率（87%）より、**軸馬が一致するか**のほうが重い。
  実測で軸の一致は12レース中9（75%）だった。買う馬が4分の1違う。
  ただし12レースでは標本が足りず、直したかどうかも判定できない。

  1列直すたびに測り直すので、**同じ条件で繰り返せる形**にしておく。

やり方
  ・開催日を複数選び、日ごとにまとめて計算する
    （本番が知り得ないのは同じ日の分だけなので、その日だけ履歴から除く）
  ・まとめ計算の費用はレース数にほぼ依存しない
    実測 12レース539秒 / 36レース592秒
  ・結果をCSVに残し、修正の前後で比較できるようにする

実行
  python measure_parity.py                既定5開催日
  python measure_parity.py --days 3       開催日を3つに
  python measure_parity.py --tag before   結果に名前を付けて保存
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
    a = sys.argv
    n_days = int(a[a.index("--days") + 1]) if "--days" in a else 5
    tag = a[a.index("--tag") + 1] if "--tag" in a else datetime.now().strftime("%m%d_%H%M")

    sys.path.insert(0, BASE_DIR)
    import features as F
    import resid_io

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    ids = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))
    meets = sorted({r[:10] for r in ids})[-n_days:]
    log(f"  対象の開催日 {len(meets)}件: {meets}")

    need = list(set(cols) | {"race_id", "馬番", "単勝オッズ"})
    F._MAX_PRED_HISTORY_OVERRIDE = None

    rows, colstat = [], {}
    for meet in meets:
        targets = sorted([r for r in ids if r[:10] == meet])
        race_df = clean[clean.race_id.isin(targets)].copy()
        race_df = race_df.drop(columns=[c for c in MASK if c in race_df.columns])
        hist = clean[~clean.race_id.isin(targets)].copy()
        t0 = time.time()
        live = F.build_features_for_prediction(race_df, hist)
        log(f"  {meet}  {len(targets)}レース  {time.time()-t0:.0f}秒")

        parts = []
        for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                              usecols=lambda c: c in need, dtype={"race_id": str},
                              chunksize=200000, low_memory=False):
            ch["race_id"] = ch["race_id"].str.replace(r"\.0$", "", regex=True)
            x = ch[ch.race_id.isin(targets)]
            if len(x):
                parts.append(x)
        if not parts:
            continue
        BT = pd.concat(parts)
        BT["bn"] = pd.to_numeric(BT["馬番"], errors="coerce")
        live = live.copy()
        live["bn"] = pd.to_numeric(live["馬番"], errors="coerce")
        # オッズはBT側に揃える（オッズ差の影響を除いて特徴量の差だけを見る）
        live = live.merge(BT[["race_id", "bn", "単勝オッズ"]]
                          .rename(columns={"単勝オッズ": "_o"}),
                          on=["race_id", "bn"], how="left")
        live["単勝オッズ"] = live["_o"]

        # 列ごとの一致を貯める
        mgc = live.merge(BT, on=["race_id", "bn"], suffixes=("_L", "_B"))
        for c in cols:
            x = pd.to_numeric(mgc.get(c + "_L"), errors="coerce")
            y = pd.to_numeric(mgc.get(c + "_B"), errors="coerce")
            if x is None or y is None:
                continue
            v = x.notna() & y.notna()
            if v.sum() < 5:
                continue
            d = (x[v] - y[v]).abs()
            s = colstat.setdefault(c, {"n": 0, "ok": 0, "sum": 0.0})
            s["n"] += int(v.sum())
            s["ok"] += int((d < 1e-6).sum())
            s["sum"] += float(d.sum())

        gl = resid_io.predict_gap(m, live)
        gb = resid_io.predict_gap(m, BT)
        if gl is None or gb is None:
            continue
        mg = (gb[["race_id", "bn", "gap"]].rename(columns={"gap": "g_BT"})
              .merge(gl[["race_id", "bn", "gap"]].rename(columns={"gap": "g_LV"}),
                     on=["race_id", "bn"]))
        for rid, g in mg.groupby("race_id"):
            ab = g.loc[g["g_BT"].idxmax()]
            al = g.loc[g["g_LV"].idxmax()]
            rows.append({
                "race_id": rid, "頭数": len(g),
                "軸一致": int(ab["bn"]) == int(al["bn"]),
                "買い一致": (g["g_BT"].max() >= 1.5) == (g["g_LV"].max() >= 1.5),
                "gap差平均": float((g["g_BT"] - g["g_LV"]).abs().mean()),
                "軸gap_BT": float(ab["g_BT"]), "軸gap_本番": float(al["g_LV"]),
            })

    R = pd.DataFrame(rows)
    log("")
    log("  ================= 結果 =================")
    log(f"    レース数        {len(R)}")
    log(f"    軸が一致        {R['軸一致'].sum()} / {len(R)}  "
        f"（{R['軸一致'].mean()*100:.1f}%）")
    log(f"    買い判断が一致   {R['買い一致'].sum()} / {len(R)}  "
        f"（{R['買い一致'].mean()*100:.1f}%）")
    log(f"    gapの平均差     {R['gap差平均'].mean():.4f}")
    # 二項の95%区間（軸一致率の精度）
    p = R["軸一致"].mean()
    se = (p * (1 - p) / len(R)) ** 0.5
    log(f"    軸一致率の95%区間 [{max(0,p-1.96*se)*100:.1f}%, {min(1,p+1.96*se)*100:.1f}%]")

    C = pd.DataFrame([{"列": c, "一致率": s["ok"] / s["n"] * 100,
                       "平均差": s["sum"] / s["n"], "件数": s["n"]}
                      for c, s in colstat.items()]).sort_values("一致率")
    log("")
    log(f"    完全一致した列 {(C['一致率'] > 99.9).sum()} / {len(C)}")
    log("    一致率が低い列（上位15）")
    for r in C.head(15).itertuples():
        log("      %-30s 一致%6.1f%%  平均差 %.4f" % (r.列[:30], r.一致率, r.平均差))

    R.to_csv(os.path.join(BASE_DIR, f"parity_race_{tag}.csv"),
             index=False, encoding="utf-8-sig")
    C.to_csv(os.path.join(BASE_DIR, f"parity_col_{tag}.csv"),
             index=False, encoding="utf-8-sig")
    log(f"\n    parity_race_{tag}.csv / parity_col_{tag}.csv に保存")


if __name__ == "__main__":
    main()
