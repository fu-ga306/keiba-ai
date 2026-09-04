# -*- coding: utf-8 -*-
"""上限を外してもなお一致しない列を特定する（2026-09-03）

履歴の上限を外すと一致率は88.4%→93.9%まで上がるが、そこで頭打ちになる。
残る約6%は上限とは別の原因。**それが何かを名指しする。**

⚠ 1レースあたり4分かかる。1レースだけで測る。
"""
import sys

for _s in (sys.stdout, sys.stderr):
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


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    sys.path.insert(0, BASE_DIR)
    import features as F

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    model_cols = m["use_cols"]

    clean = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"),
                        dtype={"race_id": str}, low_memory=False)
    clean["race_id"] = clean["race_id"].astype(str)
    rid = sorted(set(clean.loc[clean.race_id.str.startswith("2026"), "race_id"]))[-1]
    log(f"  対象 {rid}")

    need = list(set(model_cols) | {"race_id", "馬番"})
    parts = []
    for ch in pd.read_csv(os.path.join(BASE_DIR, "race_features.csv"),
                          usecols=lambda c: c in need, dtype={"race_id": str},
                          chunksize=200000, low_memory=False):
        ch["race_id"] = ch["race_id"].str.replace(r"\.0$", "", regex=True)
        x = ch[ch.race_id == rid]
        if len(x):
            parts.append(x)
    BT = pd.concat(parts)
    BT["bn"] = pd.to_numeric(BT["馬番"], errors="coerce")

    g = clean[clean.race_id == rid].copy()
    hist = clean[clean.race_id != rid].copy()
    race_df = g.drop(columns=[c for c in ("着順", "着順_num", "タイム", "上り",
                                          "通過", "着差", "賞金") if c in g.columns])
    F._MAX_PRED_HISTORY_OVERRIDE = None      # 制限なし
    log("  特徴量を計算（4分ほど・履歴の制限なし）")
    pdf = F.build_features_for_prediction(race_df, hist)
    pdf = pdf.copy()
    pdf["bn"] = pd.to_numeric(pdf["馬番"], errors="coerce")
    mg = pdf.merge(BT, on="bn", suffixes=("_本番", "_BT"))
    log(f"  突き合わせ {len(mg)}頭")

    bad = []
    for c in model_cols:
        a = pd.to_numeric(mg.get(c + "_本番"), errors="coerce")
        d = pd.to_numeric(mg.get(c + "_BT"), errors="coerce")
        if a is None or d is None:
            bad.append((c, "列が無い", np.nan, np.nan, np.nan))
            continue
        v = a.notna() & d.notna()
        if v.sum() < 3:
            bad.append((c, "比較できる行が少ない", a.isna().mean() * 100,
                        d.isna().mean() * 100, np.nan))
            continue
        agree = ((a[v] - d[v]).abs() < 1e-6).mean()
        if agree <= 0.999:
            bad.append((c, "値が違う", a.isna().mean() * 100, d.isna().mean() * 100,
                        (a[v] - d[v]).abs().mean()))

    log("")
    log(f"  === 制限を外しても一致しない列 {len(bad)} / {len(model_cols)} ===")
    log("  %-30s %-16s %8s %8s %10s" % ("列", "理由", "本番欠損", "BT欠損", "平均差"))
    log("  " + "-" * 76)
    for c, why, na1, na2, dif in bad:
        log("  %-30s %-16s %7.1f%% %7.1f%% %10.4f"
            % (c[:30], why, na1 if na1 == na1 else -1, na2 if na2 == na2 else -1,
               dif if dif == dif else -1))

    # 系統をまとめる
    import re
    log("")
    log("  === 系統ごと ===")
    pats = {"騎手": r"騎手", "調教師": r"調教師", "馬主": r"馬主",
            "賞金": r"賞金", "脚質・展開": r"脚質|先行|逃げ|差し|位置|バイアス",
            "上り・指数": r"上り|上指|タイム|速度", "その他": r"."}
    seen = set()
    for lab, p in pats.items():
        hit = [c for c, *_ in bad if re.search(p, c) and c not in seen]
        seen |= set(hit)
        if hit:
            log(f"    {lab:<12} {len(hit):>3}列  {hit[:5]}")


if __name__ == "__main__":
    main()
