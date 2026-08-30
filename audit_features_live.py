# -*- coding: utf-8 -*-
"""本番とBTで、特徴量の欠損率がどれだけ違うかを全列で見る（2026-08-30）

BT は race_features.csv を読み、本番は build_features_for_prediction が
その場で計算する。同じ323列を別のコードが作っているので、
片方だけ欠けている列が出る。それを全部あぶり出す。

  python audit_features_live.py            直近の記録で比較
  python audit_features_live.py --after HH:MM   その時刻以降の記録だけ使う
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pickle
import numpy as np
import pandas as pd

def log(m):
    print(m, flush=True)

after = None
if "--after" in sys.argv:
    after = sys.argv[sys.argv.index("--after") + 1]

s = pd.read_csv("pred_features.csv", dtype={"race_id": str}, low_memory=False)
if after:
    s = s[s["記録時刻"].astype(str).str[11:16] >= after]
m = pickle.load(open("model_resid.pkl", "rb"))
cols = [c for c in m["use_cols"] if c in s.columns]
log("  本番の記録 %d頭 / %dレース  対象 %d列" % (len(s), s.race_id.nunique(), len(cols)))

na_live = s[cols].apply(lambda x: pd.to_numeric(x, errors="coerce")).isna().mean()

# BT側は列を小分けにして読む（一括だとメモリ不足で落ちる）
BATCH = 40
tot = pd.Series(0.0, index=cols)
n = 0
for i in range(0, len(cols), BATCH):
    part = cols[i:i + BATCH]
    cnt = 0
    acc = np.zeros(len(part))
    for ch in pd.read_csv("race_features.csv", usecols=part,
                          chunksize=100000, low_memory=True):
        acc += ch.isna().sum().values
        cnt += len(ch)
    tot[part] = acc / cnt
    n = cnt
    log("    BT側 %d/%d 列を読みました" % (min(i + BATCH, len(cols)), len(cols)))
na_bt = tot

d = pd.DataFrame({"本番": na_live * 100, "BT": na_bt * 100})
d["差"] = d["本番"] - d["BT"]
d = d.sort_values("差", ascending=False)

log("")
log("  ── 本番だけ欠けている列（差が大きい順） ──")
log("  %-34s %8s %8s %8s" % ("列", "本番", "BT", "差"))
log("  " + "-" * 62)
bad = d[d["差"] > 10]
for c, r in bad.iterrows():
    log("  %-34s %7.1f%% %7.1f%% %+7.1f" % (c[:34], r["本番"], r["BT"], r["差"]))
log("")
log("  差が10ポイント超の列: %d / %d" % (len(bad), len(d)))
log("  完全に一致している列: %d" % int((d["差"].abs() < 0.5).sum()))
log("")
log("  ── BTのほうが欠けている列（参考・上位5） ──")
for c, r in d.tail(5).iterrows():
    log("  %-34s %7.1f%% %7.1f%% %+7.1f" % (c[:34], r["本番"], r["BT"], r["差"]))
