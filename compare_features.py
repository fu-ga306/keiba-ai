# -*- coding: utf-8 -*-
"""本番が計算した特徴量と、BTが読む特徴量を突き合わせる（2026-08-29）

なぜ要るか
  BT は race_features.csv を読み、本番は build_features_for_prediction が
  その場で計算する。**同じ323列を、別のコードが作っている。**
  原因④「検証と本番で同じものを計算していない」が起きうる構造。

  実際に「本番とBTで軸の選び方が違うのでは」という疑いが出たが、
  本番側の特徴量がどこにも残っていなかったため確かめられなかった。
  keiba_predict.py に pred_features.csv への保存を入れたので、
  次の週次更新で race_features.csv が伸びたら、ここで突き合わせられる。

⚠ 比較していいレースの条件
  race_features.csv に入っていて、かつ**本番が予想した時点のモデルが
  そのレースを学習に含んでいない**こと。
  含んでいると in-sample と out-of-sample を比べることになり、
  差が出るのは当たり前で、特徴量のズレの証拠にならない。
  （実際にこれで一度間違えた）

実行
  python compare_features.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import pickle

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(BASE_DIR, "pred_features.csv")
FEAT = os.path.join(BASE_DIR, "race_features.csv")


def log(m):
    print(m, flush=True)


def main():
    if not os.path.exists(SNAP):
        log("  pred_features.csv がまだありません。")
        log("  次の開催日に本番が走れば作られます（keiba_predict.py が書きます）")
        return
    s = pd.read_csv(SNAP, dtype={"race_id": str}, low_memory=False)
    s["bn"] = pd.to_numeric(s.get("馬番"), errors="coerce")
    log(f"  本番の記録 {len(s):,}頭 / {s.race_id.nunique()}レース")
    if "欠損列数" in s.columns and s["欠損列数"].max() > 0:
        log(f"  ⚠ 本番で作れなかった列がある: 最大{int(s['欠損列数'].max())}列")

    ids = set()
    for ch in pd.read_csv(FEAT, usecols=["race_id"], dtype={"race_id": str},
                          chunksize=300000):
        ids |= set(ch["race_id"].str.replace(r"\.0$", "", regex=True))
    both = sorted(set(s.race_id) & ids)
    log(f"  race_features.csv にもあるレース {len(both)}")
    if not both:
        log("  まだ突き合わせられません。週次更新で race_features.csv が伸びたら再実行を")
        return

    m = pickle.load(open(os.path.join(BASE_DIR, "model_resid.pkl"), "rb"))
    cols = [c for c in m["use_cols"] if c in s.columns]
    need = list(set(cols) | {"race_id", "馬番"})
    parts = []
    for ch in pd.read_csv(FEAT, usecols=lambda c: c in need,
                          dtype={"race_id": str}, chunksize=200000,
                          low_memory=False):
        ch["race_id"] = ch["race_id"].str.replace(r"\.0$", "", regex=True)
        x = ch[ch.race_id.isin(both)]
        if len(x):
            parts.append(x)
    B = pd.concat(parts)
    B["bn"] = pd.to_numeric(B["馬番"], errors="coerce")

    M = s[s.race_id.isin(both)].merge(B, on=["race_id", "bn"], suffixes=("_本番", "_BT"))
    log(f"  突き合わせ {len(M):,}頭\n")

    rows = []
    for c in cols:
        a = pd.to_numeric(M.get(c + "_本番"), errors="coerce")
        b = pd.to_numeric(M.get(c + "_BT"), errors="coerce")
        if a is None or b is None or a.isna().all() or b.isna().all():
            continue
        ok = a.notna() & b.notna()
        if ok.sum() < 10:
            continue
        d = (a[ok] - b[ok]).abs()
        rows.append({"列": c, "一致率": float((d < 1e-6).mean() * 100),
                     "平均差": float(d.mean()),
                     "本番の欠損率": float(a.isna().mean() * 100),
                     "BTの欠損率": float(b.isna().mean() * 100)})
    R = pd.DataFrame(rows).sort_values("一致率")
    log("  ズレている列（一致率が低い順に20列）")
    log("  " + "-" * 66)
    for r in R.head(20).itertuples():
        log("    %-28s 一致%5.1f%%  平均差%9.3f  欠損 本番%4.1f%% BT%4.1f%%"
            % (r.列[:28], r.一致率, r.平均差, r.本番の欠損率, r.BTの欠損率))
    log("")
    log(f"  完全一致した列 {(R.一致率 > 99.9).sum()} / {len(R)}")
    bad = R[R.一致率 < 99.9]
    if len(bad) == 0:
        log("  ○ 本番とBTは同じ特徴量を作っています")
    else:
        log(f"  ⚠ {len(bad)}列がズレています。上から順に原因を追ってください")


if __name__ == "__main__":
    main()
