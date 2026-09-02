# -*- coding: utf-8 -*-
"""壊れたレースクラスを、手元のデータだけで復旧する（2026-09-02）

なぜ要るか
  結果ページの取得が2026年6月から壊れ、レースクラス欄に条件文が入っていた。
  クラス_num は 7月以降 100% 欠損。600レース・7,909行。
  race_data_clean.csv は学習にも本番にも使うので、両方に効いている。

  取得側は直したが、直近2週しか取り直さない。過去分はこれで埋める。
  **スクレイピングはしない。**手元のデータだけを使う。

埋め方（この順に優先）
  ① 実測  history_marks.csv
     自分の予想システムが記録したクラス。予想した日だけ持っている。
  ② 推定  同じ馬が走った他のレースのクラスから多数決
     馬は同じクラスを走り続けるので当てられる。
     検証600レースでの実測: 確信度0.7以上に絞ると正解率99.7%（採用率56%）。
     **確信度が足りないものは埋めない。**間違ったクラスは欠損より悪い。

  どこから来た値かを クラス_出所 列に残す。あとから見分けられるようにする。

⚠ race_data_clean.csv は週次更新で作り直される。
  weekly_update.py の Step0.9 から呼んでいるので、毎週この復旧が入る。

実行
  python recover_class.py            復旧して書き戻す
  python recover_class.py --dry-run  何件埋まるか見るだけ
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from collections import Counter

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(BASE_DIR, "race_data_clean.csv")
MARKS = os.path.join(BASE_DIR, "history_marks.csv")

CONF_MIN = 0.7          # これ未満の推定は使わない（検証で正解率99.7%の水準）
NUM2NAME = {1: "未勝利", 2: "1勝クラス", 3: "2勝クラス", 4: "3勝クラス",
            5: "オープン", 6: "G3", 7: "G2", 8: "G1"}


def log(m):
    print(m, flush=True)


def main():
    dry = "--dry-run" in sys.argv
    import cleaner

    df = pd.read_csv(TARGET, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    # ⚠ copy() を付ける。付けないと元の列と同じ配列を指し、
    #   あとで書き戻したときに「前の値」まで書き換わって前後比較が嘘になる。
    cls = pd.to_numeric(df.get("クラス_num"), errors="coerce").copy()
    bad_ids = sorted(set(df.loc[cls.isna(), "race_id"]))
    log(f"  クラスが欠けているレース {len(bad_ids)} / 行 {int(cls.isna().sum()):,}")
    if not bad_ids:
        log("  復旧するものはありません")
        return

    if "クラス_出所" not in df.columns:
        df["クラス_出所"] = np.where(cls.notna(), "取得", "")

    filled = {}

    # ── ① 実測：自分の予想記録から ────────────────────────────────
    if os.path.exists(MARKS):
        m = pd.read_csv(MARKS, dtype={"race_id": str}, usecols=["race_id", "クラス"])
        m = m.dropna(subset=["クラス"]).drop_duplicates("race_id")
        for r in m.itertuples():
            if r.race_id in bad_ids:
                v = cleaner.classify_class(r.クラス)
                if pd.notna(v):
                    filled[r.race_id] = (float(v), str(r.クラス), "記録")
    log(f"  ① 予想記録から     {len(filled):>4} レース")

    # ── ② 推定：同じ馬の他レースのクラスから多数決 ──────────────
    known = df[pd.to_numeric(df["クラス_num"], errors="coerce").notna()]
    hist = {}
    for nm, g in known.groupby("馬名")["クラス_num"]:
        hist[nm] = [float(x) for x in pd.to_numeric(g, errors="coerce").dropna()]

    n_infer = n_lowconf = 0
    for rid in bad_ids:
        if rid in filled:
            continue
        g = df[df.race_id == rid]
        votes = []
        for nm in g["馬名"].dropna().astype(str):
            v = hist.get(nm)
            if v:
                votes.append(Counter(v).most_common(1)[0][0])
        if not votes:
            continue
        top, cnt = Counter(votes).most_common(1)[0]
        conf = cnt / len(votes)
        if conf >= CONF_MIN:
            filled[rid] = (float(top), NUM2NAME.get(int(top), ""), "推定")
            n_infer += 1
        else:
            n_lowconf += 1
    log(f"  ② 馬の履歴から推定 {n_infer:>4} レース（確信度{CONF_MIN}以上）")
    log(f"     確信度が足りず見送り {n_lowconf} レース")

    rest = len(bad_ids) - len(filled)
    log("")
    log(f"  埋まる {len(filled)} / {len(bad_ids)} レース（{len(filled)/len(bad_ids)*100:.1f}%）")
    log(f"  埋まらない {rest} レース ← 再取得しない限り欠損のまま")

    if dry:
        log("\n  --dry-run のため書き戻しません")
        return

    idx = df["race_id"].map(lambda r: filled.get(r))
    hit = idx.notna()
    df.loc[hit, "クラス_num"] = [v[0] for v in idx[hit]]
    df.loc[hit, "レースクラス"] = [v[1] for v in idx[hit]]
    df.loc[hit, "クラス_出所"] = [v[2] for v in idx[hit]]
    df.to_csv(TARGET, index=False, encoding="utf-8-sig")
    after = pd.to_numeric(df["クラス_num"], errors="coerce")
    log("")
    log(f"  書き戻しました。クラス_num の欠損 "
        f"{cls.isna().mean()*100:.1f}% → {after.isna().mean()*100:.1f}%")
    log("  出所の内訳: " + str(df["クラス_出所"].value_counts().to_dict()))


if __name__ == "__main__":
    main()
