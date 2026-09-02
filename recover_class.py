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


def _fill_mawari(df):
    """回りを競馬場・馬場・距離から埋める。

    取り直した行には回りが入っていない（取得関数が拾わない）。
    回りは物理的に決まっているので course_turn.csv から引ける。
    """
    p = os.path.join(BASE_DIR, "course_turn.csv")
    if "回り" not in df.columns or not os.path.exists(p):
        return
    na = df["回り"].isna()
    if not na.any():
        return
    m = pd.read_csv(p, dtype={"jyo": str})
    key = {(r.jyo, int(r.is_turf), int(r.距離)): r.回り for r in m.itertuples()}
    jyo = df["race_id"].astype(str).str[4:6]
    turf = (df.get("馬場種別").astype(str) == "芝").astype(int)
    dist = pd.to_numeric(df.get("距離"), errors="coerce")
    got = 0
    vals = df["回り"].copy()
    for i in df.index[na]:
        d = dist.at[i]
        if pd.isna(d):
            continue
        v = key.get((jyo.at[i], int(turf.at[i]), int(d)))
        if v:
            vals.at[i] = v
            got += 1
    df["回り"] = vals
    log(f"  ③ 回りを表から補完   {got:>4} 行")


def _fill_shokin(df):
    """賞金を (クラス, 着順) の中央値で埋める。

    ページの結果表に賞金の列がないので取得では埋まらない。
    JRAの賞金は定額なので、クラスと着順が分かればほぼ決まる。
    **推定値なので 賞金_出所 に印を残す。**
    """
    if "賞金" not in df.columns:
        return
    p = pd.to_numeric(df["賞金"], errors="coerce")
    na = p.isna()
    if not na.any():
        return
    cls = pd.to_numeric(df.get("クラス_num"), errors="coerce")
    ch = pd.to_numeric(df.get("着順"), errors="coerce")
    ok = p.notna() & cls.notna() & ch.notna()
    tbl = (pd.DataFrame({"c": cls[ok], "k": ch[ok], "p": p[ok]})
           .groupby(["c", "k"])["p"].median())
    if "賞金_出所" not in df.columns:
        df["賞金_出所"] = np.where(p.notna(), "取得", "")
    vals = p.copy()
    src = df["賞金_出所"].copy()
    got = 0
    for i in df.index[na]:
        c, k = cls.at[i], ch.at[i]
        if pd.isna(c) or pd.isna(k):
            continue
        v = tbl.get((c, k))
        if v is None and k >= 6:
            v = 0.0                      # 6着以下は原則ゼロ
        if v is not None:
            vals.at[i] = float(v)
            src.at[i] = "推定"
            got += 1
    df["賞金"] = vals
    df["賞金_出所"] = src
    log(f"  ④ 賞金を定額表から補完 {got:>4} 行")


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

    # ⚠ クラスが全部埋まっていても、ここで return しないこと（2026-09-02に踏んだ）
    #   回りと賞金の補完に到達せず、27.7%欠損のまま残った。
    #   クラスの復旧と、回り・賞金の補完は別の仕事。

    if "クラス_出所" not in df.columns:
        df["クラス_出所"] = np.where(cls.notna(), "取得", "")

    filled = {}

    # ── ① 実測：自分の予想記録から ────────────────────────────────
    if bad_ids and os.path.exists(MARKS):
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
    for rid in (bad_ids if bad_ids else []):
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

    if bad_ids:
        rest = len(bad_ids) - len(filled)
        log("")
        log(f"  埋まる {len(filled)} / {len(bad_ids)} レース"
            f"（{len(filled)/len(bad_ids)*100:.1f}%）")
        log(f"  埋まらない {rest} レース ← 再取得しない限り欠損のまま")

    if dry:
        log("\n  --dry-run のため書き戻しません")
        return

    _fill_mawari(df)
    _fill_shokin(df)

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
