# -*- coding: utf-8 -*-
"""出馬表の省略名を、履歴の正式名に直す（2026-08-30）

なぜ要るか
  出馬表と履歴で、騎手・調教師の表記が違っていた。

    出馬表（本番）   騎手 '岩田望'      調教師 '美浦水野'
    履歴（学習側）   騎手 '岩田望来'    調教師 '[東]水野貴広'

  照合できないので集計対象が0件になり、騎手勝率・調教師勝率が
  NaN のままモデルに渡っていた。実測で騎手90.9%・調教師100%が欠損。
  学習データ側は同じ列が0.1%しか欠けていない。

  つまりモデルは「騎手と調教師を見て学習し、見ずに予測していた」。

直し方
  履歴にある正式名の一覧を持っておき、前方一致で引く。
  調教師は地区（美浦→[東] / 栗東→[西]）で絞ってから引く。
  実測では調教師10/10、騎手10/11が一意に定まった。

  ⚠ 一意に定まらないときは**変換しない**。
    間違った人の成績を付けるくらいなら、欠損のままのほうがましなため。
    （'原' は '原優介' と '原田和真' の2人に当たるので変換しない）

使い方
  import name_resolve
  name_resolve.jockey("岩田望")     -> "岩田望来"
  name_resolve.trainer("美浦水野")  -> "[東]水野貴広"
  名前が引けなければ入力をそのまま返す。
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(BASE_DIR, "name_master.csv")
SOURCE = os.path.join(BASE_DIR, "race_data_clean.csv")

REGION = {"美浦": "[東]", "栗東": "[西]", "地方": "[地]"}

_cache = {"騎手": None, "調教師": None}


def build(force=False):
    """履歴から正式名の一覧を作る。数百行の小さな表。"""
    if os.path.exists(MASTER) and not force:
        return MASTER
    h = pd.read_csv(SOURCE, usecols=["騎手", "調教師"], low_memory=False)
    rows = []
    for kind in ("騎手", "調教師"):
        for v in sorted(set(h[kind].dropna().astype(str).str.strip())):
            if v:
                rows.append({"種別": kind, "正式名": v})
    pd.DataFrame(rows).to_csv(MASTER, index=False, encoding="utf-8-sig")
    return MASTER


def _names(kind):
    if _cache[kind] is None:
        if not os.path.exists(MASTER):
            build()
        m = pd.read_csv(MASTER)
        _cache[kind] = sorted(set(m[m["種別"] == kind]["正式名"].astype(str)))
    return _cache[kind]


def jockey(name):
    """騎手の省略名を正式名に。引けなければそのまま返す。"""
    s = str(name or "").strip()
    if not s:
        return name
    pool = _names("騎手")
    if s in pool:
        return s
    cand = [f for f in pool if f.startswith(s)]
    return cand[0] if len(cand) == 1 else name


def trainer(name):
    """調教師の省略名を正式名に。地区で絞ってから前方一致。"""
    s = str(name or "").strip()
    if not s:
        return name
    pool = _names("調教師")
    if s in pool:
        return s
    reg = next((v for k, v in REGION.items() if s.startswith(k)), None)
    body = s
    for k in REGION:
        if body.startswith(k):
            body = body[len(k):]
            break
    if reg:
        cand = [f for f in pool if f.startswith(reg) and f[len(reg):].startswith(body)]
    else:
        cand = [f for f in pool if f.startswith(body)]
    return cand[0] if len(cand) == 1 else name


def apply_to(df):
    """DataFrame の騎手・調教師の列をまとめて直す。戻り値は (変換数, 全体数)。"""
    n = tot = 0
    for col, fn in (("騎手", jockey), ("調教師", trainer)):
        if col not in df.columns:
            continue
        before = df[col].astype(str)
        after = before.map(fn)
        n += int((before != after).sum())
        tot += len(before)
        df[col] = after
    return n, tot


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = build(force="--rebuild" in sys.argv)
    print(f"  {os.path.basename(p)}  騎手{len(_names('騎手'))}人 / "
          f"調教師{len(_names('調教師'))}人")
    for x in ("岩田望", "ルメール", "原", "田辺"):
        print(f"    騎手   {x:<8} -> {jockey(x)}")
    for x in ("美浦水野", "栗東畑端", "美浦田中勝", "美浦牧"):
        print(f"    調教師 {x:<8} -> {trainer(x)}")
