# -*- coding: utf-8 -*-
"""出馬表の省略名を、履歴の正式名に直す（2026-08-30）

なぜ要るか
  出馬表と履歴で、騎手・調教師の表記が違っていた。

    出馬表（本番）   騎手 '岩田望'   '鮫島駿'   調教師 '美浦水野'
    履歴（学習側）   騎手 '岩田望来' '鮫島克駿' 調教師 '[東]水野貴広'

  照合できないので集計対象が0件になり、騎手勝率・調教師勝率が
  NaN のままモデルに渡っていた。実測で騎手90.9%・調教師100%が欠損。
  学習データ側は同じ列が0.1%しか欠けていない。

  つまりモデルは「騎手と調教師を見て学習し、見ずに予測していた」。

省略の仕方（実データから分かったこと）
  ・減量記号が付く            ☆田山 △石田 ▲森田 ◇永島
  ・姓＋名から1字。**前方一致とは限らない**
      岩田望来 → 岩田望   （名の頭）
      鮫島克駿 → 鮫島駿   （鮫島克也と区別するため名の末尾）
      角田大和 → 角田和
  ・カタカナは履歴側が5文字で切れている
      出馬表 'Ｍデムーロ' / 履歴 'Ｍ．デム'
  ・調教師は地区の書き方が違う
      出馬表 '美浦水野' / 履歴 '[東]水野貴広'

引き方
  ① 記号と句読点を落として突き合わせる
  ② 完全一致
  ③ どちらかがもう一方の先頭（カタカナの切れ対策）
  ④ 姓を固定した部分一致（順序を保って1字ずつ含む）
  ⑤ 候補が複数なら、直近に騎乗/管理している人だけに絞る

  ⚠ それでも1人に定まらなければ**変換しない**。
    間違った人の成績を付けるくらいなら、欠損のままのほうがましなため。
    （'加藤' '松本' は現役が複数いるので変換しない）

使い方
  import name_resolve
  name_resolve.jockey("鮫島駿")     -> "鮫島克駿"
  name_resolve.trainer("美浦水野")  -> "[東]水野貴広"
  引けなければ入力をそのまま返す。

作り直し
  python name_resolve.py --rebuild
"""
import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(BASE_DIR, "name_master.csv")
SOURCE = os.path.join(BASE_DIR, "race_data_clean.csv")

REGION = {"美浦": "[東]", "栗東": "[西]", "地方": "[地]"}
MARKS = "☆△▲★◇◎"          # 減量記号
RECENT_YEARS = 2             # 「直近」の定義

_cache = {}


def _norm(s):
    """記号・句読点・空白を落とす。全角Ｍなどはそのまま（履歴側も全角のため）。"""
    s = str(s or "").strip()
    s = re.sub(r"^[" + MARKS + r"]+", "", s)
    return s.replace("．", "").replace("・", "").replace(".", "").replace(" ", "").replace("　", "")


def _subseq(a, b):
    """a の文字が b に順序どおり現れるか。姓が一致していることが前提。"""
    it = iter(b)
    return all(c in it for c in a)


def build(force=False):
    """履歴から正式名の一覧を作る。直近に出ているかも持たせる。"""
    if os.path.exists(MASTER) and not force:
        return MASTER
    h = pd.read_csv(SOURCE, usecols=["race_id", "騎手", "調教師"],
                    dtype={"race_id": str}, low_memory=False)
    h["年"] = pd.to_numeric(h["race_id"].str[:4], errors="coerce")
    ymax = h["年"].max()
    rows = []
    for kind in ("騎手", "調教師"):
        v = h[kind].dropna().astype(str).str.strip()
        cnt = v.value_counts()
        rec = set(h.loc[h["年"] >= ymax - RECENT_YEARS, kind]
                  .dropna().astype(str).str.strip())
        for name, n in cnt.items():
            if name:
                rows.append({"種別": kind, "正式名": name,
                             "件数": int(n), "直近": int(name in rec)})
    pd.DataFrame(rows).to_csv(MASTER, index=False, encoding="utf-8-sig")
    _cache.clear()
    return MASTER


def _table(kind):
    if kind not in _cache:
        if not os.path.exists(MASTER):
            build()
        m = pd.read_csv(MASTER)
        m = m[m["種別"] == kind]
        _cache[kind] = [(str(r.正式名), _norm(r.正式名), int(r.直近), int(r.件数))
                        for r in m.itertuples()]
    return _cache[kind]


def _resolve(name, kind, region_hint=None):
    s = _norm(name)
    if not s:
        return name
    tbl = _table(kind)
    exact = [t for t in tbl if t[1] == s]
    if exact:
        return exact[0][0]

    pool = tbl
    if region_hint:
        rp = [t for t in tbl if t[0].startswith(region_hint)]
        if rp:
            pool = rp
            s2 = s
            for k, v in REGION.items():
                if s2.startswith(k):
                    s2 = s2[len(k):]
                    break
            s = s2

    def body(full_norm):
        for v in REGION.values():
            nv = _norm(v)
            if full_norm.startswith(nv):
                return full_norm[len(nv):]
        return full_norm

    cand = []
    for full, fn, rec, cnt in pool:
        b = body(fn)
        if not b or not s:
            continue
        if b == s or b.startswith(s) or s.startswith(b):
            cand.append((full, rec, cnt))
        elif b[0] == s[0] and _subseq(s, b):
            cand.append((full, rec, cnt))
    if len(cand) == 1:
        return cand[0][0]
    if len(cand) > 1:
        act = [c for c in cand if c[1]]           # 直近に出ている人だけ
        if len(act) == 1:
            return act[0][0]
    return name                                    # 定まらなければ触らない


def jockey(name):
    return _resolve(name, "騎手")


def trainer(name):
    s = str(name or "").strip()
    hint = next((v for k, v in REGION.items() if s.startswith(k)), None)
    return _resolve(name, "調教師", region_hint=hint)


def apply_to(df):
    """騎手・調教師の列をまとめて直す。戻り値は (変換数, 未解決数, 全体数)。"""
    n = ng = tot = 0
    for col, fn in (("騎手", jockey), ("調教師", trainer)):
        if col not in df.columns:
            continue
        before = df[col].astype(str)
        after = before.map(fn)
        n += int((before != after).sum())
        known = {t[0] for t in _table(col)}
        ng += int((~after.isin(known)).sum())
        tot += len(before)
        df[col] = after
    return n, ng, tot


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    build(force="--rebuild" in sys.argv)
    print(f"  name_master.csv  騎手{len(_table('騎手'))}人 / "
          f"調教師{len(_table('調教師'))}人")
    print("  騎手")
    for x in ("岩田望", "鮫島駿", "角田和", "石神道", "☆田山", "▲森田",
              "Ｍデムーロ", "ルメール", "原", "加藤", "松本", "◇永島"):
        r = jockey(x)
        ok = r in {t[0] for t in _table("騎手")}
        print(f"    {x:<10} -> {r}{'' if ok else '   （定まらないので変換せず）'}")
    print("  調教師")
    for x in ("美浦水野", "栗東畑端", "美浦田中勝", "美浦牧", "美浦栗田"):
        r = trainer(x)
        ok = r in {t[0] for t in _table("調教師")}
        print(f"    {x:<10} -> {r}{'' if ok else '   （定まらないので変換せず）'}")
