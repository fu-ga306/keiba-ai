# -*- coding: utf-8 -*-
"""競馬場・馬場・距離から「回り」を引く表を作る（2026-08-30）

なぜ要るか
  出馬表から回りを取る正規表現が外れていて、回り_num が常に欠損していた。
  正規表現は直したが、表記が変わればまた外れる。
  回りは**競馬場と距離で決まる物理的な事実**なので、履歴から引けば
  スクレイピングに依存しない。取れなかったときの受け皿にする。

  履歴32.6万行で確認したところ、(競馬場, 芝ダ, 距離) の103通りすべてで
  回りが一意に定まった。新潟の左と直も距離で区別できる。

実行
  python build_course_turn.py     → course_turn.csv
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

h = pd.read_csv("race_data_clean.csv",
                usecols=["race_id", "回り", "距離", "馬場種別"],
                dtype={"race_id": str}, low_memory=False)
h = h.dropna(subset=["回り"])
h["jyo"] = h["race_id"].str[4:6]
h["is_turf"] = (h["馬場種別"].astype(str) == "芝").astype(int)
h["距離"] = pd.to_numeric(h["距離"], errors="coerce")
h = h.dropna(subset=["距離"])
h["距離"] = h["距離"].astype(int)

g = h.groupby(["jyo", "is_turf", "距離"])["回り"]
amb = int((g.nunique() > 1).sum())
out = g.agg(lambda x: x.mode().iat[0]).reset_index()
out["件数"] = g.size().values
out.to_csv("course_turn.csv", index=False, encoding="utf-8-sig")
print(f"  course_turn.csv を作りました（{len(out)}通り）")
print(f"  回りが一意に定まらない組み合わせ: {amb}（0であるべき）")
print(out.head(8).to_string(index=False))
