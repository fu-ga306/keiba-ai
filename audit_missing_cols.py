# -*- coding: utf-8 -*-
"""存在しない列を参照して既定値に落ちている箇所を探す（2026-09-04）

きっかけ
  コース脚質バイアスが芝とダートを混ぜていた。

      _turf = df["馬場"]... if "馬場" in df.columns else 0

  `馬場` という列は race_features.csv にも race_data_clean.csv にも無い。
  そのため常に0（ダート扱い）になり、この特徴量はほぼ無意味な列だった。
  直したら回収率が 108.1% → 122.5% に変わった。

  **同じ型の間違いが他にもあるはず。**エラーが出ず、静かに既定値へ落ちる。

探し方
  ① features.py から df["列名"] / df.get("列名") を全部拾う
  ② 実際のデータに存在するかを照合
  ③ 存在しないのに参照している箇所を、前後の行つきで出す

⚠ 動的に作る中間列（_で始まるものなど）は途中で作られるので、
  「無い」＝バグとは限らない。**人が読んで判断する材料を出すだけ。**
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import re
from collections import defaultdict

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["features.py", "keiba_auto.py", "resid_io.py", "keiba_predict.py"]


def log(m):
    print(m, flush=True)


def main():
    # 実データに存在する列
    have = set()
    for f in ("race_features.csv", "race_data_clean.csv"):
        p = os.path.join(BASE_DIR, f)
        if os.path.exists(p):
            have |= set(pd.read_csv(p, nrows=1).columns)
    log(f"  実データの列 {len(have)}種類")

    pat_get = re.compile(r'\bdf\.get\(\s*["\']([^"\']+)["\']')
    pat_idx = re.compile(r'\bdf\[\s*["\']([^"\']+)["\']\s*\]')
    pat_in = re.compile(r'["\']([^"\']+)["\']\s+in\s+df\.columns')

    found = defaultdict(list)
    for fn in TARGETS:
        p = os.path.join(BASE_DIR, fn)
        if not os.path.exists(p):
            continue
        lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        for i, line in enumerate(lines):
            for pat in (pat_get, pat_idx, pat_in):
                for col in pat.findall(line):
                    if col.startswith("_") or col in have:
                        continue
                    # 作られる列（右辺に代入されているもの）は除く
                    if re.search(r'df\[\s*["\']' + re.escape(col) + r'["\']\s*\]\s*=', line):
                        continue
                    found[col].append((fn, i + 1, line.strip()))

    # 「無い列を条件にしている」箇所だけに絞る（else で既定値に落ちる形）
    risky = {}
    for col, hits in found.items():
        for fn, ln, src in hits:
            if ("in df.columns" in src and "else" in src) or ".get(" in src:
                risky.setdefault(col, []).append((fn, ln, src))

    log("")
    log(f"  === 実データに無い列を、既定値つきで参照している箇所 ===")
    log(f"  {len(risky)}種類")
    log("")
    for col in sorted(risky):
        log(f"  ■ 「{col}」")
        for fn, ln, src in risky[col][:3]:
            log(f"      {fn}:{ln}")
            log(f"        {src[:120]}")
        log("")

    log("  === 参考: 無い列を参照しているが既定値なしの箇所 ===")
    other = {c: h for c, h in found.items() if c not in risky}
    log(f"  {len(other)}種類: {sorted(other)[:25]}")


if __name__ == "__main__":
    main()
