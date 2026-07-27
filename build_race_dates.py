# -*- coding: utf-8 -*-
"""開催日マップ（年+場+回+日目 → 実開催日）を作る。

race_idは「年(4)+場(2)+回(2)+日目(2)+R(2)」で日付ではない。にもかかわらず
features.py は race_id 順を時系列順とみなして「過去走」を決めていたため、
同一年内で最大45.7%のレース対が逆転し、未来のレースが過去成績に混入していた
（2026-07-28に実日付で検証）。これを直すための実日付テーブル。

取得元:
  1. JV-Link YSCH（YSレコード）… 差分オプションで取れる年（現状2025-2026）
  2. db.netkeiba.com/race/list/YYYYMMDD/ … それ以前の年を補完（レート制限つき）

出力: race_dates.csv (kaisai_key=年場回日目(10桁), date)
使い方: python build_race_dates.py 2019 2024
"""
import os
import re
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "race_dates.csv")
YS_PATH = os.path.join(BASE, "data", "jv", "YSCH_YS.txt")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
MIN_INTERVAL = 3.0     # netkeibaは7/27にIPブロックされた実績あり。必ず間隔を空ける
BURST = 30
PAUSE = 60.0
JRA = {f"{i:02d}" for i in range(1, 11)}
_last = 0.0
_count = 0


def _throttle():
    global _last, _count
    w = MIN_INTERVAL - (time.time() - _last)
    if w > 0:
        time.sleep(w)
    _count += 1
    if _count % BURST == 0:
        print(f"  [レート制限] {BURST}件 → {PAUSE:.0f}秒休憩", flush=True)
        time.sleep(PAUSE)
    _last = time.time()


def from_jv():
    """JVのYSレコードから (kaisai_key, date) を得る。取れる年だけ。"""
    rows = []
    if not os.path.exists(YS_PATH):
        return pd.DataFrame(columns=["kaisai_key", "date"])
    with open(YS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("YS"):
                continue
            ymd, jyo, kai, day = line[11:19], line[19:21], line[21:23], line[23:25]
            if ymd.isdigit() and jyo in JRA:
                rows.append((ymd[:4] + jyo + kai + day, ymd))
    d = pd.DataFrame(rows, columns=["kaisai_key", "ymd"]).drop_duplicates("kaisai_key")
    d["date"] = pd.to_datetime(d["ymd"], format="%Y%m%d")
    print(f"JV(YSCH)から {len(d)}開催 ({d['ymd'].min()}〜{d['ymd'].max()})", flush=True)
    return d[["kaisai_key", "date"]]


def fetch_day(ymd):
    """その日のレース一覧から開催キー(年場回日目)を抽出。JRA開催が無ければ空。"""
    _throttle()
    try:
        r = requests.get(f"https://db.netkeiba.com/race/list/{ymd}/",
                         headers=HEADERS, timeout=20)
        if r.status_code >= 400:
            print(f"  [警告] HTTP {r.status_code} ({ymd}) → 中断", flush=True)
            return None
        r.encoding = "EUC-JP"
        ids = set(re.findall(r"/race/(\d{12})", r.text))
        return {i[:10] for i in ids if i[4:6] in JRA}
    except Exception as e:
        print(f"  [警告] {ymd}: {e}", flush=True)
        return set()


def candidate_days(y0, y1):
    """土日＋月曜(振替開催がある)を候補にする。"""
    d, end, out = date(y0, 1, 1), date(y1, 12, 31), []
    while d <= end:
        if d.weekday() in (5, 6, 0):
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    have = from_jv()
    known = set(have["kaisai_key"])
    rows = []
    days = candidate_days(y0, y1)
    print(f"netkeibaから {y0}-{y1} を補完（候補{len(days)}日・約{len(days)*MIN_INTERVAL/60:.0f}分＋休憩）",
          flush=True)
    t0 = time.time()
    for i, ymd in enumerate(days, 1):
        keys = fetch_day(ymd)
        if keys is None:      # ブロック検知 → 打ち切って途中まで保存
            print("  ブロックの可能性 → ここまでを保存して終了", flush=True)
            break
        for k in keys:
            if k not in known:
                rows.append((k, ymd))
        if i % 50 == 0:
            print(f"  {i}/{len(days)}日 取得済 開催{len(rows)}件 "
                  f"({(time.time()-t0)/60:.0f}分経過)", flush=True)
    web = pd.DataFrame(rows, columns=["kaisai_key", "ymd"]).drop_duplicates("kaisai_key")
    if len(web):
        web["date"] = pd.to_datetime(web["ymd"], format="%Y%m%d")
        web = web[["kaisai_key", "date"]]
    else:
        web = pd.DataFrame(columns=["kaisai_key", "date"])
    out = pd.concat([have, web]).drop_duplicates("kaisai_key").sort_values("kaisai_key")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n保存 → {OUT}  {len(out)}開催 "
          f"({out['date'].min().date()}〜{out['date'].max().date()})", flush=True)


if __name__ == "__main__":
    main()
