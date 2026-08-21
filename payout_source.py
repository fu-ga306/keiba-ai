# -*- coding: utf-8 -*-
"""払戻・着順の統一取得口。netkeibaスクレイピングへの依存を断つ。

取得順:
  1. jv_payouts.csv       … 過去分の払戻DB(ローカル・最速)
  2. data/jv/RACE_HR.txt  … JV-Linkで取得した直近分(fetch_jv()で更新)
  3. netkeiba             … 上記で取れない時のみ。レート制限つき

2026-07-27: netkeibaがCloudFrontでIPブロック(400)された事故を受けて新設。
JV-Link(JRA-VAN正規契約)は同じ情報を安定して返すので、照合系は原則JVを使う。
"""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
HR_PATH = os.path.join(BASE, "data", "jv", "RACE_HR.txt")
JV_CSV = os.path.join(BASE, "jv_payouts.csv")
# netkeibaから取った払戻（payout_scraper.py の出力）。今後はこちらが主。
NETKEIBA_CSV = os.path.join(BASE, "payout_data.csv")
PY32 = r"C:/Users/別府飛河/AppData/Local/Python32/Python312/python.exe"
UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}

_hr_cache = None
_csv_cache = None


def norm(kind, combo):
    p = str(combo).split("-")
    return "-".join(sorted(p) if kind in UNORDERED else p)


def fetch_jv(date_str, spec="RACE"):
    """JV-Linkで指定日(YYYYMMDD)以降のレースデータを取得し data/jv/ を更新。
    32bit PythonでしかJV-Link COMを叩けないので別プロセスで実行する。"""
    r = subprocess.run([PY32, "jv_fetch.py", spec, str(date_str), "1"],
                       cwd=BASE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    global _hr_cache
    _hr_cache = None      # 再読込させる
    return r.returncode == 0, (r.stdout or "")[-500:]


def _load_hr():
    """RACE_HR.txt を race_id -> {pay, order, win, place} に展開。"""
    global _hr_cache
    if _hr_cache is not None:
        return _hr_cache
    _hr_cache = {}
    if not os.path.exists(HR_PATH):
        return _hr_cache
    from jv_payout_parse import parse_line
    seen = {}
    with open(HR_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HR"):
                t = line.rstrip("\n")
                seen[t[11:15] + t[19:27]] = t   # 同一raceは後勝ち(確定値)
    for rid, t in seen.items():
        rows = parse_line(t)
        pay, fuku_set, tan, uma_tan = {}, [], None, None
        for r in rows:
            pay[(r["券種"], norm(r["券種"], r["組み合わせ"]))] = r["払戻金"]
            if r["券種"] == "単勝":
                tan = r["組み合わせ"]
            elif r["券種"] == "複勝":
                fuku_set.append(r["組み合わせ"])
            elif r["券種"] == "馬単":
                uma_tan = r["組み合わせ"].split("-")
        if not tan:
            continue
        second = uma_tan[1] if uma_tan and len(uma_tan) == 2 else None
        rest = [u for u in fuku_set if u not in (tan, second)]
        order = {tan: 1}
        if second:
            order[second] = 2
        if rest:
            order[rest[0]] = 3
        _hr_cache[rid] = {"pay": pay, "order": order,
                          "win": tan, "place": set(fuku_set), "rows": rows}
    return _hr_cache


def _load_csv():
    """ローカルの払戻DBを読む。netkeiba取得分とJV取得分の**両方**を見る。

    ⚠ 2026-08-22まで jv_payouts.csv しか見ていなかった。JRA-VANの契約終了に伴い
      netkeiba へ切り替えたのに、netkeiba で取った payout_data.csv を読んで
      いなかったため、新しいレースは毎回ウェブへ取りに行っていた。

    優先順は netkeiba(payout_data.csv) → JV(jv_payouts.csv)。
    同じ race_id が両方にあれば netkeiba を採る（今後はこちらが正）。
    """
    global _csv_cache
    if _csv_cache is not None:
        return _csv_cache
    _csv_cache = {}
    import pandas as pd
    # 後に読むほうが優先されるよう、JV → netkeiba の順に入れる
    for path in (JV_CSV, NETKEIBA_CSV):
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, dtype=str)
        if "払戻金" not in d.columns:
            continue
        d["払戻金"] = pd.to_numeric(d["払戻金"], errors="coerce").fillna(0).astype(int)
        for rid, g in d.groupby("race_id"):
            _csv_cache[str(rid).replace(".0", "")] = g.to_dict("records")
    return _csv_cache


# 2026-07-31: JV-VANの課金をやめる方針に伴い、直近分の取得元をnetkeibaへ戻す。
#   ・過去分(jv_payouts.csv 2019-2026)はローカルに永久に残るので今後もそのまま使う
#   ・USE_JV=False にすると JV-Link の呼び出し(fetch_jv)を試みず、
#     ローカルCSVに無いものは直接netkeibaへ行く（無駄な32bit起動と待ちを省く）
#   ・環境変数 KEIBA_USE_JV=1 で一時的に戻せる
USE_JV = os.environ.get("KEIBA_USE_JV") == "1"


def get_payout(race_id, allow_web=True):
    """[{race_id,券種,組み合わせ,払戻金,人気}] を返す。取れなければ空リスト。"""
    rid = str(race_id).replace(".0", "")
    if USE_JV:
        hr = _load_hr().get(rid)
        if hr:
            return [dict(r, race_id=rid) for r in hr["rows"]]
    csv = _load_csv().get(rid)
    if csv:
        return csv
    if not allow_web:
        return []
    from payout_scraper import get_payout as web_payout
    try:
        return web_payout(rid)
    except Exception:
        return []


def get_order(race_id):
    """race_id -> {馬番(2桁str): 着順} (1-3着のみ)。JVからのみ復元。"""
    rid = str(race_id).replace(".0", "")
    hr = _load_hr().get(rid)
    return dict(hr["order"]) if hr else {}


def available_races():
    """現在ローカルで照合可能なrace_id集合(JV分のみ)。"""
    return set(_load_hr()) | set(_load_csv())
