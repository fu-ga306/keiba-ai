# -*- coding: utf-8 -*-
"""JV-Link払戻(RACE_HR.txt)を正解データに使い、指定日の予想精度とROIを検証する。
netkeibaがIPブロック等で使えない時の照合経路。着順はHRだけで復元する:
  1着=単勝馬番 / 2着=馬単(1着→2着)の2頭目 / 3着=複勝top3の残り。

使い方:
  python analyze_accuracy_jv.py <pred.csv> <bets.csv>
    pred.csv : 対象日の today_predictions 相当（馬番・推奨ランク・予想日時を含む）
    bets.csv : 対象日の today_bets 相当（券種・組み合わせ・金額・予想日時）
  RACE_HR.txt は事前に jv_fetch.py RACE <yyyymmdd> 1 で取得しておくこと。
"""
import os
import sys
import pandas as pd
from jv_payout_parse import parse_line

BASE = os.path.dirname(os.path.abspath(__file__))
UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}  # 組を昇順ソートして突合
JYO_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
            "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}


def norm(kind, combo):
    parts = str(combo).split("-")
    if kind in UNORDERED:
        parts = sorted(parts)
    return "-".join(parts)


def load_results(hr_path):
    """race_id -> {win, place(set), order(1着,2着,3着), pay{(券種,combo):金額}}"""
    seen = {}
    with open(hr_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("HR"):
                t = line.rstrip("\n")
                seen[t[11:15] + t[19:27]] = t  # 同race後勝ち＝確定
    res = {}
    for rid, t in seen.items():
        rows = parse_line(t)
        pay = {}
        tan = fuku = uma_tan = None
        fuku_set = []
        for r in rows:
            k, c = r["券種"], norm(r["券種"], r["組み合わせ"])
            pay[(k, c)] = r["払戻金"]
            if r["券種"] == "単勝":
                tan = r["組み合わせ"]
            elif r["券種"] == "複勝":
                fuku_set.append(r["組み合わせ"])
            elif r["券種"] == "馬単":
                uma_tan = r["組み合わせ"].split("-")  # [1着,2着]
        if not tan:
            continue
        first = tan
        second = uma_tan[1] if uma_tan and len(uma_tan) == 2 else None
        third = None
        if second:
            rest = [u for u in fuku_set if u not in (first, second)]
            third = rest[0] if rest else None
        order = {first: 1}
        if second:
            order[second] = 2
        if third:
            order[third] = 3
        res[rid] = {"win": first, "place": set(fuku_set), "order": order, "pay": pay}
    return res


def latest_per_race(df):
    """race_idごとに予想日時が最新の行だけ残す（7/19二重稼働の重複を除去）"""
    df = df.copy()
    df["race_id"] = df["race_id"].astype(str)
    keep = df.groupby("race_id")["予想日時"].transform("max")
    return df[df["予想日時"] == keep]


def main():
    date_pfx = sys.argv[3] if len(sys.argv) > 3 else "2026/07/19"
    pred = pd.read_csv(sys.argv[1], dtype=str)
    bets = pd.read_csv(sys.argv[2], dtype=str)
    res = load_results(os.path.join(BASE, "data", "jv", "RACE_HR.txt"))
    # race_idの.0(float化)を除去。netkeibaとJVは開催日目が一致するので、
    # predのrace_id(例:函館day12)はJV 7/19結果と厳密一致し、JVに混入する
    # 前日(day11)分は参照されず自動排除される。
    for d in (pred, bets):
        d["race_id"] = d["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # 対象日でフィルタ（前日分の残存を除去）
    pred = pred[pred["予想日時"].astype(str).str.startswith(date_pfx)]
    bets = bets[bets["予想日時"].astype(str).str.startswith(date_pfx)]

    pred = latest_per_race(pred)
    pred = pred[pd.to_numeric(pred["馬番"], errors="coerce").notna()].copy()
    pred["馬番"] = pred["馬番"].astype(float).astype(int).astype(str).str.zfill(2)
    bets = latest_per_race(bets).drop_duplicates(["race_id", "券種", "買い方", "組み合わせ"])
    bets["金額"] = pd.to_numeric(bets["金額"], errors="coerce").fillna(0)

    races = [r for r in pred["race_id"].unique() if r in res]
    print(f"対象日: {date_pfx}  予想レース{pred['race_id'].nunique()} / JV結果照合{len(races)}")

    # ── 印別 勝率/連対率/馬券内率 ──
    MARKS = ["◎", "○", "▲", "△", "×"]
    stat = {m: {"n": 0, "win": 0, "rentai": 0, "fukusho": 0} for m in MARKS}
    sane = 0
    for rid in races:
        r = res[rid]
        pr = pred[pred["race_id"] == rid]
        if r["win"] in set(pr["馬番"]):
            sane += 1
        for m in MARKS:
            row = pr[pr["推奨ランク"] == m]
            if row.empty:
                continue
            uma = row["馬番"].iloc[0]
            st = stat[m]
            st["n"] += 1
            pos = r["order"].get(uma)
            if pos == 1:
                st["win"] += 1
            if pos in (1, 2):
                st["rentai"] += 1
            if uma in r["place"]:
                st["fukusho"] += 1
    print(f"（照合の健全性: 1着馬が予想出走表に含まれたレース {sane}/{len(races)}）")
    print("\n=== 印別精度（JV正解） ===")
    print(f"{'印':<3}{'頭数':>4}{'勝率':>8}{'連対率':>8}{'複勝率(馬券内)':>14}")
    for m in MARKS:
        s = stat[m]
        n = s["n"] or 1
        print(f"{m:<3}{s['n']:>4}{s['win']/n*100:>7.1f}%{s['rentai']/n*100:>7.1f}%{s['fukusho']/n*100:>13.1f}%")

    # ── ROI（券種別・判定別・全体） ──
    def roi_block(sub, label):
        spent = sub["金額"].sum()
        ret = 0.0
        hit = 0
        for _, b in sub.iterrows():
            rid = b["race_id"]
            if rid not in res:
                continue
            key = (b["券種"], norm(b["券種"], b["組み合わせ"]))
            p = res[rid]["pay"].get(key)
            if p:
                ret += p / 100.0 * b["金額"]
                hit += 1
        roi = ret / spent * 100 if spent else 0
        print(f"  {label:<22} 点数{len(sub):>4} 投資{int(spent):>7}円 払戻{int(ret):>8}円 "
              f"収支{int(ret-spent):>+8}円 回収{roi:>6.1f}% 的中{hit}")
        return spent, ret

    bets_r = bets[bets["race_id"].isin(res)]
    print("\n=== ROI（券種別） ===")
    ts = tr = 0
    for k in ["単勝", "複勝", "ワイド", "馬連", "馬単", "3連複", "3連単"]:
        sub = bets_r[bets_r["券種"] == k]
        if len(sub):
            s, rt = roi_block(sub, k)
            ts += s
            tr += rt
    print("\n=== ROI（判定別） ===")
    for j in bets_r["判定"].dropna().unique():
        roi_block(bets_r[bets_r["判定"] == j], f"判定:{j}")

    print("\n=== 全体 ===")
    roi_block(bets_r, "合計")


if __name__ == "__main__":
    main()
