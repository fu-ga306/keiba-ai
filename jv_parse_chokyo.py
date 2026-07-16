# -*- coding: utf-8 -*-
"""JV調教レコード(HC=坂路/WC=ウッド)をパースしてCSV化する。64bit Pythonで実行可。

HCレコード（固定長・JV-Data仕様）:
  レコード種別(2) データ区分(1) 作成年月日(8) トレセン区分(1: 0=美浦,1=栗東)
  調教年月日(8) 調教時刻(4) 血統登録番号(10)
  4F合計(4, 0.1秒) 4Fラップ(3) 3F合計(4) 3Fラップ(3) 2F合計(4) 2Fラップ(3) 1Fラップ(3)
  例: HC1 20260702 0 20260702 0507 2016104668 0680 184 0496 175 0321 163 158

出力: chokyo_hc.csv (horse_id, 調教日, トレセン, time4f, time3f, time2f, lap1f)
      ※ horse_id = 血統登録番号 = netkeibaのhorse_idと同一体系なのでそのまま結合可能。

使い方: python jv_parse_chokyo.py            # HC(坂路)
        python jv_parse_chokyo.py WC         # WC(ウッド) ※フォーマット確認後に対応
"""
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
JV_DIR = os.path.join(BASE, "data", "jv")


def parse_hc(path):
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("HC"):
                continue
            try:
                # 固定長スライス（上記仕様）
                chokyo_date = line[12:20]
                torecen = line[20:21] if line[11:12] not in ("0", "1") else line[11:12]
                # データ作成日(3:11) トレセン(11:12) 調教日(12:20) 時刻(20:24) 血統番号(24:34)
                torecen = line[11:12]
                chokyo_date = line[12:20]
                hhmm = line[20:24]
                horse_id = line[24:34]
                t4 = int(line[34:38])
                l4 = int(line[38:41])
                t3 = int(line[41:45])
                l3 = int(line[45:48])
                t2 = int(line[48:52])
                l2 = int(line[52:55])
                l1 = int(line[55:58])
                if not horse_id.isdigit():
                    bad += 1
                    continue
                rows.append({
                    "horse_id": horse_id,
                    "調教日": chokyo_date,
                    "時刻": hhmm,
                    "トレセン": "美浦" if torecen == "0" else "栗東",
                    "time4f": t4 / 10 if t4 > 0 else None,
                    "time3f": t3 / 10 if t3 > 0 else None,
                    "time2f": t2 / 10 if t2 > 0 else None,
                    "lap1f": l1 / 10 if l1 > 0 else None,
                })
            except Exception:
                bad += 1
    print(f"パース: {len(rows)}件 (不正{bad})")
    return pd.DataFrame(rows)


def parse_wc(path):
    """WC(ウッド調教・103桁固定長)。末尾24桁がHCと同じ 4F合計/ラップ...1Fラップ 構造。"""
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("WC") or len(line) < 103:
                continue
            try:
                horse_id = line[24:34]
                t4 = int(line[79:83]); t3 = int(line[86:90]); t2 = int(line[93:97])
                l1 = int(line[100:103])
                if not horse_id.isdigit():
                    bad += 1
                    continue
                rows.append({
                    "horse_id": horse_id,
                    "調教日": line[12:20],
                    "時刻": line[20:24],
                    "トレセン": "美浦" if line[11:12] == "0" else "栗東",
                    "time4f": t4 / 10 if t4 > 0 else None,
                    "time3f": t3 / 10 if t3 > 0 else None,
                    "time2f": t2 / 10 if t2 > 0 else None,
                    "lap1f": l1 / 10 if l1 > 0 else None,
                })
            except Exception:
                bad += 1
    print(f"パース: {len(rows)}件 (不正{bad})")
    return pd.DataFrame(rows)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "HC"
    if which == "HC":
        path = os.path.join(JV_DIR, "SLOP_HC.txt")
        df = parse_hc(path)
        out = os.path.join(BASE, "chokyo_hc.csv")
    else:
        df = parse_wc(os.path.join(JV_DIR, "WOOD_WC.txt"))
        out = os.path.join(BASE, "chokyo_wc.csv")
    df = df.dropna(subset=["time4f"]).sort_values(["horse_id", "調教日"])
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"保存: {out} ({len(df)}行)")
    print(df.tail(3).to_string())
    # サニティ: タイム分布
    print("\ntime4f分布:", df["time4f"].describe()[["mean", "min", "max"]].round(1).to_dict())


if __name__ == "__main__":
    main()
