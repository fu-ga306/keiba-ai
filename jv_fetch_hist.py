# -*- coding: utf-8 -*-
r"""JV-Link 歴史データ一括取得（レジューム対応・VSCodeでの手動実行用）。

目的: ラップタイム(RAレコード)と過去レースの拡充（学習データを2019〜→2010年代へ）。
      モデル強化プラン「ペース補正能力指数」の材料。

★実行はVSCodeのターミナルで（JV-Linkの認証/確認ダイアログが出るため対話環境が必要。
  Claude環境からのセットアップ取得はダイアログ待ちで固まる実績あり）:

  & "$env:LOCALAPPDATA\Python32\Python312\python.exe" jv_fetch_hist.py 2015 2018

  ・年単位で分割取得し data/jv/hist/RACE_<レコード>_<年>.txt に保存
  ・完了した年はスキップされるので、中断しても同じコマンドで再開可能
  ・1年あたり十数分〜。まず2015-2018の4年ぶんを推奨（学習データ約2倍）
  ・ダイアログが出たら「はい/OK」で進めてください（JRA-VAN正規契約の範囲）
"""
import os
import sys
import time

import win32com.client

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "jv", "hist")
os.makedirs(OUT_DIR, exist_ok=True)

# 全角→SJIS復元（jv_fetch.pyと同じ処理）
def restore_sjis(s):
    try:
        return s.encode("cp932", errors="replace").decode("cp932")
    except Exception:
        return s


def fetch_year(jv, year):
    done_flag = os.path.join(OUT_DIR, f"done_{year}.txt")
    if os.path.exists(done_flag):
        print(f"{year}: 取得済み → スキップ")
        return True
    fromtime = f"{year}0101000000"
    print(f"\n=== {year}年ぶん取得開始 (JVOpen RACE {fromtime}) ===", flush=True)
    ret = None
    for option in (4, 3):
        rc = jv.JVOpen("RACE", fromtime, option, 0, 0)
        if isinstance(rc, (list, tuple)):
            code, readcount, dlcount, lastts = rc[0], rc[1], rc[2], rc[3]
        else:
            code, readcount, dlcount = rc, 0, 0
        print(f"  JVOpen option={option} rc={code} 読込={readcount} DL={dlcount}", flush=True)
        if code == 0:
            ret = option
            break
    if ret is None:
        print("  JVOpen失敗（ダイアログ確認 or 時間をおいて再実行）")
        return False

    # ダウンロード待ち
    t0 = time.time()
    while True:
        st = jv.JVStatus()
        if st < 0:
            print(f"  JVStatus rc={st}")
            break
        if st >= dlcount:
            break
        print(f"  DL中 {st}/{dlcount} ({time.time()-t0:.0f}秒)", flush=True)
        time.sleep(5)

    files, counts, n = {}, {}, 0
    t1 = time.time()
    while True:
        r = jv.JVRead("", 200000)
        code = r[0] if isinstance(r, (list, tuple)) else r
        if code == 0:
            break
        if code == -1:      # ファイル切替
            continue
        if code < 0:
            print(f"  JVRead rc={code} → 中断")
            break
        line = restore_sjis(r[1]).rstrip("\r\n\x00")
        rectype = line[:2]
        # 年をまたぐレコードは開催年で振り分け（RA/SE/HRは開催年が11-15桁目）
        y = line[11:15] if len(line) > 15 and line[11:15].isdigit() else str(year)
        key = (rectype, y)
        if key not in files:
            files[key] = open(os.path.join(OUT_DIR, f"RACE_{rectype}_{y}.txt"),
                              "a", encoding="utf-8")
        files[key].write(line + "\n")
        counts[rectype] = counts.get(rectype, 0) + 1
        n += 1
        if n % 200000 == 0:
            print(f"  {n}件 ({time.time()-t1:.0f}秒)", flush=True)
    for f in files.values():
        f.close()
    jv.JVClose()
    with open(done_flag, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"  {year}: 完了 計{n}件 {counts}", flush=True)
    return True


def main():
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2018
    jv = win32com.client.Dispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    print(f"JVInit rc={rc}")
    for y in range(y0, y1 + 1):
        ok = fetch_year(jv, y)
        if not ok:
            print(f"{y}で停止。同じコマンドで再開できます。")
            break
    print("\n全処理終了")


if __name__ == "__main__":
    main()
