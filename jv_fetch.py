# -*- coding: utf-8 -*-
r"""JV-Link 一括取得。32bit Python (%LOCALAPPDATA%\Python32\Python312\python.exe) で実行。

使い方:
  <py32> jv_fetch.py SLOP 20190101          # 坂路調教を2019年から一括取得
  <py32> jv_fetch.py RACE 20190101          # レース情報(結果/払戻/オッズ/票数)
  <py32> jv_fetch.py WOOD 20190101          # ウッド調教
  <py32> jv_fetch.py SNAP 20260701          # 時系列系の中身確認用

出力: data/jv/<SPEC>_<レコード種別>.txt （1行=1レコード, UTF-8）
      進捗は標準出力。中断時は再実行で最初から（JV側が差分管理するのはoption=1のみ）。
"""
import os
import sys
import time
import win32com.client

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "jv")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    spec = sys.argv[1]
    fromdate = sys.argv[2]
    fromtime = fromdate + "000000" if len(fromdate) == 8 else fromdate
    os.makedirs(OUT_DIR, exist_ok=True)

    jv = win32com.client.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    if rc != 0:
        print(f"JVInit失敗 rc={rc}")
        sys.exit(1)

    # セットアップ取得（ダイアログなし=4 → ダメなら 3 にフォールバック）
    opened = False
    for option in (4, 3):
        ret = jv.JVOpen(spec, fromtime, option, 0, 0)
        rc, readcount, dlcount = ret[0], ret[1], ret[2]
        if rc == 0:
            print(f"JVOpen OK option={option} 読込対象={readcount} DL対象={dlcount}")
            opened = True
            break
        print(f"JVOpen option={option} rc={rc}")
    if not opened:
        print("JVOpen失敗")
        sys.exit(1)

    # ダウンロード完了待ち（JVStatus=DL済みファイル数）
    t0 = time.time()
    while dlcount > 0:
        st = jv.JVStatus()
        if st < 0:
            print(f"JVStatusエラー rc={st}")
            sys.exit(1)
        if st >= dlcount:
            print(f"ダウンロード完了 {st}/{dlcount}  ({time.time()-t0:.0f}秒)")
            break
        print(f"  DL中 {st}/{dlcount}  ({time.time()-t0:.0f}秒)", flush=True)
        time.sleep(5)

    # 読み込みループ。JVReadのBSTRはSJISバイト列がcp1252風に化けて返るため、
    # 1文字ずつ逆変換してバイト列に戻し cp932 でデコードし直す。
    def restore_sjis(s: str) -> str:
        out = bytearray()
        for ch in s:
            try:
                out += ch.encode("cp1252")
            except UnicodeEncodeError:
                out.append(ord(ch) & 0xFF)
        return out.decode("cp932", errors="replace")

    files = {}
    counts = {}
    n_total = 0
    t1 = time.time()
    while True:
        r = jv.JVRead("", 110000, "")
        code = r[0]
        if code == 0:          # 全件読み込み完了
            break
        if code == -1:         # ファイル切替
            continue
        if code == -3:         # ダウンロード待ち
            time.sleep(2)
            continue
        if code < 0:
            print(f"JVRead rc={code} → 中断")
            break
        try:
            line = restore_sjis(r[1]).rstrip("\r\n\x00")
        except Exception:
            continue
        rectype = line[:2]
        if rectype not in files:
            files[rectype] = open(os.path.join(OUT_DIR, f"{spec}_{rectype}.txt"),
                                  "w", encoding="utf-8")
        files[rectype].write(line + "\n")
        counts[rectype] = counts.get(rectype, 0) + 1
        n_total += 1
        if n_total % 100000 == 0:
            print(f"  {n_total}件 ({time.time()-t1:.0f}秒) {counts}", flush=True)

    for f in files.values():
        f.close()
    jv.JVClose()
    print(f"\n完了: 計{n_total}件 ({time.time()-t1:.0f}秒)")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}件 → data/jv/{spec}_{k}.txt")


if __name__ == "__main__":
    main()
