# -*- coding: utf-8 -*-
r"""JV-Link 接続プローブ。32bit Python (%LOCALAPPDATA%\Python32\Python312\python.exe) で実行。
 ① JVInit ② 各dataspecの有効性チェック(JVOpen rc) ③ RACEから数レコード読んで形式確認。
 使い方: <32bit-python> jv_probe.py
 注: makepy型定義のJVOpenはin/out引数(readcount,downloadcount)にプレースホルダ0を渡す。
"""
import sys
import win32com.client

DATASPECS = [
    ("RACE", "レース情報(RA/SE/HR払戻/H1票数/O1-O6オッズ)"),
    ("SLOP", "坂路調教(HC)"),
    ("WOOD", "ウッド調教(WC)"),
    ("BLOD", "血統(HN/SK/BT)"),
    ("DIFF", "マスタ(UM馬/KS騎手/CH調教師)"),
    ("MING", "データマイニング予想(DM/TM)"),
    ("SNAP", "時系列(スナップ)候補"),
    ("HOSN", "票数候補"),
    ("SALE", "セリ市場(HS)候補"),
    ("TCOV", "特別登録"),
    ("RCOV", "出走馬名表"),
    ("YSCH", "開催スケジュール"),
]

RC_MEAN = {
    0: "OK",
    -1: "該当データなし(仕様は有効)",
    -111: "dataspec不正",
    -112: "fromtime不正",
    -114: "キー未設定/不正",
    -201: "JVInit未実行",
    -211: "未契約(このデータは契約外)",
    -301: "認証エラー",
    -302: "キー有効期限切れ",
}


def main():
    jv = win32com.client.gencache.EnsureDispatch("JVDTLab.JVLink")
    rc = jv.JVInit("UNKNOWN")
    print(f"JVInit: rc={rc} ({'OK' if rc == 0 else 'NG'})")
    if rc != 0:
        sys.exit(1)

    fromtime = "20260714000000"   # 直近だけ=軽量プローブ
    print(f"\n--- dataspec有効性プローブ (fromtime={fromtime}, option=1) ---")
    valid = []
    for spec, desc in DATASPECS:
        try:
            ret = jv.JVOpen(spec, fromtime, 1, 0, 0)
            rc2 = ret[0] if isinstance(ret, tuple) else ret
            extra = f" 読込対象={ret[1]} DL対象={ret[2]}" if isinstance(ret, tuple) and rc2 == 0 else ""
            print(f"  {spec:5} rc={rc2:5} {RC_MEAN.get(rc2, '?'):22} {desc}{extra}")
            if rc2 in (0, -1):
                valid.append(spec)
        except Exception as e:
            print(f"  {spec:5} EXC {str(e)[:60]}")
        finally:
            try:
                jv.JVClose()
            except Exception:
                pass

    # RACE から数レコード読んで生形式を確認
    print("\n--- RACE 先頭レコード確認 ---")
    ret = jv.JVOpen("RACE", fromtime, 1, 0, 0)
    rc3 = ret[0] if isinstance(ret, tuple) else ret
    if rc3 == 0:
        seen = {}
        for _ in range(2000):
            r = jv.JVRead("", 110000, "")
            code = r[0] if isinstance(r, tuple) else r
            if code == 0:      # 全読み込み完了
                break
            if code == -1:     # ファイル切替
                continue
            if code < -1:
                print(f"  JVRead rc={code}")
                break
            buff = r[1] if isinstance(r, tuple) else ""
            rectype = buff[:2]
            seen[rectype] = seen.get(rectype, 0) + 1
            if seen[rectype] == 1:
                print(f"  初出 {rectype}: len={len(buff)} head={buff[:60]!r}")
        print("  レコード種別カウント:", seen)
    else:
        print(f"  JVOpen rc={rc3}")
    jv.JVClose()
    print("\nプローブ完了。valid specs:", valid)


if __name__ == "__main__":
    main()
