# -*- coding: utf-8 -*-
"""JVRead/JVGets 呼び出し規約の切り分け実験（32bit Pythonで実行）"""
import time
import win32com.client
import pythoncom
from win32com.client import VARIANT

jv = win32com.client.gencache.EnsureDispatch("JVDTLab.JVLink")
print("init", jv.JVInit("UNKNOWN"), flush=True)
ret = jv.JVOpen("SLOP", "20260710000000", 1, 0, 0)
print("open rc=", ret[0], "read=", ret[1], "dl=", ret[2], flush=True)
while ret[2] > 0:
    st = jv.JVStatus()
    if st >= ret[2]:
        break
    time.sleep(2)
print("DL done", flush=True)

# 実験A: JVGets に VARIANT(byref) を渡す
try:
    buf = VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_BYREF, None)
    r = jv.JVGets(buf, 110000)
    print("A JVGets(VARIANT): rc=", r[0], "buf.value type=",
          type(buf.value).__name__ if buf.value is not None else None,
          "ret types=", [type(x).__name__ for x in (r if isinstance(r, tuple) else [r])],
          flush=True)
    val = buf.value if buf.value is not None else (r[1] if isinstance(r, tuple) and len(r) > 1 else None)
    if val is not None:
        bb = bytes(bytearray(val)) if isinstance(val, (tuple, list, bytearray, memoryview)) else val
        if isinstance(bb, bytes):
            print("A bytes head:", bb[:30], flush=True)
            print("A decode cp932:", bb.decode("cp932", errors="replace")[:40], flush=True)
        else:
            print("A str repr:", repr(bb)[:80], flush=True)
except Exception as e:
    print("A EXC:", str(e)[:120], flush=True)

# 実験B: JVRead を位置引数3つ（旧probe方式）→ buffの復元を試す
try:
    r = jv.JVRead("", 110000, "")
    print("B JVRead('',n,''): rc=", r[0], "types=",
          [type(x).__name__ for x in (r if isinstance(r, tuple) else [r])], flush=True)
    b = r[1] if isinstance(r, tuple) and len(r) > 1 else None
    if isinstance(b, str) and b:
        print("B repr head:", repr(b[:30]), flush=True)
        for enc in ("cp1252", "latin-1", "mbcs"):
            try:
                dec = b.encode(enc, errors="strict").decode("cp932", errors="replace")
                print(f"B {enc}->cp932:", dec[:40], flush=True)
            except Exception as e2:
                print(f"B {enc}: NG", str(e2)[:60], flush=True)
except Exception as e:
    print("B EXC:", str(e)[:120], flush=True)

jv.JVClose()
print("done", flush=True)
