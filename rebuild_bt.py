# -*- coding: utf-8 -*-
"""検証データを現行モデルで作り直し、結果をメールで送る（2026-08-29）

なぜ要るか
  週次更新が race_features.csv と model_resid.pkl を作り直すのに、
  検証データ(resid_kinds_pred.csv)だけ作り直していなかった。
  8日間、本番と検証で違うモデルが動いていた。

  check_resid.py は「選び方が検証どおりか」しか見ないので
  「✅ 実装は検証どおり」と出し続けた。**選び方が同じでも
  モデルが違えば選ぶ馬が変わる。**

いつ走らせるか
  5年ぶんを学習し直すのでメモリを食う。予想システムが
  MFモデル3GBを抱えている時間帯（06:55〜22:30）は避ける。
  夜間に1回だけ走らせる。

何を見るか
  作り直したあとの ROI が 120.6% から動くかどうか。
  動いたなら、**これまでの120.6%は現行モデルの成績ではなかった**ということ。
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import shutil
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
PRED = os.path.join(BASE_DIR, "resid_kinds_pred.csv")


def log(m):
    print(m, flush=True)


def run(name, timeout):
    t0 = datetime.now()
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([PYTHON, os.path.join(BASE_DIR, name)],
                           cwd=BASE_DIR, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        el = (datetime.now() - t0).total_seconds() / 60
        return r.returncode == 0, r.stdout or "", r.stderr or "", el
    except subprocess.TimeoutExpired:
        return False, "", f"{timeout}秒で打ち切り", (datetime.now() - t0).total_seconds() / 60


def main():
    log(f"[{datetime.now():%Y-%m-%d %H:%M}] 検証データの作り直しを開始")

    # 予想が動いていたら走らせない。メモリを奪い合って両方失敗する
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            c = " ".join(p.info["cmdline"] or [])
            if "auto_predict_publish" in c and "python" in c.lower():
                log("  ⚠ 予想システムが稼働中。メモリを奪い合うので中止します")
                return
    except Exception:
        pass

    # 失敗しても戻せるように退避する。**上書きしてから失敗が一番困る**
    bak = ""
    if os.path.exists(PRED):
        bak = PRED + ".bak"
        shutil.copy2(PRED, bak)
        log(f"  退避: {os.path.basename(bak)}")

    ok, out, err, el = run("resid_kinds.py", 14400)
    log(f"  resid_kinds.py  {'成功' if ok else '失敗'}  {el:.0f}分")

    if not ok:
        if bak and os.path.exists(bak):
            shutil.copy2(bak, PRED)
            log("  失敗したので元に戻しました")
        body = (f"検証データの作り直しに失敗しました（{el:.0f}分）\n\n"
                f"元のファイルに戻してあります。\n\n"
                f"--- stderr ---\n{err[-2000:]}\n\n--- stdout ---\n{out[-2000:]}")
        _send("【競馬AI】検証データの作り直し 失敗", body)
        return

    ok2, out2, _, el2 = run("check_resid.py", 3600)
    log(f"  check_resid.py  {'成功' if ok2 else '失敗'}  {el2:.0f}分")

    body = (
        "検証データ(resid_kinds_pred.csv)を現行モデルで作り直しました。\n\n"
        "■ 見るところ\n"
        "  ROI が 120.6% から動いたか。\n"
        "  動いたなら、これまでの120.6%は現行モデルの成績ではなかったということです。\n"
        "  買い方（resid_io.pick_bets）は一切変えていません。\n\n"
        f"  所要 {el:.0f}分（作り直し）＋ {el2:.0f}分（照合）\n\n"
        "============================================================\n"
        "■ 作り直しの出力（末尾）\n"
        "============================================================\n"
        f"{out[-3000:]}\n\n"
        "============================================================\n"
        "■ 照合の結果\n"
        "============================================================\n"
        f"{out2[-4000:]}\n"
    )
    _send("【競馬AI】検証データを現行モデルで作り直しました", body)
    log("  メールを送信しました")


def _send(subject, body):
    try:
        import auto_predict_publish as A
        A._send_alert(subject, body)
    except Exception as e:
        log(f"  メール送信に失敗: {e}")


if __name__ == "__main__":
    main()
