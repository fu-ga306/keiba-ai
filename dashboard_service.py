# -*- coding: utf-8 -*-
"""ダッシュボード（Flask + ngrok）の稼働窓を管理する（2026-08-27）

なぜ必要か
  販売するのはダッシュボードなのに、**flask も ngrok もタスク登録されていなかった。**
  手動起動なので、PCを再起動したら復活しない。
    PC再起動 / Windows Update / 停電 → ダッシュボードは落ちたまま
  予想システムには見張り番が付いているのに、売り物にだけ付いていなかった。

  購読者から見ると「合言葉は届いたのにページが開かない」が最悪の体験になる。

なぜ24時間ではなく「土曜から」か
  開催は土日。販売noteを貼るのは土曜08:00。
  つまり読者が見に来るのは**土曜〜日曜**で、それ以外は誰も来ない。

  ⚠ 06:00に開けるのは、noteを貼る08:00より前に外から見える状態にするため。
    貼った直後にリンクが死んでいると、その号は丸ごと無駄になる。
  常時稼働はPCとネットワークを無駄に使うだけなので、窓を合わせる。

  窓: 土曜 06:00 〜 月曜 09:00
      月曜まで開けるのは、日曜の結果を月曜に見る人がいるため。

⚠ 内側の生死確認だけでは足りない
  プロセスが生きていても、**ngrokのトンネルが切れていれば外からは見えない。**
  以前「7分前メールが丸一日飛ばなかった」のと同じ、沈黙する故障の型。
  だから外形監視（自分のURLを叩いて200が返るか）も入れてある。

実行
  python dashboard_service.py status    いまの状態
  python dashboard_service.py ensure    窓の中なら起動、外なら停止（定期実行用）
  python dashboard_service.py start     窓に関係なく起動
  python dashboard_service.py stop      停止
  python dashboard_service.py probe     外形監視だけ実行
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NGROK_DOMAIN = "sway-uncanny-exonerate.ngrok-free.dev"
URL = f"https://{NGROK_DOMAIN}"
FLASK_PORT = 5000

# ── 稼働窓 ────────────────────────────────────────────────────────────
#   0=月 1=火 2=水 3=木 4=金 5=土 6=日
OPEN_DOW, OPEN_HOUR = 5, 6      # 土曜 06:00 に開ける（2026-08-28に金→土）
CLOSE_DOW, CLOSE_HOUR = 0, 9    # 月曜 09:00 に閉める


def log(m):
    print(m, flush=True)


def in_window(now=None):
    """いま稼働窓の中か。土06:00 〜 月09:00。"""
    d = now or datetime.now()
    w, h = d.weekday(), d.hour
    if w == OPEN_DOW:                   # 土曜は06:00から
        return h >= OPEN_HOUR
    if w == 6:                          # 日曜は終日
        return True
    if w == CLOSE_DOW:                  # 月曜は09:00まで
        return h < CLOSE_HOUR
    return False


def _procs():
    """flask と ngrok のプロセスを探す。"""
    out = {"flask": [], "ngrok": []}
    try:
        import psutil
    except ImportError:
        return out
    for p in psutil.process_iter(["pid", "name"]):
        try:
            cl = p.cmdline()
            joined = " ".join(cl)
            if any(a.endswith("flask_app.py") for a in cl):
                out["flask"].append(p.info["pid"])
            elif "ngrok" in (p.info["name"] or "").lower() and NGROK_DOMAIN in joined:
                out["ngrok"].append(p.info["pid"])
            elif "ngrok.exe" in joined and BASE_DIR in joined:
                out["ngrok"].append(p.info["pid"])
        except Exception:
            continue
    return out


def _stale_code(pids):
    """動いているプロセスより後に flask_app.py が更新されていないか。

    プロセスの起動時刻とファイルの更新時刻を比べるだけ。
    テンプレートも見る（HTMLだけ直した場合も反映が要るため）。
    """
    if not pids:
        return None
    try:
        import psutil
    except ImportError:
        return None
    watch = [os.path.join(BASE_DIR, "flask_app.py"),
             os.path.join(BASE_DIR, "sale_gate.py"),
             os.path.join(BASE_DIR, "sale_view.py")]
    tdir = os.path.join(BASE_DIR, "templates")
    if os.path.isdir(tdir):
        watch += [os.path.join(tdir, f) for f in os.listdir(tdir)
                  if f.endswith(".html")]
    try:
        started = min(psutil.Process(pid).create_time() for pid in pids)
    except Exception:
        return None
    newest, name = 0.0, ""
    for f in watch:
        try:
            m = os.path.getmtime(f)
        except Exception:
            continue
        if m > newest:
            newest, name = m, os.path.basename(f)
    if newest > started + 5:          # 5秒の余裕（起動直後の誤検知を防ぐ）
        from datetime import datetime as _d
        return (f"{name} が {_d.fromtimestamp(newest):%m/%d %H:%M} 更新 / "
                f"プロセスは {_d.fromtimestamp(started):%m/%d %H:%M} 起動")
    return None


def probe(timeout=15):
    """外から見えるかを確かめる。**これが本番の生死確認。**

    プロセスが生きていてもトンネルが切れていれば外からは見えないので、
    内側の確認だけでは不十分。

    ⚠ 「応答した」と「中身が見えた」を混同しないこと（2026-08-28）
      合言葉ゲートを入れた日から、/races は外部アクセスに401を返すようになった。
      401は**サーバーが生きている証拠**（ゲートが答えている）なのに、
      HTTPError を例外として拾って故障扱いにしていたため、
      20分ごとに flask と ngrok を再起動し続けていた。
      直し方は2つ入れる。
        ① ゲートの無い /sale を叩く
        ② HTTPエラーでも「ステータスが返った＝生きている」と扱う
      落ちているときは接続そのものが失敗するので、この2つで区別がつく。
    """
    import urllib.request
    import urllib.error
    req = urllib.request.Request(URL + "/sale",
                                 headers={"ngrok-skip-browser-warning": "1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        # 応答はある。502(トンネル先が落ちている)だけは故障とみなす
        if e.code in (502, 503, 504):
            return 0, f"HTTP {e.code}（トンネルの先が応答しません）"
        return e.code, ""
    except Exception as e:
        return 0, f"{type(e).__name__}: {str(e)[:120]}"


def start():
    p = _procs()
    started = []
    if not p["flask"]:
        subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "flask_app.py")],
                         cwd=BASE_DIR,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started.append("flask")
    if not p["ngrok"]:
        exe = os.path.join(BASE_DIR, "ngrok.exe")
        if os.path.exists(exe):
            subprocess.Popen([exe, "http", f"--domain={NGROK_DOMAIN}", str(FLASK_PORT)],
                             cwd=BASE_DIR,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            started.append("ngrok")
        else:
            log(f"  ⚠ {exe} がありません")
    log(f"  起動: {', '.join(started) if started else '（すでに動いています）'}")
    return started


def stop(wait=10):
    """止める。**終了を待つ。**

    ⚠ 待たないと事故る（2026-08-28に実際に発生）
      stop() の直後に start() を呼ぶと、プロセスがまだ消えていないため
      start() が「もう動いている」と誤判定して起動をスキップする。
      結果、flask だけ落ちたまま ngrok が生き、502 になった。
    """
    p = _procs()
    pids = p["flask"] + p["ngrok"]
    n = 0
    try:
        import psutil
        procs = []
        for pid in pids:
            try:
                pr = psutil.Process(pid)
                pr.terminate()
                procs.append(pr)
                n += 1
            except Exception:
                pass
        if procs:
            gone, alive = psutil.wait_procs(procs, timeout=wait)
            for pr in alive:                 # 素直に終わらないものは強制
                try:
                    pr.kill()
                except Exception:
                    pass
            psutil.wait_procs(alive, timeout=3)
    except ImportError:
        pass
    log(f"  停止: {n}プロセス")
    return n


def status():
    p = _procs()
    now = datetime.now()
    log("■ ダッシュボードの状態")
    log(f"  いま        {now:%m/%d(%a) %H:%M}")
    log(f"  稼働窓      土{OPEN_HOUR:02d}:00 〜 月{CLOSE_HOUR:02d}:00"
        f"  → いまは{'窓の中' if in_window(now) else '窓の外'}")
    log(f"  flask       {p['flask'] if p['flask'] else '× 停止'}")
    log(f"  ngrok       {p['ngrok'] if p['ngrok'] else '× 停止'}")
    code, err = probe()
    log(f"  外から見えるか "
        + (f"○ HTTP {code}" if 200 <= code < 500 else "× " + (err or f"HTTP {code}")))
    log(f"  URL         {URL}")
    return p, code


def ensure():
    """窓の中なら起動を保証し、外なら止める。定期実行から呼ぶ。

    戻り値: (状態を表す文字列, 警告が要るか)
    """
    now = datetime.now()
    p = _procs()
    alive = bool(p["flask"]) and bool(p["ngrok"])
    if in_window(now):
        if not alive:
            log(f"  窓の中だが停止していた（flask={len(p['flask'])} ngrok={len(p['ngrok'])}）→ 起動")
            start()
            return "起動しました", True
        # コードが更新されていたら立て直す（2026-08-27追加）
        #   ⚠ 「生きていて200が返れば正常」だけだと、**コードを直しても
        #     永久に反映されない。** 実際に閲覧制限を足したのに、稼働中の
        #     旧プロセスが200を返し続けるため一生有効にならない状態だった。
        #     たまたま窓の外だったので停止→再起動で助かったが、
        #     窓の中で直したら気づけない。
        stale = _stale_code(p["flask"])
        if stale:
            log(f"  {stale} → コードが新しいので立て直し")
            stop()
            start()
            return f"コード更新を反映するため再起動（{stale}）", False

        code, err = probe()
        # 「200以外＝故障」にしない。**応答があること**が生死の判定条件。
        # 401（合言葉ゲート）を故障扱いして20分ごとに再起動し続けた事故がある。
        if not (200 <= code < 500):
            # プロセスは生きているのに外から見えない＝トンネルが切れている
            log(f"  プロセスは生きているが外から見えない（{err or code}）→ 立て直し")
            stop()
            start()
            return f"外形監視に失敗したため再起動（{err or code}）", True
        log(f"  正常（外形監視 HTTP {code}）")
        return "正常", False
    else:
        if alive:
            log("  窓の外なので停止します")
            stop()
            return "窓の外のため停止", False
        log("  窓の外・停止中（正常）")
        return "窓の外・停止中", False


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else "status"
    if cmd == "status":
        status()
    elif cmd == "ensure":
        ensure()
    elif cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    elif cmd == "probe":
        code, err = probe()
        log(f"  {URL} → {'HTTP ' + str(code) if code else '× ' + err}")
    else:
        log(__doc__)


if __name__ == "__main__":
    main()
