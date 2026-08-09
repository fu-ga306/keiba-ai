# -*- coding: utf-8 -*-
"""無人運用の見張り番。異常を検知して自力で直し、直せなければメールで知らせる。

なぜ必要か（2026-08-09）
  8/8 keiba_auto が起動15秒で死んだまま一日中気づかれなかった
  8/9 スケジューラが生きたまま固まり、以降のジョブが全て止まった
  どちらも「プロセスは存在する」ので、タスクスケジューラからは正常に見える。
  人が見ていないと止まったまま何日も過ぎる。

見るもの
  ① スケジューラの心拍  … auto_predict_heartbeat.txt が更新され続けているか
  ② keiba_auto の生死   … 開催時間帯に落ちていないか
  ③ 当日データの鮮度    … 開催日なのに予想が朝から更新されていないか
  ④ ディスクの空き

できること
  ・心拍が止まっていればタスクを再起動する（8/9に手作業でやった復旧の自動化）
  ・keiba_auto が落ちていれば起動し直す
  ・直せない異常はメールで通報する
  ・同じ内容を何度も送らない（1件につき1日1回まで）

実行: タスクスケジューラに20分おきで登録する。
      手動なら python watchdog.py（--dry で通報せず判定だけ）
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

BASE_DIR = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
HEARTBEAT = os.path.join(BASE_DIR, "auto_predict_heartbeat.txt")
STATE = os.path.join(BASE_DIR, "watchdog_state.json")
LOG = os.path.join(BASE_DIR, "watchdog.log")
TASK = "競馬AI自動予想"

HEARTBEAT_STALE_MIN = 8     # 心拍がこの分数止まったらハングとみなす（待機ループは10秒周期）
DISK_MIN_GB = 2.0
QUIET_HOURS = (22, 7)       # この時間帯は通報しない（22:30に意図的に落とすため）


def log(msg):
    line = f"[{datetime.now():%m/%d %H:%M:%S}] {msg}"
    print(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(s):
    try:
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False)
    except Exception:
        pass


def notify(key, subject, body, dry=False):
    """同じ種類の通報は1日1回まで。毎回送ると読まれなくなるため。"""
    today = datetime.now().strftime("%Y-%m-%d")
    s = _state()
    if s.get(key) == today:
        log(f"  通報済み（本日分）: {key}")
        return
    if dry:
        log(f"  [dry] 通報するはず: {subject}")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = os.environ["GMAIL_ADDRESS"]
        msg["To"] = os.environ["TO_ADDRESS"]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASS"])
            srv.send_message(msg)
        s[key] = today
        _save(s)
        log(f"  通報しました: {subject}")
    except Exception as e:
        log(f"  通報に失敗: {e}")


def _is_race_day():
    """今日が開催日か。発走時刻ファイルの有無と中身で判断する。"""
    p = os.path.join(BASE_DIR, "today_race_times.json")
    if not os.path.exists(p):
        return False, {}
    # 前日のファイルが残っているだけ、という誤判定を防ぐ
    if datetime.fromtimestamp(os.path.getmtime(p)).date() != datetime.now().date():
        return False, {}
    try:
        with open(p, encoding="utf-8") as f:
            t = {k: int(v) for k, v in json.load(f).items()}
        return bool(t), t
    except Exception:
        return False, {}


def check_heartbeat(dry=False):
    """スケジューラが生きたまま固まっていないか。止まっていれば再起動する。"""
    if not os.path.exists(HEARTBEAT):
        return "心拍ファイルが無い（未起動の可能性）"
    try:
        beat = datetime.strptime(open(HEARTBEAT, encoding="utf-8").read().strip(),
                                 "%Y-%m-%d %H:%M:%S")
    except Exception:
        return "心拍ファイルを読めない"
    age = (datetime.now() - beat).total_seconds() / 60
    if age < HEARTBEAT_STALE_MIN:
        return None
    log(f"  心拍が{age:.0f}分止まっている → 再起動を試みます")
    if dry:
        return f"心拍停止 {age:.0f}分（dryのため再起動せず）"
    try:
        subprocess.run(["schtasks", "/End", "/TN", TASK],
                       capture_output=True, timeout=60)
        subprocess.run(["schtasks", "/Run", "/TN", TASK],
                       capture_output=True, timeout=60, check=True)
        log("  再起動しました")
        return None                      # 自力で直せたので通報しない
    except Exception as e:
        return f"心拍が{age:.0f}分止まっており、再起動にも失敗した: {e}"


def check_keiba_auto(times, dry=False):
    """開催時間帯に7分前ジョブが動いているか。落ちていれば起動し直す。"""
    if not times:
        return None
    cur = datetime.now().hour * 60 + datetime.now().minute
    if cur > max(times.values()):
        return None                      # 全レース終了後は起動しなくてよい
    if cur < min(times.values()) - 60:
        return None                      # 開催前は対象外
    pid_file = os.path.join(BASE_DIR, "keiba_auto.pid")
    alive = False
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file).read().strip())
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True,
                                 errors="ignore", timeout=60).stdout or ""
            alive = str(pid) in out
        except Exception:
            alive = False
    if alive:
        return None
    log("  keiba_auto が落ちている → 起動を試みます")
    if dry:
        return "keiba_auto 停止（dryのため起動せず）"
    try:
        sys.path.insert(0, BASE_DIR)
        import auto_predict_publish as A
        A.run_keiba_auto()
        return None
    except Exception as e:
        return f"keiba_auto が落ちており、起動にも失敗した: {e}"


def check_freshness(times):
    """開催日なのに当日の予想が更新されていない、という状態を拾う。"""
    if not times:
        return None
    cur = datetime.now().hour * 60 + datetime.now().minute
    # 開催時間帯だけを見る。終了後は更新が止まって当たり前なので誤検知になる。
    if not (min(times.values()) <= cur <= max(times.values()) + 30):
        return None
    p = os.path.join(BASE_DIR, "today_predictions.csv")
    if not os.path.exists(p):
        return "開催日なのに today_predictions.csv が無い"
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).total_seconds() / 60
    if age > 90:
        return f"開催中なのに予想が{age:.0f}分更新されていない"
    return None


def check_disk():
    import shutil
    free = shutil.disk_usage(BASE_DIR).free / (1024 ** 3)
    if free < DISK_MIN_GB:
        return f"ディスクの空きが{free:.1f}GBしかない"
    return None


def main():
    dry = "--dry" in sys.argv
    h = datetime.now().hour
    quiet = h >= QUIET_HOURS[0] or h < QUIET_HOURS[1]

    race_day, times = _is_race_day()
    problems = []
    for name, res in [
        ("disk", check_disk()),
        ("heartbeat", None if quiet else check_heartbeat(dry)),
        ("keiba_auto", check_keiba_auto(times, dry) if race_day else None),
        ("freshness", check_freshness(times) if race_day else None),
    ]:
        if res:
            problems.append((name, res))

    if not problems:
        log(f"正常（開催日={race_day}）")
        return 0
    for name, msg in problems:
        log(f"異常: {msg}")
        notify(f"watchdog_{name}",
               f"⚠競馬AI 異常検知: {msg[:40]}",
               f"{datetime.now():%Y/%m/%d %H:%M} 見張り番が異常を検知しました。\n\n"
               f"{msg}\n\n"
               f"確認するもの:\n"
               f"  auto_predict_heartbeat.txt … スケジューラの心拍\n"
               f"  keiba_auto_run.err … 7分前ジョブの例外\n"
               f"  today_results.log … 結果取得の記録\n"
               f"  watchdog.log … この見張り番の記録\n\n"
               f"手で直す場合:\n"
               f"  schtasks /End /TN \"{TASK}\" ; schtasks /Run /TN \"{TASK}\"\n",
               dry)
    return 1


if __name__ == "__main__":
    sys.exit(main())
