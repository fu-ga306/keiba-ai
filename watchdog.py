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
  ④ 結果取得の進み具合  … 終了後に today_results が揃っているか
  ⑤ 蓄積の成否         … その日の分が history_marks.csv に積まれたか
  ⑥ オッズ記録         … odds_history に今日の分と時間軸(分前)が入ったか
  ⑦ ディスクの空き

  ④⑤は半年かけて印を検証するための生命線。予想さえ動いていれば従来の見張り番は
  「正常」と報告し続けるので、蓄積だけが静かに止まる事故を防げなかった。

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


def check_results(times):
    """結果の取得が進んでいるか。開催日の夕方以降だけ見る。

    半年かけて印を検証するのが目的なので、蓄積が止まることが最大の事故になる。
    ところが従来の見張り番は予想(today_predictions)の鮮度しか見ておらず、
    結果取得が壊れても「正常」と報告し続けていた（2026-08-10に判明）。
    today_results が空だと archive_daily が何も残さず、half年後に分析する
    データが存在しない、という事態になる。
    """
    if not times:
        return None
    cur = datetime.now().hour * 60 + datetime.now().minute
    last = max(times.values())
    if cur < last + 45:          # 最終レースの確定と後片付けを待つ
        return None
    p = os.path.join(BASE_DIR, "today_results.csv")
    if not os.path.exists(p):
        return "開催日なのに today_results.csv が無い（結果取得が動いていない）"
    try:
        import pandas as pd
        d = pd.read_csv(p, dtype={"race_id": str})
        got = d["race_id"].nunique() if "race_id" in d.columns else 0
    except Exception as e:
        return f"today_results.csv を読めない: {str(e)[:60]}"
    if got == 0:
        return "結果が1レースも取れていない"
    if got < len(times) * 0.8:
        return f"結果取得が {got}/{len(times)}レースで止まっている"
    return None


def check_odds_history(times):
    """オッズ記録が今日の分も貯まっているか。開催日の夕方以降に見る。

    1年後にオッズ変動を特徴量にするための唯一の材料。記録が止まっても
    予想は動き続けるので、見張っていないと沈黙したまま欠落する（2026-08-11追加）。
    発走時刻・分前が入っているかも確認する。ここが空だと時間軸が作れず、
    貯まっていても使い物にならない。
    """
    if not times:
        return None
    cur = datetime.now().hour * 60 + datetime.now().minute
    if cur < max(times.values()) + 45:
        return None
    p = os.path.join(BASE_DIR, "odds_history.csv")
    if not os.path.exists(p):
        return "開催日なのに odds_history.csv が無い"
    try:
        import pandas as pd
        d = pd.read_csv(p, dtype={"race_id": str})
        today = set(times.keys())
        got = set(d["race_id"].astype(str)) & today
        if len(got) < len(today) * 0.8:
            return f"オッズ記録が {len(got)}/{len(today)}レースしかない"
        s = d[d["race_id"].astype(str).isin(today)]
        if "分前" not in s.columns or s["分前"].astype(str).str.strip().eq("").mean() > 0.5:
            return "オッズ記録に発走時刻・分前が入っていない（時間軸が作れない）"
    except Exception as e:
        return f"odds_history.csv を読めない: {str(e)[:60]}"
    return None


def check_archive(times):
    """その日の分が history_marks.csv に積まれたか。21:30以降に見る。"""
    if not times:
        return None
    cur = datetime.now().hour * 60 + datetime.now().minute
    if not (21 * 60 + 30 <= cur <= 22 * 60 + 20):
        return None
    p = os.path.join(BASE_DIR, "history_marks.csv")
    if not os.path.exists(p):
        return "開催日なのに history_marks.csv が無い（蓄積が始まっていない）"

    def _retry():
        """積まれていなければ自分で積み直す（2026-08-11追加）。
        通報だけだと人が動くまで欠落したままになる。22:30に常駐が落ちると
        その日の分は永久に失われるので、検知した時点で自動でやり直す。"""
        try:
            sys.path.insert(0, BASE_DIR)
            import importlib
            import archive_daily
            importlib.reload(archive_daily)
            n = archive_daily.archive()
            log(f"  日次アーカイブを自動で再実行 → {n}レース")
            return bool(n)
        except Exception as e:
            log(f"  日次アーカイブの再実行に失敗: {str(e)[:70]}")
            return False
    try:
        import pandas as pd
        d = pd.read_csv(p, usecols=["日付"], dtype=str)
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in set(d["日付"].astype(str)):
            if _retry():                      # 自力で直せたら通報しない
                return None
            return f"{today} 分が history_marks.csv に積まれていない（自動再実行も失敗）"
    except Exception as e:
        return f"history_marks.csv を読めない: {str(e)[:60]}"
    return None


def check_fallback():
    """MFが読めず通常モデルに落ちた形跡がないか（2026-08-11追加）。

    フォールバックしても予想は出てしまうので、黙って劣化する。
    半年間これが立たなければ通常モデル(511MB)を撤去できる。
    """
    p = os.path.join(BASE_DIR, "fallback_triggered.flag")
    if not os.path.exists(p):
        return None
    try:
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]
        today = datetime.now().strftime("%Y-%m-%d")
        hit = [l for l in lines if l.startswith(today)]
        if hit:
            return f"MFモデルが読めず通常モデルに落ちている: {hit[-1].strip()[:80]}"
    except Exception:
        pass
    return None


def check_disk():
    import shutil
    free = shutil.disk_usage(BASE_DIR).free / (1024 ** 3)
    if free < DISK_MIN_GB:
        return f"ディスクの空きが{free:.1f}GBしかない"
    return None


def check_dashboard(dry=False):
    """販売するダッシュボードが、外から見えるかを確かめる（2026-08-27追加）。

    なぜ要るか
      売り物はダッシュボードなのに、flask も ngrok もタスク登録されておらず
      手動起動だった。PCを再起動したら復活しない。
      予想システムには見張り番が付いているのに、売り物にだけ付いていなかった。

    なぜ外形監視か
      **プロセスが生きていても、ngrokのトンネルが切れていれば外からは見えない。**
      内側の生死確認では検出できない。7分前メールが丸一日飛ばなかったのと同じ、
      沈黙する故障の型。実際に自分のURLを叩いて確かめる。

    稼働窓（金06:00〜月09:00）の外なら何もしない。平日は誰も見に来ない。
    """
    try:
        import dashboard_service as ds
    except Exception as e:
        return f"dashboard_service を読めません: {type(e).__name__}"
    if dry:
        if not ds.in_window():
            return None
        code, err = ds.probe()
        return None if code == 200 else f"ダッシュボードが見えません（{err or code}）"

    # ⚠ 窓の外でも ensure() を呼ぶ（2026-08-28に修正）
    #   以前は窓の外で早期returnしていたため、**「窓の外なので停止」が
    #   一度も実行されなかった。** 結果、旧コードのプロセスが動き続け、
    #   閲覧制限を足しても反映されない状態が続いた。
    #   窓の判定は ensure() が持っているので、ここでは呼ぶだけでよい。
    msg, warn = ds.ensure()
    return f"ダッシュボード: {msg}" if warn else None


def main():
    dry = "--dry" in sys.argv
    h = datetime.now().hour
    # 22時以降は通報しないのが基本だが、蓄積の失敗だけは当日中に知りたいので
    # 22:20までは通す（22:30に常駐が落ちるとその日の分は復旧できない）。
    quiet = (h >= QUIET_HOURS[0] or h < QUIET_HOURS[1]) and not (
        h == 22 and datetime.now().minute <= 20)

    race_day, times = _is_race_day()
    problems = []
    for name, res in [
        ("disk", check_disk()),
        ("heartbeat", None if quiet else check_heartbeat(dry)),
        ("keiba_auto", check_keiba_auto(times, dry) if race_day else None),
        ("freshness", check_freshness(times) if race_day else None),
        ("results", check_results(times) if race_day else None),
        ("archive", check_archive(times) if race_day else None),
        ("odds_history", check_odds_history(times) if race_day else None),
        ("fallback", check_fallback()),
        ("dashboard", check_dashboard(dry)),
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
