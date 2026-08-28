"""
auto_predict_publish.py
────────────────────────
当日予想の自動実行 + GitHub自動プッシュスクリプト。

動作：
  ① 午前7時  → 当日全レースを予想してtoday_predictions.csvに保存 → GitHubにプッシュ
  ② 各レース40分前 → 個別レースを再予想（オッズ確定後）→ GitHubにプッシュ

使い方:
    python auto_predict_publish.py        # スケジューラー起動
    python auto_predict_publish.py test   # 即時テスト実行（スケジューラーなし）
"""

import os
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import re
import time
import subprocess
import pickle
import schedule
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"

JYO_NAMES = {
    1:"札幌", 2:"函館", 3:"福島", 4:"新潟",  5:"東京",
    6:"中山", 7:"中京", 8:"京都", 9:"阪神", 10:"小倉",
}

PYTHON = r"C:/Users/別府飛河/AppData/Local/Microsoft/WindowsApps/python3.11.exe"

DISK_MIN_GB = 2.0   # この空き未満なら予想を中止してメール警告（満杯でサイレント停止を防ぐ）

# 外部コマンドの上限時間（2026-08-09）。
#   8/9の12:40、札幌7Rのジョブでスケジューラが固まり、以降の40分前ジョブが
#   1本も動かなくなった。プロセスは生きたままCPUを消費しない状態で、
#   相乗り取得も夜の後片付けも同じスレッドなので全て道連れになる。
#   subprocess.run にタイムアウトが無く、gitやpredictが無限に待てたのが原因。
#   ここで必ず上限を切り、1つのジョブの失敗が全体を止めないようにする。
GIT_TIMEOUT = 120        # git add/status/commit/push
PREDICT_TIMEOUT = 1500   # 1レースの予想（通常2〜3分）
HEARTBEAT = os.path.join(BASE_DIR, "auto_predict_heartbeat.txt")


# ── ディスク不足アラート ──────────────────────────────────────────────────
def _send_alert(subject, body):
    """.env のGmail認証で警告メールを送る（軽量・依存少）。"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = os.environ["GMAIL_ADDRESS"]
        msg["To"] = os.environ["TO_ADDRESS"]
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASS"])
            s.send_message(msg)
        print("  警告メール送信完了")
    except Exception as e:
        print(f"  警告メール送信失敗: {e}")


def check_disk_ok():
    """C:の空きが DISK_MIN_GB 以上か。不足ならFalse＋メール警告。"""
    import shutil
    free_gb = shutil.disk_usage(BASE_DIR).free / (1024 ** 3)
    if free_gb < DISK_MIN_GB:
        msg = (f"競馬AI: ディスク空きが {free_gb:.2f}GB しかありません（閾値{DISK_MIN_GB}GB）。\n"
               f"予想を中止しました。C:ドライブを空けてください（例: 管理者cmdで powercfg /h off、"
               f"ディスククリーンアップ、不要ファイル削除）。空けたら予想が再開します。")
        print(f"  ⚠ ディスク不足 {free_gb:.2f}GB → 予想中止・メール警告")
        _send_alert(f"⚠競馬AI ディスク不足 残{free_gb:.1f}GB 予想中止", msg)
        return False
    return True


# ── 心拍（スケジューラが動いていることの証跡）────────────────────────────
def _run_with_beat(cmd, timeout, interval=20):
    """長い子プロセスを、心拍を打ちながら待つ（2026-08-15追加）。

    なぜ必要か
      朝の一括予想は36レースで20分前後かかる。従来は subprocess.run で
      待つ間、待機ループが止まるので心拍も止まっていた。見張り番は8分で
      ハングとみなして再起動するため、正常な予想が途中で殺されていた
      （2026-08-15に発生。札幌8Rまでで中断し、新潟・中京の24レースが未予想）。

    心拍を打ち続けるので「動いている長い処理」と「本当のハング」を区別できる。
    timeout を超えたら kill するので、本当に固まった場合は従来どおり止まる。
    戻り値は subprocess.run と同じく returncode を持つオブジェクト。
    """
    import subprocess as _sp
    p = _sp.Popen(cmd, cwd=BASE_DIR, text=True)
    waited = 0
    while True:
        try:
            p.wait(timeout=interval)
            break
        except _sp.TimeoutExpired:
            _beat()                     # ← 生きていることを外に伝える
            waited += interval
            if waited >= timeout:
                p.kill()
                p.wait()
                raise _sp.TimeoutExpired(cmd, timeout)
    _beat()
    return p


def _beat():
    """待機ループを1周するたびに時刻を書く。

    プロセスが生きていてもジョブが進まない状態（2026-08-09に発生）は、
    プロセス一覧を見ても分からない。このファイルが更新され続けているか
    どうかで、スケジューラ本体が動いているかを外から確認できる。
    """
    try:
        with open(HEARTBEAT, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass


# ── ダッシュボード即時更新通知 ────────────────────────────────────────────
def notify_dashboard():
    """Flaskダッシュボードのキャッシュをクリアして最新データを即時反映する"""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000/api/refresh", timeout=3)
        print("  [Dashboard] キャッシュクリア → 最新データに更新")
    except Exception:
        pass  # Flask未起動の場合はスキップ（エラーにしない）


# ── Git自動プッシュ ───────────────────────────────────────────────────────
def _git_env():
    """認証プロンプトを一切出さない環境を作る（2026-08-09）。

    credential.helper=manager は認証が切れるとGUIのダイアログを出し、
    無人運用では誰も応答しないので git が永久に待つ。8/9のハングは
    これが原因の可能性が高い。プロンプトを禁止しておけば、認証切れは
    「待ち続ける」ではなく「即エラー」になり、次のジョブへ進める。
    認証情報自体は Windows資格情報マネージャーに保存済みなので、
    通常運転でプロンプトが必要になることはない。
    """
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never",
            "GIT_ASKPASS": "", "SSH_ASKPASS": ""}


def git_push(message: str):
    """today_predictions.csv と prediction_record_v2.csv をGitHubにプッシュ"""
    try:
        os.chdir(BASE_DIR)
        _env = _git_env()

        # 変更があるファイルだけ追加
        files = ["today_predictions.csv", "prediction_record_v2.csv", "today_bets.csv",
                 "odds_history.csv",   # オッズ変動特徴の蓄積データ（追記式・バックアップ用）
                 "today_results.csv",  # 同じ競馬場の終了レースの着順・払戻（2026-08-07）
                 "history_marks.csv"]  # 印と着順の履歴・1行1頭（2026-08-09）
        for f in files:
            path = os.path.join(BASE_DIR, f)
            if os.path.exists(path):
                subprocess.run(["git", "add", f], cwd=BASE_DIR, check=True,
                               timeout=GIT_TIMEOUT, env=_env)

        # 変更がなければスキップ。
        #   2026-08-09: ここは git status（リポジトリ全体）を見ていたため、
        #   実験用スクリプトなど無関係な未追跡ファイルがあると「変更あり」と
        #   誤判定し、対象ファイルに差分が無いのに commit を叩いて必ず失敗して
        #   いた。ステージした分だけを見るように直す。
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=GIT_TIMEOUT,
            env=_env
        )
        if not result.stdout.strip():
            print("  [Git] 対象ファイルに変更なし・スキップ")
            return

        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=BASE_DIR, check=True, timeout=GIT_TIMEOUT, env=_env
        )
        subprocess.run(
            ["git", "push"],
            cwd=BASE_DIR, check=True, timeout=GIT_TIMEOUT, env=_env
        )
        print(f"  [Git] プッシュ完了: {message}")

    except subprocess.TimeoutExpired:
        print(f"  [Git] {GIT_TIMEOUT}秒を超えたため中断（スケジューラを止めない）")
    except subprocess.CalledProcessError as e:
        print(f"  [Git] エラー: {e}")
    except Exception as e:
        print(f"  [Git] 予期せぬエラー: {e}")


# ── 全レース予想（午前7時実行） ───────────────────────────────────────────
def run_morning_prediction():
    """当日の全レースを予想してCSVに保存・GitHubにプッシュ"""
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    print(f"\n{'='*50}")
    print(f"[{now}] 朝の一括予想開始")
    print(f"{'='*50}")

    if not check_disk_ok():   # ディスク不足なら中止（満杯でのサイレント停止/クラッシュを防止）
        return

    # 開催日ガード: 非開催日は予想・プッシュをスキップ（get_today_racesが
    # 最小レース数チェック込みで空を返す→誤メール/誤予想を防止）。
    try:
        from keiba_auto import get_today_races
        if not get_today_races():
            print(f"[{now}] 本日は非開催（レース取得0件）→ 朝の一括予想をスキップ")
            return
    except Exception as e:
        print(f"  開催日チェックでエラー（続行）: {e}")

    # today_predictions.csv / today_bets.csv をリセット
    # （betsを消し忘れると前日レースの買い目が残存し、照合・ダッシュボードに混入する）
    for _f in ("today_predictions.csv", "today_bets.csv"):
        _p = os.path.join(BASE_DIR, _f)
        if os.path.exists(_p):
            os.remove(_p)

    # keiba_predict.py の today モードを実行
    #   ⚠ 2026-08-15: ここは36レースで20分前後かかる。従来は subprocess.run で
    #     待つ間、待機ループが回らず心拍が止まっていた。見張り番は8分止まったら
    #     ハングとみなして再起動するので、正常に動いている朝の予想が
    #     途中（札幌8R時点）で強制終了された。
    #     長い処理の間も心拍を打ち続けるようにして誤検知を防ぐ。
    try:
        result = _run_with_beat(
            [PYTHON, os.path.join(BASE_DIR, "keiba_predict.py"), "today"],
            timeout=PREDICT_TIMEOUT * 3,   # 36レース分。1レースあたりの想定×余裕
        )
        if result.returncode == 0:
            print(f"\n[{datetime.now().strftime('%H:%M')}] 朝の一括予想完了")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M')}] 予想エラー（returncode={result.returncode}）")
    except Exception as e:
        print(f"  予想実行エラー: {e}")
        return

    # GitHubにプッシュ → ダッシュボード即時更新
    date_str = datetime.now().strftime("%Y/%m/%d")
    git_push(f"当日予想更新 {date_str} 07:00")
    notify_dashboard()

    # 一括予想が終わってから keiba_auto.py を起動する（2026-08-08）。
    # 先に起動すると両方が同時に model.pkl を読み、メモリ不足で落ちる。
    run_keiba_auto()

    # note公開用のダイジェストをメール送信（2026-08-05追加）。
    # 買い推奨レースをMarkdownで整形して送るだけ。公開は手動で行う。
    # 失敗しても予想の処理は止めない。
    try:
        subprocess.run([PYTHON, os.path.join(BASE_DIR, "note_digest.py")],
                       cwd=BASE_DIR, timeout=300,
                       env=dict(os.environ, PYTHONUTF8="1"))
    except Exception as e:
        print(f"  note用ダイジェストの送信に失敗（続行）: {e}")


# ── 個別レース予想（各レース40分前実行） ──────────────────────────────────
# 締切の何分前にオッズだけ記録するか（2026-08-14）。
#   1分前は通信遅延で発走後になりかねないので2分前にする。
ODDS_SNAP_MIN = 2


def run_odds_snapshot(race_id: str, race_time: str):
    """締切直前のオッズだけを記録する（予想・メール・プッシュはしない）。

    なぜ必要か
      バックテストは確定オッズで買い目を決めるが、実運用は7分前で決める。
      この差だけでEV方式は 119.6% → 88.4%（-28.7pt）に落ちる。
      原因はモデルではなく「賭ける時刻」なので、締切に近づけるほど縮むはず。
      しかし締切直前のオッズを一度も記録していないため、縮み具合が測れない。
      半年貯めてから「投票を遅らせる価値があるか」を判定する。

    アクセスは1レースにつき1回だけ増える。予想処理は走らせない。
    """
    jyo_cd = int(str(race_id)[4:6])
    race_no = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
    now = datetime.now().strftime("%H:%M")
    print(f"[{now}] {jyo_name} {race_no}R オッズ記録のみ（発走: {race_time}）")
    try:
        subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "keiba_predict.py"), race_id,
             "nomail", "oddsonly"],
            cwd=BASE_DIR, capture_output=False, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⚠ {jyo_name} {race_no}R のオッズ記録がタイムアウト（予想には影響なし）")
    except Exception as e:
        print(f"  ⚠ オッズ記録に失敗（予想には影響なし）: {e}")


def run_race_prediction(race_id: str, race_time: str):
    """個別レースの予想を実行してGitHubにプッシュ"""
    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

    now = datetime.now().strftime("%H:%M")
    print(f"\n[{now}] {jyo_name} {race_no}R 予想開始（発走: {race_time}）")

    if not check_disk_ok():   # ディスク不足なら中止
        return

    try:
        result = subprocess.run(
            [PYTHON, os.path.join(BASE_DIR, "keiba_predict.py"), race_id, "nomail"],
            cwd=BASE_DIR,
            capture_output=False,
            text=True,
            timeout=PREDICT_TIMEOUT,
        )
        if result.returncode == 0:
            print(f"  {jyo_name} {race_no}R 予想完了")
        else:
            print(f"  {jyo_name} {race_no}R 予想エラー")
            return
    except subprocess.TimeoutExpired:
        print(f"  {jyo_name} {race_no}R 予想が{PREDICT_TIMEOUT}秒を超えたため中断")
        return
    except Exception as e:
        print(f"  予想実行エラー: {e}")
        return

    # 同じ競馬場の終わったレースの結果を取る（2026-08-07追加）。
    #   ダッシュボードに前レースの着順・払戻を出すため。
    #   ⚠️まだ取っていないレースだけを1件ずつ取る設計。毎回全部取りに行くと
    #     netkeibaのIPブロックを招く（2026-07-27に実際に400を食らった）。
    #   ここで貯めたものは21時の結果照合でも再利用され、取得が二重にならない。
    try:
        import today_results
        today_results.update_for_race(race_id)
    except Exception as e:
        print(f"  前レース結果の取得に失敗（続行）: {e}")

    # GitHubにプッシュ → ダッシュボード即時更新
    git_push(f"{jyo_name} {race_no}R 予想更新 {datetime.now().strftime('%H:%M')}")
    notify_dashboard()


# ── スケジュール設定 ──────────────────────────────────────────────────────
def setup_schedule():
    """当日のレーススケジュールを設定する"""
    from keiba_auto import get_today_races

    print("当日レース一覧を取得中...")
    race_info = get_today_races()
    if not race_info:
        print("本日のレースが取得できませんでした")
        return 0

    # 発走時刻を保存しておく（結果の後片付けジョブが「他のスクレイピングと
    # 被らない時刻か」を判定するのに使う。2026-08-07）
    try:
        import today_results
        today_results.save_race_times(race_info)
    except Exception as e:
        print(f"  発走時刻の保存に失敗（続行）: {e}")

    now = datetime.now()
    scheduled = 0

    # 前日に登録したレースジョブを破棄する（2026-08-02）。
    # 下の登録は every().day.at() ＝毎日繰り返しなので、常駐が日をまたぐと
    # 「昨日のrace_idを今日の同時刻に再予想してpush」が起き、ダッシュボードが
    # 前日のまま固まる。tagを付けてここで消すことで、その日の分だけが残る。
    schedule.clear("race")

    for race_id, race_time in sorted(race_info.items()):
        time_match = re.search(r"(\d{1,2}):(\d{2})", race_time)
        if not time_match:
            continue

        h = int(time_match.group(1))
        m = int(time_match.group(2)) - 40  # 40分前
        if m < 0:
            m += 60
            h -= 1

        notify_time  = f"{h:02d}:{m:02d}"
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if scheduled_dt < now:
            print(f"  スキップ（過去）: {race_id} {notify_time}")
            continue

        schedule.every().day.at(notify_time).do(
            run_race_prediction,
            race_id=race_id,
            race_time=race_time,
        ).tag("race")

        # 締切直前のオッズを記録する（2026-08-14追加）。
        #   なぜ必要か: バックテストは確定オッズで買い目を決めるが、実運用は
        #   7分前で決める。この差だけでEV方式は119.6%→88.4%に落ちる。
        #   差の原因は「賭ける時刻」なので、締切に近づけるほど縮むはず。
        #   しかし1分前のオッズを一度も記録していないため、縮み具合が測れない。
        #   予想は出さず、オッズだけを1回取る（アクセスは1レース1回だけ増える）。
        h2 = int(time_match.group(1))
        m2 = int(time_match.group(2)) - ODDS_SNAP_MIN
        if m2 < 0:
            m2 += 60
            h2 = (h2 - 1) % 24
        snap_dt = now.replace(hour=h2, minute=m2, second=0, microsecond=0)
        if snap_dt > now:
            schedule.every().day.at(f"{h2:02d}:{m2:02d}").do(
                run_odds_snapshot, race_id=race_id, race_time=race_time,
            ).tag("race")

        jyo_cd   = int(str(race_id)[4:6])
        race_no  = int(str(race_id)[10:12])
        jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
        print(f"  予約: {jyo_name} {race_no}R → {notify_time}（発走: {race_time}）")
        scheduled += 1

    return scheduled


# ── メイン ────────────────────────────────────────────────────────────────
def _free_mem_gb():
    """物理メモリの空き(GB)。取れなければ None。"""
    try:
        import ctypes

        class _S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = _S()
        s.dwLength = ctypes.sizeof(_S)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


def keiba_auto_alive():
    """keiba_auto.py が生きているか。pidファイルだけでは判定しない。"""
    pid_file = os.path.join(BASE_DIR, "keiba_auto.pid")
    if not os.path.exists(pid_file):
        return False
    try:
        pid = int(open(pid_file).read().strip())
    except Exception:
        return False
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                           capture_output=True, text=True, errors="ignore")
        return str(pid) in (r.stdout or "")
    except Exception:
        return False


def run_keiba_auto():
    """keiba_auto.py をサブプロセスで起動（7分前の直前更新・メール送信）。

    2026-08-08: 06:58に起動していたが、07:00の一括予想と model.pkl(数GB)の
    読込がぶつかり、メモリ不足で起動直後に死んでいた（pidファイルだけ残り、
    プロセスは存在しない状態）。出力を捨てていたため気づけなかった。
      ・起動を一括予想の完了後に移す（同時にモデルを読ませない）
      ・空きメモリが足りないときは起動を見送る
      ・出力を keiba_auto_run.log/.err に残す
    """
    if keiba_auto_alive():
        print("  keiba_auto.py は既に稼働中 → 起動しない")
        return
    free = _free_mem_gb()
    if free is not None and free < 1.5:
        print(f"  空きメモリ {free:.1f}GB → keiba_auto.py の起動を見送り"
              f"（動作中の予想を巻き込まないため。次の点検で再試行）")
        return
    print(f"\n[{datetime.now().strftime('%H:%M')}] keiba_auto.py 起動"
          f"（空きメモリ {free:.1f}GB）" if free is not None else "")
    try:
        # 出力を捨てると死因が分からなくなるのでファイルに残す
        out = open(os.path.join(BASE_DIR, "keiba_auto_run.log"), "a",
                   encoding="utf-8", errors="ignore")
        err = open(os.path.join(BASE_DIR, "keiba_auto_run.err"), "a",
                   encoding="utf-8", errors="ignore")
        out.write(f"\n===== 起動 {datetime.now():%Y/%m/%d %H:%M:%S} =====\n")
        out.flush()
        result = subprocess.Popen(
            [PYTHON, os.path.join(BASE_DIR, "keiba_auto.py")],
            cwd=BASE_DIR, stdout=out, stderr=err,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        print(f"  keiba_auto.py 起動完了 (PID: {result.pid})")
        with open(os.path.join(BASE_DIR, "keiba_auto.pid"), "w") as f:
            f.write(str(result.pid))
    except Exception as e:
        print(f"  keiba_auto.py 起動エラー: {e}")


def ensure_keiba_auto():
    """keiba_auto.py が落ちていたら復帰させる（15分おきの点検）。

    起動直後に死んでも、次の点検で拾い直せるようにする。最終レースを
    過ぎていれば何もしない。
    """
    if keiba_auto_alive():
        return
    try:
        import today_results
        times = today_results._race_times()
    except Exception:
        times = {}
    if not times:
        return                      # 非開催日、または発走時刻がまだ未登録
    cur = datetime.now().hour * 60 + datetime.now().minute
    if cur > max(times.values()):
        return                      # 全レース終了後は復帰させない
    print(f"[{datetime.now().strftime('%H:%M')}] keiba_auto.py が停止している"
          f"→ 復帰を試みます")
    run_keiba_auto()


def stop_keiba_auto():
    """keiba_auto.py を停止"""
    pid_file = os.path.join(BASE_DIR, "keiba_auto.pid")
    if not os.path.exists(pid_file):
        print("  keiba_auto.pid なし → スキップ")
        return
    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())
        import signal, subprocess
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  keiba_auto.py 停止完了 (PID: {pid})")
            else:
                print(f"  keiba_auto.py 既に終了済み (PID: {pid})")
        except Exception:
            os.kill(pid, signal.SIGTERM)
            print(f"  keiba_auto.py 停止完了 (PID: {pid})")
    except (OSError, PermissionError) as e:
        print(f"  keiba_auto.py 既に終了済み (無視): {e}")
    except Exception as e:
        print(f"  keiba_auto.py 停止エラー: {e}")
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)


def main():
    print(f"=== 自動予想・公開システム 起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}] ===\n")

    # 多重起動ガード（2026-07-20）: タスクスケジューラ6:55の自動起動と手動起動が
    # 重なると全ジョブが二重になる（7/19発生）。既存プロセスがいれば即終了。
    from keiba_auto import ensure_single_instance
    ensure_single_instance("auto_predict_publish.py")

    # 午前7時の一括予想をスケジュール
    schedule.every().day.at("07:00").do(run_morning_prediction)
    print("  [済] 朝7時の一括予想をスケジュール登録")

    # keiba_auto.py の停止（21時）。起動は run_morning_prediction の最後で行う。
    #   2026-08-08: 06:58起動だと07:00の一括予想とモデル読込が重なり、
    #   メモリ不足で起動直後に死んでいた。一括予想が終わってから起動する。
    schedule.every().day.at("21:00").do(stop_keiba_auto)
    print("  [済] keiba_auto.py 停止(21:00)をスケジュール登録")

    # keiba_auto.py の生存点検（15分おき）。落ちていたら復帰させる。
    schedule.every(15).minutes.do(ensure_keiba_auto)
    print("  [済] keiba_auto.py 生存点検(15分おき)をスケジュール登録")

    # 当日レースの個別予想スケジュールを設定
    now = datetime.now()

    # 個別レースの登録は毎日7:05に行う（2026-08-02に毎日化）。
    # 以前は一度きり(CancelJob)だったため、常駐が日をまたぐと翌日のレースが
    # 1本も登録されないまま動き続けていた。
    schedule.every().day.at("07:05").do(setup_and_register)
    print("  [済] 個別レース登録(07:05・毎日)をスケジュール登録")

    # 7時以降に手動起動した場合は、7:05を待たず今すぐ当日分を登録する
    if now.hour >= 7:
        n = setup_schedule()
        print(f"  {n}レースをスケジュール登録（起動時）")

    # 日次で自ら終了する（2026-08-02）。
    # レースジョブは every().day.at() ＝毎日繰り返しのため、常駐が生き続けると
    # 前日分のジョブが翌日も動く。タスクスケジューラが毎朝6:55に起動し直す設計に
    # 合わせ、夜に必ず落として翌朝まっさらな状態から始める。
    # 結果の後片付け（2026-08-07）。相乗り取得では後続レースのない最終レースが
    # 漏れるため、レース終了後の空き時間に回収する。today_results.sweep() が
    # 「発走±15分に入っていないか」を毎回自分で確認するので、ここでは10分おきに
    # 声をかけるだけでよい（17:00〜20:40の外なら何もせず戻る）。
    schedule.every(10).minutes.do(run_result_sweep)
    print("  [済] 結果の後片付け(17:00-20:40・10分おき)をスケジュール登録")

    # その日の予想＋結果を1行1頭で履歴に積む（2026-08-09）。
    #   today_*.csv は翌朝上書きされるため、印ごとの成績や評価グレードの精度を
    #   後から検証できるのはこの履歴だけ。21:00の結果照合の後に実行する。
    schedule.every().day.at("21:10").do(run_daily_archive)
    print("  [済] 日次アーカイブ(21:10)をスケジュール登録")

    # 当日ぶんの払戻取得（2026-08-24追加）。日次点検(21:20)より前に置く。
    #   ワイドの払戻は today_results.csv にも history_marks.csv にも無いので、
    #   ここで取らないと週次まで的中が分からない。
    schedule.every().day.at("21:05").do(run_payout_today)
    print("  [済] 当日払戻の取得(21:05)をスケジュール登録")

    # 日次点検（2026-08-24追加）。アーカイブ(21:10)の後、終了(22:30)の前。
    #   audit_system.py はどのスケジュールにも入っていなかった。書いてあるのに
    #   回していなければ、落ちていても壊れていても分からない。
    schedule.every().day.at("21:20").do(run_daily_check)
    print("  [済] 日次点検(21:20)をスケジュール登録")

    schedule.every().day.at("22:30").do(_nightly_exit)
    print("  [済] 日次終了(22:30)をスケジュール登録")

    print(f"\n待機中... (Ctrl+Cで停止)\n")
    while True:
        _beat()
        schedule.run_pending()
        time.sleep(10)


def run_result_sweep():
    """レース終了後、まだ取れていない当日結果を回収してダッシュボードに反映。"""
    try:
        import today_results
        n = today_results.sweep()
    except Exception as e:
        print(f"  結果の後片付けに失敗（続行）: {e}")
        return
    if n:
        git_push(f"当日結果を更新 {datetime.now().strftime('%H:%M')}")


def run_daily_archive():
    """その日の予想＋結果を history_marks.csv に積んでプッシュする。"""
    try:
        import archive_daily
        n = archive_daily.archive()
    except Exception as e:
        print(f"  日次アーカイブに失敗（続行）: {e}")
        return
    if n:
        git_push(f"履歴を蓄積 {datetime.now().strftime('%m/%d')}")


def run_payout_today():
    """その日のレースの払戻だけを取得する（2026-08-24追加）。

    なぜ要るか
      払戻表(payout_data.csv)は週次でしか取りに行かない。単勝の払戻は
      today_results.csv / history_marks.csv にあるので即日照合できるが、
      **ワイドはどちらにも無い**ので、週次まで的中も外れも分からなかった。
      8/22-8/23のワイド10点が丸ごと評価保留になっていた。
      その日のぶんだけ取れば、翌朝には全券種の収支が出る。

    アクセスは当日の36レースぶんだけ。既に払戻表にあるレースは飛ばす。
    payout_scraper 側のレート制限(_throttle)をそのまま使う。
    """
    try:
        import payout_scraper as ps
    except Exception as e:
        print(f"  払戻取得を読み込めません: {e}")
        return
    # 今日どのレースを予想したかは paper_resid.csv が持っている
    src = os.path.join(BASE_DIR, "paper_resid.csv")
    if not os.path.exists(src):
        print("  paper_resid.csv が無いので払戻取得をスキップ")
        return
    d = pd.read_csv(src, dtype={"race_id": str})
    today = datetime.now().strftime("%Y/%m/%d")
    ids = sorted(set(d.loc[d["記録時刻"].astype(str).str.startswith(today), "race_id"]))
    if not ids:
        print("  今日の記録が無いので払戻取得をスキップ")
        return
    done = ps._load_done_race_ids()
    todo = [r for r in ids if r not in done]
    print(f"  払戻取得: 今日{len(ids)}レース中 {len(todo)}レースが未取得")
    # 時間上限（2026-08-24）。21:10のアーカイブ、21:20の点検と同じスレッドで
    #   直列に走るので、ここが長引くと後ろが全部ずれる。8/9に1つのジョブが
    #   固まって以降のジョブが全滅した事故と同じ形。
    #   36レースなら通常2.8分だが、Seleniumに落ちると倍以上かかる。
    #   途中で切り上げても、残りは週次(Step0.55)が拾い直す。
    PAYOUT_BUDGET = 240      # 秒
    t0 = time.time()
    got = rows_all = 0
    for rid in todo:
        if time.time() - t0 > PAYOUT_BUDGET:
            print(f"  時間上限{PAYOUT_BUDGET}秒に達したので中断"
                  f"（残り{len(todo)-got}レースは週次で取得）")
            break
        try:
            rows = ps.get_payout(rid)
        except Exception as e:
            print(f"  {rid} 取得失敗: {type(e).__name__}")
            continue
        if rows == "BLOCKED":
            # ブロックされたら即やめる。粘るとIP遮断が長引く。週次で拾い直せる。
            print("  ⚠ netkeibaにブロックされたため中止（週次で取り直します）")
            break
        if rows:
            ps._save(rows, append=True)
            got += 1
            rows_all += len(rows)
    try:
        ps._close_fallback_driver()
    except Exception:
        pass
    print(f"  払戻取得完了: {got}レース / {rows_all}行")
    if got:
        git_push(f"払戻を取得 {datetime.now():%m/%d}")


def run_daily_check():
    """日次の自己点検。audit_system と paper_report を実際に走らせて結果を送る。

    なぜ要るか（2026-08-24）
      8/22-8/23の2日間で11件の不具合が見つかったが、自動検知はゼロだった。
      全部あとから手で調べて出たもの。特に危ないのは「沈黙する故障」で、
      たとえば7分前メールが丸一日1通も飛んでいなかったのに、
      ログにもエラーは出ず、誰も気づかなかった。
      能動的に数えて突きつけないと分からない種類の壊れ方をする。

      audit_system.py は書いてあったのに、どのスケジュールにも入っていなかった
      （grepして0件）。週次にも日次にも無い。書いただけで回していなかった。
    """
    import subprocess as _sp
    parts, ng_high, ng_mid = [], 0, 0
    for name in ("audit_system.py", "paper_report.py"):
        try:
            # ⚠ 子プロセスにUTF-8を明示する（2026-08-24）
            #   これが無いと子の標準出力は cp932 になり、
            #   ①「⚠」(U+26A0)を print した瞬間に UnicodeEncodeError で落ちる
            #   ②こちらは encoding="utf-8" で読むので、届いた分も文字化けする
            #   実際に初回の日次点検で両方起き、2本とも異常終了した。
            #   スクリプト側にも同じ対策を入れてあるが、環境変数でも二重に塞ぐ。
            _env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            r = _sp.run([PYTHON, os.path.join(BASE_DIR, name)],
                        cwd=BASE_DIR, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=900,
                        env=_env)
            out = (r.stdout or "") + (r.stderr or "")
            if r.returncode != 0:
                out += f"\n⚠ {name} が異常終了しました（コード{r.returncode}）"
                ng_high += 1
        except Exception as e:
            out = f"⚠ {name} を実行できませんでした: {type(e).__name__}: {e}"
            ng_high += 1
        # 件数を拾って件名に出す（本文を読まなくても異常が分かるように）
        m = re.search(r"【重要度高】(\d+)件", out)
        if m:
            ng_high += int(m.group(1))
        m = re.search(r"【重要度中】(\d+)件", out)
        if m:
            ng_mid += int(m.group(1))
        parts.append(f"{'=' * 60}\n■ {name}\n{'=' * 60}\n{out}")

    # 7分前メールが何通飛んだか。0通なら沈黙する故障を疑う
    mail_line = ""
    try:
        lp = os.path.join(BASE_DIR, "keiba_auto_run.log")
        t = open(lp, encoding="utf-8", errors="ignore").read()
        i = t.rfind("件を予約しました")
        today = t[i:] if i > 0 else ""
        sent = len(re.findall(r"メール送信完了", today))
        skip = len(re.findall(r"メール非送信", today))
        ran = len(re.findall(r"予測開始", today))
        mail_line = (f"7分前ジョブ: 予想{ran}レース / メール送信{sent}通 / 非送信{skip}件")
        if ran and sent == 0:
            mail_line += "\n  ⚠ 1通も飛んでいません。買い判定が全滅か、配信経路の故障を疑ってください。"
            ng_high += 1
    except Exception:
        pass

    # 販売ページの合言葉。期間が変わる月だけ、note に貼る文面を添える
    #   ⚠ note には投稿APIが無いので、ここだけは手で貼るしかない。
    #     せめて「そのまま貼れる完成形」で渡し、考える手間をゼロにする。
    sale_line = ""
    try:
        import sale_gate
        d = datetime.now()
        # 配信は金曜（2026-08-27に月曜から変更）。
        #   開催は土日なので、直前の金曜に出すほうが読まれる。
        #   月曜に出すと週末までに忘れられる。
        #   合言葉はISO週で切り替わり、土日は同じ週の終わりに入る。
        #   つまり金曜に配る合言葉は、その週末をそのままカバーする。
        SALE_DOW = 4          # 0=月 4=金
        soon = (d.day <= 3) if sale_gate.ROTATE == "month" else (d.weekday() == SALE_DOW)
        if soon:
            # ⚠ .env をここで読む（2026-08-28）
            #   _send_alert が load_dotenv するのはメール送信の直前で、
            #   この行より**後**。先に environ を見ても SALE_URL は入っておらず、
            #   案内文のURLが「（ダッシュボードのURL）」のまま配信されていた。
            #   note_weekly.py は自分で .env を読むので下書き側は正しく、
            #   **同じメールの中でURLが食い違う**という気づきにくい形で出ていた。
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(BASE_DIR, ".env"))
            except Exception:
                pass
            url = os.environ.get("SALE_URL", "（ダッシュボードのURL）")
            sale_line = ("\n" + "=" * 60
                         + "\n■ 販売ページ：note に貼る文面（合言葉が変わりました）\n"
                         + "=" * 60 + "\n"
                         + sale_gate.note_block(url, d) + "\n")
            # 記事の下書きも丸ごと添える。**そのまま貼れる完成形**にして、
            # 毎週の手作業を「コピーして貼る」だけにする。
            try:
                import subprocess as _sp2
                _e2 = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
                _r2 = _sp2.run([PYTHON, os.path.join(BASE_DIR, "note_weekly.py"),
                                "--print"], cwd=BASE_DIR, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=600, env=_e2)
                if _r2.returncode == 0 and _r2.stdout.strip():
                    sale_line += ("\n" + "=" * 60
                                  + "\n■ 今週の販売note（下書き・そのまま貼れます）\n"
                                  + "=" * 60 + "\n" + _r2.stdout.strip() + "\n")
                else:
                    sale_line += "\n（note下書きの生成に失敗しました。実績データを確認してください）\n"
            except Exception as _e3:
                sale_line += f"\n（note下書きを作れませんでした: {type(_e3).__name__}）\n"
    except Exception:
        pass

    # 撤退条件の判定日までの残り（2026-08-27に事前登録）
    #   金銭で測ると労力に見合わないので期限を切った。
    #   だらだら続けて消耗するのを避けるため、残り日数を毎日突きつける。
    #   条件は 撤退条件.md に固定してある。**判定日まで動かさない。**
    dead_line = ""
    try:
        from datetime import date
        _left = (date(2026, 11, 30) - date.today()).days
        if _left >= 0:
            dead_line = (f"撤退条件の判定日まで あと{_left}日"
                         f"（2026/11/30・撤退条件.md）\n")
        else:
            dead_line = ("⚠ 撤退条件の判定日を過ぎています。"
                         "撤退条件.md の記録欄を埋めて判断してください\n")
    except Exception:
        pass

    head = (f"競馬AI 日次点検 {datetime.now():%Y/%m/%d}\n\n"
            f"重要度高 {ng_high}件 / 重要度中 {ng_mid}件\n"
            + (mail_line + "\n" if mail_line else "")
            + dead_line
            + sale_line
            + "\n（このメールは毎晩21:20に自動送信されます）\n\n")
    mark = "⚠" if ng_high else "○"
    _send_alert(f"【競馬AI 日次点検 {mark}高{ng_high}/中{ng_mid}】"
                f"{datetime.now():%m/%d}", head + "\n\n".join(parts))
    print(f"  日次点検を送信（高{ng_high}/中{ng_mid}）")


def _nightly_exit():
    """翌朝6:55のタスク起動に備えて常駐を終了する。"""
    print(f"\n[{datetime.now().strftime('%H:%M')}] 日次終了。"
          f"翌朝6:55にタスクスケジューラが起動し直します。")
    sys.stdout.flush()
    sys.exit(0)


def setup_and_register():
    """7時5分に個別レーススケジュールを設定（一括予想完了後・毎日実行）"""
    n = setup_schedule()
    print(f"  {n}レースをスケジュール登録（7:05）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # テスト：即時実行
        print("テストモード: 朝の一括予想を即時実行します")
        run_morning_prediction()
    elif len(sys.argv) > 1 and sys.argv[1] == "schedule":
        # スケジュールのみ設定（一括予想なし）
        n = setup_schedule()
        print(f"\n{n}レースをスケジュール登録。待機中...\n")
        while True:
            _beat()
            schedule.run_pending()
            time.sleep(10)
    else:
        main()