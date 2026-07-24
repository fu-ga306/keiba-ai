"""
start_dashboard.py
Flask + ngrok を起動してダッシュボードURLをメールで通知する
"""
import os
import sys
import subprocess
import smtplib
import time
from email.mime.text import MIMEText
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASS = os.environ["GMAIL_APP_PASS"]
TO_ADDRESS = os.environ["TO_ADDRESS"]

NGROK_DOMAIN = "sway-uncanny-exonerate.ngrok-free.dev"
DASHBOARD_URL = f"https://{NGROK_DOMAIN}"
FLASK_PORT = 5000


def send_email(url: str):
    msg = MIMEText(
        f"競馬AIダッシュボードが起動しました。\n\n"
        f"📱 スマホでのURL:\n{url}\n\n"
        f"🔄 データは5分ごとにキャッシュ更新されます。",
        "plain",
        "utf-8",
    )
    msg["Subject"] = "🏇 競馬AI予想 ダッシュボード起動"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = TO_ADDRESS

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
        server.send_message(msg)
    print(f"メール送信完了 → {TO_ADDRESS}")


def _already_running():
    """Flask(5000番)が既に起動済みかを判定（自動起動での二重起動防止）。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", FLASK_PORT)) == 0


def main():
    print("=" * 50)
    print("  競馬AI予想 ダッシュボード起動")
    print("=" * 50)

    if _already_running():
        print(f"\nダッシュボードは既に起動中（ポート{FLASK_PORT}使用中）→ 二重起動せず終了")
        return

    # Flask 起動
    print("\n[1] Flask 起動中...")
    flask_proc = subprocess.Popen(
        ["python", "flask_app.py"],
        cwd=BASE_DIR,
        env={**os.environ, "PYTHONUTF8": "1"},
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    time.sleep(2)
    print(f"    -> http://localhost:{FLASK_PORT}")

    # ngrok 起動（固定ドメイン）
    print(f"\n[2] ngrok 起動中（固定ドメイン）...")
    ngrok_exe = os.path.join(BASE_DIR, "ngrok.exe")
    ngrok_proc = subprocess.Popen(
        [ngrok_exe, "http", f"--domain={NGROK_DOMAIN}", str(FLASK_PORT)],
        cwd=BASE_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    time.sleep(3)
    print(f"    -> {DASHBOARD_URL}")

    # メール送信
    print("\n[3] メール送信中...")
    try:
        send_email(DASHBOARD_URL)
    except Exception as e:
        print(f"    メール送信失敗: {e}")

    print("\n" + "=" * 50)
    print(f"  URL: {DASHBOARD_URL}")
    print("  このウィンドウは閉じてもOKです")
    print("  （Flask/ngrokのウィンドウは開いたまま）")
    print("=" * 50)
    # 自動起動（タスクスケジューラ等・非対話）ではinputで待たない。
    # Flask/ngrokは別プロセスで起動済みなので、このプロセスはそのまま終了してよい。
    if sys.stdin and sys.stdin.isatty():
        input("\nEnterで終了...")


if __name__ == "__main__":
    main()
