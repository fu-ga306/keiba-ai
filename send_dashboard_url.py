# -*- coding: utf-8 -*-
"""ダッシュボードの公開URL(固定ngrokドメイン)をオンデマンドでメール送信する。
start_dashboardが既に起動中(二重起動ガードでメールをスキップ)でもURLを受け取れる。
使い方: python send_dashboard_url.py
"""
from start_dashboard import send_email, DASHBOARD_URL

if __name__ == "__main__":
    try:
        send_email(DASHBOARD_URL)
        print(f"公開URLをメール送信しました → {DASHBOARD_URL}")
    except Exception as e:
        print(f"メール送信失敗: {e}")
