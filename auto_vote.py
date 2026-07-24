# -*- coding: utf-8 -*-
"""自動投票モジュール。7分前ジョブが確定した today_bets.csv（直前オッズで確定した買い目）を
IPATへ投票する。**安全第一：既定OFF＋ドライラン**。トグルはこのファイル冒頭を1行変更するだけ。

━━ 安全設計（多重ガード）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. AUTO_VOTE_ENABLED=False なら完全無効（何もしない）。既定はこれ。
  2. VOTE_MODE="dryrun" は実投票せず vote_log.csv に「投票予定」だけ記録（検証用）。
  3. 実投票 "ipat" は次の3つが全て揃った時のみ発動。1つでも欠ければ自動でドライランに降格:
        ① VOTE_MODE="ipat"
        ② プロジェクト直下に AUTO_VOTE_ARMED ファイルが存在（物理的な最終スイッチ）
        ③ .env に IPAT_INET_ID / IPAT_SUBSCRIBER / IPAT_PASSWORD / IPAT_PARS が設定済み
  4. 日次予算上限 DAILY_BUDGET_MAX / 1レース上限 PER_RACE_MAX / 日次損失ストップ DAILY_LOSS_STOP。
  5. すべての投票（予定/実行/スキップ）を vote_log.csv に記録。

  ※IPAT自動化はJRAの利用規約上グレー。自己資金・自己アカウントの個人利用が前提。
    実投票は必ず少額で挙動確認してから本運用すること。
"""
import os
import csv
from datetime import datetime

BASE_DIR = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

# ━━ トグル（ここだけ変えれば切替できる）━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTO_VOTE_ENABLED = True         # マスタースイッチ。Trueで投票処理が動く
VOTE_MODE = "dryrun"             # "dryrun"=記録のみ(実弾ゼロ) / "ipat"=実投票（要ARMED＋認証）
#   ↑2026-07-24: 今週から予想検証のためドライランON。実投票にはVOTE_MODE="ipat"＋
#     AUTO_VOTE_ARMEDファイル＋IPAT認証＋_submit_ipatのSETUP_DONE=True が全て必要。
# ━━ 安全弁 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DAILY_BUDGET_MAX = 30000         # 1日の投票総額上限(円)。超過分は以降スキップ
PER_RACE_MAX     = 5000          # 1レースの投票上限(円)。today_bets側の予算と二重チェック
DAILY_LOSS_STOP  = None          # 1日の実損失がこれを超えたら以降停止(円)。Noneで無効

ARMED_FILE = os.path.join(BASE_DIR, "AUTO_VOTE_ARMED")
VOTE_LOG   = os.path.join(BASE_DIR, "vote_log.csv")
LOG_COLS   = ["日時", "race_id", "券種", "買い方", "組み合わせ", "金額", "mode", "status", "備考"]


def _today():
    return datetime.now().strftime("%Y/%m/%d")


def _log_rows(rows):
    new = not os.path.exists(VOTE_LOG)
    with open(VOTE_LOG, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LOG_COLS})


def _today_spent():
    """本日すでに投票（実行 or dryrun予定）した合計額。日次予算のカウントに使う。"""
    if not os.path.exists(VOTE_LOG):
        return 0
    total = 0
    with open(VOTE_LOG, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("日時", "").startswith(_today()) and r.get("status") in ("投票", "投票予定"):
                try:
                    total += int(float(r.get("金額", 0) or 0))
                except ValueError:
                    pass
    return total


def _read_today_bets(race_id):
    """today_bets.csv から当該レースの買い目行（金額つき）を読む。"""
    import pandas as pd
    path = os.path.join(BASE_DIR, "today_bets.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype={"race_id": str})
    df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    sub = df[df["race_id"] == str(race_id)]
    return sub.to_dict("records")


def _effective_mode():
    """実際に効くモードを返す。ipatでも条件が欠ければdryrunへ降格。"""
    if VOTE_MODE != "ipat":
        return "dryrun", ""
    if not os.path.exists(ARMED_FILE):
        return "dryrun", "ARMEDファイル無し→dryrun降格"
    need = ["IPAT_INET_ID", "IPAT_SUBSCRIBER", "IPAT_PASSWORD", "IPAT_PARS"]
    if any(not os.environ.get(k) for k in need):
        return "dryrun", "IPAT認証情報不足→dryrun降格"
    return "ipat", ""


def place_race_bets(race_id):
    """7分前ジョブから呼ぶ入口。today_betsの買い目を投票（既定は無効/ドライラン）。"""
    if not AUTO_VOTE_ENABLED:
        return
    race_id = str(race_id)
    bets = _read_today_bets(race_id)
    if not bets:
        print(f"  [自動投票] {race_id}: 買い目なし・スキップ")
        return

    mode, downgrade = _effective_mode()
    spent = _today_spent()
    race_total = sum(int(float(b.get("金額", 0) or 0)) for b in bets)

    # ── 安全弁 ──
    if race_total > PER_RACE_MAX:
        print(f"  [自動投票] {race_id}: 1レース{race_total}円 > 上限{PER_RACE_MAX}円 → スキップ")
        _log_rows([{"日時": datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "race_id": race_id,
                    "金額": race_total, "mode": mode, "status": "スキップ", "備考": "1レース上限超過"}])
        return
    if spent + race_total > DAILY_BUDGET_MAX:
        print(f"  [自動投票] {race_id}: 本日累計{spent}+{race_total} > 日次上限{DAILY_BUDGET_MAX}円 → スキップ")
        _log_rows([{"日時": datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "race_id": race_id,
                    "金額": race_total, "mode": mode, "status": "スキップ", "備考": "日次予算超過"}])
        return

    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if mode == "ipat":
        try:
            ok = _submit_ipat(race_id, bets)
            status = "投票" if ok else "失敗"
        except Exception as e:
            print(f"  [自動投票] IPAT投票エラー: {e} → 記録のみ")
            status = "失敗"
    else:
        status = "投票予定"   # dryrun

    rows = [{"日時": now, "race_id": race_id, "券種": b.get("券種"), "買い方": b.get("買い方"),
             "組み合わせ": b.get("組み合わせ"), "金額": b.get("金額"), "mode": mode,
             "status": status, "備考": downgrade} for b in bets]
    _log_rows(rows)
    tag = "実投票" if mode == "ipat" and status == "投票" else ("ドライラン" if mode == "dryrun" else status)
    print(f"  [自動投票] {race_id}: {len(bets)}点 計{race_total}円 → {tag}"
          f"（本日累計{spent + race_total}円/上限{DAILY_BUDGET_MAX}）"
          + (f" ※{downgrade}" if downgrade else ""))


# ══ IPAT実投票バックエンド ═══════════════════════════════════════════════
#   実弾。selectorはIPATのUI変更で壊れやすく、実ページでの検証が必須のため、
#   既定では未接続（下の SETUP_DONE=False の間は必ず失敗を返してドライランに留める）。
#   ユーザーが少額で挙動確認しながら selector を確定させたら SETUP_DONE=True にする。
SETUP_DONE = False   # 実投票フローを検証済みにしたら True（未検証のうちは絶対に投票しない）


def _submit_ipat(race_id, bets):
    """IPATへ実投票する。戻り値True=成功。未検証(SETUP_DONE=False)なら投票せずFalse。"""
    if not SETUP_DONE:
        print("  [自動投票] IPATバックエンド未検証(SETUP_DONE=False) → 実投票せず。"
              "少額での検証・selector確定後に有効化してください。")
        return False
    # ── 以下、実装時に少額検証しながら確定させる（IPAT通常購入フロー）──
    #   1. Selenium(実Chrome)でIPATログイン(INET-ID→加入者番号/暗証番号/P-ARS番号)
    #   2. 「通常購入」→ 場・レース・式別・馬番・金額 を bets から入力
    #   3. 「購入予定リスト」で金額合計を検算（race_total と一致確認）
    #   4. 一致時のみ「購入する」を実行、結果を確認
    #   ※各ステップでログ・スクショを残し、失敗時は投票中止(False)
    raise NotImplementedError("IPAT購入フローは少額検証しながら実装・有効化する")
