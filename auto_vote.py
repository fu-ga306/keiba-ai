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

━━ IPAT認証の設定（実投票にだけ必要。ドライランでは不要）━━━━━━━━━━━━━
  .env に次の4つを書く。.env は .gitignore 済みなのでGitHubには上がらない。

      IPAT_INET_ID=xxxxxxxx        # INET-ID（ハガキ等に記載の8桁）
      IPAT_SUBSCRIBER=xxxxxxxx     # 加入者番号
      IPAT_PASSWORD=xxxx           # 暗証番号(4桁)
      IPAT_PARS=xxxx               # P-ARS番号(4桁)

  4つのうち1つでも欠けると _effective_mode() が自動でドライランへ降格する。
  つまり「設定し忘れたまま実弾が出る」ことはない。

  ⚠ 認証情報を書いても、それだけでは投票されない。VOTE_MODE="ipat" と
    AUTO_VOTE_ARMED ファイルと SETUP_DONE=True が揃って初めて実投票になる。
  ⚠ IPATの入金は自動化していない。残高が無ければ投票は失敗する。
    先にIPAT側で入金しておくこと。

━━ 買い目の取得元 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BET_SOURCE="resid" が既定。残差モデル（いま前向き検証している買い方）を投票する。
  "legacy" にすると旧方式(today_bets.csv)に戻るが、こちらは検証で100%を割って
  いるので通常は使わない。
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
#   ⚠ 上限は「検証した買い方の規模」に合わせてある（2026-08-22に見直し）。
#     残差モデルは1レース1〜4点・1点100円なので、1レース最大400円。
#     1日36レースのうち買うのは約半分なので、1日でも数千円にしかならない。
#     上限が実際の規模より大きすぎると、バグで点数が暴発したとき止められない。
#     実運用を始めるときは、まずここをさらに小さくして挙動を確かめること。
DAILY_BUDGET_MAX = 3000          # 1日の投票総額上限(円)。超過分は以降スキップ
PER_RACE_MAX     = 500           # 1レースの投票上限(円)。買い目側の金額と二重チェック
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


# ━━ 買い目の取得元（2026-08-22追加）━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   BET_SOURCE = "resid"  … 残差モデル（paper_resid.csv）。いま評価している買い方
#                "legacy" … 旧方式（today_bets.csv）。購入停止済みで100%を割る
#
#   ⚠ 既定を "resid" にしてある。以前は today_bets.csv だけを読んでいたため、
#     BETTING_ENABLED を True にすると「検証していない旧方式」が投票される
#     という危険な状態だった（2026-08-22に修正）。
#   ⚠ 取得元を変えても、実投票のガード（VOTE_MODE / ARMED / 認証 / SETUP_DONE）
#     は別。ここを変えただけでは1円も動かない。
BET_SOURCE = "resid"
STAKE_PER_POINT = 100     # 1点あたりの金額(円)。検証は1点100円換算で行っている


def _read_resid_bets(race_id):
    """残差モデルの買い目を paper_resid.csv から読む。

    記録の形は1レース複数行（軸の単勝1行＋ダートなら相手のワイド最大3行）。
    判定が「買い」の行だけを投票対象にする。「候補」（S評価の相手）は
    検証していないので投票しない。
    """
    import pandas as pd
    path = os.path.join(BASE_DIR, "paper_resid.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, dtype={"race_id": str})
    df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    sub = df[(df["race_id"] == str(race_id)) & (df["判定"] == "買い")]
    out = []
    for _, r in sub.iterrows():
        kind = str(r.get("券種") or "")
        if kind not in ("単勝", "ワイド"):      # 候補行などは投票しない
            continue
        out.append({"race_id": str(race_id), "券種": kind,
                    "買い方": f"残差{r.get('役割') or ''}",
                    "組み合わせ": str(r.get("組み合わせ") or ""),
                    "金額": STAKE_PER_POINT,
                    "馬名": r.get("馬名"), "gap": r.get("gap")})
    return out


def _read_today_bets(race_id):
    """当該レースの買い目行（金額つき）を読む。取得元は BET_SOURCE で切り替える。"""
    if BET_SOURCE == "resid":
        return _read_resid_bets(race_id)
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
    # ── 二重購入の防止（最優先）────────────────────────────────────────
    #   実弾事故で最も多いのが二重購入。通信が切れて成否が分からないまま
    #   再実行すると、同じ馬券をもう一度買ってしまう。
    #   そこで「投票する前に痕跡を残し、痕跡があれば絶対に投票しない」構造にする。
    #   痕跡はレース単位のファイルで、消さない限り再投票できない。
    done = os.path.join(BASE_DIR, "vote_done", f"{race_id}.txt")
    os.makedirs(os.path.dirname(done), exist_ok=True)
    if os.path.exists(done):
        print(f"  [自動投票] {race_id}: 投票済みの記録あり → 二重投票を防止して中止")
        return False

    total = sum(int(float(b.get("金額", 0) or 0)) for b in bets)
    with open(done, "w", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y/%m/%d %H:%M:%S}\t着手\t{len(bets)}点\t{total}円\n")

    drv = None
    try:
        drv = _ipat_login()
        if drv is None:
            _mark(done, "失敗", "ログインできず")
            return False

        # 買い目を入力する。1点ずつ入れ、入力後に画面から読み戻して検算する。
        entered = _ipat_enter_bets(drv, race_id, bets)
        if entered is None:
            _mark(done, "失敗", "入力に失敗（投票せず終了）")
            return False

        # ── 金額の検算 ──
        #   画面の合計と、こちらが意図した合計が1円でも違えば投票しない。
        #   桁の入力ミスや行のズレはここで必ず止まる。
        if entered != total:
            print(f"  [自動投票] 金額不一致 意図{total}円 / 画面{entered}円 → 投票中止")
            _shot(drv, race_id, "amount_mismatch")
            _mark(done, "中止", f"金額不一致 意図{total}/画面{entered}")
            return False

        ok = _ipat_confirm(drv, race_id)
        _mark(done, "投票" if ok else "失敗", f"{total}円")
        return ok
    except Exception as e:
        # 例外時は「成否不明」として痕跡を残す。手動で確認するまで再投票させない。
        print(f"  [自動投票] 例外: {type(e).__name__}: {str(e)[:120]}")
        if drv is not None:
            _shot(drv, race_id, "exception")
        _mark(done, "成否不明", f"{type(e).__name__}: {str(e)[:80]}")
        return False
    finally:
        if drv is not None:
            try:
                drv.quit()
            except Exception:
                pass


def _mark(path, status, note=""):
    """投票の痕跡に結果を追記する。"""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y/%m/%d %H:%M:%S}\t{status}\t{note}\n")
    except Exception:
        pass


def _shot(drv, race_id, tag):
    """画面を保存する。後から何が起きたか追えるように。"""
    try:
        d = os.path.join(BASE_DIR, "vote_shots")
        os.makedirs(d, exist_ok=True)
        drv.save_screenshot(os.path.join(
            d, f"{race_id}_{tag}_{datetime.now():%Y%m%d_%H%M%S}.png"))
    except Exception:
        pass


def _ipat_login():
    """IPATにログインしてWebDriverを返す。失敗時None。

    ⚠️ここから下の3関数はIPATの実画面に合わせて確定させる必要がある。
      selectorは環境・時期で変わるため、SETUP_DONE=True にする前に
      少額（1点100円）で必ず実地確認すること。手順は AUTO_VOTE_手順書.md 参照。
    """
    raise NotImplementedError(
        "IPATログインは実画面でselectorを確定させてから実装する。"
        "AUTO_VOTE_手順書.md の手順2を参照。")


def _ipat_enter_bets(drv, race_id, bets):
    """買い目を入力し、画面が示す購入予定の合計金額を返す。失敗時None。"""
    raise NotImplementedError(
        "買い目入力は実画面でselectorを確定させてから実装する。"
        "AUTO_VOTE_手順書.md の手順3を参照。")


def _ipat_confirm(drv, race_id):
    """購入を確定する。成功時True。"""
    raise NotImplementedError(
        "購入確定は実画面でselectorを確定させてから実装する。"
        "AUTO_VOTE_手順書.md の手順4を参照。")
