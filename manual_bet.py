# -*- coding: utf-8 -*-
"""手動購入の記録と検証（2026-08-22 新規）

なぜ手動なのか
  IPATの自動投票は3関数とも未実装（_ipat_login / _ipat_enter_bets / _ipat_confirm）で、
  実装するとJRAの規約上グレーな領域に入る。一方で確かめたいことは
  「買い目が実際に買えるか」「払戻が正しく紐づくか」「収支が合うか」であって、
  それは手動で買っても全部確かめられる。自動化は実測で100%超を確認してからでよい。

使い方
  python manual_bet.py                    今日の買い目（発走前のみ）を発走順に表示
  python manual_bet.py all                発走済みも含めて全部表示
  python manual_bet.py record 202601020107 単勝 04 100
                                          実際に買ったものを記録する
  python manual_bet.py check              払戻と突き合わせて収支を出す

記録先: manual_bets.csv（購入したものだけ。買っていないものは入れない）
  paper_resid.csv は「モデルが買えと言った」記録。
  manual_bets.csv は「実際に買った」記録。この2つは別物として分けている。
  混ぜると、買い忘れたレースを的中扱いにしてしまう事故が起きる。
"""
import os
import re
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(BASE_DIR, "paper_resid.csv")
MANUAL = os.path.join(BASE_DIR, "manual_bets.csv")
RUNLOG = os.path.join(BASE_DIR, "keiba_auto_run.log")
COLS = ["購入日時", "race_id", "jyo", "race_no", "券種", "組み合わせ",
        "馬名", "金額", "購入時オッズ", "備考"]


def log(m):
    print(m, flush=True)


def _post_times():
    """発走時刻を実行ログから拾う。スクレイピングはしない。"""
    t = {}
    if not os.path.exists(RUNLOG):
        return t
    with open(RUNLOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"予約: .*?\((\d{12})\).*?発走: (\d{1,2}:\d{2})", line)
            if m:
                t[m.group(1)] = m.group(2)   # 後勝ち＝最新の予約
    return t


def _buys():
    if not os.path.exists(PAPER):
        log("paper_resid.csv がありません。予想がまだ走っていません。")
        return None
    d = pd.read_csv(PAPER, dtype={"race_id": str, "組み合わせ": str})
    return d[d["判定"] == "買い"].copy()


def show(show_all=False):
    b = _buys()
    if b is None:
        return
    if b.empty:
        log("今日は買い判定のレースがありません。")
        return
    pt = _post_times()
    b["発走"] = b.race_id.map(pt).fillna("--:--")
    now = datetime.now().strftime("%H:%M")
    if not show_all:
        before = b[b["発走"] > now]
        if before.empty:
            log(f"現在 {now}。発走前の買い目はもうありません。")
            log("全部見るには: python manual_bet.py all")
            return
        b = before
    b = b.sort_values(["発走", "race_id"])

    done = set()
    if os.path.exists(MANUAL):
        m = pd.read_csv(MANUAL, dtype={"race_id": str, "組み合わせ": str})
        done = {(r.race_id, r.券種, str(r.組み合わせ)) for r in m.itertuples()}

    log(f"■ 今日の買い目（現在 {now}）")
    log(f"  {'発走':<7}{'場':<5}{'R':>3}  {'券種':<5}{'買い目':<8}{'馬名':<20}"
        f"{'オッズ':>7}{'人気':>5}{'gap':>7}  済")
    log("  " + "-" * 74)
    for r in b.itertuples():
        mark = "●" if (r.race_id, r.券種, str(r.組み合わせ)) in done else ""
        log(f"  {r.発走:<7}{r.jyo:<5}{r.race_no:>3}  {r.券種:<5}"
            f"{str(r.組み合わせ):<8}{str(r.馬名)[:18]:<20}"
            f"{r.単勝オッズ:>7.1f}{r.人気:>5.0f}{r.gap:>7.2f}  {mark}")
    log(f"\n  {len(b)}点。1点100円なら計 {len(b) * 100:,}円")
    log("\n  ⚠ ここに出るオッズは予想を出した時点のもの。発走までに動きます。")
    log("    軸の選び方はオッズに依存しませんが、gapの絶対値は動きます。")
    log("    直前の予想メール／ダッシュボードの方が新しいので、そちらを優先してください。")
    log("\n  買ったら記録する:")
    log("    python manual_bet.py record <race_id> <券種> <組み合わせ> <金額>")


def record(race_id, kind, combi, amount, note=""):
    b = _buys()
    if b is None:
        return
    hit = b[(b.race_id == race_id) & (b["券種"] == kind)
            & (b["組み合わせ"].astype(str) == str(combi))]
    if hit.empty:
        log(f"⚠ {race_id} {kind} {combi} は今日の買い目にありません。")
        log("  買い目一覧: python manual_bet.py")
        log("  それでも記録する場合は、組み合わせの表記（馬番の0埋め等）を確認してください。")
        return
    r = hit.iloc[0]
    row = {
        "購入日時": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "race_id": race_id, "jyo": r["jyo"], "race_no": r["race_no"],
        "券種": kind, "組み合わせ": str(combi), "馬名": r["馬名"],
        "金額": int(amount), "購入時オッズ": r["単勝オッズ"], "備考": note,
    }
    df = pd.DataFrame([row], columns=COLS)
    if os.path.exists(MANUAL):
        old = pd.read_csv(MANUAL, dtype={"race_id": str, "組み合わせ": str})
        dup = old[(old.race_id == race_id) & (old["券種"] == kind)
                  & (old["組み合わせ"].astype(str) == str(combi))]
        if not dup.empty:
            log(f"⚠ 同じ買い目がすでに記録されています（{dup.iloc[0]['購入日時']}）。")
            log("  二重に記録しないよう中止します。訂正するなら manual_bets.csv を直接編集してください。")
            return
        df = pd.concat([old, df], ignore_index=True)[COLS]
    df.to_csv(MANUAL, index=False, encoding="utf-8-sig")
    log(f"○ 記録しました: {r['jyo']}{r['race_no']}R {kind} {combi} "
        f"{r['馬名']} {amount}円")
    log(f"  結果は払戻取得後に: python manual_bet.py check")


def check():
    if not os.path.exists(MANUAL):
        log("manual_bets.csv がありません。まだ何も記録されていません。")
        return
    m = pd.read_csv(MANUAL, dtype={"race_id": str, "組み合わせ": str})
    if m.empty:
        log("記録が空です。")
        return
    frames = []
    for path in ("jv_payouts.csv", "payout_data.csv"):
        fp = os.path.join(BASE_DIR, path)
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, dtype=str))
    if not frames:
        log("払戻データがありません。")
        return
    jv = pd.concat(frames, ignore_index=True)
    jv["race_id"] = jv["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    jv["払戻金"] = pd.to_numeric(jv["払戻金"], errors="coerce").fillna(0)
    jv = jv.drop_duplicates(["race_id", "券種", "組み合わせ"], keep="last")
    PAY = {(r.race_id, r.券種, str(r.組み合わせ)): r.払戻金 for r in jv.itertuples()}
    done = set(jv.race_id)

    log("■ 手動購入の収支")
    log(f"  {'購入日時':<20}{'場':<5}{'R':>3}  {'券種':<5}{'買い目':<8}"
        f"{'金額':>7}{'払戻':>8}  結果")
    log("  " + "-" * 66)
    inv = ret = 0
    pend = 0
    for r in m.itertuples():
        key = (r.race_id, r.券種, str(r.組み合わせ))
        if r.race_id not in done:
            res, p = "結果待ち", None
            pend += 1
        else:
            p = PAY.get(key, 0.0)
            res = "的中" if p > 0 else "外れ"
            inv += r.金額
            ret += p * r.金額 / 100.0        # 払戻金は100円あたり
        log(f"  {r.購入日時:<20}{r.jyo:<5}{r.race_no:>3}  {r.券種:<5}"
            f"{str(r.組み合わせ):<8}{r.金額:>7,}"
            f"{('' if p is None else format(int(p * r.金額 / 100), ',')):>8}  {res}")
    log("  " + "-" * 66)
    if inv:
        log(f"  確定分: 投資 {inv:,}円 / 払戻 {int(ret):,}円 "
            f"/ 収支 {int(ret) - inv:+,}円 / 回収率 {ret / inv * 100:.1f}%")
    if pend:
        log(f"  結果待ち {pend}点（払戻は週次のnetkeiba取得後に反映されます）")
    log("\n  ※ 少数の的中で回収率を判断しないこと。目安は的中100本。")


def main():
    a = sys.argv[1:]
    if not a:
        show()
    elif a[0] == "all":
        show(show_all=True)
    elif a[0] == "check":
        check()
    elif a[0] == "record" and len(a) >= 5:
        record(a[1], a[2], a[3], a[4], a[5] if len(a) > 5 else "")
    else:
        log(__doc__)


if __name__ == "__main__":
    main()
