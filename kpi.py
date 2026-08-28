# -*- coding: utf-8 -*-
"""収益の手前の数字を週次で記録する（2026-08-28）

なぜ作るか
  施策が6つに増えて収益が0円、という状態が続いていました。
  原因は「何が動いていないか」を測っていなかったことです。
  閲覧が増えても問い合わせが0なら、それは動いていません。
  収益に近い側の数字が動いたかどうかだけを見ます。

取り方
  各サイトの閲覧数は**手入力**です。スクレイピングはしません
  （規約と、過去にアカウント凍結が起きているため）。
  売上や受注は自分で分かるので、これも手入力です。

実行
  python kpi.py                          いまの状態を表示
  python kpi.py --set note_v=120 x_f=3   今週の数字を記録
  python kpi.py --history                これまでの推移
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import csv
from datetime import datetime, date, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KPI = os.path.join(BASE_DIR, "kpi.csv")

# 判定日。事前に固定して動かさない（撤退条件.md と同じ考え方）
JUDGE_DAY = date(2026, 9, 25)

# 列の定義。キー / 表示名 / 段
#   段 = ファネルのどこか。収益に近いほど大きい
FIELDS = [
    ("note_v",   "note 閲覧",        1),
    ("zenn_v",   "Zenn 閲覧",        1),
    ("x_f",      "X フォロワー",      1),
    ("cc_v",     "ココナラ 出品閲覧",  2),
    ("cc_msg",   "ココナラ 問い合わせ", 3),
    ("cc_ord",   "ココナラ 受注",      4),
    ("note_paid","note 有料記事 本数",  2),
    ("note_buy", "note 有料 購入",    4),
    ("yen",      "売上（円）",        5),
]
KEYS = [k for k, _, _ in FIELDS]
LABEL = {k: n for k, n, _ in FIELDS}
TIER = {k: t for k, _, t in FIELDS}

TIER_NAME = {1: "見られたか", 2: "出品が見られたか",
             3: "声がかかったか", 4: "仕事になったか", 5: "お金になったか"}


def log(m):
    print(m, flush=True)


def _w(t):
    """表示幅。全角は2桁として数える。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "FWA" else 1 for c in str(t))


def pad(t, n):
    return str(t) + " " * max(0, n - _w(t))


def load():
    if not os.path.exists(KPI):
        return []
    with open(KPI, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: r.get("date", ""))


def num(r, k):
    """未入力と0を区別する。0は測って0、空欄は測っていない。"""
    v = (r.get(k) or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def save(rows):
    with open(KPI, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + KEYS)
        w.writeheader()
        w.writerows(rows)


def do_set(args):
    """今日の日付で1行を作る/更新する。同じ日なら上書き。"""
    vals = {}
    for a in args:
        if "=" not in a:
            log(f"  無視: {a}（key=値 の形で指定します）")
            continue
        k, v = a.split("=", 1)
        k = k.strip()
        if k not in KEYS:
            log(f"  そんな項目はありません: {k}")
            log(f"  使えるのは: {', '.join(KEYS)}")
            return
        vals[k] = v.strip()
    if not vals:
        log("  記録するものがありません")
        return

    rows = load()
    today = datetime.now().strftime("%Y-%m-%d")
    cur = next((r for r in rows if r.get("date") == today), None)
    if cur is None:
        cur = {"date": today}
        rows.append(cur)
    for k in KEYS:
        cur.setdefault(k, "")
    cur.update(vals)
    save(sorted(rows, key=lambda r: r["date"]))
    log(f"  {today} に記録しました: " + " / ".join(f"{LABEL[k]}={v}" for k, v in vals.items()))


def show():
    rows = load()
    log("")
    log("  収益までの段")
    log("  " + "-" * 56)

    if not rows:
        log("  まだ1件も記録がありません。")
        log("")
        log("  最初の記録:")
        log("    python kpi.py --set note_v=42 zenn_v=0 x_f=0 yen=0")
        log("")
        return

    last, prev = rows[-1], (rows[-2] if len(rows) >= 2 else None)

    cur_tier = 0
    for tier in sorted(TIER_NAME):
        ks = [k for k in KEYS if TIER[k] == tier]
        log(f"  【{tier}】{TIER_NAME[tier]}")
        for k in ks:
            v = num(last, k)
            if v is None:
                log(f"      {pad(LABEL[k], 22)} 未測定")
                continue
            d = ""
            if prev is not None:
                p = num(prev, k)
                if p is not None:
                    dv = v - p
                    d = f"  {dv:+,.0f}" if dv else "  ±0"
            log(f"      {pad(LABEL[k], 22)}{v:>8,.0f}{d}")
            if v > 0:
                cur_tier = max(cur_tier, tier)
        log("")

    # 売り物が無いのに購入0件を「売れない証拠」と読まない。
    # 過去に何度もやった「0件＝正常ではないかもしれない」の裏返しで、
    # ここでは「0件＝異常ではない」を見落とさないための注記。
    npaid = num(last, "note_paid")
    if npaid is not None and npaid == 0 and num(last, "note_buy") == 0:
        log("  ※ 有料記事が0本なので、購入0件は情報を持ちません。")
        log("     閲覧がいくら増えても、買うものが無ければ売上は立ちません。")
        log("")

    log("  " + "-" * 56)
    log(f"  いま到達している段: 【{cur_tier}】{TIER_NAME.get(cur_tier, '（まだ0段）')}")
    nxt = cur_tier + 1
    if nxt in TIER_NAME:
        log(f"  次に動かすべき数字: {TIER_NAME[nxt]}")
        log(f"    → " + " / ".join(LABEL[k] for k in KEYS if TIER[k] == nxt))
    else:
        log("  最上段に到達しています。単価を上げる段階です。")

    # 期限
    d = (JUDGE_DAY - date.today()).days
    log("")
    log(f"  判定日 {JUDGE_DAY:%Y/%m/%d} まで あと{d}日")
    if num(last, "yen") in (None, 0):
        log("  売上0円のまま判定日を迎えたら、出品文か価格か経路を変えます。")
    log("")


def history():
    rows = load()
    if not rows:
        log("  記録がありません")
        return
    hdr = ["date"] + KEYS
    w = [10] + [max(7, len(k) + 2) for k in KEYS]
    log("")
    log("  " + " ".join(pad(h, x) for h, x in zip(hdr, w)))
    for r in rows:
        cells = [r.get("date", "")]
        for k in KEYS:
            v = num(r, k)
            cells.append("-" if v is None else f"{v:,.0f}")
        log("  " + " ".join(pad(c, x) for c, x in zip(cells, w)))
    log("")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--set":
        do_set(a[1:])
        show()
    elif a and a[0] == "--history":
        history()
    else:
        show()


if __name__ == "__main__":
    main()
