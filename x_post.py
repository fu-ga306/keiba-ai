# -*- coding: utf-8 -*-
"""X（旧Twitter）へ公式APIで投稿する（2026-08-25）

なぜ公式APIか
  ブラウザ自動操作は規約がグレーで、凍結リスクを負うのは本人。
  X APIは従量課金で **投稿1件 $0.01（約1.5円）**。初回に最低$5のチャージが要るが、
  1日3投稿なら月90件＝約$0.90（135円）。壁になる金額ではない。
  公式APIなら規約上の問題も無い。

安全設計（実弾＝投稿してしまう事故を防ぐ）
  ① X_POST_ARMED ファイルが無ければ投稿しない（物理スイッチ）
  ② DRY_RUN=True が既定。実投稿には明示的に False にする必要がある
  ③ 同じ本文は二度投稿しない（x_posted.csv に痕跡を残す）
  ④ 1日の上限を持つ（暴発しても被害が限定される）

.env に入れるもの（X Developer Portal で発行）
  X_API_KEY=...
  X_API_SECRET=...
  X_ACCESS_TOKEN=...
  X_ACCESS_SECRET=...
  ※ 1つでも欠ければ自動でドライランに落ちる

使い方
  python x_post.py "投稿する本文"        1件投稿（既定はドライラン）
  python x_post.py --queue               投稿キュー(x_queue.txt)を処理
  python x_post.py --status              設定と本日の投稿数を表示
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import hashlib
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARMED_FILE = os.path.join(BASE_DIR, "X_POST_ARMED")
LOG_CSV = os.path.join(BASE_DIR, "x_posted.csv")
QUEUE_TXT = os.path.join(BASE_DIR, "x_queue.txt")

# ── 安全弁 ────────────────────────────────────────────────────────────
DRY_RUN = True          # True の間は絶対に投稿しない。実投稿時に False にする
DAILY_MAX = 5           # 1日の投稿上限。暴発しても被害を限定する
MAX_LEN = 280           # Xの上限。日本語は1文字=1としてカウントされる

_KEYS = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"]


def log(m):
    print(m, flush=True)


def _load_env():
    """.env を読む。インラインコメントは落とす。"""
    v = {k: os.environ.get(k, "") for k in _KEYS}
    p = os.path.join(BASE_DIR, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, val = line.split("=", 1)
            k = k.strip()
            # ⚠ 手順書の例をコメントごとコピーされる事故が実際に起きたので必ず落とす
            val = val.split("#", 1)[0].strip().strip('"').strip("'").strip()
            if k in _KEYS and not v.get(k):
                v[k] = val
    return v


def _effective_mode():
    """実際に効くモードを返す。条件が欠ければ dryrun へ降格。"""
    if DRY_RUN:
        return "dryrun", "DRY_RUN=True"
    if not os.path.exists(ARMED_FILE):
        return "dryrun", "X_POST_ARMED が無い"
    miss = [k for k, v in _load_env().items() if not v]
    if miss:
        return "dryrun", f"認証情報が不足({len(miss)}件)"
    return "live", ""


def _today_count():
    """本日すでに何件投稿したか。"""
    if not os.path.exists(LOG_CSV):
        return 0
    today = datetime.now().strftime("%Y/%m/%d")
    n = 0
    for line in open(LOG_CSV, encoding="utf-8", errors="ignore"):
        if line.startswith(today) and ",live," in line:
            n += 1
    return n


def _seen(text):
    """同じ本文を投稿済みか。二重投稿の防止。"""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if os.path.exists(LOG_CSV):
        for line in open(LOG_CSV, encoding="utf-8", errors="ignore"):
            if h in line:
                return True, h
    return False, h


def _record(text, h, mode, note=""):
    head = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", encoding="utf-8-sig") as f:
        if head:
            f.write("日時,ハッシュ,モード,文字数,備考,本文先頭\n")
        safe = text.replace("\n", " ").replace(",", "、")[:60]
        f.write(f"{datetime.now():%Y/%m/%d %H:%M:%S},{h},{mode},"
                f"{len(text)},{note},{safe}\n")


def post(text, force=False):
    """1件投稿する。戻り値 True=投稿した（ドライラン含め処理成功）。"""
    text = text.strip()
    if not text:
        log("  本文が空です")
        return False
    if len(text) > MAX_LEN:
        log(f"  ⚠ {len(text)}文字（上限{MAX_LEN}）。投稿しません")
        return False

    dup, h = _seen(text)
    if dup and not force:
        log(f"  同じ本文が投稿済み（{h}）→ 二重投稿を防止して中止")
        return False

    mode, why = _effective_mode()
    n = _today_count()
    if mode == "live" and n >= DAILY_MAX:
        log(f"  本日{n}件で上限{DAILY_MAX}に達しています → 中止")
        _record(text, h, "skip", "日次上限")
        return False

    if mode == "dryrun":
        log(f"  [ドライラン] {len(text)}文字（{why}）")
        log("  ---")
        for ln in text.split("\n"):
            log(f"  | {ln}")
        log("  ---")
        _record(text, h, "dryrun", why)
        return True

    # ── ここから実投稿 ──
    try:
        from requests_oauthlib import OAuth1Session
    except ImportError:
        log("  requests_oauthlib が必要です: pip install requests_oauthlib")
        return False
    c = _load_env()
    try:
        s = OAuth1Session(c["X_API_KEY"], c["X_API_SECRET"],
                          c["X_ACCESS_TOKEN"], c["X_ACCESS_SECRET"])
        r = s.post("https://api.x.com/2/tweets", json={"text": text}, timeout=30)
        if r.status_code in (200, 201):
            tid = (r.json().get("data") or {}).get("id", "")
            log(f"  ○ 投稿しました（id={tid}）本日{n+1}/{DAILY_MAX}件目")
            _record(text, h, "live", tid)
            return True
        # 402=課金切れ / 429=レート上限 は原因が違うので分けて出す
        hint = {402: "チャージ残高を確認してください",
                429: "レート上限。時間を空けてください",
                401: "認証情報を確認してください"}.get(r.status_code, "")
        log(f"  ⚠ 投稿失敗 HTTP {r.status_code} {hint}")
        log(f"    {r.text[:200]}")
        _record(text, h, "fail", f"HTTP{r.status_code}")
        return False
    except Exception as e:
        log(f"  ⚠ 例外: {type(e).__name__}: {str(e)[:150]}")
        _record(text, h, "fail", type(e).__name__)
        return False


def run_queue():
    """x_queue.txt を処理する。空行区切りで1投稿。

    処理した分は先頭から削除するので、失敗したものは残る。
    """
    if not os.path.exists(QUEUE_TXT):
        log(f"  {os.path.basename(QUEUE_TXT)} がありません")
        return
    raw = open(QUEUE_TXT, encoding="utf-8").read()
    items = [x.strip() for x in raw.split("\n\n") if x.strip()]
    if not items:
        log("  キューは空です")
        return
    log(f"  キュー {len(items)}件")
    rest = []
    for i, t in enumerate(items, 1):
        log(f"\n[{i}/{len(items)}]")
        if not post(t):
            rest.append(t)
    with open(QUEUE_TXT, "w", encoding="utf-8") as f:
        f.write("\n\n".join(rest))
    log(f"\n  残り {len(rest)}件")


def status():
    mode, why = _effective_mode()
    c = _load_env()
    log("■ X投稿の設定")
    log(f"  DRY_RUN        {DRY_RUN}")
    log(f"  X_POST_ARMED   {'あり' if os.path.exists(ARMED_FILE) else 'なし'}")
    for k in _KEYS:
        log(f"  {k:<18}{'設定あり' if c[k] else '未設定'}")
    log(f"  実効モード       {mode}" + (f"（{why}）" if why else ""))
    log(f"  本日の投稿       {_today_count()}/{DAILY_MAX}件")
    log(f"\n  実投稿するには: DRY_RUN=False にして、"
        f"{os.path.basename(ARMED_FILE)} を置き、認証情報4つを .env に入れる")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "--status":
        status()
    elif a[0] == "--queue":
        run_queue()
    else:
        post(" ".join(a))


if __name__ == "__main__":
    main()
