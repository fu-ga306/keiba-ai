# -*- coding: utf-8 -*-
"""販売用ページの合言葉ゲート（2026-08-27）

なぜ合言葉方式か
  ダッシュボードを購読制にしたいが、認証・課金・常時稼働の3つを同時に抱えると
  作り込みが大きくなり、「本当に買う人がいるか」を確かめる前に消耗する。

  そこで**課金は note に任せる**。
    note の有料記事に「今週の合言葉」を書く
    → 買った人だけが合言葉を知っている
    → ダッシュボードの有料部分が見られる

  Stripe も会員DBも要らない。今の仕組みのまま始められる。
  規模が大きくなってから作り込めばよい。

合言葉の作り方
  週ごとに自動で変わる。ISO週番号から機械的に決まるので、
  「今週の合言葉」を note に書くだけで運用できる。
  秘密鍵（SALE_SALT）を .env に置き、そこから導出する。

  ⚠ これは**強固な認証ではありません。** 合言葉が共有されれば誰でも見られます。
    それでも「無料で全部見える」よりはるかにマシで、小規模なら十分です。
    購読者が増えて実害が出たら、そのとき本物の認証に移行してください。
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

# 合言葉に使う語。読みやすく、口頭でも伝えられるものにする
_WORDS = [
    "さくら", "うみ", "そら", "やま", "かぜ", "つき", "ほし", "かわ",
    "もり", "はる", "なつ", "あき", "ふゆ", "あさ", "ゆう", "みなみ",
]


def _salt():
    """.env の SALE_SALT を読む。無ければ既定値（そのままでも動くが推奨しない）。"""
    p = os.path.join(BASE_DIR, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line.startswith("SALE_SALT=") and "=" in line:
                v = line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
                if v:
                    return v
    return "keiba-ai-default-salt"


def passphrase(when=None):
    """その週の合言葉を返す。日曜始まりのISO週で切り替わる。

    形式: ひらがな2語＋数字2桁（例: さくら-うみ-73）
    覚えやすく、かつ総当たりされにくい程度の長さにしている。
    """
    d = when or datetime.now()
    y, w, _ = d.isocalendar()
    h = hashlib.sha256(f"{_salt()}|{y}-{w:02d}".encode("utf-8")).hexdigest()
    a = _WORDS[int(h[0:2], 16) % len(_WORDS)]
    b = _WORDS[int(h[2:4], 16) % len(_WORDS)]
    n = int(h[4:6], 16) % 100
    return f"{a}-{b}-{n:02d}"


def check(given, when=None):
    """合言葉が合っているか。前週のものも1週間だけ通す。

    なぜ前週も通すか
      週の境目に買った人が、切り替わりで見られなくなるのを防ぐため。
      「日曜に買ったのに月曜に見られない」は問い合わせの元になる。
    """
    if not given:
        return False
    g = str(given).strip().lower().replace(" ", "").replace("　", "")
    d = when or datetime.now()
    from datetime import timedelta
    for delta in (0, -7):
        if g == passphrase(d + timedelta(days=delta)).lower():
            return True
    return False


def week_label(when=None):
    """note に書くときの表示用（例: 2026年 第35週（8/25〜8/31））。"""
    from datetime import timedelta
    d = when or datetime.now()
    y, w, dow = d.isocalendar()
    mon = d - timedelta(days=dow - 1)
    sun = mon + timedelta(days=6)
    return f"{y}年 第{w}週（{mon:%-m/%-d}〜{sun:%-m/%-d}）" if os.name != "nt" \
        else f"{y}年 第{w}週（{mon.month}/{mon.day}〜{sun.month}/{sun.day}）"


def main():
    from datetime import timedelta
    now = datetime.now()
    print("■ 販売用ページの合言葉")
    print(f"  salt: {'設定あり' if _salt() != 'keiba-ai-default-salt' else '⚠ 既定値のまま（.env に SALE_SALT を設定してください）'}")
    print()
    print(f"  今週  {week_label(now)}")
    print(f"        合言葉: {passphrase(now)}")
    print()
    print("  先の週（note の予約投稿に使えます）")
    for i in (1, 2, 3):
        d = now + timedelta(weeks=i)
        print(f"    {week_label(d)}  {passphrase(d)}")
    print()
    print("  ※ 前週の合言葉も1週間は通ります（週の境目に買った人のため）")


if __name__ == "__main__":
    main()
