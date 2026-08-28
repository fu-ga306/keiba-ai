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

# ── 合言葉の更新周期 ────────────────────────────────────────────────
#   "week"  : 毎週変わる。漏れても被害は1週間で切れるが、**毎週noteを書き換える手間**が出る
#   "month" : 毎月変わる。手間は月1回。購読者が少ないうちはこれで十分
#   "fixed" : 変えない。手間ゼロ。漏れたら手動で SALE_SALT を変える
#
#   2026-08-27: 利用者の判断で "week" を採用。
#   手間は増えるが、貼る文面は月曜の点検メールに完成形で届くのでコピーだけで済む。
#   競馬の開催は土日で、ISO週では両方とも週の終わりに入る。
#   つまり**週の切り替わり（月曜）が開催を分断しない**ので、この単位は都合がよい。
ROTATE = "week"

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


def _period_key(d):
    """更新周期に応じた期間の識別子。"""
    if ROTATE == "fixed":
        return "fixed"
    if ROTATE == "month":
        return f"{d.year}-{d.month:02d}"
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def passphrase(when=None):
    """その期間の合言葉を返す。ROTATE で切り替わる単位が決まる。

    形式: ひらがな2語＋数字2桁（例: さくら-うみ-73）
    覚えやすく、かつ総当たりされにくい程度の長さにしている。
    """
    d = when or datetime.now()
    h = hashlib.sha256(f"{_salt()}|{_period_key(d)}".encode("utf-8")).hexdigest()
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
    # 前の期間のものも通す。切り替わり直後に見られなくなるのを防ぐため
    #   （「買った翌日に見られない」は問い合わせの元になる）
    #
    # ⚠ ただし週跨ぎは**月〜水だけ**にする（2026-08-28）
    #   それまでは前週の合言葉を丸7日通していた。開催は土日なので、
    #   8/29-30ぶんを買った人が翌週の 9/5-6 も見られてしまう。
    #   隔週で買えば全開催をカバーできることになり、売り物が成立しない。
    #   防ぎたいのは「日曜に買って月曜に見られない」なので、
    #   次の開催（土曜）に届かない水曜までで足りる。
    backs = {"week": (0,), "month": (0, -32), "fixed": (0,)}[ROTATE]
    if ROTATE == "week" and d.weekday() <= 2:      # 月火水のみ前週を通す
        backs = (0, -7)
    for delta in backs:
        if g == passphrase(d + timedelta(days=delta)).lower():
            return True
    return False


def period_label(when=None):
    """note に書くときの表示用。"""
    from datetime import timedelta
    d = when or datetime.now()
    if ROTATE == "fixed":
        return "（固定）"
    if ROTATE == "month":
        return f"{d.year}年{d.month}月"
    # 表示は「開催の2日間」にする（2026-08-28）
    #   合言葉はISO週で切り替わるが、読者に意味があるのは**どの開催で使えるか**。
    #   「第35週（8/24〜8/30）」では、平日を含む7日間に見えて実態と合わない。
    #   実際に使えるのは土日の2日ぶん。そこを書く。
    y, w, dow = d.isocalendar()
    sat = d + timedelta(days=5 - (dow - 1))     # 同じISO週の土曜
    sun = sat + timedelta(days=1)
    return f"{sat.month}/{sat.day}(土)・{sun.month}/{sun.day}(日)"


def ensure_salt():
    """SALE_SALT が無ければ自動生成して .env に書く。

    種は一度決めれば変える必要がない（合言葉のほうが期間ごとに変わるため）。
    人が考える必要も無いので、自動で作る。
    ⚠ 既にあれば絶対に上書きしない。上書きすると既存の合言葉が全部変わる。
    """
    import secrets
    if _salt() != "keiba-ai-default-salt":
        return False, "既に設定済み（変更しません）"
    p = os.path.join(BASE_DIR, ".env")
    v = secrets.token_urlsafe(24)
    try:
        cur = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        if cur and not cur.endswith("\n"):
            cur += "\n"
        cur += ("\n# 販売ページの合言葉の種（自動生成・変更すると合言葉が全部変わります）\n"
                f"SALE_SALT={v}\n")
        with open(p, "w", encoding="utf-8") as f:
            f.write(cur)
        return True, "生成して .env に書きました"
    except Exception as e:
        return False, f"書けませんでした: {type(e).__name__}: {e}"


def note_block(url_base, when=None):
    """note に貼るだけの文面を作る。ここが毎回の手作業になるので、完成形で渡す。"""
    d = when or datetime.now()
    rot = {"month": "毎月", "week": "毎週", "fixed": ""}[ROTATE]
    return "\n".join([
        f"【{period_label(d)}の閲覧情報】",
        "",
        f"合言葉： {passphrase(d)}",
        "",
        "下のリンクを開き、合言葉を入力すると全頭の評価が表示されます。",
        url_base,
        "",
        f"※ 合言葉は{rot}変わります。前の期間のものもしばらく有効です。",
        "※ 3着以内に入る確率を示したもので、的中・利益を保証するものではありません。",
    ])


def main():
    from datetime import timedelta
    a = sys.argv[1:]
    if a and a[0] == "--init":
        ok, msg = ensure_salt()
        print(f"  SALE_SALT: {msg}")
        if ok:
            print("  → これ以降、合言葉はこの環境に固有のものになります")
        return

    now = datetime.now()
    print("■ 販売用ページの合言葉")
    print(f"  更新周期: {ROTATE}"
          + {"month": "（毎月・手間は月1回）", "week": "（毎週）",
             "fixed": "（固定・手間ゼロ）"}[ROTATE])
    ok = _salt() != "keiba-ai-default-salt"
    print(f"  種(SALE_SALT): {'設定あり' if ok else '⚠ 既定値のまま → python sale_gate.py --init'}")
    print()
    print(f"  今 {period_label(now)}   合言葉: {passphrase(now)}")
    print()
    print("  先の期間（note の予約投稿に使えます）")
    for i in (1, 2, 3):
        if ROTATE == "month":
            # ⚠ 32日ずつ足すと月が飛ぶ（8/27+96日=12/1 で11月が抜けた）。
            #   月そのものを進めて、各月の1日を使う。
            m = now.month + i
            d = now.replace(year=now.year + (m - 1) // 12,
                            month=(m - 1) % 12 + 1, day=1)
        else:
            d = now + timedelta(days=7 * i)
        print(f"    {period_label(d):<16}{passphrase(d)}")
    print()
    print("  前の期間のものも通ります（切り替わり直後に見られなくなるのを防ぐため）")


if __name__ == "__main__":
    main()
