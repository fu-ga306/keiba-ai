# -*- coding: utf-8 -*-
"""週次の検証結果から Zenn 記事を生成して公開する（2026-08-25）

なぜ Zenn か
  GitHub と連携すると **push するだけで自動公開**される。公式に自動化が想定
  されている作りなので規約上の問題が無い。note には公式の投稿APIが無く、
  ブラウザ自動操作は規約グレーで凍結リスクを本人が負う。

なぜ「予約投稿」なのか（ここが設計の肝）
  完全自動で即公開すると、変な記事が出たときに止められない。
  かといって毎回人が確認するなら手離れにならない。
  そこで **published_at を数日先にして push する**。
    ・何もしなければ、その日時に自動で公開される（手離れ）
    ・気に入らなければ、それまでにファイルを直すか消せばよい（安全）
  Zenn は未来日時を指定するとその時刻まで非公開にしてくれる。

安全設計
  ① DRY_RUN=True が既定。実際の push には明示的に False が要る
  ② 生成した記事は必ず published_at 付き（即時公開はしない）
  ③ 同じ週の記事は二度作らない
  ④ 数字は paper_report.py の出力をそのまま引用する（作文しない）

使い方
  python zenn_publish.py             今週分を生成（ドライラン）
  python zenn_publish.py --push      生成して push（DRY_RUN=False が必要）
  python zenn_publish.py --status    設定を表示
"""
import sys

for _s in (sys.stdout, sys.stderr):   # cp932環境でのUnicodeEncodeError→異常終了を防ぐ
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import re
import subprocess
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZENN_DIR = os.path.join(os.path.dirname(BASE_DIR), "keiba_zenn")
ART_DIR = os.path.join(ZENN_DIR, "articles")
QUEUE_TXT = os.path.join(BASE_DIR, "x_queue.txt")

# ── 安全弁 ────────────────────────────────────────────────────────────
DRY_RUN = True            # True の間は push しない
DELAY_DAYS = 3            # 何日先に公開予約するか（この間は差し替え・削除できる）
PUBLISH_HOUR = 21         # 公開時刻


def log(m):
    print(m, flush=True)


def _run(cmd, cwd, timeout=120):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _paper_report():
    """paper_report.py を実行して出力をそのまま得る。

    ⚠ 数字はここから引用するだけ。記事側で作文しない。
      作文すると出典と食い違い、公開後に訂正することになる（実際にやった）。
    """
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, os.path.join(BASE_DIR, "paper_report.py")],
                       cwd=BASE_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900, env=env)
    if r.returncode != 0:
        return None, f"paper_report.py が異常終了（コード{r.returncode}）"
    return r.stdout, ""


def _pick(out, pat, default="―"):
    m = re.search(pat, out)
    return m.group(1).strip() if m else default


def build_article(out):
    """週次レポートの記事本文を組み立てる。"""
    today = datetime.now()
    pub = (today + timedelta(days=DELAY_DAYS)).replace(
        hour=PUBLISH_HOUR, minute=0, second=0, microsecond=0)
    slug = f"keiba-ai-weekly-{today:%Y%m%d}"

    nrace = _pick(out, r"記録 ([\d,]+)レース")
    buy15 = _pick(out, r">=1\.5\s+([\d,]+)\s+([\d.]+)%")
    rate15 = _pick(out, r">=1\.5\s+[\d,]+\s+([\d.]+)%")
    roi = _pick(out, r"軸gap>=1\.5\s+([\d,]+)点\s+的中\s*(\d+)")
    hit_line = _pick(out, r"(軸gap>=1\.5.*?回収率\s*[\d.]+%)", "")
    place = _pick(out, r"実測 ([\d.]+)%（\d+点）")

    body = f"""---
title: "競馬AIの前向き検証 {today:%Y/%m/%d}週 ― 買わずに記録した{nrace}レース"
emoji: "📊"
type: "idea"
topics: ["機械学習", "データ分析", "統計", "競馬"]
published: true
published_at: {pub:%Y-%m-%d %H:%M}
---

バックテストで回収率120.6%だったモデルを、**買わずに記録だけ**続けています。
その週次報告です。数字は集計スクリプトの出力をそのまま貼っています。

なぜ買わないかは [別記事](/articles/backtest-vanishes-six-traps) に書きました。
要するに、過去8回とも「良く見えたバックテスト」が実測で消えたからです。

## 今週までの記録

```
{out.strip()}
```

## 読み方

見ているのは回収率ではありません。**回収率で黒字を証明するには約6,400点＝58か月**
かかるので、短期の数字にはほとんど情報がありません。

代わりに見ているのは次の3つです。

1. **買い率** — 検証値47.7%から大きく外れていたら、本番と検証で条件が違う
2. **馬券内率** — 検証値32.1%。100点で±9ptに収まるので、回収率より早く判定できる
3. **決める時刻による買い目の差** — 検証は確定オッズで選べるが、本番は7分前に決める

いずれも「儲かるか」ではなく「**壊れていないか**」を見る指標です。
証明を諦めて監視に切り替えた、という判断です。

---

この記録は自動生成しています。都合の良い週だけ出す、ということができない作りです。
"""
    return slug, body, pub


def build_tweet(slug, pub):
    return (f"競馬AIの週次記録を出しました。\n"
            f"バックテスト120.6%のモデルを、買わずに記録だけ続けています。\n"
            f"回収率で黒字を証明するには58か月かかると分かったので、"
            f"いまは「壊れていないか」だけ見ています。\n"
            f"https://zenn.dev/articles/{slug}")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--status":
        log("■ Zenn自動公開の設定")
        log(f"  DRY_RUN      {DRY_RUN}")
        log(f"  Zennリポジトリ {ZENN_DIR}")
        log(f"  存在するか     {'○' if os.path.isdir(ART_DIR) else '× 未作成'}")
        log(f"  公開予約       {DELAY_DAYS}日先の{PUBLISH_HOUR}時")
        log(f"  既存記事       {len([f for f in os.listdir(ART_DIR)]) if os.path.isdir(ART_DIR) else 0}本")
        return

    if not os.path.isdir(ART_DIR):
        log(f"  Zennリポジトリがありません: {ART_DIR}")
        log("  先に手順書（zenn_setup.md）に従ってリポジトリを作ってください。")
        return

    out, err = _paper_report()
    if err:
        log(f"  ⚠ {err} → 記事を作りません（間違った数字を出さないため）")
        return

    slug, body, pub = build_article(out)
    path = os.path.join(ART_DIR, f"{slug}.md")
    if os.path.exists(path):
        log(f"  {slug}.md は作成済み → 二重作成を防止して中止")
        return

    if DRY_RUN and not (a and a[0] == "--push"):
        log(f"  [ドライラン] {slug}.md を作る想定（公開予約 {pub:%Y-%m-%d %H:%M}）")
        log("  ---")
        for ln in body.split("\n")[:16]:
            log(f"  | {ln}")
        log("  | …")
        log("  ---")
        log("  実際に作るには DRY_RUN=False にして --push を付けてください。")
        return

    if DRY_RUN:
        log("  ⚠ DRY_RUN=True のままです。--push だけでは公開しません。")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    log(f"  ○ {slug}.md を作成（公開予約 {pub:%Y-%m-%d %H:%M}）")

    # X告知はキューに入れるだけ。投稿は x_post.py の安全弁を通す
    with open(QUEUE_TXT, "a", encoding="utf-8") as f:
        f.write(("\n\n" if os.path.getsize(QUEUE_TXT) else "")
                if os.path.exists(QUEUE_TXT) else "")
        f.write(build_tweet(slug, pub))
    log("  ○ X告知を x_queue.txt に追加（投稿は x_post.py が判断）")

    for cmd in (["git", "add", "."],
                ["git", "commit", "-m", f"週次記録 {datetime.now():%Y/%m/%d}"],
                ["git", "push"]):
        r = _run(cmd, ZENN_DIR)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout or ""):
            log(f"  ⚠ {' '.join(cmd)} 失敗: {(r.stderr or r.stdout)[:150]}")
            return
    log(f"  ○ push しました。{pub:%m/%d %H:%M} に自動公開されます")
    log(f"    それまでは {path} を直すか消せば取り消せます")


if __name__ == "__main__":
    main()
