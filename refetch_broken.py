# -*- coding: utf-8 -*-
"""レースクラスが壊れた600レースを取り直す（2026-09-02）

なぜ要るか
  2026年6月から結果ページの取得が壊れ、レースクラス欄に条件文が入り、
  レース名と賞金が空になっていた。600レース・7,909行。
  取得側は直したが、週次のStep0は直近2週しか取り直さない。
  推定で416レースは埋めたが、レース名と賞金は埋まらないままだった。

⚠ 取得のしかた
  ・**逐次**。並列にしない（ブロックの原因になる）
  ・2.5秒あける（既定の1.5秒より遅くする）
  ・既存の取得関数(update_data.scrape_races)を再利用する。
    パースを書き直すと、また別の取り違えを作る
  ・古い壊れた行を**先に消す**。
    scrape_races は drop_duplicates(keep=first) なので、
    消さないと古いほうが勝って何も変わらない

実行
  python refetch_broken.py            取り直す
  python refetch_broken.py --dry-run  対象件数だけ見る
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import shutil
import time
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LARGE = os.path.join(BASE_DIR, "race_data_large.csv")
CLEAN = os.path.join(BASE_DIR, "race_data_clean.csv")
SLEEP = 2.5


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def broken_ids():
    """取り直すべきレース。クラスが取得由来でない、またはレース名が空。"""
    d = pd.read_csv(CLEAN, dtype={"race_id": str},
                    usecols=lambda c: c in ("race_id", "クラス_出所", "レース名"),
                    low_memory=False)
    bad = set()
    if "クラス_出所" in d.columns:
        bad |= set(d.loc[~d["クラス_出所"].isin(["取得"]), "race_id"])
    # ⚠ レース名を条件に入れないこと（2026-09-02に踏んだ）
    #   update_data.scrape_races はレース名を拾わないので、
    #   取り直しても永久に欠損のまま。条件に入れると毎回600件全部を
    #   取り直すことになる。**レース名はモデルが使っていない。**
    return sorted(bad)


def main():
    dry = "--dry-run" in sys.argv
    ids = broken_ids()
    log(f"取り直す対象 {len(ids)} レース  想定所要 {len(ids)*SLEEP/60:.0f}分")
    if not ids:
        log("対象がありません")
        return
    if dry:
        log("--dry-run のため取得しません")
        log(f"  例: {ids[:5]}")
        return

    bak = LARGE + f".bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(LARGE, bak)
    log(f"退避 {os.path.basename(bak)}")

    # 古い壊れた行を消す。消さないと drop_duplicates で古いほうが残る
    big = pd.read_csv(LARGE, dtype={"race_id": str}, low_memory=False)
    before = len(big)
    big = big[~big["race_id"].astype(str).isin(ids)]
    big.to_csv(LARGE, index=False, encoding="utf-8-sig")
    log(f"壊れた行を除去 {before:,} → {len(big):,}（-{before-len(big):,}）")
    del big

    sys.path.insert(0, BASE_DIR)
    import update_data

    # 既定の1.5秒より遅くする。ブロックを避けるため
    _orig_sleep = time.sleep

    def _slow(sec):
        _orig_sleep(SLEEP if sec and sec <= 2.0 else sec)

    update_data.time.sleep = _slow
    log(f"取得開始（逐次・{SLEEP}秒あけ）")
    t0 = datetime.now()
    n = update_data.scrape_races(ids)
    update_data.time.sleep = _orig_sleep
    log(f"取得完了 {n:,}行  所要 {(datetime.now()-t0).total_seconds()/60:.1f}分")

    log("クリーニング")
    from cleaner import clean_race_data
    clean_race_data(input_csv=LARGE, output_csv=CLEAN)

    log("残りをクラス復旧で埋める")
    import subprocess
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "recover_class.py")],
                   cwd=BASE_DIR, env={**os.environ, "PYTHONUTF8": "1"})

    d = pd.read_csv(CLEAN, dtype={"race_id": str}, low_memory=False)
    d26 = d[d["race_id"].str[:4] == "2026"]
    log("結果（2026年）")
    for c in ("クラス_num", "レース名", "賞金"):
        if c in d.columns:
            log(f"  {c:<10} 欠損 {pd.to_numeric(d26[c], errors='coerce').isna().mean()*100:5.1f}%"
                if c != "レース名" else
                f"  {c:<10} 欠損 {d26[c].isna().mean()*100:5.1f}%")
    if "クラス_出所" in d.columns:
        log(f"  クラスの出所: {d26['クラス_出所'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
