# -*- coding: utf-8 -*-
"""
payout_scraper.py
─────────────────
netkeiba のレース結果ページから全券種の払戻データを取得する。
同着（2着同着など）で1券種に複数の払戻が発生するケースに対応。

保存形式（縦持ち・案B）:
  race_id, 券種, 組み合わせ, 払戻金, 人気
  例（安田記念・2着同着）:
    202605030211, 馬連, 04-11, 3630, 20
    202605030211, 馬連, 04-14, 1100, 9   ← 同着で2行

対象レース:
  race_features.csv の race_id を使う（全レース）。
  payout_data.csv に既にあるrace_idはスキップ（再開可能）。

使い方:
  python payout_scraper.py          # 全レース取得
  python payout_scraper.py 202605030211  # 単一レースのテスト
"""
import os
import sys
import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FEAT_CSV    = os.path.join(BASE_DIR, "race_features.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR, "payout_data.csv")

# 券種名の正規化（netkeibaのclass名 → 日本語券種名）
BET_TYPE_MAP = {
    "Tansho":  "単勝",
    "Fukusho": "複勝",
    "Wakuren": "枠連",
    "Umaren":  "馬連",
    "Wide":    "ワイド",
    "Umatan":  "馬単",
    "Fuku3":   "3連複",
    "Tan3":    "3連単",
}


def create_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    options.add_argument(f"--user-agent={ua}")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
    except Exception:
        pass
    return driver


def is_blocked(page_source: str) -> bool:
    """IPブロック/アクセス制限の検知（horse_scraperと同じ方針）。"""
    if page_source is None:
        return True
    if len(page_source) >= 5000:
        lower = page_source.lower()
        hard = ["アクセスが集中して", "アクセスを制限させて",
                "ただいまアクセスが集中", "too many requests",
                "service temporarily unavailable"]
        return any(s in lower for s in hard)
    lower = page_source.lower()
    short = ["403 forbidden", "access denied", "forbidden",
             "too many requests", "429 ", "service unavailable",
             "アクセスが集中", "アクセスを制限"]
    if any(s in lower for s in short):
        return True
    if len(page_source) < 500:
        return True
    return False


def parse_payout(soup, race_id):
    """
    払戻表をパースして、同着対応で全券種・全組み合わせを取り出す。
    返り値: [{race_id, 券種, 組み合わせ, 払戻金, 人気}, ...]
    """
    rows = []

    # 払戻テーブルは Payout_Detail_Table クラスの table が複数ある
    tables = soup.find_all("table", class_="Payout_Detail_Table")
    if not tables:
        return rows

    for table in tables:
        for tr in table.find_all("tr"):
            # 券種を判定（tr or td の class に Tansho/Umaren 等が入る）
            bet_type = None
            # tr自体のclass、または最初のth/tdのclassから券種を特定
            classes = " ".join(tr.get("class", []))
            for key, jp in BET_TYPE_MAP.items():
                if key in classes:
                    bet_type = jp
                    break
            # thのclassでも判定
            if bet_type is None:
                th = tr.find("th")
                if th:
                    th_classes = " ".join(th.get("class", []))
                    for key, jp in BET_TYPE_MAP.items():
                        if key in th_classes:
                            bet_type = jp
                            break
            if bet_type is None:
                continue

            # 組み合わせ（Result）・払戻金（Payout）・人気（Ninki）のtdを取得
            result_td = tr.find("td", class_="Result")
            payout_td = tr.find("td", class_="Payout")
            ninki_td  = tr.find("td", class_="Ninki")

            if result_td is None or payout_td is None:
                continue

            # ── 同着対応 ──
            # Result内: 同着があると <ul><li>...</li><li>...</li></ul> 等で複数
            # Payout内: <span>金額</span> が同着の数だけ並ぶ（<br>区切り）
            # Ninki内:  人気も同着の数だけ並ぶ

            # 組み合わせの抽出（馬番のまとまりごと）
            combos = parse_result_combos(result_td)
            # 払戻金の抽出（円→数値、同着分すべて）
            payouts = parse_payout_amounts(payout_td)
            # 人気の抽出
            ninkis = parse_ninki(ninki_td) if ninki_td else []

            # 組み合わせ数と払戻数を対応付け（同着で複数）
            n = max(len(combos), len(payouts))
            for i in range(n):
                combo  = combos[i] if i < len(combos) else (combos[-1] if combos else "")
                payout = payouts[i] if i < len(payouts) else None
                ninki  = ninkis[i] if i < len(ninkis) else None
                if payout is None:
                    continue
                rows.append({
                    "race_id":   race_id,
                    "券種":      bet_type,
                    "組み合わせ": combo,
                    "払戻金":    payout,
                    "人気":      ninki,
                })

    return rows


def parse_result_combos(result_td):
    """Resultセルから組み合わせを抽出。券種ごとに「組」の数が異なる。

    netkeibaの実構造:
      - ul基盤(枠連/馬連/馬単/ワイド/3連複/3連単): <ul>が1組に対応し、
        ul内の<li><span>馬番</span></li>がその組の構成馬。
        ワイドや同着では<ul>が複数 → 複数組を返す。
      - div基盤(単勝/複勝): <div><span>馬番</span></div>が並び、
        digitを持つspanが1頭=1組（複勝は3頭=3組、各馬別払戻）。
    例: 馬連 → ["03-10"]、ワイド → ["03-10","03-05","05-10"]、複勝 → ["03","10","05"]
    """
    combos = []
    uls = result_td.find_all("ul")
    if uls:
        # ul = 1組。ul内のliのspan(馬番)を連結して1組とする。
        for ul in uls:
            nums = [s.get_text(strip=True) for s in ul.find_all("span")]
            nums = [n for n in nums if n.isdigit()]
            if nums:
                combos.append("-".join(n.zfill(2) for n in nums))
        return combos

    # div基盤（単勝・複勝）: digitを持つspanが1頭=1組
    nums = [s.get_text(strip=True) for s in result_td.find_all("span")]
    nums = [n for n in nums if n.isdigit()]
    return [n.zfill(2) for n in nums]


def parse_payout_amounts(payout_td):
    """Payoutセルから払戻金（円）を数値リストで返す。
    複勝/ワイドは1spanに <br> 区切りで複数金額が入る。
    例: "570円<br>200円<br>1,130円" → [570, 200, 1130]
    """
    txt = payout_td.get_text(separator="\n")
    return [int(m.replace(",", "")) for m in re.findall(r"([\d,]+)円", txt)]


def parse_ninki(ninki_td):
    """Ninkiセルから人気を数値リストで返す。複勝は3個・同着は複数。
    例: "9人気1人気12人気" → [9, 1, 12]
    """
    txt = ninki_td.get_text(separator="\n")
    return [int(m) for m in re.findall(r"(\d+)人気", txt)]


# ── HTTP取得（requests優先・Seleniumフォールバック）──────────────────────
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
_session = None
_fallback_driver = None

# ── レート制限（2026-07-27追加）───────────────────────────────────────────
# 0.2秒間隔で37レースを連続取得した結果、CloudFrontにIP単位でブロックされ
# race/db.netkeiba.com が全て400になる事故が発生。最低間隔と連続取得上限を設ける。
MIN_INTERVAL = 3.0     # 1リクエストあたり最低間隔(秒)
BURST_LIMIT = 30       # この件数ごとに長い休憩を挟む
BURST_PAUSE = 60.0     # 休憩の長さ(秒)
_last_req = 0.0
_req_count = 0


def _throttle():
    """netkeibaへ出す全リクエストの直前に呼ぶ。間隔を空けブロックを防ぐ。"""
    global _last_req, _req_count
    wait = MIN_INTERVAL - (time.time() - _last_req)
    if wait > 0:
        time.sleep(wait)
    _req_count += 1
    if _req_count % BURST_LIMIT == 0:
        print(f"  [レート制限] {BURST_LIMIT}件取得 → {BURST_PAUSE:.0f}秒休憩")
        time.sleep(BURST_PAUSE)
    _last_req = time.time()


def _get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(_HEADERS)
    return _session


def _get_fallback_driver():
    """Seleniumフォールバック用ドライバを遅延生成（requestsがブロック時のみ使用）。"""
    global _fallback_driver
    if _fallback_driver is None:
        _fallback_driver = create_driver()
    return _fallback_driver


def _close_fallback_driver():
    global _fallback_driver
    if _fallback_driver is not None:
        try:
            _fallback_driver.quit()
        except Exception:
            pass
        _fallback_driver = None


def get_payout(race_id):
    """1レースの払戻データを取得。requests優先、ブロック/失敗時のみSelenium。
    返り値: 行リスト / [](払戻表なし=結果未確定/存在しない) / "BLOCKED"（ブロック検知）。"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    # ① requests優先（軽量・高速・ブロックされにくい）
    try:
        _throttle()
        r = _get_session().get(url, timeout=15)
        # CloudFrontのIPブロックは本文なしの4xxで来る。HTMLパースしても無意味なので即中断。
        if r.status_code >= 400:
            print(f"  [警告] netkeiba HTTP {r.status_code}（IPブロックの可能性）→ 取得中止")
            return "BLOCKED"
        r.encoding = r.apparent_encoding
        if not is_blocked(r.text):
            return parse_payout(BeautifulSoup(r.text, "html.parser"), race_id)
    except Exception:
        pass
    # ② Seleniumフォールバック（requestsがブロック/失敗時のみ）
    try:
        d = _get_fallback_driver()
        d.get(url)
        time.sleep(random.uniform(3.5, 6.0))
        page = d.page_source
        if is_blocked(page):
            return "BLOCKED"
        return parse_payout(BeautifulSoup(page, "html.parser"), race_id)
    except Exception:
        return "BLOCKED"


def _load_done_race_ids():
    """既に払戻を持っている race_id を返す（再開用）。

    payout_data.csv（netkeiba取得分）だけでなく jv_payouts.csv（JV-Link取得分）も
    見る。JRA-VANの契約終了で取得元をnetkeibaへ移すが、過去分の払戻はローカルに
    残っているので取り直す必要がない。

    ⚠ 2026-08-22まで payout_data.csv しか見ておらず、JVで持っている
      23,340レースを再取得しようとしていた。1レース2〜3.5秒待つので、
      無駄なアクセスが数万回になるところだった。
    """
    done = set()
    for path in (OUTPUT_CSV, os.path.join(BASE_DIR, "jv_payouts.csv")):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, low_memory=False, usecols=["race_id"])
            done |= set(df["race_id"].astype(str)
                        .str.replace(r"\.0$", "", regex=True).unique())
        except Exception:
            continue
    return done


def _save(rows, append=True):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if append and os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def build_payout_data(year=None):
    """払戻データを取得する。year を指定するとその年（例:'2025'）のレースのみ取得。
    バックテスト用途なら year='2025' で約10時間・3千レースに絞れる。"""
    print("race_features.csv を読み込み中...")
    df = pd.read_csv(FEAT_CSV, low_memory=False, usecols=["race_id"])
    all_ids = sorted(df["race_id"].astype(str).unique())
    if year:
        all_ids = [r for r in all_ids if str(r).startswith(str(year))]
        print(f"{year}年に絞り込み: {len(all_ids)}レース")
    else:
        print(f"全レース数: {len(all_ids)}")

    done = _load_done_race_ids()
    targets = [r for r in all_ids if r not in done]
    print(f"取得済み: {len(done)} → スキップ")
    print(f"新規取得対象: {len(targets)}")

    if not targets:
        print("全レース取得済みです")
        return

    buffer = []
    consecutive_blocks = 0

    for i, race_id in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {race_id}", end=" ")
        result = get_payout(race_id)   # requests優先・自動フォールバック

        if result == "BLOCKED":
            consecutive_blocks += 1
            print(f"⚠️ ブロック検知（連続{consecutive_blocks}回目）")
            if consecutive_blocks >= 3:
                print("\n連続ブロックのため中断します。取得済みは保存済み。")
                print("数時間〜1日空けて再実行してください。")
                _save(buffer); buffer = []
                _close_fallback_driver()
                return
            cooldown = random.uniform(600, 1200)
            print(f"  {cooldown/60:.0f}分クールダウン...")
            _close_fallback_driver()
            time.sleep(cooldown)
            continue
        else:
            consecutive_blocks = 0

        if result:
            buffer.extend(result)
            print(f"✓ {len(result)}件")
        else:
            print("- 払戻取得できず")

        # 礼儀待機（requests優先で軽量。ブロック回避のため適度に）
        time.sleep(random.uniform(2.0, 3.5))

        # 100レースごとに小休憩
        if (i + 1) % 100 == 0:
            rest = random.uniform(20, 40)
            print(f"  [{i+1}レース完了] {rest:.0f}秒休憩...")
            time.sleep(rest)

        # 200レースごとに中間保存
        if (i + 1) % 200 == 0 and buffer:
            _save(buffer)
            buffer = []
            print("--- 中間保存 ---")

    if buffer:
        _save(buffer)
    _close_fallback_driver()
    print(f"\n完了！ → {OUTPUT_CSV}")


def test_single(race_id):
    """単一レースのテスト取得（同着対応・組み合わせ分離の確認用）。"""
    try:
        result = get_payout(race_id)
        if result == "BLOCKED":
            print("ブロックされました")
            return
        print(f"\n{race_id} の払戻データ（{len(result)}件）:")
        print("-" * 50)
        for r in result:
            print(f"  {r['券種']:6} {r['組み合わせ']:12} "
                  f"{r['払戻金']:>8}円  {r['人気']}人気")
    finally:
        _close_fallback_driver()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # 4桁数字なら年フィルタ（例: python payout_scraper.py 2025）
        if arg.isdigit() and len(arg) == 4:
            build_payout_data(year=arg)
        else:
            test_single(arg)  # 12桁race_idなら単一テスト
    else:
        build_payout_data()
