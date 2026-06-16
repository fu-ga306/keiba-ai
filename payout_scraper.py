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
import time
import random
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
    """Resultセルから組み合わせを抽出。同着なら複数返す。
    馬番は <span> または <div> 単位でまとまっている。
    例: 馬連同着 → ["04-11", "04-14"]
    """
    combos = []
    # liごと（同着が別liに入る構造）
    lis = result_td.find_all("li")
    if lis:
        for li in lis:
            nums = [s.get_text(strip=True) for s in li.find_all("span")]
            nums = [n for n in nums if n.isdigit()]
            if nums:
                combos.append("-".join(n.zfill(2) for n in nums))
        if combos:
            return combos

    # divごと（Result_Box が複数あるパターン）
    divs = result_td.find_all("div", recursive=False)
    if len(divs) > 1:
        for div in divs:
            nums = [s.get_text(strip=True) for s in div.find_all("span")]
            nums = [n for n in nums if n.isdigit()]
            if nums:
                combos.append("-".join(n.zfill(2) for n in nums))
        if combos:
            return combos

    # span を直接並べるパターン（単勝・複勝など、または同着なしの組み合わせ）
    spans = [s.get_text(strip=True) for s in result_td.find_all("span")]
    spans = [s for s in spans if s.isdigit()]
    if spans:
        # 複勝のように1頭ずつが別組（=別払戻）の場合と、
        # 馬連のように複数頭で1組の場合がある。
        # ここでは「全部つなげて1組」とし、複勝など1頭単位は
        # 呼び出し側の払戻数で分割対応する。
        combos.append("-".join(s.zfill(2) for s in spans))
    return combos


def parse_payout_amounts(payout_td):
    """Payoutセルから払戻金（円）を数値リストで返す。同着なら複数。
    例: "3,630円1,100円" → [3630, 1100]
    """
    amounts = []
    # spanごとに金額が入る
    spans = payout_td.find_all("span")
    if spans:
        for sp in spans:
            txt = sp.get_text(strip=True).replace(",", "").replace("円", "")
            if txt.isdigit():
                amounts.append(int(txt))
    if amounts:
        return amounts

    # spanがない場合、テキストを<br>や円で分割
    raw = payout_td.get_text(separator="\n", strip=True)
    for part in raw.replace("円", "\n").split("\n"):
        p = part.replace(",", "").strip()
        if p.isdigit():
            amounts.append(int(p))
    return amounts


def parse_ninki(ninki_td):
    """Ninkiセルから人気を数値リストで返す。同着なら複数。"""
    ninkis = []
    txt = ninki_td.get_text(separator="\n", strip=True)
    for part in txt.replace("人気", "\n").split("\n"):
        p = part.strip()
        if p.isdigit():
            ninkis.append(int(p))
    return ninkis


def get_payout(driver, race_id):
    """1レースの払戻データを取得。BLOCKEDならその旨を返す。"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    driver.get(url)
    time.sleep(random.uniform(3.5, 6.0))
    page = driver.page_source
    if is_blocked(page):
        return "BLOCKED"
    soup = BeautifulSoup(page, "html.parser")
    return parse_payout(soup, race_id)


def _load_done_race_ids():
    """既に取得済みのrace_idを返す（再開用）。"""
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            return set(df["race_id"].astype(str).unique())
        except Exception:
            return set()
    return set()


def _save(rows, append=True):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if append and os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def build_payout_data():
    print("race_features.csv を読み込み中...")
    df = pd.read_csv(FEAT_CSV, low_memory=False, usecols=["race_id"])
    all_ids = sorted(df["race_id"].astype(str).unique())
    print(f"全レース数: {len(all_ids)}")

    done = _load_done_race_ids()
    targets = [r for r in all_ids if r not in done]
    print(f"取得済み: {len(done)} → スキップ")
    print(f"新規取得対象: {len(targets)}")

    if not targets:
        print("全レース取得済みです")
        return

    driver = create_driver()
    buffer = []
    consecutive_blocks = 0

    for i, race_id in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {race_id}", end=" ")
        try:
            result = get_payout(driver, race_id)
        except Exception as e:
            print(f"エラー、再起動: {e}")
            try:
                driver.quit()
            except Exception:
                pass
            driver = create_driver()
            continue

        if result == "BLOCKED":
            consecutive_blocks += 1
            print(f"⚠️ ブロック検知（連続{consecutive_blocks}回目）")
            if consecutive_blocks >= 3:
                print("\n連続ブロックのため中断します。取得済みは保存済み。")
                print("数時間〜1日空けて再実行してください。")
                _save(buffer); buffer = []
                try: driver.quit()
                except Exception: pass
                return
            cooldown = random.uniform(600, 1200)
            print(f"  {cooldown/60:.0f}分クールダウン...")
            try: driver.quit()
            except Exception: pass
            time.sleep(cooldown)
            driver = create_driver()
            continue
        else:
            consecutive_blocks = 0

        if result:
            buffer.extend(result)
            n_doutaku = sum(1 for r in result if result.count(r) > 1)
            print(f"✓ {len(result)}件")
        else:
            print("- 払戻取得できず")

        # 待機（保守的）
        time.sleep(random.uniform(8.0, 14.0))

        # 50レースごとに休憩
        if (i + 1) % 50 == 0:
            rest = random.uniform(90, 180)
            print(f"  [{i+1}レース完了] {rest:.0f}秒休憩...")
            time.sleep(rest)

        # 100レースごとに中間保存
        if (i + 1) % 100 == 0 and buffer:
            _save(buffer)
            buffer = []
            print("--- 中間保存 ---")

    if buffer:
        _save(buffer)
    try: driver.quit()
    except Exception: pass
    print(f"\n完了！ → {OUTPUT_CSV}")


def test_single(race_id):
    """単一レースのテスト取得（同着対応の確認用）。"""
    driver = create_driver()
    try:
        result = get_payout(driver, race_id)
        if result == "BLOCKED":
            print("ブロックされました")
            return
        print(f"\n{race_id} の払戻データ（{len(result)}件）:")
        print("-" * 50)
        for r in result:
            print(f"  {r['券種']:6} {r['組み合わせ']:12} "
                  f"{r['払戻金']:>8}円  {r['人気']}人気")
    finally:
        try: driver.quit()
        except Exception: pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single(sys.argv[1])
    else:
        build_payout_data()
