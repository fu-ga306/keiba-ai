# -*- coding: utf-8 -*-
"""
診断用: Seleniumがnetkeibaの馬ページで実際に何を取得しているか確認する。
is_blockedの誤検知か、本当のbot検知かを切り分ける。
"""
import time
import random as _r
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# テスト対象（実在する有名馬: イクイノックス 2019105219）
TEST_HORSE_ID = "2019105219"
URL = f"https://db.netkeiba.com/horse/{TEST_HORSE_ID}/"


def make_driver(headless=True):
    options = Options()
    if headless:
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


def diagnose(headless):
    mode = "headless" if headless else "通常(画面表示)"
    print(f"\n{'='*50}")
    print(f"診断モード: {mode}")
    print(f"{'='*50}")
    driver = make_driver(headless=headless)
    try:
        driver.get(URL)
        time.sleep(_r.uniform(3.5, 6.0))
        page = driver.page_source
        soup = BeautifulSoup(page, "html.parser")

        print(f"  取得したHTML長: {len(page)} 文字")
        title = soup.find("title")
        print(f"  ページタイトル: {title.get_text(strip=True) if title else '(なし)'}")
        h1 = soup.find("h1")
        print(f"  h1: {h1.get_text(strip=True) if h1 else '(なし)'}")
        print(f"  テーブル数: {len(soup.find_all('table'))}")
        blood = soup.find("table", class_="blood_table")
        print(f"  blood_table: {'あり' if blood else 'なし'}")
        if blood:
            tds = blood.find_all("td")
            if len(tds) >= 1:
                a = tds[0].find("a")
                print(f"  父馬: {a.get_text(strip=True) if a else tds[0].get_text(strip=True)}")

        # ブロック兆候のチェック
        text = page.lower()
        signals = ["403", "access denied", "アクセスが集中", "アクセスを制限",
                   "too many requests", "429", "しばらく時間"]
        found = [s for s in signals if s in text]
        if found:
            print(f"  ⚠️ ブロック兆候検出: {found}")
        elif len(page) < 500:
            print(f"  ⚠️ ページが短すぎる（{len(page)}文字）→ ブロック疑い")
        else:
            print(f"  ✅ 正常にページ取得できている")

        # HTML冒頭を表示（何が返ってきているか目視確認用）
        print(f"\n  --- HTML冒頭300文字 ---")
        print("  " + page[:300].replace("\n", " "))
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    # headlessモードで診断
    diagnose(headless=True)
    print("\n\n次に画面表示モードでも試します（ブラウザが開きます）...")
    time.sleep(2)
    # 画面表示モードで診断（headless検知の切り分け）
    diagnose(headless=False)
