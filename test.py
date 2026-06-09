"""
horse_idが引退馬・地方馬などでページ構造が異なる場合にNoneになる原因を調査
"""
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Noneになっていたhorse_idをいくつかテスト
test_ids = ["2017101922", "2017100870", "2013102687", "2011103940"]

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--log-level=3")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

for horse_id in test_ids:
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    print(f"\n=== {horse_id} ===")
    print(f"URL: {url}")

    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # blood_tableの存在確認
    blood = soup.find("table", class_="blood_table")
    print(f"blood_table: {blood is not None}")

    if blood:
        tds = blood.find_all("td")
        print(f"td数: {len(tds)}")
        for i, td in enumerate(tds[:6]):
            print(f"  td[{i}]: {td.get_text(strip=True)[:30]}")
    else:
        # blood_tableがない場合、全テーブルを確認
        tables = soup.find_all("table")
        print(f"テーブル数: {len(tables)}")
        for i, t in enumerate(tables[:5]):
            cls = t.get("class", [])
            tds = t.find_all("td")
            first = tds[0].get_text(strip=True)[:20] if tds else ""
            print(f"  table[{i}] class={cls}  first_td={first}")

        # ページタイトル確認（404やエラーページの可能性）
        h1 = soup.find("h1")
        print(f"h1: {h1.get_text(strip=True) if h1 else 'なし'}")

driver.quit()