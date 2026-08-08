import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import smtplib
import re
import schedule   # 7分前ジョブの登録に必須。コメントアウトすると main() が
                  # NameError で落ちる（2026-08-08〜09に発生）
import time
import os
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# コンソール出力をUTF-8化（cp932環境での罫線/絵文字によるUnicodeEncodeError→異常終了を防ぐ）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from result_tracker import record_from_prediction
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from model import LambdaRankWrapper  # pickle読み込みに必要

# ── 設定 ──────────────────────────────────────────────────────────────────
# 【セキュリティ修正】パスワード・パスは環境変数から取得
# 実行前に .env ファイルか OS の環境変数に設定してください:
#   GMAIL_ADDRESS=your@gmail.com
#   GMAIL_APP_PASS=xxxx xxxx xxxx xxxx
#   TO_ADDRESS=your@gmail.com
#   KEIBA_BASE_DIR=C:\Users\...\keiba_ai
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未インストール時は os.environ のみ参照

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS", "")
TO_ADDRESS    = os.environ.get("TO_ADDRESS", GMAIL_ADDRESS)

# メール配信フラグ（2026-07-18: 買い目のあるレース＝勝負/買い/堅実 のときだけ配信）
#   True  = 全レース配信 / False = 全レース非配信 / "buy_only" = 買うレースのみ配信
# 7分前ジョブは予想更新（オッズ直前反映→today_predictions/today_bets更新→push）として継続する。
SEND_EMAIL = "buy_only"

# オッズ取得の範囲（2026-07-18・IPブロック対策 案A）
#   False = 単勝(b1)のみ取得しChrome起動を5→1に激減（買い判定は単勝オッズ+人気で完結するため十分）。
#   True  = 単勝/複勝/ワイド/馬連/馬単の5種すべて取得（EV表示用・負荷5倍）。
# 将来 案B(requests化) で本質解決したらこのフラグは不要になる。
FETCH_ALL_ODDS = False

# 実際に取得するオッズページ（2026-08-05）。
#   3年検証で買うと決めたのは 単勝(b1) と 馬単(b6) の2つだけ。
#   複勝(b2)/ワイド(b4)/馬連(b5) は買わないので取得しない。
#   7/18にIPブロックを受けてChrome起動を5→1に減らした経緯があるため、
#   5種全部に戻さず必要な2種に絞る（起動1→2回）。ブロック再発は
#   単勝の運用まで止めるので、負荷は必要最小限にとどめる。
FETCH_ODDS_TYPES = ("b1", "b6")
BASE_DIR      = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

JYO_NAMES = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟",  5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉",
}


def ensure_single_instance(script_name: str):
    """多重起動ガード（2026-07-20）。
    同じスクリプトを実行中の別プロセスがいれば、この場で終了する。
    2026-07-19にスケジューラ手動起動×タスクスケジューラ6:55起動が重なり、
    keiba_autoが2本並走してメール二重・予想連発が発生した再発防止。"""
    import sys
    if "--force" in sys.argv:
        print("  [多重起動ガード] --force指定 → ガードをスキップ（既存が昇格ハング時の復旧用）")
        return
    try:
        import psutil
    except ImportError:
        return  # psutil無しなら判定不能のためスキップ
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            cl = p.info["cmdline"] or []
            if any(str(a).endswith(script_name) for a in cl):
                print(f"[多重起動ガード] {script_name} は既に PID{p.info['pid']} で実行中 → 起動を中止します")
                sys.exit(0)
        except Exception:
            continue

# ── 戦略の期待値しきい値（案B: 生確率ベース） ──────────────────
# model.py の期待値スイープで判明した最適値（2026/06/15確定）。
# 生確率は正規化版より小さく出るが、スイープの結果 +0.3 が
# トータル利益最大（251回 回収率129.2%）と判明したため踏襲。
EV_THRESHOLD_MAIN  = 0.3   # 戦略A/C/F/H等の主要しきい値
EV_THRESHOLD_SUB   = 0.2   # 戦略D等のサブしきい値
EV_THRESHOLD_TAIKO = 0.1   # ○対抗の印選出しきい値

# model.py の FEATURE_COLS と完全一致させること（フォールバック用）
FEATURE_COLS = [
    "枠番", "馬番", "斤量", "斤量_相対",
    "年齢", "is_male", "is_female", "is_castrated",
    "馬体重", "体重増減", "馬体重_相対",
    "人気", "出走頭数", "競馬場cd", "レース番号",
    "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
    "過去平均上り", "直近3走平均着順",
    "過去平均タイム秒", "直近3走平均タイム秒", "過去最速タイム秒",
    "直近3走平均上り", "過去平均体重増減",
    "体重増減_過去標準偏差", "体重増減_異常度",
    "距離カテゴリ", "距離別過去平均着順",
    "騎手勝率", "騎手複勝率", "調教師勝率", "調教師複勝率",
    "距離", "馬場状態_num", "is_turf", "クラス_num",
    "前走間隔", "同距離過去勝率", "同距離過去平均着順",
    "良馬場勝率", "重馬場勝率",
    "過去最速上り", "上り偏差", "距離別過去平均上り",
    "斤量変化", "乗り替わり", "連闘", "休み明け", "負担率",
    "レース内_過去勝率ランク", "レース内_直近3走平均着順ランク",
    "レース内_過去平均上りランク", "レース内_騎手勝率ランク",
    "競馬場距離過去勝率", "競馬場距離過去平均着順",
    "競馬場過去勝率", "競馬場過去平均着順",
    "過去平均先行指数", "先行馬フラグ",
    "想定先行馬数", "想定先行馬率", "他馬想定先行馬数", "差し馬×ハイペース想定",
    "開催月", "開催季節",
    "前走着順", "前走上り", "前走距離", "距離変化",
    "連続複勝フラグ", "連続勝利フラグ", "近走改善度",
    "平均タイム差", "騎手競馬場勝率",
    "距離×馬場_過去勝率", "距離×馬場_過去平均着順",
    "距離×クラス_過去勝率", "芝ダート×先行_過去勝率",
    "前走好走×人気薄", "前走着順×人気_乖離", "斤量×年齢_負担",
    "距離延長×前走好走", "距離短縮×前走好走",
    "距離変化比率", "大幅延長フラグ", "大幅短縮フラグ",
    "距離延長幅", "長距離フラグ", "大幅延長×長距離", "距離延長×先行",
    "経験最長距離", "経験範囲超過", "延長×距離経験不足",
    "長距離複勝率", "前走余力", "延長×前走余力",
    "騎手直近勝率", "騎手直近複勝率", "騎手調子トレンド",
    "直近5走勝利数", "直近5走複勝数", "直近5走平均着順",
    "過去獲得賞金累計", "過去平均獲得賞金",
    "回り_num",
    "父系_今回距離適性", "母父系_今回距離適性",
    "父系_長距離勝率", "母父系_長距離勝率",
    "父系_芝ダ適性", "父系_複勝率",
]


# ── ユーティリティ ────────────────────────────────────────────────────────
def _make_chrome_driver():
    """ヘッドレス Chrome ドライバーを生成して返す"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def calc_place_prob_harvill(win_probs: np.ndarray) -> np.ndarray:
    """
    ハーヴィルモデルで複勝確率（3着以内）を近似計算する。
    win_probs: 正規化済みの勝ち確率配列
    """
    n = len(win_probs)
    place_probs = np.zeros(n)
    for i in range(n):
        pi = win_probs[i]
        # 自分が1着の確率
        place_probs[i] += pi
        # 自分が2着・3着になる確率（ハーヴィル近似）
        for j in range(n):
            if j == i:
                continue
            pj = win_probs[j]
            s_excl_j = 1.0 - pj
            if s_excl_j <= 0:
                continue
            # j が1着、残りから i が2着
            p2 = pj * (pi / s_excl_j)
            place_probs[i] += p2
            # j が1着、k が2着、i が3着
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = win_probs[k]
                s_excl_jk = s_excl_j - pk
                if s_excl_jk <= 0:
                    continue
                place_probs[i] += pj * (pk / s_excl_j) * (pi / s_excl_jk)
    return place_probs


def kelly_fraction(win_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    1/4 ケリー基準で推奨賭け率を返す（0〜1）。
    fraction=0.25 で破産リスクを抑えた保守的設定。
    """
    if odds <= 1 or win_prob <= 0:
        return 0.0
    b = odds - 1
    q = 1.0 - win_prob
    k = (b * win_prob - q) / b
    return max(0.0, k * fraction)


# ── Step1: 当日レース一覧取得 ──────────────────────────────────────────────
def get_today_races():
    print("当日レース一覧を取得中...")
    today = datetime.now().strftime("%Y%m%d")
    driver = _make_chrome_driver()
    try:
        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={today}"
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        race_info = {}
        for a in soup.find_all("a", href=re.compile(r"race_id=\d{12}")):
            race_id = re.search(r"race_id=(\d{12})", a["href"]).group(1)
            if race_id in race_info:
                continue
            parent = a.find_parent()
            text = parent.get_text(strip=True) if parent else ""
            time_match = re.search(r"(\d{1,2}:\d{2})", text)
            if time_match:
                race_info[race_id] = time_match.group(1)
        # 開催日ガード: JRA実開催は必ず12レース/場以上。0<件数<閾値 は
        # 非開催日にnetkeibaが前売り/次開催の出馬表を一時的に見せた誤検出とみなし、
        # 「本日は非開催」として空を返す（メール送信・予想・プッシュを全経路で抑止）。
        MIN_RACE_DAY = 6
        if 0 < len(race_info) < MIN_RACE_DAY:
            print(f"  ⚠ レース数{len(race_info)}件は異常（非開催日/前売り誤検出）"
                  f"→ 本日は非開催として扱い、予想・メールを抑止します")
            return {}
        return race_info
    finally:
        driver.quit()


def classify_race_class(race_name, cond_text, grade_icon=None):
    """レース名(グレード)＋条件テキストから (クラス名, クラス_num 1-8) を判定。
    序列: 1 新馬/未勝利 / 2 1勝 / 3 2勝 / 4 3勝 / 5 オープン特別・L / 6 G3 / 7 G2 / 8 G1。
    netkeibaはグレードをローマ数字(GⅢ)やアイコン(Icon_GradeTypeN)で出すため複数表記に対応し、
    グレードは「オープン」より優先。全角数字も正規化する。"""
    z2h = str.maketrans("０１２３４５６７８９", "0123456789")
    name = (race_name or "").translate(z2h)
    cond = (cond_text or "").translate(z2h)
    # グレード: レース名の表記（ローマ数字・括弧）or アイコンクラスの番号
    grade = None
    if "GⅠ" in name or "(GI)" in name or "（GI）" in name:
        grade = "G1"
    elif "GⅡ" in name or "(GII)" in name or "（GII）" in name:
        grade = "G2"
    elif "GⅢ" in name or "(GIII)" in name or "（GIII）" in name:
        grade = "G3"
    elif grade_icon in ("1", "2", "3"):
        grade = {"1": "G1", "2": "G2", "3": "G3"}[grade_icon]
    if grade:
        return grade, {"G1": 8, "G2": 7, "G3": 6}[grade]
    if "未勝利" in cond:
        return "未勝利", 1
    if "新馬" in cond:
        return "新馬", 1
    if "1勝クラス" in cond or "500万" in cond:
        return "1勝クラス", 2
    if "2勝クラス" in cond or "1000万" in cond:
        return "2勝クラス", 3
    if "3勝クラス" in cond or "1600万" in cond:
        return "3勝クラス", 4
    if "オープン" in cond or "リステッド" in cond or "(L)" in name or "（L）" in name:
        return "オープン", 5
    return None, None


# ── Step2: 出馬表・オッズ取得 ──────────────────────────────────────────────
def get_race_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        # netkeibaがUTF-8に移行したため、固定指定ではなく自動判定にする
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")

        race_info = {}

        race_div = soup.find("div", class_="RaceData01")
        if race_div:
            text = race_div.get_text().replace("\xa0", " ")
            print(f"  RaceData01: {text[:80].encode('cp932', errors='replace').decode('cp932')}")
            dist_match = re.search(r"(\d{3,4})m", text)
            race_info["距離"] = int(dist_match.group(1)) if dist_match else None
            race_info["馬場種別"] = "芝" if "芝" in text else "ダート"
            race_info["is_turf"] = 1.0 if "芝" in text else 0.0

            # 回り（右/左/直）を抽出 — モデルの学習データと一致させる
            m_turn = re.search(r"[芝ダ](右|左|直)", text)
            race_info["回り"] = m_turn.group(1) if m_turn else None
            race_info["回り_num"] = {"右": 1, "左": 2, "直": 3}.get(race_info["回り"])

            race_info["馬場状態"] = None
            full_text = "".join(
                elem for elem in race_div.parent.find_all(string=True)
            ).replace("\n", "").replace(" ", "").replace("\xa0", "")
            for condition in ["不良", "稍重", "重", "良"]:
                if condition in full_text:
                    race_info["馬場状態"] = condition
                    break
            # 保険: RaceData01の局所テキストで拾えない場合(函館ダート等でNaN化していた)、
            # ページ全体からラベル付きで拾う。netkeibaは芝/ダートの状態を別表示するため、
            # ダート戦はダート側、芝戦は芝側の状態を優先する。1レース1ページなので誤検出しにくい。
            if race_info["馬場状態"] is None:
                page = soup.get_text()
                is_dirt = race_info.get("is_turf") == 0.0
                labels = ([r"ダート?[:：]", r"馬場[:：]"] if is_dirt
                          else [r"芝[:：]", r"馬場[:：]"])
                for lab in labels:
                    m = re.search(lab + r"\s*(不良|稍重|重|良)", page)
                    if m:
                        race_info["馬場状態"] = m.group(1)
                        break
            race_info["馬場状態_num"] = {"良": 1, "稍重": 2, "重": 3, "不良": 4}.get(race_info["馬場状態"])
        else:
            print("  RaceData01: 見つからず")

        race_div2 = soup.find("div", class_="RaceData02")
        if race_div2:
            text2 = race_div2.get_text()
            # レース名(グレード)を取得: G1/G2/G3はローマ数字やアイコンで出るため必須。
            name_div = soup.find("div", class_="RaceName")
            race_name = name_div.get_text(strip=True) if name_div else ""
            grade_icon = None
            if name_div:
                icon = name_div.find("span", class_=re.compile(r"Icon_GradeType\d"))
                if icon:
                    m = re.search(r"Icon_GradeType(\d)", " ".join(icon.get("class", [])))
                    grade_icon = m.group(1) if m else None
            # クラス名＋クラス_num(1-8) を優先順位付きで判定（グレード>条件>オープン）
            cls_name, cls_num = classify_race_class(race_name, text2, grade_icon)
            race_info["レースクラス"] = cls_name
            race_info["クラス_num"] = cls_num

        table = soup.find("table", class_="Shutuba_Table")
        print(f"  Shutuba_Table: {table is not None}")
        if table is None:
            return None

        rows = []
        for tr in table.find_all("tr"):
            td = tr.find_all("td")
            if len(td) >= 8:
                # 馬名セル(td[3])のリンクから horse_id を取得（血統結合に使う）。
                # netkeibaのURL構造 /horse/数字 は文字コード変更後も不変。
                horse_id = None
                a_tag = td[3].find("a", href=re.compile(r"/horse/\d+"))
                if a_tag and a_tag.has_attr("href"):
                    m = re.search(r"/horse/(\d+)", a_tag["href"])
                    if m:
                        horse_id = m.group(1)
                rows.append({
                    "枠番":   td[0].get_text(strip=True),
                    "馬番":   td[1].get_text(strip=True),
                    "馬名":   td[3].get_text(strip=True),
                    "horse_id": horse_id,
                    "性齢":   td[4].get_text(strip=True),
                    "斤量":   td[5].get_text(strip=True),
                    "騎手":   td[6].get_text(strip=True),
                    "調教師": td[7].get_text(strip=True),
                    "馬体重": td[8].get_text(strip=True) if len(td) > 8 else "",
                })

        print(f"  取得行数: {len(rows)}")
        if not rows:
            return None

        df = pd.DataFrame(rows)
        for key, val in race_info.items():
            df[key] = val
        df["race_id"] = str(race_id)
        df["上り"] = np.nan

    except Exception as e:
        import traceback
        print(f"出馬表取得エラー: {race_id}")
        traceback.print_exc()
        return None

    # ── オッズ・人気取得（Selenium）──
    # ── オッズ取得（単勝・複勝・ワイド・馬連）────────────────────────────
    def fetch_odds_page(race_id, odds_type, sleep_sec=3):
        """netkeibaのオッズページをSeleniumで取得。
        FETCH_ODDS_TYPES に無い券種はChromeを起動せず空を返す
        （下流は空マップで正常動作）。FETCH_ALL_ODDS=True なら全種取得。"""
        if not FETCH_ALL_ODDS and odds_type not in FETCH_ODDS_TYPES:
            return BeautifulSoup("", "html.parser")
        driver = _make_chrome_driver()
        try:
            url = (
                f"https://race.netkeiba.com/odds/index.html"
                f"?race_id={race_id}&type={odds_type}"
            )
            driver.get(url)
            time.sleep(sleep_sec)
            return BeautifulSoup(driver.page_source, "html.parser")
        finally:
            driver.quit()

    # ── 単勝・人気取得 ──
    try:
        soup_tan = fetch_odds_page(race_id, "b1")
        tables = soup_tan.find_all("table", class_="RaceOdds_HorseList_Table")
        odds_table = tables[0] if tables else None
        if odds_table:
            temp = []
            for tr in odds_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 6:
                    umaban = td[1].get_text(strip=True)
                    try:
                        odds_val = float(td[5].get_text(strip=True))
                    except ValueError:
                        odds_val = np.nan
                    temp.append((umaban, odds_val))
            valid = [(u, o) for u, o in temp if not np.isnan(o)]
            valid_sorted = sorted(valid, key=lambda x: x[1])
            ninki_map = {u: i+1 for i, (u, _) in enumerate(valid_sorted)}
            odds_map  = {u: (o, ninki_map.get(u, np.nan)) for u, o in temp}
            df["単勝オッズ"] = df["馬番"].astype(str).map(lambda x: odds_map.get(x, (np.nan, np.nan))[0])
            df["人気"]      = df["馬番"].astype(str).map(lambda x: odds_map.get(x, (np.nan, np.nan))[1])
            print(f"  単勝オッズ取得成功: {len(valid)}頭分")
        else:
            df["単勝オッズ"] = np.nan
            df["人気"]      = np.nan
    except Exception as e:
        print(f"  単勝オッズ取得エラー: {e}")
        df["単勝オッズ"] = np.nan
        df["人気"]      = np.nan

    # ── 複勝オッズ取得 ──
    # 2026-07-31: netkeibaのb1ページは単勝と複勝を同じテーブル群で返すので、
    #   b2を別途叩かずに soup_tan を再利用する（リクエスト増ゼロ・Chrome起動も増えない）。
    #   FETCH_ALL_ODDS=False でb2が空になり複勝が常に欠測だった問題の解消も兼ねる。
    #   b1に複勝表が無い版に当たった場合だけ、従来どおりb2へフォールバックする。
    try:
        tables = soup_tan.find_all("table", class_="RaceOdds_HorseList_Table")
        fuku_table = tables[1] if len(tables) > 1 else None
        if fuku_table is None:
            soup_fuku = fetch_odds_page(race_id, "b2")
            tables = soup_fuku.find_all("table", class_="RaceOdds_HorseList_Table")
            fuku_table = tables[0] if tables else None
        fuku_map = {}
        if fuku_table:
            for tr in fuku_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 6:
                    umaban = td[1].get_text(strip=True)
                    # 複勝は最低・最高オッズの範囲で表示される場合あり
                    odds_text = td[5].get_text(strip=True)
                    try:
                        # "1.5 - 2.3" のような形式に対応
                        odds_vals = [float(x) for x in odds_text.replace("－", "-").split("-") if x.strip()]
                        fuku_min = min(odds_vals)
                        fuku_max = max(odds_vals)
                        fuku_map[umaban] = (fuku_min, fuku_max)
                    except:
                        fuku_map[umaban] = (np.nan, np.nan)
        df["複勝オッズ_min"] = df["馬番"].astype(str).map(lambda x: fuku_map.get(x, (np.nan, np.nan))[0])
        df["複勝オッズ_max"] = df["馬番"].astype(str).map(lambda x: fuku_map.get(x, (np.nan, np.nan))[1])
        print(f"  複勝オッズ取得成功: {len(fuku_map)}頭分")
    except Exception as e:
        print(f"  複勝オッズ取得エラー: {e}")
        df["複勝オッズ_min"] = np.nan
        df["複勝オッズ_max"] = np.nan

    # ── ワイド・馬連オッズ取得（組み合わせ形式） ──
    try:
        soup_wide = fetch_odds_page(race_id, "b4")
        # ワイドは馬番ペアごとのオッズ
        wide_map = {}
        wide_table = soup_wide.find("table", id="odds_wide_block")
        if wide_table is None:
            wide_tables = soup_wide.find_all("table", class_="RaceOdds_HorseList_Table")
            wide_table = wide_tables[0] if wide_tables else None
        if wide_table:
            for tr in wide_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 4:
                    try:
                        uma1 = td[0].get_text(strip=True)
                        uma2 = td[1].get_text(strip=True)
                        odds_text = td[3].get_text(strip=True)
                        odds_vals = [float(x) for x in odds_text.replace("－","-").split("-") if x.strip()]
                        wide_map[(uma1, uma2)] = (min(odds_vals), max(odds_vals))
                    except:
                        pass
        df["_wide_map"] = [wide_map] * len(df)
        print(f"  ワイドオッズ取得成功: {len(wide_map)}組み合わせ")
    except Exception as e:
        print(f"  ワイドオッズ取得エラー: {e}")
        df["_wide_map"] = [{}] * len(df)

    try:
        soup_umaren = fetch_odds_page(race_id, "b5")
        umaren_map = {}
        umaren_tables = soup_umaren.find_all("table", class_="RaceOdds_HorseList_Table")
        umaren_table = umaren_tables[0] if umaren_tables else None
        if umaren_table:
            for tr in umaren_table.find_all("tr")[1:]:
                td = tr.find_all("td")
                if len(td) >= 4:
                    try:
                        uma1 = td[0].get_text(strip=True)
                        uma2 = td[1].get_text(strip=True)
                        odds_val = float(td[3].get_text(strip=True))
                        umaren_map[(uma1, uma2)] = odds_val
                    except:
                        pass
        df["_umaren_map"] = [umaren_map] * len(df)
        print(f"  馬連オッズ取得成功: {len(umaren_map)}組み合わせ")
    except Exception as e:
        print(f"  馬連オッズ取得エラー: {e}")
        df["_umaren_map"] = [{}] * len(df)

    # ── 馬単オッズ取得（順序あり：1着→2着） ──
    try:
        soup_umatan = fetch_odds_page(race_id, "b6")
        umatan_map = {}
        # 2026-08-05: netkeibaのHTML構造が変わっていた。
        #   旧: table.RaceOdds_HorseList_Table を1つ探して4列(1着/2着/?/オッズ)を読む
        #   新: table.Odds_Table が1着馬ごとに1つ。thが1着馬番、
        #       各trが (2着馬番, オッズ) の2列。
        # 旧コードは該当テーブルが0件で、馬単オッズが常に空だった。
        for tbl in soup_umatan.find_all("table", class_="Odds_Table"):
            th = tbl.find("th")
            if th is None:
                continue
            uma1 = th.get_text(strip=True)
            if not uma1.isdigit():
                continue
            for tr in tbl.find_all("tr"):
                td = tr.find_all("td")
                if len(td) < 2:
                    continue
                uma2 = td[0].get_text(strip=True)
                if not uma2.isdigit():
                    continue
                try:
                    # 1,424.8 のようにカンマ区切りで来る
                    odds_val = float(td[1].get_text(strip=True).replace(",", ""))
                except ValueError:
                    continue
                umatan_map[(uma1, uma2)] = odds_val  # (1着馬番, 2着馬番) -> オッズ
        df["_umatan_map"] = [umatan_map] * len(df)
        print(f"  馬単オッズ取得成功: {len(umatan_map)}組み合わせ")
    except Exception as e:
        print(f"  馬単オッズ取得エラー: {e}")
        df["_umatan_map"] = [{}] * len(df)

    return df


# ── Step3: 特徴量構築 ──────────────────────────────────────────────────────
def build_features(race_df, history_df):
    # ── 馬ごとの過去成績を集計 ──
    horse_stats = {}
    for horse, group in history_df.groupby("馬名"):
        valid = group.dropna(subset=["着順_num"])
        if len(valid) == 0:
            continue
        # 【バグ修正】race_id の最大値で最新レースを特定（ソート保証）
        valid = valid.sort_values("race_id")
        last3 = valid.tail(3)

        if "馬場状態_num" in valid.columns:
            goods = valid[valid["馬場状態_num"] == 1]
            heavy = valid[valid["馬場状態_num"] >= 3]
        else:
            goods = pd.DataFrame()
            heavy = pd.DataFrame()

        horse_stats[horse] = {
            "過去出走数":           len(valid),
            "過去平均着順":         valid["着順_num"].mean(),
            "過去勝率":             (valid["着順_num"] == 1).mean(),
            "過去複勝率":           (valid["着順_num"] <= 3).mean(),
            "過去平均上り":         valid["上り"].mean() if "上り" in valid.columns else np.nan,
            "直近3走平均着順":     last3["着順_num"].mean(),
            "過去平均タイム秒":     valid["タイム秒"].mean() if "タイム秒" in valid.columns else np.nan,
            "直近3走平均タイム秒": last3["タイム秒"].mean() if "タイム秒" in last3.columns else np.nan,
            "過去最速タイム秒":     valid["タイム秒"].min()  if "タイム秒" in valid.columns else np.nan,
            "直近3走平均上り":     last3["上り"].mean() if "上り" in last3.columns else np.nan,
            "過去平均体重増減":     valid["体重増減"].mean() if "体重増減" in valid.columns else np.nan,
            "過去最速上り":         valid["上り"].min() if "上り" in valid.columns else np.nan,
            "上り偏差":             valid["上り"].std() if "上り" in valid.columns else np.nan,
            "良馬場勝率":           (goods["着順_num"] == 1).mean() if len(goods) > 0 else 0.0,
            "重馬場勝率":           (heavy["着順_num"] == 1).mean() if len(heavy) > 0 else 0.0,
            # 追加特徴量用
            "直近斤量":             valid["斤量"].iloc[-1] if "斤量" in valid.columns else np.nan,
            "直近騎手":             str(valid["騎手"].iloc[-1]) if "騎手" in valid.columns else "",
            "valid_df":             valid,
        }

    # ── 騎手・調教師の累積成績 ──
    jockey_stats  = {}
    trainer_stats = {}
    for _, row in history_df.sort_values("race_id").iterrows():
        j = str(row.get("騎手", ""))
        t = str(row.get("調教師", ""))
        chakujun = row.get("着順_num", np.nan)
        if pd.isna(chakujun):
            continue
        for stats, key in [(jockey_stats, j), (trainer_stats, t)]:
            if key not in stats:
                stats[key] = {"runs": 0, "wins": 0, "top3": 0}
            stats[key]["runs"] += 1
            if chakujun == 1:
                stats[key]["wins"] += 1
            if chakujun <= 3:
                stats[key]["top3"] += 1

    # ── 各馬の特徴量を構築 ──
    rows = []
    for _, row in race_df.iterrows():
        horse   = str(row.get("馬名", ""))
        jockey  = str(row.get("騎手", ""))
        trainer = str(row.get("調教師", ""))
        current_dist = pd.to_numeric(row.get("距離", np.nan), errors="coerce")

        feat = {
            "馬名":       horse,
            "race_id":     row.get("race_id", ""),
            "単勝オッズ": pd.to_numeric(row.get("単勝オッズ", np.nan), errors="coerce"),
            "人気":       pd.to_numeric(row.get("人気", np.nan), errors="coerce"),
            "枠番":       pd.to_numeric(row.get("枠番", np.nan), errors="coerce"),
            "馬番":       pd.to_numeric(row.get("馬番", np.nan), errors="coerce"),
            "斤量":       pd.to_numeric(row.get("斤量", np.nan), errors="coerce"),
            "距離":       current_dist,
            "is_turf":    1 if row.get("馬場種別") == "芝" else 0,
            "馬場状態":   row.get("馬場状態", None),
        }

        baba_map = {"良": 1, "稍重": 2, "重": 3, "不良": 4}
        feat["馬場状態_num"] = baba_map.get(row.get("馬場状態"), np.nan)

        class_map = {
            "新馬": 1, "未勝利": 1,
            "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
            "オープン": 5, "G3": 6, "G2": 7, "G1": 8,
        }
        feat["クラス_num"] = class_map.get(row.get("レースクラス"), np.nan)
        feat["回り_num"] = row.get("回り_num")

        seire = str(row.get("性齢", ""))
        feat["is_male"]      = 1 if "牡" in seire else 0
        feat["is_female"]    = 1 if "牝" in seire else 0
        feat["is_castrated"] = 1 if "セ" in seire else 0
        age_match = re.search(r"(\d+)", seire)
        feat["年齢"] = int(age_match.group(1)) if age_match else np.nan

        weight_raw = str(row.get("馬体重", ""))
        w_match = re.search(r"(\d+)", weight_raw)
        feat["馬体重"] = int(w_match.group(1)) if w_match else np.nan
        d_match = re.search(r"\(([+-]?\d+)\)", weight_raw)
        feat["体重増減"] = int(d_match.group(1)) if d_match else np.nan

        # 馬の過去成績
        hs = horse_stats.get(horse, {})
        for key in [
            "過去出走数", "過去平均着順", "過去勝率", "過去複勝率",
            "過去平均上り", "直近3走平均着順", "過去平均タイム秒",
            "直近3走平均タイム秒", "過去最速タイム秒",
            "直近3走平均上り", "過去平均体重増減", "良馬場勝率", "重馬場勝率",
            "過去最速上り", "上り偏差",
        ]:
            feat[key] = hs.get(key, np.nan)

        # 【バグ修正】同距離過去成績：feature.py と同じく ±200m でマッチ
        feat["同距離過去勝率"]     = 0.0
        feat["同距離過去平均着順"] = np.nan
        if "valid_df" in hs and pd.notna(current_dist):
            v_df = hs["valid_df"]
            if "距離" in v_df.columns:
                same_dist_df = v_df[
                    (pd.to_numeric(v_df["距離"], errors="coerce") >= current_dist - 200) &
                    (pd.to_numeric(v_df["距離"], errors="coerce") <= current_dist + 200)
                ]
                if len(same_dist_df) > 0:
                    feat["同距離過去勝率"]     = (same_dist_df["着順_num"] == 1).mean()
                    feat["同距離過去平均着順"] = same_dist_df["着順_num"].mean()

        # 【バグ修正】前走間隔：ソート済み valid_df の最終行を使う
        if "valid_df" in hs and "race_id" in hs["valid_df"].columns and len(hs["valid_df"]) > 0:
            try:
                current_date = datetime.strptime(
                    str(row.get("race_id", ""))[:8], "%Y%m%d"
                )
                last_race_id = str(hs["valid_df"]["race_id"].astype(str).max())[:8]
                last_date    = datetime.strptime(last_race_id, "%Y%m%d")
                feat["前走間隔"] = (current_date - last_date).days / 7
            except Exception:
                feat["前走間隔"] = np.nan
        else:
            feat["前走間隔"] = np.nan

        # ── 距離変化系特徴量（学習時features.pyと同じロジック）──
        prev_dist = np.nan
        if "valid_df" in hs and len(hs["valid_df"]) > 0 and "距離" in hs["valid_df"].columns:
            try:
                prev_dist = pd.to_numeric(
                    hs["valid_df"]["距離"].iloc[-1], errors="coerce"
                )
            except Exception:
                prev_dist = np.nan
        feat["前走距離"] = prev_dist
        if pd.notna(current_dist) and pd.notna(prev_dist):
            feat["距離変化"]     = current_dist - prev_dist
            feat["距離変化比率"] = current_dist / prev_dist if prev_dist > 0 else np.nan
            feat["大幅延長フラグ"] = 1.0 if (current_dist - prev_dist) > 200 else 0.0
            feat["大幅短縮フラグ"] = 1.0 if (current_dist - prev_dist) < -200 else 0.0
            feat["距離延長幅"]   = max(current_dist - prev_dist, 0)
        else:
            feat["距離変化"]     = np.nan
            feat["距離変化比率"] = np.nan
            feat["大幅延長フラグ"] = np.nan
            feat["大幅短縮フラグ"] = np.nan
            feat["距離延長幅"]   = np.nan
        feat["長距離フラグ"] = 1.0 if (pd.notna(current_dist) and current_dist >= 2001) else 0.0
        feat["大幅延長×長距離"] = (
            (feat["大幅延長フラグ"] if pd.notna(feat["大幅延長フラグ"]) else 0)
            * feat["長距離フラグ"]
        )
        # 距離延長×先行は先行馬フラグ確定後に計算（下流で補完）

        prev_jockey    = hs.get("直近騎手", "")
        feat["乗り替わり"] = 0 if (prev_jockey == "" or prev_jockey == current_jockey) else 1

        prev_kinryo    = hs.get("直近斤量", np.nan)
        current_kinryo = feat["斤量"]
        feat["斤量変化"] = (
            current_kinryo - prev_kinryo
            if pd.notna(prev_kinryo) and pd.notna(current_kinryo)
            else np.nan
        )

        interval = feat.get("前走間隔", np.nan)
        feat["連闘"]   = 1 if pd.notna(interval) and interval <= 1  else 0
        feat["休み明け"] = 1 if pd.notna(interval) and interval >= 12 else 0
        feat["負担率"] = (
            current_kinryo / feat["馬体重"]
            if pd.notna(current_kinryo) and pd.notna(feat.get("馬体重")) and feat["馬体重"] > 0
            else np.nan
        )

        # 騎手・調教師成績
        js = jockey_stats.get(jockey, {})
        feat["騎手勝率"]   = js["wins"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        feat["騎手複勝率"] = js["top3"] / js["runs"] if js.get("runs", 0) > 0 else np.nan
        ts = trainer_stats.get(trainer, {})
        feat["調教師勝率"]   = ts["wins"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan
        feat["調教師複勝率"] = ts["top3"] / ts["runs"] if ts.get("runs", 0) > 0 else np.nan

        rows.append(feat)

    pdf = pd.DataFrame(rows)
    pdf["出走頭数"]    = len(pdf)
    pdf["馬体重_相対"] = pdf["馬体重"] - pdf["馬体重"].mean()
    pdf["斤量_相対"]   = pdf["斤量"]   - pdf["斤量"].mean()
    pdf["人気順位"]    = pdf["人気"].rank()
    pdf["人気_inv"]    = 1 / pdf["人気"].clip(lower=1)

    race_id_str = str(pdf["race_id"].iloc[0])
    pdf["競馬場cd"]   = int(race_id_str[4:6])
    pdf["レース番号"] = int(race_id_str[10:12])
    pdf["距離カテゴリ"] = pd.cut(
        pdf["レース番号"], bins=[0, 4, 8, 12], labels=[1, 2, 3]
    ).astype(float)
    pdf["距離別過去平均着順"] = np.nan
    pdf["上り"] = np.nan

    # 距離別過去平均上り（horse_statsから距離カテゴリ別に取得）
    def get_dist_avg_agari(row):
        horse = row["馬名"]
        hs = horse_stats.get(horse, {})
        v_df = hs.get("valid_df", pd.DataFrame())
        if len(v_df) == 0 or "上り" not in v_df.columns:
            return np.nan
        dist_cat = row.get("距離カテゴリ", np.nan)
        if pd.isna(dist_cat):
            return np.nan
        # 距離カテゴリに対応するレース番号範囲でフィルタ
        bins = {1.0: (1, 4), 2.0: (5, 8), 3.0: (9, 12)}
        rng = bins.get(float(dist_cat))
        if rng is None:
            return np.nan
        if "レース番号" in v_df.columns:
            same = v_df[
                (pd.to_numeric(v_df["レース番号"], errors="coerce") >= rng[0]) &
                (pd.to_numeric(v_df["レース番号"], errors="coerce") <= rng[1])
            ]
            return same["上り"].mean() if len(same) > 0 else np.nan
        return np.nan

    pdf["距離別過去平均上り"] = pdf.apply(get_dist_avg_agari, axis=1)

    # レース内ランク特徴量
    pdf["レース内_過去勝率ランク"] = pdf["過去勝率"].rank(
        ascending=False, method="min"
    )
    pdf["レース内_直近3走平均着順ランク"] = pdf["直近3走平均着順"].rank(
        ascending=True, method="min"
    )
    pdf["レース内_過去平均上りランク"] = pdf["過去平均上り"].rank(
        ascending=True, method="min"
    )
    pdf["レース内_騎手勝率ランク"] = pdf["騎手勝率"].rank(
        ascending=False, method="min"
    )

    # 全 FEATURE_COLS を数値型に統一
    for col in FEATURE_COLS:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    return pdf


# ── Step4: メール本文作成 ──────────────────────────────────────────────────
def make_email_body(race_id, pdf):
    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
    now      = datetime.now().strftime("%Y/%m/%d %H:%M")

    lines = [f"【競馬AI予測】{jyo_name} {race_no}R  {now}\n"]
    lines.append("=" * 40)

    baba = "不明"
    if "馬場状態" in pdf.columns:
        val = pdf["馬場状態"].iloc[0]
        if pd.notna(val) and str(val).strip() not in ("", "None"):
            baba = str(val).strip()

    dist = (
        int(pdf["距離"].iloc[0])
        if "距離" in pdf.columns and pd.notna(pdf["距離"].iloc[0])
        else "不明"
    )
    turf = "芝" if pdf["is_turf"].iloc[0] == 1 else "ダート"
    lines.append(f"馬場: {turf} {dist}m  状態: {baba}\n")

    # ── 購入推奨度（買い指数・レース単位／honest backtestで単勝回収率に較正済み）──
    try:
        _kai_s = pdf["買い指数"].dropna() if "買い指数" in pdf.columns else pd.Series(dtype=float)
        if len(_kai_s) > 0:
            _ki  = int(_kai_s.iloc[0])
            _rec = str(pdf["購入推奨"].iloc[0]) if "購入推奨" in pdf.columns else ""
            _roi = str(pdf["想定単回収"].iloc[0]) if "想定単回収" in pdf.columns else ""
            _mk  = {"勝負": "🔥", "買い": "✅", "堅実": "🟢", "少額": "⚠", "見送り": "❌"}.get(_rec, "")
            lines.append(f"【購入推奨度】{_mk} {_rec}   買い指数 {_ki}/100   想定回収 {_roi}")
            lines.append("")
    except Exception:
        pass

    # ── 推奨買い目（_race_bet_plan・today_betsと同一の確定メニュー）────────────
    try:
        from keiba_predict import _build_bet_rows, _race_bet_plan
        _plan = _race_bet_plan(pdf)
        _rows = _build_bet_rows(pdf, str(race_id))
        if _rows:
            _myo = pdf[pdf["妙味軸"] == "◎妙"] if "妙味軸" in pdf.columns else pdf.iloc[0:0]
            if len(_myo):
                _m = _myo.iloc[0]
                _mp = f"{int(_m['人気'])}番人気" if pd.notna(_m.get("人気")) else "-"
                lines.append(f"★推奨買い目（軸: ◎妙 {_m['馬番']}番 {_m['馬名']} {_mp}）")
            else:
                _hn = pdf[pdf["印"] == "◎"]
                _hs = f"{int(_hn.iloc[0]['馬番'])}番 {_hn.iloc[0]['馬名']}" if len(_hn) else ""
                lines.append(f"★推奨買い目（軸: ◎ {_hs}・両モデル合意）")
            _grp = {}
            for _r in _rows:
                _g = _grp.setdefault(_r["買い方"], [0, 0, _r["券種"], _r.get("BT回収率", "")])
                _g[0] += 1
                _g[1] += _r.get("金額", 100)
            for _name, (_pt, _amt, _kind, _bt) in _grp.items():
                _combos = [x["組み合わせ"] for x in _rows if x["買い方"] == _name]
                _cs = _combos[0] if _pt == 1 else f"{_combos[0]} 他{_pt-1}点"
                lines.append(f"  {_kind:4}: {_cs}  {_pt}点/{_amt:,}円 [BT{_bt}%]")
            _tot = sum(x.get("金額", 100) for x in _rows)
            lines.append(f"  ── 合計 {len(_rows)}点 / {_tot:,}円 ──")
            lines.append("  ※相手（人気◯位内）は直前オッズの人気で自動決定")
            lines.append("")
        elif _plan.get("判定") == "見送り":
            lines.append(f"★このレースは見送り推奨（{_plan.get('理由','')}）")
            lines.append("")
    except Exception as _e:
        print(f"  メール買い目セクションskip: {_e}")

    for mark, label in [("◎", "◎本命"), ("○", "○対抗"), ("▲", "▲穴馬")]:
        row_match = pdf[pdf["印"] == mark]
        if row_match.empty:
            continue
        row = row_match.iloc[0]

        odds     = row.get("単勝オッズ", np.nan)
        pop      = row.get("人気", np.nan)
        weight   = row.get("馬体重", np.nan)
        weight_d = row.get("体重増減", np.nan)
        win_p    = row.get("勝ち確率", np.nan)
        place_p  = row.get("複勝確率", np.nan)
        ev_win   = row.get("単勝期待値", np.nan)
        ev_place = row.get("複勝期待値", np.nan)
        winrate  = row.get("過去勝率", np.nan)
        runs     = int(row.get("過去出走数", 0)) if pd.notna(row.get("過去出走数")) else 0
        strategy = row.get("該当戦略", "")
        kelly    = row.get("推奨賭け率", np.nan)

        odds_str     = f"{odds}倍" if pd.notna(odds) and odds > 0 else "未確定"
        pop_str      = f"{int(pop)}番人気" if pd.notna(pop) and pop > 0 else "未確定"
        weight_str   = (
            f"{int(weight)}kg({weight_d:+.0f})"
            if pd.notna(weight) and pd.notna(weight_d)
            else "当日発表待ち"
        )
        win_p_str    = f"{win_p*100:.1f}%"   if pd.notna(win_p)    else "-"
        place_p_str  = f"{place_p*100:.1f}%" if pd.notna(place_p)  else "-"
        ev_win_str   = f"{ev_win:+.2f}"       if pd.notna(ev_win)   else "-"
        ev_place_str = f"{ev_place:+.2f}"     if pd.notna(ev_place) else "-"
        winrate_str  = (
            f"{winrate*100:.0f}%({runs}走)"
            if pd.notna(winrate) and runs > 0
            else "データなし"
        )
        kelly_str    = f"{kelly*100:.1f}%" if pd.notna(kelly) and kelly > 0 else "-"

        lines.append(f"\n{label} {row['馬名']}")
        lines.append(f"  オッズ: {odds_str}  {pop_str}")
        lines.append(f"  馬体重: {weight_str}")
        lines.append(f"  過去勝率: {winrate_str}")
        lines.append(f"  AI勝ち確率: {win_p_str}  複勝確率: {place_p_str}")
        lines.append(f"  単勝期待値: {ev_win_str}  複勝期待値: {ev_place_str}")
        lines.append(f"  推奨賭け率（1/4Kelly）: {kelly_str}")

        # 人気帯別バックテスト回収率コメント（MF◎ 3,144レース実績）
        if mark == "◎" and pd.notna(pop) and pop > 0:
            pop_i = int(pop)
            if pop_i == 1:
                lines.append("  [BT] 1番人気◎ → 単勝回収率97.8%（赤字） / 単勝見送り推奨")
                lines.append("       複勝・ワイドの軸 or 人気薄との組み合わせで勝負")
            elif pop_i <= 3:
                lines.append(f"  [BT] 2-3番人気◎ → 単勝回収率133.2%（黒字）")
            elif pop_i <= 6:
                lines.append(f"  [BT] 4-6番人気◎ → 単勝回収率172.3%（妙味あり）")
            else:
                lines.append(f"  [BT] 7番人気以下◎ → 単勝回収率311.9%（積極的に狙う）")

        if strategy:
            verdict = f"【高回収率該当】{strategy}"
        elif pd.notna(ev_win):
            if win_p < 0.03:
                verdict = "[--] 見送り（超大穴・ノイズ判定）"
            elif ev_win >= 1.0 and win_p >= 0.12:
                verdict = "[++] 強く買い推奨"
            elif ev_win >= 0.5 and win_p >= 0.06:
                verdict = "[~] 買い推奨"
            elif ev_win >= 0.2 and win_p >= 0.04:
                verdict = "[OK] 買い検討"
            else:
                verdict = "[--] 見送り"
        else:
            verdict = "[?] オッズ確定後に判断"

        lines.append(f"  【判定】{verdict}")

    lines.append("\n" + "=" * 40)

    valid_pdf = pdf[pdf["勝ち確率"] >= 0.03]

    lines.append("\n[単勝期待値ランキング] TOP3（勝率3%以上）")
    for _, row in valid_pdf.sort_values("単勝期待値", ascending=False).head(3).iterrows():
        ev    = row.get("単勝期待値", np.nan)
        odds  = row.get("単勝オッズ", np.nan)
        win_p = row.get("勝ち確率", np.nan)
        if pd.notna(ev):
            lines.append(
                f"  {row['馬名']}: 期待値{ev:+.2f} "
                f"(確率{win_p*100:.1f}% × {odds}倍)"
            )

    lines.append("\n[複勝期待値ランキング] TOP3（勝率3%以上）")
    for _, row in valid_pdf.sort_values("複勝期待値", ascending=False).head(3).iterrows():
        ev_p    = row.get("複勝期待値", np.nan)
        place_p = row.get("複勝確率", np.nan)
        est_odds = row.get("複勝推定オッズ", np.nan)
        if pd.notna(ev_p):
            lines.append(
                f"  {row['馬名']}: 期待値{ev_p:+.2f} "
                f"(確率{place_p*100:.1f}% × {est_odds:.2f}倍推定)"
            )

    lines.append("\n※複勝オッズはハーヴィルモデルで推定。実際と異なる場合があります。")

    # ── 馬券種別 買い目提案 ──────────────────────────────────────────
    lines.append("\n" + "=" * 40)
    lines.append("[買い目提案]")
    lines.append("=" * 40)

    # 単勝推奨
    lines.append("\n【単勝】")
    strategy_horses = pdf[pdf["該当戦略"] != ""].sort_values("単勝期待値", ascending=False)
    if not strategy_horses.empty:
        for _, row in strategy_horses.head(2).iterrows():
            ev  = row.get("単勝期待値", np.nan)
            odd = row.get("単勝オッズ", np.nan)
            kelly = row.get("推奨賭け率", np.nan)
            kelly_s = f"{kelly*100:.1f}%" if pd.notna(kelly) and kelly > 0 else "-"
            lines.append(
                f"  [+] {row['馬名']} {odd}倍 "
                f"期待値{ev:+.2f} Kelly:{kelly_s} "
                f"({row['該当戦略']})"
            )
    else:
        lines.append("  （戦略該当馬なし）")

    # 複勝推奨
    lines.append("\n【複勝】")
    valid_fuku = pdf[pdf["複勝確率"] >= 0.35].copy()
    if "複勝期待値_実" in pdf.columns:
        valid_fuku = valid_fuku[valid_fuku["複勝期待値_実"] >= 0.1]
        valid_fuku = valid_fuku.sort_values("複勝期待値_実", ascending=False)
        col = "複勝期待値_実"
    else:
        valid_fuku = valid_fuku.sort_values("複勝確率", ascending=False)
        col = "複勝確率"
    if not valid_fuku.empty:
        for _, row in valid_fuku.head(3).iterrows():
            fuku_min = row.get("複勝オッズ_min", np.nan)
            fuku_max = row.get("複勝オッズ_max", np.nan)
            if pd.notna(fuku_min):
                odds_s = f"{fuku_min}〜{fuku_max}倍"
            else:
                odds_s = "オッズ確定前"
            pp = row["複勝確率"]
            ev = row.get(col, np.nan)
            lines.append(
                f"  [+] {row['馬名']} {odds_s} "
                f"複勝確率{pp*100:.1f}% 期待値{ev:+.2f}"
            )
    else:
        lines.append("  （推奨馬なし）")

    # ワイド買い目
    wide_bets = pdf["_wide_bets"].iloc[0] if "_wide_bets" in pdf.columns else []
    if wide_bets:
        lines.append("\n【ワイド】")
        sorted_wide = sorted(wide_bets, key=lambda x: x["ワイド期待値"], reverse=True)
        for bet in sorted_wide[:3]:
            u1 = bet["馬番1"]
            u2 = bet["馬番2"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["ワイド期待値"]
            mark = "[+]" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} × {n2} "
                f"{bet['ワイドオッズ_min']}〜{bet['ワイドオッズ_max']}倍 "
                f"的中率{bet['ワイド的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    # 馬連買い目
    umaren_bets = pdf["_umaren_bets"].iloc[0] if "_umaren_bets" in pdf.columns else []
    if umaren_bets:
        lines.append("\n【馬連】")
        sorted_umaren = sorted(umaren_bets, key=lambda x: x["馬連期待値"], reverse=True)
        for bet in sorted_umaren[:3]:
            u1 = bet["馬番1"]
            u2 = bet["馬番2"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["馬連期待値"]
            mark = "[+]" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} × {n2} "
                f"{bet['馬連オッズ']}倍 "
                f"的中率{bet['馬連的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    # 馬単買い目
    umatan_bets = pdf["_umatan_bets"].iloc[0] if "_umatan_bets" in pdf.columns else []
    if umatan_bets:
        lines.append("\n【馬単】")
        sorted_umatan = sorted(umatan_bets, key=lambda x: x["馬単期待値"], reverse=True)
        for bet in sorted_umatan[:3]:
            u1 = bet["馬番1着"]
            u2 = bet["馬番2着"]
            n1 = pdf[pdf["馬番"].astype(str) == u1]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u1]) > 0 else u1
            n2 = pdf[pdf["馬番"].astype(str) == u2]["馬名"].iloc[0] if len(pdf[pdf["馬番"].astype(str) == u2]) > 0 else u2
            ev = bet["馬単期待値"]
            mark = "[+]" if ev >= 0 else "△"
            lines.append(
                f"  {mark} {n1} → {n2} "
                f"{bet['馬単オッズ']}倍 "
                f"的中率{bet['馬単的中確率']*100:.1f}% 期待値{ev:+.2f}"
            )

    lines.append("\n※AIによる予測です。投資は自己責任でお願いします。")
    return "\n".join(lines)


# ── Step5: メール送信 ──────────────────────────────────────────────────────
def send_email(subject, body):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        print("  メール設定が未完了（環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS を設定）")
        return
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = TO_ADDRESS
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.send_message(msg)
        print(f"  メール送信完了 → {TO_ADDRESS}")
    except Exception as e:
        print(f"  メール送信エラー: {e}")


# ── Step6: 1レース実行 ────────────────────────────────────────────────────
def run_single_race(race_id, history_df, models_pack):
    print(f"\n[{datetime.now().strftime('%H:%M')}] {race_id} 予測開始...")
    try:
        from keiba_predict import predict_race_pdf
        pdf = predict_race_pdf(race_id, history_df=history_df, models_pack=models_pack)
    except Exception as e:
        import traceback
        print(f"  予測エラー: {e}")
        traceback.print_exc()
        return
    if pdf is None:
        print(f"  出馬表取得失敗: {race_id}")
        return

    jyo_cd   = int(str(race_id)[4:6])
    race_no  = int(str(race_id)[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))

    body    = make_email_body(race_id, pdf)
    print(body)
    # 買い目の有無・判定を取得（buy_only配信の判定に使う）
    _has_bets, _verdict = False, ""
    try:
        from keiba_predict import _build_bet_rows, _race_bet_plan
        _has_bets = len(_build_bet_rows(pdf, str(race_id))) > 0
        _verdict = _race_bet_plan(pdf).get("判定", "")
    except Exception:
        pass
    _do_send = (SEND_EMAIL is True) or (SEND_EMAIL == "buy_only" and _has_bets)
    if _do_send:
        _vmap = {"勝負": "🔥勝負", "買い": "✅買い", "堅実": "🟢堅実"}
        subject = f"【競馬AI {_vmap.get(_verdict, '')}】{jyo_name} {race_no}R 買い目"
        send_email(subject, body)
    else:
        _why = "非開催/見送りレース" if SEND_EMAIL == "buy_only" else "メール配信OFF"
        print(f"  （メール非送信: {_why}・ダッシュボードで確認）")
    try:
        record_from_prediction(race_id, pdf)
    except Exception as e:
        print(f"  記録保存エラー: {e}")

    # 自動投票（直前オッズで確定した today_bets を投票）。既定OFF＋ドライランで安全。
    try:
        import auto_vote
        auto_vote.place_race_bets(str(race_id))
    except Exception as e:
        print(f"  自動投票スキップ（予想は継続）: {e}")

    # 直前更新をダッシュボードに反映（predict_race_pdfが更新した
    # today_predictions.csv / today_bets.csv をプッシュ→キャッシュクリア）
    _push_latest(f"{jyo_name} {race_no}R 直前更新 {datetime.now().strftime('%H:%M')}")


def _push_latest(message: str):
    """予想・買い目CSVをGitHubへプッシュしダッシュボードを更新する（7分前ジョブ用）。"""
    import subprocess
    try:
        for f in ("today_predictions.csv", "today_bets.csv", "prediction_record_v2.csv",
                  "odds_history.csv"):   # オッズ変動蓄積データもバックアップ
            p = os.path.join(BASE_DIR, f)
            if os.path.exists(p):
                subprocess.run(["git", "add", f], cwd=BASE_DIR, capture_output=True)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=BASE_DIR,
                            capture_output=True, text=True)
        if not st.stdout.strip():
            return
        subprocess.run(["git", "commit", "-m", message], cwd=BASE_DIR, capture_output=True)
        subprocess.run(["git", "push"], cwd=BASE_DIR, capture_output=True)
        print(f"  [Git] 直前更新プッシュ: {message}")
    except Exception as e:
        print(f"  [Git] プッシュエラー: {e}")
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000/api/refresh", timeout=3)
    except Exception:
        pass


# ── Step7: スケジューラー ─────────────────────────────────────────────────
def main():
    print(f"=== 競馬AI自動予測 起動 [{datetime.now().strftime('%Y/%m/%d %H:%M')}] ===\n")
    ensure_single_instance("keiba_auto.py")

    # レース有無を先に確認(2026-07-27〜)。モデル読込(数GB)より前に判定することで、
    # レースの無い日(月曜など)に無駄な大容量ロードでRAMを圧迫しない。
    race_info = get_today_races()
    if not race_info:
        print("本日のレースが取得できませんでした")
        return

    print("モデル読み込み中...")
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    is_multi = saved.get("format") == "multi_v1" and saved.get("place2") and saved.get("place3")
    models_pack = {
        "win":    {"models": saved.get("win", {}).get("models", saved["models"]),
                   "use_cols": saved.get("win", {}).get("use_cols", saved["use_cols"])},
        "place2": saved["place2"] if is_multi else None,
        "place3": saved["place3"] if is_multi else None,
    }
    print("  3モデル構成 → 独立予想を使用" if is_multi else "  旧モデル構成 → ハーヴィル変換を使用")

    import mf_model_io
    if mf_model_io.exists(BASE_DIR):
        # MFは妙味検出の心臓部。読込失敗すると妙が一切出ず判定が全て堅実/見送りに
        # 劣化する（2026-07-19に発生）。3回リトライし、それでも失敗なら大警告を出す。
        # 2026-07-27〜: model_mf_parts/があれば逐次読込(ピークRAM激減)、無ければ従来pkl。
        mf_saved = None
        for _try in range(3):
            try:
                mf_saved = mf_model_io.load_mf(BASE_DIR)
                break
            except Exception as _e:
                print(f"  MF読込失敗({_try+1}/3): {_e} → 5秒後リトライ")
                time.sleep(5)
        if mf_saved is not None:
            models_pack["mf"] = mf_saved
            print("  市場フリーモデル読み込み完了")
        else:
            print("  " + "!" * 60)
            print("  !! 警告: MFモデル読込に3回失敗。妙味検出が無効のまま稼働します")
            print("  !! → 判定が堅実/見送りだけに劣化します。model_mf.pkl/model_mf_partsを確認してください")
            print("  " + "!" * 60)
            models_pack["mf"] = None
    else:
        models_pack["mf"] = None

    print("履歴データ読み込み中（しばらくかかります）...")
    history_df = pd.read_csv(
        os.path.join(BASE_DIR, "race_data_clean.csv"), low_memory=False
    )
    print(f"読み込み完了: {len(history_df)}行\n")

    print(f"\n{len(race_info)}レースをスケジュール登録中...")
    now = datetime.now()
    scheduled = 0

    for race_id, race_time in sorted(race_info.items()):
        time_match = re.search(r"(\d{1,2}):(\d{2})", race_time)
        if not time_match:
            continue

        h = int(time_match.group(1))
        m = int(time_match.group(2)) - 7
        if m < 0:
            m += 60
            h -= 1

        notify_time  = f"{h:02d}:{m:02d}"
        scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if scheduled_dt < now:
            print(f"  スキップ（過去）: {race_id} {notify_time}")
            continue

        schedule.every().day.at(notify_time).do(
            run_single_race,
            race_id=race_id,
            history_df=history_df,
            models_pack=models_pack,
        )
        jyo_cd   = int(str(race_id)[4:6])
        race_no  = int(str(race_id)[10:12])
        jyo_name = JYO_NAMES.get(jyo_cd, str(jyo_cd))
        print(
            f"  予約: {jyo_name} {race_no}R ({race_id}) "
            f"→ {notify_time}（発走: {race_time}）"
        )
        scheduled += 1

    if scheduled == 0:
        print("予約できるレースがありませんでした")
        return

    print(f"\n{scheduled}件を予約しました。待機中... (Ctrl+Cで停止)\n")
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    # 自動スケジュール待機モード（各レース発走7分前に予想・メール送信）
    main()