# -*- coding: utf-8 -*-
"""IPATの画面構造を調べる（投票はしない・絶対に確定しない）

なぜ必要か
  自動投票の3関数(_ipat_login / _ipat_enter_bets / _ipat_confirm)を書くには、
  実画面のinput名・ボタンの位置が要る。推測で書いたselectorはまず当たらないし、
  当たらないまま本番に置くと「静かに投票されない」か「違う馬券を買う」。
  だから先に画面を見る。ここでは一切購入しない。

安全設計
  ① 確定系のボタンは絶対に押さない（_FORBIDDEN の語を含む要素はクリック禁止）
  ② 金額入力もしない。買い目も入れない。見るだけ
  ③ 出力から認証情報を伏字にする（このファイルを共有しても漏れない）
  ④ 各段階でスクリーンショットを残す

使い方（あなたの環境で実行してください）
  1) .env に4つ入れる
       IPAT_INET_ID=...
       IPAT_SUBSCRIBER=...
       IPAT_PASSWORD=...
       IPAT_PARS=...
  2) 発売時間中（開催日の朝〜最終レース前）に実行する
       python ipat_probe.py login     ログイン前の画面だけ見る（認証不要）
       python ipat_probe.py inside    ログインして中の画面構造を見る
  3) ipat_probe/ に出た *.txt と *.png を確認する

  ⚠ headless=False（画面が出る）で動きます。何が起きているか自分の目で見るため。
     途中で止めたければウィンドウを閉じるかCtrl+Cで構いません。
"""
import os
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ipat_probe")
LOGIN_URL = "https://www.ipat.jra.go.jp/"

# この語を含むボタン・リンクは自動でクリックしない（実弾防止の最終線）
_FORBIDDEN = ["購入", "確定", "投票する", "発売", "決定", "実行", "送信", "OK"]

_CRED_KEYS = ["IPAT_INET_ID", "IPAT_SUBSCRIBER", "IPAT_PASSWORD", "IPAT_PARS"]


def log(m):
    print(m, flush=True)


def _load_env():
    """.env を読む。os.environ が優先。"""
    v = {k: os.environ.get(k, "") for k in _CRED_KEYS}
    p = os.path.join(BASE_DIR, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, val = line.split("=", 1)
            k, val = k.strip(), val.strip().strip('"').strip("'")
            if k in _CRED_KEYS and not v.get(k):
                v[k] = val
    return v


def _mask(text, creds):
    """出力から認証情報を消す。共有しても漏れないようにするため。"""
    for k, val in creds.items():
        if val and len(val) >= 3:
            text = text.replace(val, f"<{k}>")
    return text


def _driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    o = Options()
    # headlessにしない。何が起きているか目で見えることを優先する。
    o.add_argument("--window-size=1280,1000")
    o.add_argument("--log-level=3")
    o.add_experimental_option("excludeSwitches", ["enable-logging"])
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                            options=o)


def _dump(drv, tag, creds):
    """今の画面の構造をテキストに落とす。"""
    from selenium.webdriver.common.by import By
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    lines = [f"# {tag}  {datetime.now():%Y/%m/%d %H:%M:%S}",
             f"URL   : {drv.current_url}",
             f"TITLE : {drv.title}", ""]

    for label, sel in [("input", "input"), ("select", "select"),
                       ("button", "button"), ("link", "a"),
                       ("form", "form"), ("iframe", "iframe")]:
        try:
            els = drv.find_elements(By.CSS_SELECTOR, sel)
        except Exception:
            continue
        if not els:
            continue
        lines.append(f"── {label}  {len(els)}個 " + "─" * 40)
        for i, e in enumerate(els[:60]):
            try:
                a = {k: (e.get_attribute(k) or "")
                     for k in ("type", "name", "id", "class", "value",
                               "href", "placeholder")}
                txt = (e.text or "").strip().replace("\n", " ")[:40]
                vis = "見える" if e.is_displayed() else "隠れ"
                danger = " ⚠確定系" if any(w in (txt + a["value"])
                                        for w in _FORBIDDEN) else ""
                lines.append(
                    f"  [{i:>2}] {vis} type={a['type']!r} name={a['name']!r} "
                    f"id={a['id']!r} value={a['value']!r} text={txt!r}"
                    f"{danger}")
                if label == "link" and a["href"]:
                    lines[-1] += f"\n        href={a['href'][:110]}"
            except Exception:
                continue
        if len(els) > 60:
            lines.append(f"  …他{len(els)-60}個")
        lines.append("")

    body = _mask("\n".join(lines), creds)
    fp = os.path.join(OUT_DIR, f"{ts}_{tag}.txt")
    open(fp, "w", encoding="utf-8").write(body)
    try:
        drv.save_screenshot(os.path.join(OUT_DIR, f"{ts}_{tag}.png"))
    except Exception:
        pass
    log(f"  → {os.path.basename(fp)} と .png を保存")
    return fp


def probe_login():
    """ログイン前の画面だけ見る。認証情報は不要。"""
    creds = _load_env()
    log("■ ログイン画面の構造を調べます（認証情報は使いません）")
    drv = _driver()
    try:
        drv.get(LOGIN_URL)
        time.sleep(3)
        _dump(drv, "01_login_page", creds)
        log("\n  この画面のinput欄が INET-ID の入力欄です。")
        log("  name/id を控えれば _ipat_login の第一段が書けます。")
        log("\n  20秒待ちます。画面を目で確認してください。")
        time.sleep(20)
    finally:
        drv.quit()


def probe_inside():
    """ログインして中の構造を見る。確定は絶対にしない。"""
    creds = _load_env()
    missing = [k for k in _CRED_KEYS if not creds.get(k)]
    if missing:
        log("⚠ 認証情報が足りません: " + ", ".join(missing))
        log("  .env に4つ入れてから実行してください。")
        log("  （このスクリプトは認証情報を画面にも出力にも残しません）")
        return

    from selenium.webdriver.common.by import By
    log("■ ログインして画面構造を調べます")
    log("  ⚠ 購入・確定は一切しません。押さないボタン: " + " / ".join(_FORBIDDEN))
    drv = _driver()
    try:
        drv.get(LOGIN_URL)
        time.sleep(3)
        _dump(drv, "01_login_page", creds)

        # ── 第1段: INET-ID ──
        # 実画面のname属性が不明なので、見えているtext/password欄を上から使う。
        # 構造が分かったら固定selectorに書き換える。
        boxes = [e for e in drv.find_elements(
            By.CSS_SELECTOR, "input[type=text],input[type=password]")
            if e.is_displayed()]
        if not boxes:
            log("  ⚠ 入力欄が見つかりません。ダンプを見てselectorを決めます。")
            time.sleep(15)
            return
        log(f"  入力欄 {len(boxes)}個を検出 → 1つ目に INET-ID を入れます")
        boxes[0].send_keys(creds["IPAT_INET_ID"])
        _dump(drv, "02_inetid_filled", creds)

        # 送信ボタンを探す。確定系の語を含むものは押さない。
        btns = [e for e in drv.find_elements(
            By.CSS_SELECTOR, "button,input[type=submit],a") if e.is_displayed()]
        cand = [e for e in btns
                if "ログイン" in ((e.text or "") + (e.get_attribute("value") or ""))
                and not any(w in ((e.text or "") + (e.get_attribute("value") or ""))
                            for w in _FORBIDDEN)]
        if not cand:
            log("  ⚠ ログインボタンを特定できません。ダンプを確認してください。")
            log("  30秒待ちます。手動でログインを進めてもらえれば、その先も記録します。")
            time.sleep(30)
            _dump(drv, "03_after_manual", creds)
            return
        cand[0].click()
        time.sleep(4)
        _dump(drv, "03_after_inetid", creds)

        # ── 第2段: 加入者番号 / 暗証番号 / P-ARS ──
        boxes = [e for e in drv.find_elements(
            By.CSS_SELECTOR, "input[type=text],input[type=password],"
                             "input[type=tel],input[type=number]")
            if e.is_displayed()]
        log(f"  第2段の入力欄 {len(boxes)}個")
        vals = [creds["IPAT_SUBSCRIBER"], creds["IPAT_PASSWORD"], creds["IPAT_PARS"]]
        for e, v in zip(boxes, vals):
            e.send_keys(v)
        _dump(drv, "04_second_filled", creds)

        log("\n  ここから先は手動で進めてください（ログインボタンを押す→通常投票へ）。")
        log("  60秒間、10秒ごとに画面構造を記録します。")
        log("  ⚠ 購入確定は押さないでください。買い目の入力画面まででいいです。")
        for i in range(6):
            time.sleep(10)
            _dump(drv, f"05_manual_{i+1}", creds)
    finally:
        try:
            drv.quit()
        except Exception:
            pass
    log("\n■ 完了。ipat_probe/ の .txt を確認してください。")
    log("  認証情報は伏字にしてあるので、そのまま共有できます。")


def main():
    a = sys.argv[1:]
    if a and a[0] == "login":
        probe_login()
    elif a and a[0] == "inside":
        probe_inside()
    else:
        log(__doc__)


if __name__ == "__main__":
    main()
