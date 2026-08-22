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
            k = k.strip()
            # ⚠ インラインコメントを落とす（2026-08-22）
            #   手順書に「IPAT_INET_ID=xxxxxxxx  # INET-ID（8桁）」と例示したため、
            #   コメントごとコピーされると値が25文字になり、そのまま入力欄に
            #   打ち込まれる。さらに伏字は「.envの値」を探すので実際のIDは
            #   伏字にならず、ダンプに平文で残った。実際にそれが起きた。
            val = val.split("#", 1)[0]
            val = val.strip().strip('"').strip("'").strip()
            if k in _CRED_KEYS and not v.get(k):
                v[k] = val
    return v


# 期待する書式。合わなければ実行前に止める（打ち込んでから気づくのを避ける）
_CRED_FORMAT = {
    "IPAT_INET_ID":    (8,  "INET-ID（英数8桁）"),
    "IPAT_SUBSCRIBER": (8,  "加入者番号（数字8桁）"),
    "IPAT_PASSWORD":   (4,  "暗証番号（数字4桁）"),
    "IPAT_PARS":       (4,  "P-ARS番号（数字4桁）"),
}


def _check_creds(creds):
    """書式を確認する。値そのものは絶対に表示しない。"""
    bad = []
    for k, (ln, desc) in _CRED_FORMAT.items():
        v = creds.get(k, "")
        if not v:
            bad.append(f"  × {k:<18}未設定  期待: {desc}")
        elif len(v) != ln:
            bad.append(f"  × {k:<18}{len(v)}文字  期待: {desc}"
                       + ("  ← # 以降のコメントが混ざっていませんか"
                          if len(v) > ln else ""))
    return bad


# 値を絶対に出力しないフィールド名。IPATの実画面から確認した。
#   inetid=INET-ID / i=加入者番号 / p=暗証番号 / r=P-ARS
#   uh,u,nm,mzj,fidf,mli,ckn,reqid=セッション識別子
_SECRET_FIELDS = {"inetid", "i", "p", "r", "uh", "u", "nm",
                  "mzj", "fidf", "mli", "ckn", "reqid"}


def _mask(text, creds):
    """出力から認証情報を消す。共有しても漏れないようにするため。

    ⚠ .envの値との一致だけに頼らない（2026-08-22）
      手動でIDを打ち直した場合、.envの値と画面の値が違う。実際にそれで
      INET-IDが平文でダンプに残った。値ではなくフィールド名で伏せる方が確実。
      _dump 側で value を伏せてから、こちらで念のため値でも消す二重がけ。
    """
    for k, val in creds.items():
        if val and len(val) >= 3:
            text = text.replace(val, f"<{k}>")
    return text


def _click_login(drv, form_name):
    """ログインボタンを押す。確定系の語を含む要素は絶対に押さない。

    IPATのログインボタンはテキストの無い <a href="...#"> （画像ボタン）で、
    onclickでJSのsubmitを呼ぶ。文字で探せないので、
      ① そのフォーム内の、hrefが '#' で終わる <a>
      ② 無ければ フォームのsubmit()
    の順に試す。②はJSのバリデーションを飛ばすので最後の手段。
    """
    from selenium.webdriver.common.by import By
    for e in drv.find_elements(By.CSS_SELECTOR, "a"):
        try:
            if not e.is_displayed():
                continue
            href = e.get_attribute("href") or ""
            label = (e.text or "") + (e.get_attribute("value") or "")
            if any(w in label for w in _FORBIDDEN):
                continue                      # 確定系は押さない
            if href.endswith("#"):
                e.click()
                log("  ログインボタン（画像リンク）を押しました")
                return True
        except Exception:
            continue
    try:
        drv.execute_script(
            f"document.forms['{form_name}'].submit();")
        log(f"  {form_name}.submit() で送信しました")
        return True
    except Exception:
        return False


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
                # 認証・セッション系は値を出さない。password型も一律伏せる。
                if a["name"] in _SECRET_FIELDS or a["type"] == "password":
                    if a["value"]:
                        a["value"] = "<伏字>"
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
    bad = _check_creds(creds)
    if bad:
        log("⚠ .env の認証情報が期待した書式ではありません:")
        for b in bad:
            log(b)
        log("\n  .env には値だけを書いてください（# のコメントは付けない）:")
        log("      IPAT_INET_ID=A1B2C3D4      ← このように値だけ")
        log("  値が違うまま実行すると、入力欄に誤った文字列が打ち込まれます。")
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
        #   2026-08-22の実画面で確認した構造:
        #     form name='FORM1' / input type=text name='inetid'
        #   ログインボタンは <button> でも input[type=submit] でもなく、
        #   テキストの無い <a href="...#"> （画像ボタン）。だから
        #   「"ログイン"という文字を探す」やり方では見つからなかった。
        try:
            box = drv.find_element(By.CSS_SELECTOR, "input[name='inetid']")
        except Exception:
            log("  ⚠ input[name=inetid] が見つかりません。画面構造が変わった可能性。")
            time.sleep(15)
            return
        box.clear()
        box.send_keys(creds["IPAT_INET_ID"])
        log("  INET-IDを入力（値は表示しません）")
        _dump(drv, "02_inetid_filled", creds)

        if not _click_login(drv, "FORM1"):
            log("  ⚠ ログインボタンを押せません。30秒待つので手動で進めてください。")
            time.sleep(30)
            _dump(drv, "03_after_manual", creds)
            return
        time.sleep(4)
        _dump(drv, "03_after_inetid", creds)

        # ── 第2段: 加入者番号 / 暗証番号 / P-ARS ──
        #   実画面で確認: pw_080_i.cgi の form name='FORM4' に
        #     input name='i' (加入者番号) / 'p' (暗証番号,password) / 'r' (P-ARS)
        try:
            for nm, key in (("i", "IPAT_SUBSCRIBER"),
                            ("p", "IPAT_PASSWORD"),
                            ("r", "IPAT_PARS")):
                e = drv.find_element(By.CSS_SELECTOR, f"input[name='{nm}']")
                e.clear()
                e.send_keys(creds[key])
            log("  加入者番号・暗証番号・P-ARSを入力（値は表示しません）")
        except Exception as ex:
            log(f"  ⚠ 第2段の入力欄が見つかりません: {type(ex).__name__}")
            time.sleep(20)
            _dump(drv, "04_second_failed", creds)
            return
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
