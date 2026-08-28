# -*- coding: utf-8 -*-
"""販売noteの見出し画像を作る（2026-08-29）

考え方
  競馬の予想画像は「馬・芝の緑・炎・的中！」に寄る。同じ土俵に乗ると、
  実績ゼロのこちらは必ず負ける。**同じ見た目にしないことが最大の差別化**になる。

  この商品が他と違うのは1点だけで、「当てる」ではなく「確率が正しい」。
  だから絵柄も、競馬場ではなく**計器**に寄せる。
  夜のオッズ表示板のような濃い地に、数字を大きく置く。

  文字は実データから入れる。**画像でだけ盛る、が起きないようにする。**

出力（note の見出し画像は 1280x670）
  note_thumb/A_数字.png    実測の数字で殴る。本命
  note_thumb/B_主張.png    「当たる予想は売っていません」
  note_thumb/C_較正表.png  記事の中に置く用。予測と実測を並べた表

実行
  python note_thumb.py
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "note_thumb")
W, H = 1280, 670

# 夜の表示板。芝の緑を使わないことで「よくある競馬画像」から外す
INK    = "#0E1319"   # 地
PANEL  = "#161D26"   # 一段明るい面
RULE   = "#2A3644"   # 罫
TEXT   = "#E6EAF0"
MUTED  = "#7C8899"
AMBER  = "#F0A63C"   # 表示板の数字。**強調はここだけに使う**
GREEN  = "#4FB477"   # 実測が予測に一致したことだけに使う

FONTS = "C:/Windows/Fonts/"


def f(name, size):
    return ImageFont.truetype(FONTS + name, size)


def log(m):
    print(m, flush=True)


def tw(d, t, ft):
    b = d.textbbox((0, 0), t, font=ft)
    return b[2] - b[0], b[3] - b[1]


def base():
    im = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(im)
    # 細い横罫を薄く敷く。馬柱の列組みを思わせるが、模様にはしない
    for y in range(0, H, 42):
        d.line([(0, y), (W, y)], fill="#131A22", width=1)
    # 左の縦バー。計器のマーカー
    d.rectangle([0, 0, 10, H], fill=AMBER)
    return im, d


def label(d, x, y, t, c=MUTED, s=26):
    d.text((x, y), t, font=f("BIZ-UDGothicB.ttc", s), fill=c)


def save(im, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    im.save(p, "PNG")
    log(f"  {os.path.relpath(p, BASE_DIR)}  {im.size[0]}x{im.size[1]}")


def art_a(d0):
    """数字で殴る。小さく表示されても『57 → 32』だけは読める大きさにする。"""
    im, d = base()
    b = d0["bands"][0]
    label(d, 64, 62, f"AIが出した「3着以内に入る確率」を検証した記録")
    label(d, 64, 104, f"{d0['last']}　{d0['races']}レース　{d0['horses']}頭", MUTED, 24)

    d.line([(64, 158), (W - 64, 158)], fill=RULE, width=2)

    label(d, 64, 186, f"「確率{b['th']}%以上」と出した馬", TEXT, 34)

    big = f("YuGothB.ttc", 168)
    d.text((60, 232), f"{b['n']}頭", font=big, fill=TEXT)
    w1, _ = tw(d, f"{b['n']}頭", big)

    arrow = f("YuGothB.ttc", 96)
    d.text((60 + w1 + 34, 288), "→", font=arrow, fill=MUTED)
    w2, _ = tw(d, "→", arrow)

    d.text((60 + w1 + w2 + 68, 232), f"{b['hit']}頭", font=big, fill=AMBER)

    label(d, 68, 432, "が実際に3着以内に入りました", TEXT, 36)

    # 右下に到達率。ここだけ緑を使う
    pct = f"{b['rate']:.0f}%"
    fp = f("YuGothB.ttc", 116)
    wp, hp = tw(d, pct, fp)
    d.rectangle([W - 64 - wp - 56, 470, W - 64, 470 + hp + 56], fill=PANEL)
    d.text((W - 64 - wp - 28, 494), pct, font=fp, fill=GREEN)

    label(d, 68, 528, "売っているのは「当たる予想」ではありません。", MUTED, 27)
    label(d, 68, 566, "確率が正しい表です。", MUTED, 27)
    save(im, "A_数字.png")


def art_b(d0):
    """主張型。数字より言葉が刺さる人向け。文字数を絞って大きくする。"""
    im, d = base()
    label(d, 64, 74, "競馬AI　検証の記録")

    big = f("YuGothB.ttc", 104)
    d.text((60, 148), "「当たる予想」は", font=big, fill=TEXT)
    d.text((60, 276), "売っていません。", font=big, fill=TEXT)

    d.line([(64, 424), (W - 64, 424)], fill=RULE, width=2)

    label(d, 64, 456, "売っているのは", MUTED, 30)
    mid = f("YuGothB.ttc", 62)
    d.text((60, 496), "確率が正しい表", font=mid, fill=AMBER)
    w, _ = tw(d, "確率が正しい表", mid)
    d.text((60 + w + 16, 512), "です。", font=f("YuGothB.ttc", 44), fill=TEXT)

    if d0["cal"]:
        label(d, 64, 596, f"補正に使っていない日で検証。最大のズレ {d0['cal_worst']:.1f}ポイント",
              MUTED, 25)
    save(im, "B_主張.png")


def art_c(d0):
    """較正表。記事の中に置く用。**これが商品の根拠そのもの**。"""
    if not d0["cal"]:
        log("  較正データが無いのでCは作りません")
        return
    im, d = base()
    label(d, 64, 56, "予測した確率は、実際にその割合で当たっているか")
    label(d, 64, 96, f"補正に使っていない日（{d0['cal_day']}）で検証", MUTED, 24)

    x0, y0 = 64, 156
    colw = [268, 150, 252, 300, 182]   # 合計1152＝左右64の余白を除いた幅ぴったり
    heads = ["人気", "頭数", "AIの予測", "実際に3着以内", "差"]
    fh = f("BIZ-UDGothicB.ttc", 27)
    x = x0
    for i, hd in enumerate(heads):
        d.text((x, y0), hd, font=fh, fill=MUTED)
        x += colw[i]
    d.line([(x0, y0 + 40), (x0 + sum(colw), y0 + 40)], fill=AMBER, width=2)

    fr = f("YuGothB.ttc", 30)
    y = y0 + 60
    for r in d0["cal"]:
        diff = r["act"] - r["pred"]
        vals = [r["band"], f"{r['n']}", f"{r['pred']:.1f}%",
                f"{r['act']:.1f}%", f"{diff:+.1f}"]
        x = x0
        for i, v in enumerate(vals):
            c = TEXT
            if i == 3:
                c = AMBER
            if i == 4:
                c = GREEN if abs(diff) < 5 else MUTED
            d.text((x, y), v, font=fr, fill=c)
            x += colw[i]
        y += 52
        d.line([(x0, y - 10), (x0 + sum(colw), y - 10)], fill="#1D2833", width=1)

    label(d, x0, y + 18,
          f"最大のズレは {d0['cal_worst']:.1f}ポイント。"
          "「40%と出た馬は、だいたい40%来る」状態です。", TEXT, 26)
    save(im, "C_較正表.png")


def main():
    import note_weekly
    d0 = note_weekly.gather()
    if d0 is None:
        log("  実績データが足りません。画像は作りません。")
        log("  （数字の裏付けが無い画像は出さない、という方針です）")
        return
    log("  使う実データ")
    if d0["bands"]:
        b = d0["bands"][0]
        log(f"    確率{b['th']}%以上 {b['n']}頭 → 的中{b['hit']}頭（{b['rate']:.1f}%）")
    log(f"    較正の最大ズレ {d0['cal_worst']:.1f}pt（{d0['cal_day']}で検証）")
    log("")
    art_a(d0)
    art_b(d0)
    art_c(d0)


if __name__ == "__main__":
    main()
