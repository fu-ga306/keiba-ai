# -*- coding: utf-8 -*-
"""朝の一括予想のあと、買い推奨レースをnoteに貼りやすい形にしてメール送信する。

ユーザー要望(2026-08-05): 朝の一括予想時に買い推奨レースを一覧にして、
noteへ公開しやすい形でメール配信。公開自体は手動で行う。

出力の考え方:
  ・noteはMarkdownの見出し・表・箇条書きをそのまま貼れる。装飾は最小限にする
  ・買う理由（人気とモデル評価の食い違い）が読者に伝わる並びにする
  ・免責を必ず入れる。買い目を公開する＝馬券推奨に当たるため
    （開発紹介の段階では不要だったが、この配信からは必要）

使い方:
  python note_digest.py            … today_bets.csv から作ってメール送信
  python note_digest.py --print    … 送信せず内容を表示するだけ（確認用）
  python note_digest.py --save     … note_digest.md に保存もする
"""
import os
import sys
from datetime import datetime

import pandas as pd

BASE_DIR = os.environ.get("KEIBA_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
JYO = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
       6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}

DISCLAIMER = """
---

**ご確認ください**

- 本記事はAIによる予想の記録であり、的中や利益を保証するものではありません
- 掲載する買い目は筆者自身の検証記録です。購入の判断はご自身の責任でお願いします
- 過去の成績は将来の結果を約束しません。実際、検証期間5年のうち1年は負け越しています
- **掲載の買い目は朝の時点のものです。** オッズの変動で発走前に増減します
- 20歳未満の方は勝馬投票券を購入できません
"""


def _fmt_odds(v):
    try:
        return f"{float(v):.1f}倍"
    except (TypeError, ValueError):
        return "-"


def build_digest(bets: pd.DataFrame, preds: pd.DataFrame) -> str:
    """noteに貼れるMarkdownを組み立てる。"""
    today = datetime.now().strftime("%Y年%m月%d日")
    if bets.empty:
        return (f"# AI競馬予想 {today}\n\n"
                "朝の時点では、購入条件を満たすレースがありませんでした。\n\n"
                "条件を満たさない日は買わない、というのもこの手法の一部です。\n\n"
                "ただしオッズは発走直前まで動くので、"
                "直前に条件を満たすレースが出てくることはあります。\n"
                + DISCLAIMER)

    lines = [f"# AI競馬予想 {today}", ""]
    races = bets["race_id"].drop_duplicates().tolist()
    lines += [f"朝の時点での推奨は **{len(races)}レース** です。", "",
              "市場の評価とAIの評価が食い違っている馬を軸に選んでいます。",
              "人気薄でもAIが高く評価し、かつオッズに妙味がある場合だけ買います。", "",
              # 2026-08-10: 朝の一覧と最終の買い目が食い違う件を明記する。
              #   8/9は朝5レース→最終2レースまで減った。オッズが動いて期待値が
              #   条件を下回るため。書いておかないと「消えた」と受け取られる。
              "## ⚠️ この一覧は発走前に変わります", "",
              "買うかどうかは **期待値（AIの勝率 × 単勝オッズ）** で決めています。",
              "オッズは発走直前まで動くので、**朝は条件を満たしていた馬が、"
              "直前には満たさなくなる**ことがよくあります。"
              "逆に、朝は対象外だったレースが加わることもあります。", "",
              "実際、2026年8月9日は **朝5レース → 最終2レース** まで減りました。", "",
              "オッズは下がる方向に動きやすく、特に**走る馬ほど直前に買われます**。"
              "下がれば期待値も同じだけ下がり、条件から外れます。", "",
              "**最終的な買い目は各レースの発走7分前に確定します。**"
              "この一覧は「朝の時点での候補」としてお読みください。", ""]

    for rid in races:
        b = bets[bets["race_id"] == rid]
        p = preds[preds["race_id"] == rid] if preds is not None else pd.DataFrame()
        try:
            jyo = JYO.get(int(str(rid)[4:6]), str(rid)[4:6])
            rno = int(str(rid)[10:12])
        except (ValueError, IndexError):
            jyo, rno = "?", "?"

        # 軸は単勝の馬番。単勝が無い場合（旧仕様の買い目など）は、
        # 最初の買い目の先頭馬番を軸とみなす。券種構成が変わっても壊れないように。
        tan = b[b["券種"] == "単勝"]
        if len(tan):
            axis_no = str(tan["組み合わせ"].iloc[0])
        else:
            first = str(b["組み合わせ"].iloc[0]) if len(b) else ""
            axis_no = first.split("-")[0] if first else ""
        row = p[p["馬番"].astype(str) == axis_no] if len(p) else pd.DataFrame()

        lines.append(f"## {jyo}{rno}R")
        if len(row):
            r = row.iloc[0]
            name = str(r.get("馬名", ""))
            pop = r.get("人気", "")
            od = _fmt_odds(r.get("単勝オッズ"))
            gap = pd.to_numeric(r.get("乖離"), errors="coerce")
            mr = pd.to_numeric(r.get("MF複勝順位"), errors="coerce")
            lines.append(f"**軸: {axis_no}番 {name}**（{int(pop) if pd.notna(pop) else '-'}番人気 / {od}）")
            lines.append("")
            if pd.notna(mr) and pd.notna(gap):
                lines.append(f"- 市場の評価: {int(pop) if pd.notna(pop) else '-'}番人気")
                lines.append(f"- AIの評価: {int(mr)}番手")
                lines.append(f"- 評価の差: **{gap:+.0f}**（AIのほうが高く見ている）")
            lines.append("")
        else:
            lines.append(f"**軸: {axis_no}番**")
            lines.append("")

        lines.append("| 券種 | 買い目 |")
        lines.append("|---|---|")
        for kind, g in b.groupby("券種", sort=False):
            combos = "、".join(str(c) for c in g["組み合わせ"])
            lines.append(f"| {kind} | {combos} |")
        lines.append("")

    lines += ["## この予想について", "",
              "市場（オッズ）が見落としている馬をAIで探し、期待値が一定以上の場合だけ買う手法です。",
              "オッズを見ずに能力だけで評価するモデルを作り、その評価と人気の差を狙います。", "",
              "過去5年の検証では通算113.7%でしたが、**月単位では半分が負け**です。",
              "当たるのは12回に1回ほどで、当たったときの配当で取り返す形になります。", ""]
    return "\n".join(lines) + DISCLAIMER


def main():
    bets_p = os.path.join(BASE_DIR, "today_bets.csv")
    pred_p = os.path.join(BASE_DIR, "today_predictions.csv")
    if not os.path.exists(bets_p):
        print("today_bets.csv がありません（朝の一括予想の後に実行してください）")
        bets = pd.DataFrame(columns=["race_id", "券種", "組み合わせ"])
    else:
        bets = pd.read_csv(bets_p, dtype={"race_id": str, "組み合わせ": str})
        # 当日の買い目だけを対象にする（前日分が残っていることがある）
        if "予想日時" in bets.columns:
            today = datetime.now().strftime("%Y/%m/%d")
            same = bets["予想日時"].astype(str).str[:10] == today
            if same.any():
                bets = bets[same]
            else:
                print(f"  today_bets.csv は当日のものではありません"
                      f"（{bets['予想日時'].astype(str).str[:10].max()}）→ 空として扱います")
                bets = bets.iloc[0:0]
    preds = (pd.read_csv(pred_p, dtype={"race_id": str})
             if os.path.exists(pred_p) else None)

    text = build_digest(bets, preds)

    if "--print" in sys.argv or "--save" in sys.argv:
        print(text)
    if "--save" in sys.argv:
        out = os.path.join(BASE_DIR, "note_digest.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n保存 → {out}")
    if "--print" not in sys.argv:
        try:
            from keiba_predict import send_email
            n = bets["race_id"].nunique() if not bets.empty else 0
            subj = f"【note用】{datetime.now():%m/%d} 朝の推奨 {n}レース（発走前に変動）"
            send_email(subj, text)
            print(f"メール送信: {subj}")
        except Exception as e:
            print(f"メール送信に失敗: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
