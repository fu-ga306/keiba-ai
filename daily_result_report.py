# -*- coding: utf-8 -*-
"""当日の結果報告をメール配信する。vote_log(自動投票の記録)の買い目を実払戻(netkeiba)と
突き合わせ、的中レース一覧＋全体の投資/払戻/回収率をまとめて送る。
18:00の照合タスク(analyze_accuracy)末尾から呼ばれる。単独実行も可: python daily_result_report.py [YYYY/MM/DD]
"""
import os
import sys
import time
import smtplib
import warnings
from datetime import datetime
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")
import pandas as pd
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
UNORDERED = {"馬連", "ワイド", "3連複", "枠連"}
JYO = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
       "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def _norm(kind, combo):
    p = str(combo).split("-")
    return "-".join(sorted(p) if kind in UNORDERED else p)


def build_report(date_str=None):
    """当日の的中レポート本文(文字列)と件名を返す。買い目が無ければ (None, None)。"""
    date_str = date_str or datetime.now().strftime("%Y/%m/%d")
    path = os.path.join(BASE, "vote_log.csv")
    if not os.path.exists(path):
        return None, None
    v = pd.read_csv(path, dtype=str)
    v = v[v["日時"].astype(str).str.startswith(date_str)]
    v = v[v["status"].isin(["投票予定", "投票"])]
    v["金額"] = pd.to_numeric(v["金額"], errors="coerce").fillna(0)
    # 同一レースが再予想で複数回記録される場合の重複を除去（最新の1回分だけ残す）
    v = v.sort_values("日時").drop_duplicates(["race_id", "券種", "組み合わせ"], keep="last")
    if v.empty:
        return None, None

    from payout_scraper import get_payout
    import collections
    races = sorted(v["race_id"].dropna().unique())
    hit_rows, tot_inv, tot_ret, tot_pts, tot_hits = [], 0, 0, 0, 0
    kind_stats = collections.defaultdict(lambda: [0, 0, 0, 0])  # 券種->[投資,払戻,点数,的中]
    for rid in races:
        try:
            # キーは必ず(券種, 正規化組み合わせ)。券種を含めないと単勝06と複勝06、
            # 馬単06-07とワイド06-07が衝突して誤的中になる。
            pay = {(p["券種"], _norm(p["券種"], p["組み合わせ"])): int(p["払戻金"]) for p in get_payout(rid)}
        except Exception:
            pay = {}
        time.sleep(0.3)
        sub = v[v["race_id"] == rid]
        hits = []
        for _, b in sub.iterrows():
            amt = int(b["金額"])
            pv = pay.get((b["券種"], _norm(b["券種"], b["組み合わせ"])), 0)
            ret_b = int(pv / 100 * amt) if pv > 0 else 0
            ks = kind_stats[b["券種"]]
            ks[0] += amt; ks[1] += ret_b; ks[2] += 1; ks[3] += (pv > 0)
            if pv > 0:
                hits.append((b["券種"], b["組み合わせ"], amt, ret_b))
        inv = int(sub["金額"].sum())
        ret = sum(h[3] for h in hits)
        tot_inv += inv; tot_ret += ret; tot_pts += len(sub); tot_hits += len(hits)
        if hits:
            hit_rows.append((JYO.get(rid[4:6], rid[4:6]), int(rid[10:12]), inv, ret, hits))

    roi = tot_ret / tot_inv * 100 if tot_inv else 0
    lines = [f"【競馬AI 結果報告】{date_str}", ""]
    lines.append(f"■ 全体: 投資{tot_inv:,}円 → 払戻{tot_ret:,}円  収支{tot_ret - tot_inv:+,}円  回収率{roi:.1f}%")
    lines.append(f"　購入{tot_pts}点 / 的中{tot_hits}点 (点数的中率{tot_hits / tot_pts * 100:.1f}%) / 的中レース{len(hit_rows)}")
    lines.append("")
    lines.append("── 券種別 収支 ──")
    for k in ["単勝", "複勝", "ワイド", "馬連", "馬単", "3連複", "3連単"]:
        if k in kind_stats:
            kinv, kret, kpts, khits = kind_stats[k]
            kroi = kret / kinv * 100 if kinv else 0
            lines.append(f"  {k:<4} 投資{kinv:,}→払戻{kret:,}  収支{kret - kinv:+,}  回収{kroi:.0f}%  ({khits}/{kpts}的中)")
    lines.append("")
    lines.append("── 的中レース（収支順）──")
    if not hit_rows:
        lines.append("　的中なし")
    for jyo, rno, inv, ret, hits in sorted(hit_rows, key=lambda x: -(x[3] - x[2])):
        lines.append(f"■ {jyo}{rno}R  投資{inv:,}→払戻{ret:,}円 (収支{ret - inv:+,})")
        for k, c, a, r in sorted(hits, key=lambda x: -x[3]):
            lines.append(f"    {k:<4} {c:<10} {a}円→{r:,}円")
    lines.append("")
    lines.append("※ vote_log(自動投票の記録)×netkeiba実払戻で照合。dryrun中は実際の投票はしていません。")
    subject = f"【競馬AI 結果】{date_str} 回収{roi:.0f}% 収支{tot_ret - tot_inv:+,}円 的中{len(hit_rows)}R"
    return "\n".join(lines), subject


def send(body, subject):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = os.environ["GMAIL_ADDRESS"]
    msg["To"] = os.environ["TO_ADDRESS"]
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASS"])
        s.send_message(msg)


def main(date_str=None):
    body, subject = build_report(date_str)
    if not body:
        print("結果報告: 当日の買い目記録なし → スキップ")
        return
    print(body)
    try:
        send(body, subject)
        print(f"\n結果報告をメール送信しました → {os.environ.get('TO_ADDRESS')}")
    except Exception as e:
        print(f"結果報告メール送信失敗: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
