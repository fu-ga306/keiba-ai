"""
keiba_predict.py
────────────────
指定レースの全馬詳細予想を出力 + メール送信するスクリプト。
keiba_auto.py と同じ特徴量・モデルを使用。

使い方:
    python keiba_predict.py 202606050811
    python keiba_predict.py  # → TARGET_RACE_ID を直接編集して実行
"""

import os
import sys
import glob
import pickle
import smtplib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore")

# ── ★ここを変更するだけ★ ─────────────────────────────────────────────────
TARGET_RACE_ID = "202606050811"   # 予想したいレースID（12桁）
# ────────────────────────────────────────────────────────────────────────

# ── メール設定（環境変数 or .env から取得） ──────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASS = os.environ.get("GMAIL_APP_PASS", "")
TO_ADDRESS    = os.environ.get("TO_ADDRESS", GMAIL_ADDRESS)
BASE_DIR      = os.environ.get("KEIBA_BASE_DIR",
                    os.path.dirname(os.path.abspath(__file__)))

JYO_NAMES = {
    1:"札幌", 2:"函館", 3:"福島", 4:"新潟",  5:"東京",
    6:"中山", 7:"中京", 8:"京都", 9:"阪神", 10:"小倉",
}


# ── ハーヴィルモデルで複勝・3着内確率を計算 ─────────────────────────────
def calc_place_probs_harvill(win_probs: np.ndarray):
    n = len(win_probs)
    place2 = np.zeros(n)
    place3 = np.zeros(n)

    for i in range(n):
        pi = win_probs[i]
        place2[i] += pi
        place3[i] += pi
        for j in range(n):
            if j == i:
                continue
            pj = win_probs[j]
            s_ij = 1.0 - pj
            if s_ij <= 0:
                continue
            p2 = pj * (pi / s_ij)
            place2[i] += p2
            place3[i] += p2
            for k in range(n):
                if k == i or k == j:
                    continue
                pk = win_probs[k]
                s_ijk = s_ij - pk
                if s_ijk <= 0:
                    continue
                place3[i] += pj * (pk / s_ij) * (pi / s_ijk)

    return place2, place3


def kelly_fraction(win_prob: float, odds: float, fraction: float = 0.25) -> float:
    if odds <= 1 or win_prob <= 0:
        return 0.0
    b = odds - 1
    q = 1.0 - win_prob
    k = (b * win_prob - q) / b
    return max(0.0, k * fraction)


# ── 表示ユーティリティ ────────────────────────────────────────────────────
W = 76  # 罫線幅

def sep(char="═"):
    return char * W

def header(text, char="═"):
    lines = []
    lines.append(sep(char))
    lines.append(f"  {text}")
    lines.append(sep(char))
    return "\n".join(lines)


# ── メール送信 ────────────────────────────────────────────────────────────
def send_email(subject: str, body: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        print("  ⚠ メール設定未完了（環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS を設定）")
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
        print(f"  ✅ メール送信完了 → {TO_ADDRESS}")
    except Exception as e:
        print(f"  ❌ メール送信エラー: {e}")


# ── レポート本文生成 ──────────────────────────────────────────────────────
def build_report(pdf, race_id, jyo_name, race_no,
                 dist, turf, baba, cls, n_horse):
    lines = []

    def _ev(v):
        return f"{v:+.2f}" if pd.notna(v) else "-"
    def _odds(v):
        return f"{v:.1f}倍" if pd.notna(v) else "未確定"

    # ヘッダー
    from datetime import datetime
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    lines.append(sep())
    lines.append(f"  🏇 競馬AI 詳細予想レポート  {now}")
    lines.append(f"  レースID : {race_id}   {jyo_name} {race_no}R")
    lines.append(sep())
    lines.append("")

    # レース概要
    lines.append(header("📋 レース概要", "─"))
    lines.append(f"  コース   : {turf} {dist}m")
    lines.append(f"  馬場状態 : {baba}")
    lines.append(f"  クラス   : {cls}")
    lines.append(f"  出走頭数 : {n_horse}頭")
    lines.append("")

    # ── 全馬詳細一覧 ──────────────────────────────────────────────────────
    lines.append(header("🐴 全馬詳細評価（予測順位順）", "─"))
    lines.append("")

    # 列ヘッダー
    col = (
        f"{'順':>2} {'馬番':>2} {'馬名':<13} "
        f"{'単勝':>6} {'人':>2}人 "
        f"{'勝率':>5} {'連対':>5} {'複勝':>5} {'3着内':>5} "
        f"{'期待値':>6} {'Kelly':>5} "
        f"{'過去勝率':>6} {'走':>3} "
        f"{'間隔':>5} {'乖離':>4}  戦略"
    )
    lines.append(col)
    lines.append("─" * W)

    pdf_sorted = pdf.sort_values("予測順位")
    for _, row in pdf_sorted.iterrows():
        odds     = row.get("単勝オッズ", np.nan)
        pop      = row.get("人気", np.nan)
        win_p    = row["勝ち確率"]
        place2_p = row["連対確率"]
        place_p  = row["複勝確率"]
        place3_p = row["3着内確率"]
        ev       = row["単勝期待値"]
        kelly    = row["推奨賭け率"]
        past_wr  = row.get("過去勝率", np.nan)
        runs     = int(row.get("過去出走数", 0)) if pd.notna(row.get("過去出走数")) else 0
        interval = row.get("前走間隔", np.nan)
        strategy = row.get("該当戦略", "")

        odds_s     = f"{odds:.1f}倍" if pd.notna(odds) else "  -  "
        pop_s      = f"{int(pop)}"   if pd.notna(pop)  else " -"
        interval_s = f"{interval:.0f}W" if pd.notna(interval) else "初戦"
        past_wr_s  = f"{past_wr*100:.0f}%" if pd.notna(past_wr) else "  -"
        kelly_s    = f"{kelly*100:.1f}%" if kelly > 0 else "  - "
        ev_s       = f"{ev:+.2f}" if pd.notna(ev) else "  - "
        mark       = "◀" if row["予測順位"] == 1 else "  "

        gap = row.get("乖離スコア", np.nan)
        gap_s = f"{gap:+.0f}" if pd.notna(gap) else "  -"
        line = (
            f"{int(row['予測順位']):>2} {int(row['馬番']):>2} {str(row['馬名']):<13} "
            f"{odds_s:>6} {pop_s:>2}人 "
            f"{win_p*100:>4.1f}% {place2_p*100:>4.1f}% {place_p*100:>4.1f}% {place3_p*100:>4.1f}% "
            f"{ev_s:>6} {kelly_s:>5} "
            f"{past_wr_s:>5} {runs:>3}走 "
            f"{interval_s:>4} {gap_s:>4}  {strategy} {mark}"
        )
        lines.append(line)

    lines.append("─" * W)
    lines.append("  ※ 連対=2着以内確率  複勝=3着以内確率  3着内=ハーヴィル近似")
    lines.append("")

    # ── 推奨馬（2軸） ────────────────────────────────────────────────────
    def rec_block(title, icon, df_top, note):
        block = []
        block.append(header(f"{icon} {title}", "─"))
        block.append(f"  {note}")
        block.append("")
        marks = ["◎", "○", "▲", "△", "×"]
        for i, (_, row) in enumerate(df_top.head(5).iterrows()):
            mk       = marks[i]
            odds     = row.get("単勝オッズ", np.nan)
            pop      = row.get("人気", np.nan)
            win_p    = row["勝ち確率"]
            place2_p = row["連対確率"]
            place_p  = row["複勝確率"]
            place3_p = row["3着内確率"]
            ev       = row["単勝期待値"]
            kelly    = row["推奨賭け率"]
            past_wr  = row.get("過去勝率", np.nan)
            runs     = int(row.get("過去出走数", 0)) if pd.notna(row.get("過去出走数")) else 0
            weight   = row.get("馬体重", np.nan)
            weight_d = row.get("体重増減", np.nan)
            interval = row.get("前走間隔", np.nan)
            strategy = row.get("該当戦略", "")

            odds_s     = f"{odds:.1f}倍"              if pd.notna(odds)                          else "未確定"
            pop_s      = f"{int(pop)}番人気"           if pd.notna(pop)                           else "未確定"
            weight_s   = f"{int(weight)}kg({weight_d:+.0f})" if pd.notna(weight) and pd.notna(weight_d) else "当日発表"
            interval_s = f"{interval:.1f}週"           if pd.notna(interval)                      else "初戦"
            past_wr_s  = f"{past_wr*100:.0f}%({runs}走)" if pd.notna(past_wr) and runs > 0        else "データなし"
            kelly_s    = f"{kelly*100:.1f}%"           if kelly > 0                               else "-"

            block.append(f"  {mk} 馬番{int(row['馬番'])}番 【{row['馬名']}】")
            block.append(f"     オッズ    : {odds_s}  {pop_s}")
            block.append(f"     馬体重    : {weight_s}  前走間隔: {interval_s}")
            block.append(f"     勝率{win_p*100:.1f}%  連対率{place2_p*100:.1f}%  複勝率{place_p*100:.1f}%  3着内{place3_p*100:.1f}%")
            block.append(f"     単勝期待値: {_ev(ev)}  推奨賭け率(1/4Kelly): {kelly_s}")
            block.append(f"     過去勝率  : {past_wr_s}")
            if strategy:
                block.append(f"     🔥 バックテスト戦略該当: {strategy}")
            block.append("")
        return block

    top5_ai = pdf.sort_values("勝ち確率", ascending=False).head(5)
    top5_ev = pdf[pdf["勝ち確率"] >= 0.03].sort_values("単勝期待値", ascending=False).head(5)

    lines += rec_block(
        "純粋AI予想 TOP5", "🤖", top5_ai,
        "モデルが算出した勝ち確率のみで選出。オッズ・人気は一切考慮しない。"
    )
    lines += rec_block(
        "期待値ベース推奨 TOP5", "💰", top5_ev,
        "期待値（勝ち確率×オッズ-1）が高い順。馬券的妙味を重視した選出。"
    )

    # ── 2軸比較サマリー ──────────────────────────────────────────────────
    lines.append(header("🔍 2軸比較サマリー", "─"))
    ai_names = top5_ai["馬名"].tolist()
    ev_names = top5_ev["馬名"].tolist()
    both    = [h for h in ai_names if h in ev_names]
    ai_only = [h for h in ai_names if h not in ev_names]
    ev_only = [h for h in ev_names if h not in ai_names]

    lines.append("  【両方に選ばれた馬】← 最注目")
    if both:
        for h in both:
            row = pdf[pdf["馬名"] == h].iloc[0]
            st  = f"  🔥{row['該当戦略']}" if row.get("該当戦略") else ""
            lines.append(
                f"    ✅ {h}  勝率{row['勝ち確率']*100:.1f}%  "
                f"連対率{row['連対確率']*100:.1f}%  複勝率{row['複勝確率']*100:.1f}%  "
                f"期待値{row['単勝期待値']:+.2f}  {row['単勝オッズ']:.1f}倍{st}"
            )
    else:
        lines.append("    （なし）")

    lines.append("")
    lines.append("  【純粋AI予想のみ】← モデル高評価だがオッズが低い可能性")
    if ai_only:
        for h in ai_only:
            row = pdf[pdf["馬名"] == h].iloc[0]
            lines.append(
                f"    🤖 {h}  勝率{row['勝ち確率']*100:.1f}%  "
                f"期待値{_ev(row['単勝期待値'])}  {_odds(row['単勝オッズ'])}"
            )
    else:
        lines.append("    （なし）")

    lines.append("")
    lines.append("  【期待値ベースのみ】← 穴馬候補・妙味あり")
    if ev_only:
        for h in ev_only:
            row = pdf[pdf["馬名"] == h].iloc[0]
            lines.append(
                f"    💰 {h}  勝率{row['勝ち確率']*100:.1f}%  "
                f"期待値{_ev(row['単勝期待値'])}  {_odds(row['単勝オッズ'])}"
            )
    else:
        lines.append("    （なし）")

    lines.append("")

    # ── 最終予想（2軸を統合した結論） ──────────────────────────────────
    lines.append(header("🏆 最終予想（総合判定）", "═"))
    lines.append("  【判定ロジック】")
    lines.append("  ① 両方選出 × 戦略該当  → 最強推奨（◎）")
    lines.append("  ② AI予測順位が上位（期待値はマイナスでも可）")
    lines.append("  ③ 該当戦略があれば加点して優先度UP")
    lines.append("  ※ 期待値・オッズは参考情報として表示（印の決定には使用しません）")
    lines.append("")

    # スコアリング（100点満点）
    # AI予測順位（勝ち確率）80点 + 戦略該当ボーナス20点
    # 期待値は印の決定には使わず、参考情報として表示するのみ
    pdf["_ai_rank"]  = pdf["勝ち確率"].rank(ascending=False)
    n = len(pdf)
    pdf["_score"] = (
        (1 - (pdf["_ai_rank"] - 1) / n) * 80 +
        pdf["該当戦略"].apply(lambda s: 20 if s else 0)
    )
    final_top = pdf.sort_values("_score", ascending=False).head(5)

    final_marks  = ["◎", "○", "▲", "△", "×"]
    final_labels = ["最強推奨", "強く推奨", "推奨", "穴候補", "注目"]

    lines.append("  ┌─────────────────────────────────────────────────────────┐")
    for i, (_, row) in enumerate(final_top.iterrows()):
        if i >= 5:
            break
        mk    = final_marks[i]
        lbl   = final_labels[i]
        odds  = row.get("単勝オッズ", np.nan)
        pop   = row.get("人気", np.nan)
        wp    = row["勝ち確率"]
        ev    = row["単勝期待値"]
        score = row["_score"]
        strat = row.get("該当戦略", "")
        in_ai = row["馬名"] in ai_names
        in_ev = row["馬名"] in ev_names

        tag = []
        if in_ai and in_ev:
            tag.append("両軸◎")
        elif in_ai:
            tag.append("AI予想")
        elif in_ev:
            tag.append("期待値")
        if strat:
            tag.append(strat)

        odds_s = f"{odds:.1f}倍" if pd.notna(odds) else "未確定"
        pop_s  = f"{int(pop)}人気" if pd.notna(pop) else "-"
        tag_s  = " / ".join(tag)

        lines.append(
            f"  │ {mk}【{lbl}】 馬番{int(row['馬番'])}番 {str(row['馬名']):<12}"
            f"  {odds_s} {pop_s}"
            f"  勝率{wp*100:.1f}%  EV{_ev(ev)}"
            f"  総合{score:.0f}点"
        )
        if tag_s:
            lines.append(f"  │           {tag_s}")
        if i < 4:
            lines.append("  ├─────────────────────────────────────────────────────────┤")
    lines.append("  └─────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── 券種推奨（3モデルによる役割判定） ────────────────────────────
    if "券種推奨" in pdf.columns and (pdf["券種推奨"] != "").any():
        lines.append(header("🎯 券種推奨（勝ち・連対・複勝の3モデル判定）", "─"))
        lines.append("  軸◎=勝てる本命  軸(人気)=実力上位だが妙味薄  相手○=連対候補  穴▲=複勝妙味")
        lines.append("")
        role_order = {"軸◎": 0, "軸(人気)": 1, "相手○": 2, "穴▲": 3}
        rec_df = pdf[pdf["券種推奨"] != ""].copy()
        rec_df["_ord"] = rec_df["券種推奨"].map(role_order).fillna(9)
        rec_df = rec_df.sort_values("_ord")
        for _, row in rec_df.iterrows():
            role  = row["券種推奨"]
            odds  = row.get("単勝オッズ", np.nan)
            pop   = row.get("人気", np.nan)
            wp    = row["勝ち確率"]
            p2    = row["連対確率"]
            p3    = row["複勝確率"]
            ev    = row["単勝期待値"]
            odds_s = f"{odds:.1f}倍" if pd.notna(odds) else "未確定"
            pop_s  = f"{int(pop)}人気" if pd.notna(pop) else "-"
            ev_s   = f"{ev:+.2f}" if pd.notna(ev) else "-"
            lines.append(
                f"  {role:<7} 馬番{int(row['馬番']):>2} {str(row['馬名']):<12} "
                f"{odds_s:>7} {pop_s:>5}"
            )
            lines.append(
                f"           勝率{wp*100:4.1f}%  連対率{p2*100:4.1f}%  "
                f"複勝率{p3*100:4.1f}%  EV{ev_s}"
            )
        lines.append("")
        lines.append("  ※ 連対率・複勝率は専用モデルが各馬独立に予想した値です。")
        lines.append("")

    # ── 買い目サマリー（複勝・馬連・ワイド・馬単・3連複） ──────────────
    lines.append("  【買い目サマリー】")
    honmei = final_top.iloc[0]
    taiko  = final_top.iloc[1] if len(final_top) > 1 else None
    ana    = final_top.iloc[2] if len(final_top) > 2 else None

    h_odds = honmei.get("単勝オッズ", np.nan)
    lines.append(
        (f"  単勝   : ◎{honmei['馬名']}  "
         f"（{h_odds:.1f}倍 / EV{_ev(honmei['単勝期待値'])}）")
        if pd.notna(h_odds) else f"  単勝   : ◎{honmei['馬名']}"
    )

    # 複勝：◎の複勝確率×想定複勝オッズ(単勝オッズの目安1/3〜1/4)で期待値推定
    h_place_p = honmei.get("複勝確率", np.nan)
    if pd.notna(h_place_p) and pd.notna(h_odds):
        est_fuku_odds = max(h_odds / 4, 1.05)  # 簡易推定（実オッズはkeiba_auto側で実測）
        fuku_ev = h_place_p * est_fuku_odds - 1
        lines.append(
            f"  複勝   : ◎{honmei['馬名']}  "
            f"（複勝率{h_place_p*100:.1f}% / 推定オッズ約{est_fuku_odds:.1f}倍 / 推定EV{fuku_ev:+.2f}）"
        )

    if taiko is not None:
        h_p2 = honmei.get("連対確率", np.nan)
        t_p1 = taiko.get("勝ち確率", np.nan)
        t_p2 = taiko.get("連対確率", np.nan)
        h_p1 = honmei.get("勝ち確率", np.nan)

        # ワイド的中確率（◎○ともに3着以内）の近似 = 複勝確率の積に補正係数
        h_pl = honmei.get("複勝確率", np.nan)
        t_pl = taiko.get("複勝確率", np.nan)
        if pd.notna(h_pl) and pd.notna(t_pl):
            wide_p = min(h_pl * t_pl * 1.5, 0.95)
            lines.append(
                f"  ワイド : ◎{honmei['馬名']} ─ ○{taiko['馬名']}  "
                f"（的中率約{wide_p*100:.1f}%）"
            )

        # 馬連的中確率 = P(◎1着,○2着) + P(○1着,◎2着) の近似
        if pd.notna(h_p1) and pd.notna(t_p1) and pd.notna(h_p2) and pd.notna(t_p2):
            umaren_p = h_p1 * (t_p2 / max(1 - h_p1, 1e-6)) + t_p1 * (h_p2 / max(1 - t_p1, 1e-6))
            umaren_p = min(umaren_p, 0.95)
            lines.append(
                f"  馬連   : ◎{honmei['馬名']} ─ ○{taiko['馬名']}  "
                f"（的中率約{umaren_p*100:.1f}%）"
            )

        # 馬単的中確率 = P(◎1着 → ○2着) のみ（順序固定）
        if pd.notna(h_p1) and pd.notna(t_p2):
            umatan_p = h_p1 * (t_p2 / max(1 - h_p1, 1e-6))
            umatan_p = min(umatan_p, 0.95)
            lines.append(
                f"  馬単   : ◎{honmei['馬名']} → ○{taiko['馬名']}  "
                f"（的中率約{umatan_p*100:.1f}%）"
            )

    if ana is not None:
        lines.append(
            f"  3連複  : ◎{honmei['馬名']} ─ ○{taiko['馬名']} ─ ▲{ana['馬名']}"
        )
    lines.append("  ※ ワイド・馬連・馬単の的中率は確率モデルによる近似値です。")
    lines.append("  ※ 複勝オッズは実オッズと異なる場合があります（keiba_auto.pyは実オッズ使用）。")
    lines.append("")

    lines.append(sep("─"))
    lines.append("  ※ AIによる予測です。投資は自己責任でお願いします。")
    lines.append("  ※ 推奨賭け率は1/4ケリー基準（保守的設定）です。")
    lines.append("  ※ 複勝・3着内確率はハーヴィルモデルによる近似値です。")
    lines.append(sep())

    return "\n".join(lines)


# ── メイン ────────────────────────────────────────────────────────────────
def predict_race(race_id: str):
    race_id = str(race_id).strip()
    if len(race_id) != 12 or not race_id.isdigit():
        print("エラー: race_id は12桁の数字で指定してください（例: 202606050811）")
        sys.exit(1)

    jyo_cd   = int(race_id[4:6])
    race_no  = int(race_id[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, f"競馬場{jyo_cd}")

    # モデル読み込み
    model_path = os.path.join(BASE_DIR, "model.pkl")
    print(f"モデル読み込み中...")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    models   = saved["models"]      # 旧互換（=winモデル）
    use_cols = saved["use_cols"]

    # 3モデル（win/place2/place3）を取得。新形式ならそれぞれ別モデル、
    # 旧形式なら place2/place3 は None（ハーヴィル変換にフォールバック）
    win_models    = saved.get("win", {}).get("models", models)
    win_cols      = saved.get("win", {}).get("use_cols", use_cols)
    place2_models = saved.get("place2", {}).get("models")
    place2_cols   = saved.get("place2", {}).get("use_cols")
    place3_models = saved.get("place3", {}).get("models")
    place3_cols   = saved.get("place3", {}).get("use_cols")
    is_multi = saved.get("format") == "multi_v1" and place2_models and place3_models
    if is_multi:
        print("  3モデル構成(win/place2/place3)を検出 → 独立予想を使用")
    else:
        print("  旧モデル構成 → 連対率・複勝率はハーヴィル変換を使用")

    # 市場フリーモデル（存在する場合のみ）
    mf_models = None
    mf_cols   = None
    mf_path   = os.path.join(BASE_DIR, "model_mf.pkl")
    if os.path.exists(mf_path):
        try:
            with open(mf_path, "rb") as f:
                mf_saved = pickle.load(f)
            mf_models = mf_saved["models"]
            mf_cols   = mf_saved["use_cols"]
            print("  市場フリーモデル読み込み完了")
        except Exception as e:
            print(f"  市場フリーモデルスキップ: {e}")

    # 履歴データ読み込み
    hist_path = os.path.join(BASE_DIR, "race_features.csv")
    print(f"履歴データ読み込み中...")
    history_df = pd.read_csv(hist_path, low_memory=False)
    print(f"  読み込み完了: {len(history_df)}行")

    # 出馬表・オッズ取得
    from keiba_auto import get_race_data, build_features
    print("出馬表・オッズ取得中...")
    race_df = get_race_data(race_id)
    if race_df is None:
        print("エラー: 出馬表を取得できませんでした")
        sys.exit(1)
    print(f"  {len(race_df)}頭分取得完了")

    # 特徴量構築
    print("特徴量構築中...")
    pdf = build_features(race_df, history_df)

    # 予測
    print("予測中...")
    X     = pdf.reindex(columns=win_cols)
    preds = np.mean([m.predict_proba(X)[:, 1] for m in win_models], axis=0)
    pdf["予測スコア"] = preds
    pdf["予測順位"]   = pdf["予測スコア"].rank(ascending=False).astype(int)

    # 案B: 勝ち確率は「生確率」をそのまま使う（正規化しない）。
    # winモデルが出す「この馬が1着になる絶対確率」。
    win_probs = np.clip(np.nan_to_num(preds, nan=0.0), 0, 1)
    pdf["勝ち確率"] = win_probs

    if is_multi:
        # ── 連対率・複勝率を独立モデルで予想（生確率） ──
        X_p2 = pdf.reindex(columns=place2_cols)
        X_p3 = pdf.reindex(columns=place3_cols)
        p2_raw = np.mean([m.predict_proba(X_p2)[:, 1] for m in place2_models], axis=0)
        p3_raw = np.mean([m.predict_proba(X_p3)[:, 1] for m in place3_models], axis=0)
        p2_raw = np.clip(np.nan_to_num(p2_raw, nan=0.0), 0, 1)
        p3_raw = np.clip(np.nan_to_num(p3_raw, nan=0.0), 0, 1)
        # 案B整合性: 勝率 ≤ 連対率 ≤ 複勝率（包含関係）を生確率で保証する
        place2 = np.maximum(p2_raw, win_probs)
        place3 = np.clip(np.maximum(p3_raw, place2), 0, 1)
        pdf["連対確率"]  = place2
        pdf["複勝確率"]  = place3
        pdf["3着内確率"] = place3
        # 独立モデルによる連対・複勝の順位（券種推奨に使う）
        pdf["連対順位"] = pd.Series(place2, index=pdf.index).rank(ascending=False)
        pdf["複勝順位"] = pd.Series(place3, index=pdf.index).rank(ascending=False)
        print("  連対率・複勝率を独立モデルで予想完了（生確率・包含関係保証）")
    else:
        # ── 旧来のハーヴィル変換（生の勝率ベース） ──
        place2, place3 = calc_place_probs_harvill(win_probs)
        place2 = np.maximum(place2, win_probs)
        place3 = np.maximum(place3, place2)
        pdf["連対確率"]  = place2
        pdf["複勝確率"]  = place3
        pdf["3着内確率"] = place3
        pdf["連対順位"] = pd.Series(place2, index=pdf.index).rank(ascending=False)
        pdf["複勝順位"] = pd.Series(place3, index=pdf.index).rank(ascending=False)

    pdf["単勝期待値"] = pdf["勝ち確率"] * pdf["単勝オッズ"] - 1
    pdf["推奨賭け率"] = pdf.apply(
        lambda r: kelly_fraction(r["勝ち確率"], r["単勝オッズ"]), axis=1
    )

    # 市場フリーモデルで乖離スコアを計算
    pdf["MF予測順位"] = np.nan
    pdf["乖離スコア"] = np.nan
    pdf["MF勝ち確率"] = np.nan
    if mf_models is not None and mf_cols is not None:
        try:
            X_mf  = pdf.reindex(columns=mf_cols)
            mf_preds = np.mean([m.predict_proba(X_mf)[:, 1] for m in mf_models], axis=0)
            pdf["MF予測順位"] = pd.Series(mf_preds).rank(ascending=False).values
            pdf["乖離スコア"] = pdf["予測順位"] - pdf["MF予測順位"]
            # 純粋AI勝ち確率（レース内で正規化）
            mf_raw = np.clip(np.nan_to_num(mf_preds, nan=0.0), 0, None)
            pdf["MF勝ち確率"] = mf_raw / mf_raw.sum() if mf_raw.sum() > 0 else np.ones(len(mf_raw)) / len(mf_raw)
            print("  市場フリー予測成功")
        except Exception as e:
            print(f"  市場フリー予測エラー（スキップ）: {e}")

    def check_strategy(row):
        s = []
        if row["予測順位"] == 1 and row["単勝期待値"] >= 0.3 and 1.5 <= row["単勝オッズ"] <= 20:
            s.append("戦略A")
            if row["人気"] != 1:
                s.append("戦略A-2")
        if row["人気"] >= 3 and row["予測順位"] == 1 and row["単勝期待値"] >= 0.3:
            s.append("戦略C")
        if pd.notna(row.get("前走間隔")) and 2 <= row["前走間隔"] <= 4 \
                and row["予測順位"] == 1 and row["単勝期待値"] >= 0.2:
            s.append("戦略D")
        # 戦略E：一時無効化（交互作用特徴量追加後、乖離スコアの有効性が崩壊したため）
        # if (pd.notna(row.get("乖離スコア"))
        #         and row.get("乖離スコア", 0) >= 5
        #         and row.get("MF予測順位", 99) == 1):
        #     s.append("戦略E(市場見落とし)")
        return " / ".join(s) if s else ""

    pdf["該当戦略"] = pdf.apply(check_strategy, axis=1)

    # ── 券種推奨（軸◎/軸(人気)/相手○/穴▲）──────────────────────────
    # 軸馬   : 勝率1位 かつ 勝ち期待値≥0
    # 軸(人気): 勝率1位 だが 勝ち期待値<0（実力はあるが妙味なし）
    # 相手   : 連対率が高い（軸以外の上位2頭）
    # 穴馬   : 複勝率が高くオッズ妙味あり、または勝ち期待値が高い一発
    pdf["券種推奨"] = ""
    try:
        win_rank   = pdf["勝ち確率"].rank(ascending=False)
        place2_rk  = pdf["連対順位"]
        place3_rk  = pdf["複勝順位"]

        # 軸馬（勝率1位）
        axis_idx = win_rank.idxmin()
        axis_ev  = pdf.loc[axis_idx, "単勝期待値"]
        if pd.notna(axis_ev) and axis_ev >= 0:
            pdf.at[axis_idx, "券種推奨"] = "軸◎"
        else:
            pdf.at[axis_idx, "券種推奨"] = "軸(人気)"

        # 相手○（連対率上位、軸を除く上位2頭）
        aite_count = 0
        for idx in place2_rk.sort_values().index:
            if idx == axis_idx:
                continue
            if pdf.at[idx, "券種推奨"] == "":
                pdf.at[idx, "券種推奨"] = "相手○"
                aite_count += 1
            if aite_count >= 2:
                break

        # 穴馬▲（複勝率上位 かつ オッズ妙味 or 勝ち期待値プラス、未割当の馬）
        for idx in place3_rk.sort_values().index:
            if pdf.at[idx, "券種推奨"] != "":
                continue
            odds = pdf.at[idx, "単勝オッズ"]
            ev   = pdf.at[idx, "単勝期待値"]
            fuku = pdf.at[idx, "複勝確率"]
            # 妙味判定: 単勝10倍以上で複勝率が一定以上、またはEVプラス
            if (pd.notna(odds) and odds >= 7 and pd.notna(fuku) and fuku >= 0.35) \
               or (pd.notna(ev) and ev >= 0.2):
                pdf.at[idx, "券種推奨"] = "穴▲"
                break
    except Exception as e:
        print(f"  券種推奨の計算でエラー（スキップ）: {e}")

    # レース概要
    dist   = int(pdf["距離"].iloc[0]) if pd.notna(pdf["距離"].iloc[0]) else "不明"
    turf   = "芝" if pdf["is_turf"].iloc[0] == 1 else "ダート"
    baba_inv = {1:"良", 2:"稍重", 3:"重", 4:"不良"}
    baba   = baba_inv.get(pdf["馬場状態_num"].iloc[0], "不明")
    cls_inv = {1:"新馬",2:"未勝利",3:"1勝クラス",4:"2勝クラス",
               5:"3勝クラス",6:"オープン",7:"G3",8:"G2",9:"G1"}
    cls    = cls_inv.get(pdf["クラス_num"].iloc[0], "不明")
    n_horse = len(pdf)

    # レポート生成・表示
    report = build_report(
        pdf, race_id, jyo_name, race_no,
        dist, turf, baba, cls, n_horse
    )
    print("\n" + report)

    # ── CSV保存（GitHub経由でダッシュボードに表示） ──────────────────
    try:
        # 総合スコア計算（build_reportと同じロジック）
        pdf["_ai_rank"] = pdf["勝ち確率"].rank(ascending=False)
        n = len(pdf)
        pdf["総合スコア"] = (
            (1 - (pdf["_ai_rank"] - 1) / n) * 80 +
            pdf["該当戦略"].apply(lambda s: 20 if s else 0)
        )
        final_top = pdf.sort_values("総合スコア", ascending=False).head(5)
        marks = ["◎", "○", "▲", "△", "×"]
        pdf["推奨ランク"] = ""
        for i, (idx, _) in enumerate(final_top.iterrows()):
            if i < len(marks):
                pdf.at[idx, "推奨ランク"] = marks[i]

        # 保存する列を選択
        save_cols = [
            "race_id", "馬名", "馬番", "枠番",
            "単勝オッズ", "人気",
            "勝ち確率", "連対確率", "複勝確率", "3着内確率",
            "単勝期待値", "推奨賭け率",
            "乖離スコア", "MF予測順位", "MF勝ち確率",
            "該当戦略", "推奨ランク", "総合スコア", "券種推奨",
            "予測順位", "連対順位", "複勝順位",
            "過去勝率", "過去出走数", "前走間隔",
        ]
        save_cols = [c for c in save_cols if c in pdf.columns]
        save_df = pdf[save_cols].copy()

        # レース情報を追加
        save_df["jyo"]      = jyo_name
        save_df["race_no"]  = race_no
        save_df["距離"]     = dist
        save_df["馬場"]     = turf
        save_df["馬場状態"] = baba
        save_df["クラス"]   = cls
        save_df["予想日時"] = datetime.now().strftime("%Y/%m/%d %H:%M")

        # today_predictions.csv に追記（当日分をまとめる）
        out_path = os.path.join(BASE_DIR, "today_predictions.csv")
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            # 同じrace_idは上書き
            existing = existing[existing["race_id"].astype(str) != str(race_id)]
            save_df = pd.concat([existing, save_df], ignore_index=True)
        save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  予想データ保存完了 → {out_path}")
    except Exception as e:
        print(f"  予想データ保存エラー: {e}")

    # メール送信
    subject = f"【競馬AI詳細予想】{jyo_name} {race_no}R"
    print("\nメール送信中...")
    send_email(subject, report)


def _run_predict_safe(race_id):
    try:
        predict_race(race_id)
    except Exception as e:
        print(f"  {race_id} エラー: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "today":
        # 当日の全レースを一括予想してCSVに保存
        # auto_predict_publish.py の朝7時一括予想から呼び出される
        from keiba_auto import get_today_races
        print("当日レース一覧を取得中...")
        race_info = get_today_races()
        if not race_info:
            print("本日のレースが取得できませんでした")
            sys.exit(1)
        print(f"{len(race_info)}レースを予想します")
        for rid in sorted(race_info.keys()):
            _run_predict_safe(rid)
        print("\n全レース予想完了 → today_predictions.csv")
    else:
        # 個別レース予想（auto_predict_publish.py の発走40分前実行から呼び出される）
        race_id = sys.argv[1] if len(sys.argv) > 1 else TARGET_RACE_ID
        predict_race(race_id)