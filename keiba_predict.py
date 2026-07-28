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

# コンソール出力をUTF-8化（レポートの罫線'═'等がcp932環境でUnicodeEncodeError→
# 予想プロセスが異常終了しgit push漏れになるのを防ぐ。errors="replace"で絶対に落とさない）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from model import LambdaRankWrapper  # pickle読み込みに必要

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


def read_pred_csv(path, **kw):
    """race_idを必ず文字列で読み込む。全数字のrace_idはpandasがint/float化し、
    NaN混入時は float→'202602011201.0' と化けて結果照合(netkeiba/JV)が全滅する
    （2026-07-19の照合失敗の真因）。dtype=str＋末尾'.0'除去で恒久防止する。"""
    df = pd.read_csv(path, dtype={"race_id": str}, **kw)
    if "race_id" in df.columns:
        df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def record_odds_snapshot(pdf, race_id):
    """予想実行の都度、現時点のオッズ/人気を odds_history.csv に追記（重複除去せず時系列蓄積）。
    朝・発走40分前・直前の各実行で自然に貯まり、後日「オッズ変動（金の流れ）」特徴の材料になる。
    新規スクレイピングはせず既に取得済みのpdfのオッズを使う（IP負荷ゼロ）。記録失敗で予想は止めない。"""
    try:
        cols = [c for c in ("馬番", "馬名", "単勝オッズ", "人気", "複勝オッズ_min") if c in pdf.columns]
        if "単勝オッズ" not in cols or pdf["単勝オッズ"].isna().all():
            return  # オッズ未取得なら記録しない（NaN行で汚さない）
        snap = pdf[cols].copy()
        snap.insert(0, "race_id", str(race_id))
        snap["記録時刻"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        path = os.path.join(BASE_DIR, "odds_history.csv")
        snap.to_csv(path, mode="a", header=not os.path.exists(path),
                    index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  オッズ記録スキップ: {e}")


def fmt_bet_combos(kind, combos):
    """買い目の組み合わせ群を「軸→相手」で表示。相手(複勝上位)の馬番が全部見えるようにする。
    例: 馬単 09-04,09-01,… → "9→4・1・7・10・3・2"。ダッシュボードと同じ整形。"""
    lists = [str(c).split("-") for c in combos if str(c)]
    if not lists:
        return ""
    common = set(lists[0])
    for p in lists[1:]:
        common &= set(p)

    def _i(x):
        try:
            return str(int(x))
        except (ValueError, TypeError):
            return str(x)

    if 1 <= len(common) <= 2 and len(lists) >= 2:
        axis = "・".join(_i(x) for x in lists[0] if x in common)
        partners = []
        for p in lists:
            for x in p:
                if x not in common and x not in partners:
                    partners.append(x)
        arrow = "→" if kind in ("馬単", "3連単") else "-"
        return f"{axis}{arrow}" + "・".join(_i(x) for x in partners)
    if len(lists) <= 6:
        return " / ".join("-".join(_i(x) for x in p) for p in lists)
    return f"{'-'.join(_i(x) for x in lists[0])} 他{len(lists) - 1}点"


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


# ── 推奨馬の組み合わせ的中率（フォーメーション）─────────────────────────────
_FORMATION_CALIB = None

def _load_formation_calib():
    """isotonicキャリブレーター（formation_calibrators.pkl）を1回だけ読む。無ければ空。"""
    global _FORMATION_CALIB
    if _FORMATION_CALIB is None:
        path = os.path.join(BASE_DIR, "formation_calibrators.pkl")
        try:
            with open(path, "rb") as f:
                _FORMATION_CALIB = pickle.load(f)
        except Exception:
            _FORMATION_CALIB = {}
    return _FORMATION_CALIB


def formation_probs(win_probs, idx3, idx5):
    """勝ち確率(Plackett-Luce)から推奨馬集合の組み合わせ確率を計算する。
    idx3=推奨3頭(◎○▲)の位置, idx5=推奨5頭(◎○▲△×)の位置。
    戻り値: dict(s3_2ren, s3_2fuku, s5_2ren, s5_3fuku) の生確率。"""
    from itertools import permutations
    p = np.array(win_probs, dtype=float)
    p = np.clip(np.nan_to_num(p, nan=0.0), 1e-12, None)
    p = p / p.sum()
    n = len(p)
    S3, S5 = set(idx3), set(idx5)

    def top2_in(S):
        tot = 0.0
        for i in S:
            for j in S:
                if i == j:
                    continue
                d = 1 - p[i]
                if d > 1e-12:
                    tot += p[i] * p[j] / d
        return min(tot, 1.0)

    a2 = 0.0   # S3のうち2頭以上がtop3(複勝圏)
    b2 = 0.0   # top3すべてS5内(5頭で3着以内独占)
    for i, j, k in permutations(range(n), 3):
        d1 = 1 - p[i]
        d2 = 1 - p[i] - p[j]
        if d1 <= 1e-12 or d2 <= 1e-12:
            continue
        pr = p[i] * (p[j] / d1) * (p[k] / d2)
        t3 = {i, j, k}
        if len(t3 & S3) >= 2:
            a2 += pr
        if t3 <= S5:
            b2 += pr
    return {"s3_2ren": top2_in(S3), "s3_2fuku": a2,
            "s5_2ren": top2_in(S5), "s5_3fuku": b2}


def formation_probs_calibrated(win_probs, idx3, idx5):
    """組み合わせ確率を実績ベース(isotonic)で補正して返す。補正器が無ければ生確率。"""
    raw = formation_probs(win_probs, idx3, idx5)
    calib = _load_formation_calib()
    out = {}
    for k, v in raw.items():
        iso = calib.get(k)
        if iso is not None:
            try:
                out[k] = float(iso.predict([v])[0])
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


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
    lines.append(f"  [競馬AI 詳細予想レポート]  {now}")
    lines.append(f"  レースID : {race_id}   {jyo_name} {race_no}R")
    lines.append(sep())
    lines.append("")

    # レース概要
    lines.append(header("[レース概要]", "─"))
    lines.append(f"  コース   : {turf} {dist}m")
    lines.append(f"  馬場状態 : {baba}")
    lines.append(f"  クラス   : {cls}")
    lines.append(f"  出走頭数 : {n_horse}頭")
    lines.append("")

    # ── 全馬詳細一覧 ──────────────────────────────────────────────────────
    lines.append(header("[全馬詳細評価（予測順位順）]", "─"))
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
        # 状況フラグ（前走情報から派生）
        _flags = []
        _cls_chg = row.get("クラス変化", np.nan)
        if pd.notna(_cls_chg) and _cls_chg > 0:
            _flags.append("昇級")
        elif pd.notna(_cls_chg) and _cls_chg < 0:
            _flags.append("降級▲")
        _prev_diff = row.get("前走着差_秒", np.nan)
        if pd.notna(_prev_diff) and _prev_diff <= 0.2 and pd.notna(row.get("前走着順")) and row["前走着順"] != 1:
            _flags.append("接戦")
        if row.get("重賞出走フラグ", 0) == 1.0:
            _flags.append("重賞歴")
        flag_str = "[" + "/".join(_flags) + "]" if _flags else ""

        line = (
            f"{int(row['予測順位']):>2} {int(row['馬番']):>2} {str(row['馬名']):<13} "
            f"{odds_s:>6} {pop_s:>2}人 "
            f"{win_p*100:>4.1f}% {place2_p*100:>4.1f}% {place_p*100:>4.1f}% {place3_p*100:>4.1f}% "
            f"{ev_s:>6} {kelly_s:>5} "
            f"{past_wr_s:>5} {runs:>3}走 "
            f"{interval_s:>4} {gap_s:>4}  {strategy}{flag_str} {mark}"
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
                block.append(f"     [戦略該当] {strategy}")
            block.append("")
        return block

    # top5_ai: 総合スコア(MF60%+通常40%+EVペナルティ)の上位5頭 = 印と同じ基準
    top5_ai = pdf.sort_values("総合スコア", ascending=False).head(5)
    # top5_ev: 通常モデル高評価かつEV<0（市場評価と近い）→ 安定した信頼馬
    top5_ev = pdf[(pdf["単勝期待値"] < 0) & (pdf["勝ち確率"] >= 0.03)].sort_values("勝ち確率", ascending=False).head(5)
    if top5_ev.empty:
        top5_ev = pdf[pdf["勝ち確率"] >= 0.03].sort_values("勝ち確率", ascending=False).head(5)

    lines += rec_block(
        "AI総合予想 TOP5", "[AI]", top5_ai,
        "MF60%+通常40%ブレンド+EVペナルティで選出。印の◎○▲と同じ基準。"
    )
    lines += rec_block(
        "安定高評価 TOP5", "[安定]", top5_ev,
        "通常モデル高評価かつEV<0（市場評価と合致）。信頼度の高い実力馬。"
    )

    # ── 2軸比較サマリー ──────────────────────────────────────────────────
    lines.append(header("[2軸比較サマリー]", "─"))
    ai_names = top5_ai["馬名"].tolist()
    ev_names = top5_ev["馬名"].tolist()
    both    = [h for h in ai_names if h in ev_names]
    ai_only = [h for h in ai_names if h not in ev_names]
    ev_only = [h for h in ev_names if h not in ai_names]

    lines.append("  【両方に選ばれた馬】← 最注目")
    if both:
        for h in both:
            row = pdf[pdf["馬名"] == h].iloc[0]
            st  = f"  [!!]{row['該当戦略']}" if row.get("該当戦略") else ""
            lines.append(
                f"    [+] {h}  勝率{row['勝ち確率']*100:.1f}%  "
                f"連対率{row['連対確率']*100:.1f}%  複勝率{row['複勝確率']*100:.1f}%  "
                f"期待値{_ev(row['単勝期待値'])}  {_odds(row['単勝オッズ'])}{st}"
            )
    else:
        lines.append("    （なし）")

    lines.append("")
    lines.append("  【純粋AI予想のみ】← モデル高評価だがオッズが低い可能性")
    if ai_only:
        for h in ai_only:
            row = pdf[pdf["馬名"] == h].iloc[0]
            lines.append(
                f"    [AI] {h}  勝率{row['勝ち確率']*100:.1f}%  "
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
                f"    [EV] {h}  勝率{row['勝ち確率']*100:.1f}%  "
                f"期待値{_ev(row['単勝期待値'])}  {_odds(row['単勝オッズ'])}"
            )
    else:
        lines.append("    （なし）")

    lines.append("")

    # ── 最終予想（2軸を統合した結論） ──────────────────────────────────
    lines.append(header("[最終予想（総合判定）]", "═"))
    lines.append("  【判定ロジック】")
    lines.append("  ① 両方選出 × 戦略該当  → 最強推奨（◎）")
    lines.append("  ② AI予測順位が上位（期待値はマイナスでも可）")
    lines.append("  ③ 該当戦略があれば加点して優先度UP")
    lines.append("  ※ 期待値・オッズは参考情報として表示（印の決定には使用しません）")
    lines.append("")

    # 最終予想の順位: predict_race_pdf が付与した印列（◎○▲△×）をそのまま表示する。
    # ◎○=総合スコア(勝ち軸)、▲△=複勝確率(3着内堅い)、×=人気薄の穴、と役割分離済み。
    # ここで総合スコア順に再導出すると印列と不一致になるため、印列を正順で並べる。
    _mk_ord = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
    final_top = pdf[pdf["印"].isin(_mk_ord)].copy()
    final_top["_mk_ord"] = final_top["印"].map(_mk_ord)
    final_top = final_top.sort_values("_mk_ord").head(5)

    # ── AI信頼度スコア ─────────────────────────────────────────────────
    # TOP1とTOP2の総合スコア差。大きいほど◎が突出しており、AIが確信を持っている。
    _scores_sorted = pdf["総合スコア"].nlargest(2).values
    _score_gap = float(_scores_sorted[0] - _scores_sorted[1]) if len(_scores_sorted) >= 2 else 0.0
    _honmei_ev_val = pdf.loc[pdf["印"] == "◎", "単勝期待値"].values
    _honmei_ev = float(_honmei_ev_val[0]) if len(_honmei_ev_val) > 0 and pd.notna(_honmei_ev_val[0]) else None
    _mf_top = pdf["MF勝ち確率"].idxmax() if ("MF勝ち確率" in pdf.columns and pdf["MF勝ち確率"].notna().any()) else None
    _win_top = pdf["勝ち確率"].idxmax()
    _models_agree = (_mf_top == _win_top) if _mf_top is not None else False

    # 少頭数補正: 8頭以下はランダム性が高く予測精度が下がる
    _n_horses = len(pdf)
    _is_small_field = _n_horses <= 8

    # ◎の前走情報（着差・クラス変化）
    _honmei_row = pdf[pdf["印"] == "◎"]
    _honmei_prev_diff = None
    _honmei_class_chg = None
    if not _honmei_row.empty:
        _hr = _honmei_row.iloc[0]
        if "前走着差_秒" in _hr.index and pd.notna(_hr.get("前走着差_秒")):
            _honmei_prev_diff = float(_hr["前走着差_秒"])
        if "クラス変化" in _hr.index and pd.notna(_hr.get("クラス変化")):
            _honmei_class_chg = float(_hr["クラス変化"])

    # 信頼度判定（少頭数・前走情報でペナルティ/ボーナス）
    _effective_gap = _score_gap
    if _is_small_field:
        _effective_gap *= 0.8  # 少頭数は信頼度を20%ダウン
    if _honmei_class_chg is not None and _honmei_class_chg > 0:
        _effective_gap *= 0.85  # 昇級戦は15%ダウン
    if _honmei_prev_diff is not None and _honmei_prev_diff <= 0.2:
        _effective_gap *= 1.1  # 前走接戦（0.2秒以内）は10%アップ

    if _effective_gap >= 15 and _models_agree and (_honmei_ev is not None and _honmei_ev < 0.1):
        _confidence = "高★★★"
        _confidence_note = "→ 積極的に買い推奨"
    elif _effective_gap >= 8 or (_models_agree and _honmei_ev is not None and _honmei_ev < 0.1):
        _confidence = "中★★"
        _confidence_note = "→ 通常推奨"
    else:
        _confidence = "低★"
        _confidence_note = "→ 伯仲レース・少額か見送り推奨"

    # 付加情報ラベル
    _extra_notes = []
    if _is_small_field:
        _extra_notes.append(f"少頭数({_n_horses}頭)")
    if _honmei_class_chg is not None and _honmei_class_chg > 0:
        _extra_notes.append("◎昇級戦")
    elif _honmei_class_chg is not None and _honmei_class_chg < 0:
        _extra_notes.append("◎降級(有利)")
    if _honmei_prev_diff is not None:
        _extra_notes.append(f"◎前走着差{_honmei_prev_diff:.2f}秒")
    _extra_str = " / ".join(_extra_notes)

    lines.append(f"  [AI信頼度] {_confidence}  (スコア差:{_score_gap:.1f} / モデル合意:{'YES' if _models_agree else 'NO'}) {_confidence_note}")
    if _extra_str:
        lines.append(f"             {_extra_str}")
    # 購入推奨度（買い指数・honest backtestで単勝回収率に較正）
    if "買い指数" in pdf.columns and pdf["買い指数"].notna().any():
        _ki  = int(pdf["買い指数"].dropna().iloc[0])
        _rec = str(pdf["購入推奨"].iloc[0]) if "購入推奨" in pdf.columns else ""
        _roi = str(pdf["想定単回収"].iloc[0]) if "想定単回収" in pdf.columns else ""
        lines.append(f"  [購入推奨度] {_rec}  買い指数 {_ki}/100  想定単勝回収 {_roi}")
    lines.append("")

    # 役割ラベル（印の意味に対応）
    final_label_map = {"◎": "本命", "○": "対抗", "▲": "3着内堅い",
                       "△": "3着内", "×": "穴"}

    lines.append("  ┌─────────────────────────────────────────────────────────┐")
    for i, (_, row) in enumerate(final_top.iterrows()):
        if i >= 5:
            break
        mk    = str(row["印"])
        lbl   = final_label_map.get(mk, "")
        odds  = row.get("単勝オッズ", np.nan)
        pop   = row.get("人気", np.nan)
        wp    = row["勝ち確率"]
        ev    = row["単勝期待値"]
        score = row.get("総合スコア", row.get("_score", 0.0))
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
        # ◎は馬券内軸(place3-1位・BT複勝率64.2%)。単勝の主役は◎妙。
        if i == 0:
            lines.append("  │  [役割] ◎=馬券内軸(BT複勝率64.2%)。単勝・高配当の主役は◎妙(下の妙味軸)")
        if i < 4:
            lines.append("  ├─────────────────────────────────────────────────────────┤")
    lines.append("  └─────────────────────────────────────────────────────────┘")
    lines.append("")
    # バックテスト要約（B印・2025 honest 3,144レース・実払戻）
    lines.append("  [BT] B印 2025 honest(3144R・実払戻)")
    lines.append("  ◎複勝率64.2% / ◎妙:単勝167%・馬単妙→総流し148%・◎-妙ワイド115%")
    lines.append("  【鉄則】◎妙が出たレースを厚く買う。重賞14頭以上は見送り(AI優位なし)")
    lines.append("")

    # ── 券種推奨（3モデルによる役割判定） ────────────────────────────
    if "券種推奨" in pdf.columns and (pdf["券種推奨"] != "").any():
        lines.append(header("[券種推奨（3モデル判定）]", "─"))
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

    # ── [妙味重視の狙い目（回収率重視）] ────────────────────────────
    lines.append(header("[妙味重視の狙い目（回収率重視）]", "─"))
    lines.append("  市場(人気)を出し抜ける可能性のある買い方。妙味がなければ「見送り推奨」。")
    lines.append("")

    # ⓪ 回収率用◎（妙味軸）: MF最上位が人気馬寄りの◎と別馬のとき提示
    #    BT(2025): 発生率42%・複勝率44.9%・◎平均人気4.6。的中用◎とは別の「妙味の軸」。
    lines.append("  ⓪ 回収率用◎（妙味軸）── 的中用◎(人気寄り)とは別に、MFが推す価値馬")
    if "妙味軸" in pdf.columns and (pdf["妙味軸"] == "◎妙").any():
        m_row = pdf[pdf["妙味軸"] == "◎妙"].iloc[0]
        m_odds = m_row.get("単勝オッズ", np.nan)
        m_pop  = m_row.get("人気", np.nan)
        m_fuku = m_row.get("複勝確率", np.nan)
        m_ev   = m_row.get("単勝期待値", np.nan)
        m_odds_s = f"{m_odds:.1f}倍" if pd.notna(m_odds) else "未確定"
        m_pop_s  = f"{int(m_pop)}番人気" if pd.notna(m_pop) else "-"
        lines.append(
            f"     ◎妙 {m_row['馬名']}（{m_odds_s} {m_pop_s}） "
            f"複勝率{m_fuku*100:.1f}% 期待値{_ev(m_ev)}"
        )
        lines.append("     → 市場が過小評価。複勝・ワイドの軸、or 単勝妙味として狙う価値。")
    else:
        lines.append("     該当なし（MFの本命＝的中用◎と一致 → 軸に一本化。素直に◎を信頼）。")
    lines.append("")

    # ① ◎の手堅い複勝（単勝は難しいが複勝率が高い本命）
    _honmei_mark_df = pdf[pdf["印"] == "◎"]
    honmei_row = _honmei_mark_df.iloc[0] if not _honmei_mark_df.empty else pdf.sort_values("総合スコア", ascending=False).iloc[0]
    h_win  = honmei_row["勝ち確率"]
    h_fuku = honmei_row["複勝確率"]
    h_odds = honmei_row.get("単勝オッズ", np.nan)
    lines.append("  ① 本命の手堅い複勝")
    if pd.notna(h_fuku) and h_fuku >= 0.45:
        lines.append(
            f"     ◎{honmei_row['馬名']} → 複勝率{h_fuku*100:.1f}%。"
            f"単勝(勝率{h_win*100:.1f}%)は割れても複勝なら手堅い。"
        )
    else:
        lines.append(
            f"     ◎{honmei_row['馬名']} の複勝率{h_fuku*100:.1f}%。"
            f"複勝でも確実とは言えず、軸の信頼度は中程度。"
        )
    lines.append("")

    # ② 妙味のある人気薄馬（期待値≥0 かつ 人気4番手以下 or 勝率ランク<人気ランク）
    lines.append("  ② 妙味のある人気薄（市場が見落とした可能性）")
    pdf_myumi = pdf.copy()
    pdf_myumi["_勝率ランク"] = pdf_myumi["勝ち確率"].rank(ascending=False)
    candidates = pdf_myumi[
        (pdf_myumi["単勝期待値"] >= 0)
        & (
            (pdf_myumi["人気"] >= 4)
            | (pdf_myumi["_勝率ランク"] < pdf_myumi["人気"])
        )
    ].sort_values("単勝期待値", ascending=False)
    if len(candidates) > 0:
        for _, row in candidates.head(3).iterrows():
            odds = row.get("単勝オッズ", np.nan)
            pop  = row.get("人気", np.nan)
            ev   = row["単勝期待値"]
            wp   = row["勝ち確率"]
            fuku = row["複勝確率"]
            odds_s = f"{odds:.1f}倍" if pd.notna(odds) else "未確定"
            pop_s  = f"{int(pop)}番人気" if pd.notna(pop) else "-"
            lines.append(
                f"     ★ {row['馬名']}（{odds_s} {pop_s}）"
                f" 勝率{wp*100:.1f}% 複勝率{fuku*100:.1f}% 期待値{_ev(ev)}"
            )
        lines.append("     → モデル評価の割にオッズが妙味。単勝or複勝で狙う価値あり。")
    else:
        lines.append("     該当なし。このレースは人気薄に妙味がなく、無理は禁物。")
    lines.append("")

    # ③ 総合判断（AI信頼度 + 妙味有無で総合判定）
    lines.append("  ③ レース総合判断")
    has_myumi = len(candidates) > 0
    # 信頼度は既に計算済み（_confidence, _confidence_note, _score_gap, _models_agree）
    _ev_str = f"EV{_honmei_ev:+.2f}" if _honmei_ev is not None else "EV不明"
    lines.append(
        f"     AI信頼度 {_confidence} (スコア差:{_score_gap:.1f} / モデル合意:{'YES' if _models_agree else 'NO'} / ◎{_ev_str})"
    )
    if _confidence == "高★★★":
        msg = "AIが強く確信 → 単勝・馬連を中心に積極推奨。"
    elif _confidence == "中★★":
        msg = "AI評価は安定 → 印通りに購入。複勝で安全策も有効。"
    else:
        msg = "伯仲レース(AIの確信が低い) → 少額か◎複勝のみ推奨。見送りも選択肢。"
    if has_myumi:
        msg += f" 妙味馬あり({len(candidates)}頭)。"
    lines.append(f"     {msg}")
    lines.append("")

    # ── AI合意馬推奨（MF×通常モデル両方が上位と判断した馬）────────────
    # EV>=0.4は実績で勝率0%だったため廃止。MFと通常モデル両方がTop3以内の馬を推奨。
    has_mf_col = "MF勝ち確率" in pdf.columns and pdf["MF勝ち確率"].notna().any()
    if has_mf_col:
        mf_top3   = set(pdf["MF勝ち確率"].nlargest(3).index)
        win_top3  = set(pdf["勝ち確率"].nlargest(3).index)
        agree_idx = mf_top3 & win_top3
        agree_df  = pdf.loc[list(agree_idx)].sort_values("MF勝ち確率", ascending=False)
        agree_df  = agree_df[agree_df["単勝オッズ"] >= 1.5]
    else:
        agree_df = pdf[pdf["予測順位"] <= 3].sort_values("勝ち確率", ascending=False)
        agree_df = agree_df[agree_df["単勝オッズ"] >= 1.5]
    lines.append("  【[AI合意] MF×通常モデル両方がTop3評価（双方が認めた本命候補）】")
    if len(agree_df) > 0:
        for _, r in agree_df.head(3).iterrows():
            pop_r = r.get("人気", np.nan)
            pop_comment = ""
            if pd.notna(pop_r):
                if int(pop_r) == 1:
                    pop_comment = " [1番人気]"
                elif int(pop_r) >= 7:
                    pop_comment = " [穴]"
                elif int(pop_r) >= 4:
                    pop_comment = " [中穴]"
            lines.append(
                f"     ★ {r['馬名']}（{r['単勝オッズ']:.1f}倍 {int(pop_r) if pd.notna(pop_r) else '-'}人気）"
                f"  MF勝率{r['MF勝ち確率']*100:.1f}% 通常勝率{r['勝ち確率']*100:.1f}%{pop_comment}"
            )
        lines.append("     ※ 両モデル合意馬が◎の基本軸。単独で◎が両モデル合意なら信頼度高。")
    else:
        lines.append("     該当なし（モデル意見が分かれているレース）。印を参考に判断。")
    lines.append("")

    # ── 買い目サマリー（複勝・馬連・ワイド・馬単・3連複） ──────────────
    lines.append("  【買い目サマリー（印ベース・参考）】")
    honmei = final_top.iloc[0]
    taiko  = final_top.iloc[1] if len(final_top) > 1 else None
    ana    = final_top.iloc[2] if len(final_top) > 2 else None

    h_odds = honmei.get("単勝オッズ", np.nan)
    h_pop  = honmei.get("人気", np.nan)
    _tan_comment = " ← ◎単勝はBT105%の薄利。単勝の主役は◎妙(BT167%)"
    lines.append(
        (f"  単勝   : ◎{honmei['馬名']}  "
         f"（{h_odds:.1f}倍 / EV{_ev(honmei['単勝期待値'])}）{_tan_comment}")
        if pd.notna(h_odds) else f"  単勝   : ◎{honmei['馬名']}{_tan_comment}"
    )

    # 複勝：◎の複勝確率×想定複勝オッズ(単勝オッズの目安1/3〜1/4)で期待値推定
    h_place_p = honmei.get("複勝確率", np.nan)
    if pd.notna(h_place_p) and pd.notna(h_odds):
        est_fuku_odds = max(h_odds / 4, 1.05)  # 簡易推定（実オッズはkeiba_auto側で実測）
        fuku_ev = h_place_p * est_fuku_odds - 1
        lines.append(
            f"  複勝   : ◎{honmei['馬名']}  "
            f"（複勝率{h_place_p*100:.1f}% / 推定オッズ約{est_fuku_odds:.1f}倍 / 推定EV{_ev(fuku_ev)}）"
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

    # ── 💰 レース判定と買い目（_race_bet_planに完全連動・2026-07-16）──
    #   レース単位でマトリクス判定(クラス×妙人気帯×MF自信度×頭数)し、買い目と厚さを自動調整。
    lines.append("")
    _plan = _race_bet_plan(pdf)
    _badge = {"勝負": "🔥勝負", "買い": "✅買い", "堅実": "🟢堅実", "少額": "⚠少額",
              "見送り": "❌見送り"}.get(_plan["判定"], _plan["判定"])
    lines.append(f"  ── 💰 レース判定: {_badge}（指数{_plan['指数']}） ──")
    lines.append(f"  理由: {_plan['理由']}")
    _myo = pdf[pdf["妙味軸"] == "◎妙"] if "妙味軸" in pdf.columns else pdf.iloc[0:0]
    if _plan["menu"]:
        _mk_no = {}
        for _mk2 in ("◎", "○", "▲", "△"):
            _r2 = pdf[pdf["印"] == _mk2]
            if len(_r2) and pd.notna(_r2.iloc[0]["馬番"]):
                _mk_no[_mk2] = int(_r2.iloc[0]["馬番"])
        _hno2 = _mk_no.get("◎", "-")
        _mno = None
        # 複妙(place系の軸)馬番。無ければ勝率妙にフォールバック
        _mno_p = None
        if "MF複勝順位" in pdf.columns and pd.to_numeric(pdf["MF複勝順位"], errors="coerce").notna().any():
            _rp2 = pdf[pd.to_numeric(pdf["MF複勝順位"], errors="coerce") == 1]
            if len(_rp2) and pd.notna(_rp2.iloc[0]["馬番"]):
                _mno_p = int(_rp2.iloc[0]["馬番"])
        if len(_myo):
            _m = _myo.iloc[0]
            _mp_s = f"{int(_m['人気'])}番人気" if pd.notna(_m.get("人気")) else "-"
            _mno = int(_m["馬番"])
            if _mno_p is None:
                _mno_p = _mno
            lines.append(f"  ◎妙 {_m['馬名']}（馬番{_mno} {_mp_s}）を軸に買う。資金配分: {_plan['サイズ']}")
        else:
            lines.append(f"  ◎（馬番{_hno2}・両モデル合意）を軸に買う。資金配分: {_plan['サイズ']}")
        # 実際の買い目行（資金設定・券種スキップ・予算トリム反映後）から点数と金額を取得
        try:
            _rid_disp = str(pdf["race_id"].iloc[0]) if "race_id" in pdf.columns else "0"
            _rows_bet = _build_bet_rows(pdf, _rid_disp)
        except Exception:
            _rows_bet = []
        _grp, _combos = {}, {}
        for rr in _rows_bet:
            g0 = _grp.setdefault(rr["買い方"], [0, 0])
            g0[0] += 1
            g0[1] += rr.get("金額", 100)
            _combos.setdefault(rr["買い方"], []).append(str(rr["組み合わせ"]))
        for kind, name, roi in _plan["menu"]:
            if _rows_bet and name not in _grp:
                continue           # KIND_SKIP/予算トリムで除外された買い方は表示しない
            # 実際の買い目(today_bets)の組み合わせから「軸→相手(馬番)」を展開表示
            combo = fmt_bet_combos(kind, _combos.get(name, []))
            _pts_s = ""
            if name in _grp:
                _pts_s = f"  {_grp[name][0]}点/{_grp[name][1]:,}円"
            lines.append(f"   {kind:4}: {combo:22} [BT{roi}%]{_pts_s}")
        if _rows_bet:
            _tot_amt = sum(r.get("金額", 100) for r in _rows_bet)
            lines.append(f"   ── 合計 {len(_rows_bet)}点 / {_tot_amt:,}円 ──")
        lines.append("   ※ 相手（人気上位/◯位内）は直前オッズの人気で自動決定＝レースごとに最適化")
    elif _plan["判定"] == "見送り":
        lines.append("  このレースは購入非推奨。資金は🔥勝負/✅買い/🟢堅実レースに温存。")
    lines.append("  ※ BT=2025実払戻3144R。判定/買い目はtoday_bets.csvに保存され、翌日照合されます。")
    lines.append("")

    # ── 推奨馬の組み合わせ的中率（フォーメーション・実績補正済み）──────────
    try:
        pdf_pos = pdf.reset_index(drop=True)
        marks = pdf_pos["印"].astype(str).values
        wp = pd.to_numeric(pdf_pos["勝ち確率"], errors="coerce").values
        idx3 = [i for i, m in enumerate(marks) if m in ("◎", "○", "▲")]
        idx5 = [i for i, m in enumerate(marks) if m in ("◎", "○", "▲", "△", "×")]
        if len(idx3) >= 3 and len(idx5) >= 5 and np.isfinite(wp).sum() >= 5:
            fc = formation_probs_calibrated(wp, idx3, idx5)
            lines.append("  【推奨馬の組み合わせ的中率（過去実績で補正済み）】")
            lines.append(f"    推奨3頭のうち2頭が連対（1-2着独占）: {fc['s3_2ren']*100:.1f}%")
            lines.append(f"    推奨3頭のうち2頭が複勝圏内        : {fc['s3_2fuku']*100:.1f}%")
            lines.append(f"    推奨5頭のうち2頭が連対            : {fc['s5_2ren']*100:.1f}%")
            lines.append(f"    推奨5頭のうち3頭が複勝圏内        : {fc['s5_3fuku']*100:.1f}%")
            lines.append("  ※ 推奨3頭=◎○▲、推奨5頭=◎○▲△×。2025年実績でキャリブレーション済み。")
            lines.append("")
    except Exception as e:
        pass  # 組み合わせ確率の計算失敗時はスキップ（本体予想は継続）

    lines.append(sep("─"))
    lines.append("  ※ AIによる予測です。投資は自己責任でお願いします。")
    lines.append("  ※ 推奨賭け率は1/4ケリー基準（保守的設定）です。")
    lines.append("  ※ 複勝・3着内確率はハーヴィルモデルによる近似値です。")
    lines.append(sep())

    return "\n".join(lines)


# ── 予測コア（auto/predict 共通エンジン） ────────────────────────────────
def predict_race_pdf(race_id: str, *, history_df: pd.DataFrame, models_pack: dict):
    """
    出馬表取得 → 特徴量構築 → 予測 → 印付け → CSV保存 → pdf を返す。

    models_pack のキー:
        "win"    : {"models": [...], "use_cols": [...]}  必須
        "place2" : {"models": [...], "use_cols": [...]}  なければ None
        "place3" : {"models": [...], "use_cols": [...]}  なければ None
        "mf"     : {"models": [...], "use_cols": [...]}  なければ None
    """
    race_id  = str(race_id).strip()
    jyo_cd   = int(race_id[4:6])
    race_no  = int(race_id[10:12])
    jyo_name = JYO_NAMES.get(jyo_cd, f"競馬場{jyo_cd}")

    win_pack    = models_pack["win"]
    place2_info = models_pack.get("place2")
    place3_info = models_pack.get("place3")
    mf_info     = models_pack.get("mf")

    win_models    = win_pack["models"]
    win_cols      = win_pack["use_cols"]
    win_weights   = win_pack.get("weights")
    place2_models = place2_info["models"]   if place2_info else None
    place2_cols   = place2_info["use_cols"] if place2_info else None
    place2_weights = place2_info.get("weights") if place2_info else None
    place3_models = place3_info["models"]   if place3_info else None
    place3_cols   = place3_info["use_cols"] if place3_info else None
    place3_weights = place3_info.get("weights") if place3_info else None
    mf_models     = mf_info["models"]       if mf_info else None
    mf_cols       = mf_info["use_cols"]     if mf_info else None
    mf_weights    = mf_info.get("weights")  if mf_info else None
    # MF複勝(place3)モデル: place系買い目の妙軸に使う（勝率軸より馬券内率・ROIが高い）
    mf_p3_info    = mf_info.get("place3") if mf_info else None
    mf_p3_models  = mf_p3_info["models"]   if mf_p3_info else None
    mf_p3_cols    = mf_p3_info["use_cols"] if mf_p3_info else None
    mf_p3_weights = mf_p3_info.get("weights") if mf_p3_info else None
    is_multi = place2_models is not None and place3_models is not None

    def _wavg(models, X, weights):
        """重み付きアンサンブル予測（weights=Noneなら均等平均）"""
        preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
        if weights is None:
            return preds.mean(axis=1)
        w = np.asarray(weights, dtype=float); w = w / w.sum()
        return preds @ w

    # 出馬表・オッズ取得
    from keiba_auto import get_race_data, build_features
    print("出馬表・オッズ取得中...")
    race_df = get_race_data(race_id)
    if race_df is None:
        print(f"エラー: 出馬表を取得できませんでした ({race_id})")
        return None
    print(f"  {len(race_df)}頭分取得完了")

    # 特徴量構築（一本化: 学習と同じパイプライン。失敗時は従来法にフォールバック）
    print("特徴量構築中...")
    try:
        import features as _features
        pdf = _features.build_features_for_prediction(race_df, history_df)
        if pdf is None or len(pdf) == 0 or "馬名" not in pdf.columns:
            raise ValueError("一本化パイプラインの出力が不正")
        print(f"  特徴量構築成功(一本化): {len(pdf)}行")
    except Exception as e_uni:
        print(f"  一本化パイプライン不可({e_uni}) → 従来法にフォールバック")
        pdf = build_features(race_df, history_df)
        print(f"  特徴量構築成功(従来): {len(pdf)}行")

    # ── 勝ち確率（レース内正規化）
    print("予測中...")
    X     = pdf.reindex(columns=win_cols)
    preds = _wavg(win_models, X, win_weights)
    pdf["予測スコア"] = preds
    pdf["予測順位"]   = pd.Series(preds).rank(ascending=False).astype(int).values
    win_raw   = np.clip(np.nan_to_num(preds, nan=0.0), 0, 1)
    win_sum   = win_raw.sum()
    win_probs = win_raw / win_sum if win_sum > 0 else np.ones(len(win_raw)) / len(win_raw)
    pdf["勝ち確率"]    = win_probs
    pdf["勝ち確率_生"] = win_raw

    # ── 連対率・複勝率
    if is_multi:
        X_p2   = pdf.reindex(columns=place2_cols)
        X_p3   = pdf.reindex(columns=place3_cols)
        p2_raw = np.clip(np.nan_to_num(
            _wavg(place2_models, X_p2, place2_weights), nan=0.0), 0, 1)
        p3_raw = np.clip(np.nan_to_num(
            _wavg(place3_models, X_p3, place3_weights), nan=0.0), 0, 1)
        n_runners = len(pdf)
        target2 = min(2, n_runners)
        target3 = min(3, n_runners)
        p2_sum, p3_sum = p2_raw.sum(), p3_raw.sum()
        place2 = np.maximum(
            np.clip((p2_raw / p2_sum * target2) if p2_sum > 0 else np.full(n_runners, target2 / n_runners), 0, 1),
            win_probs)
        place3 = np.clip(np.maximum(
            np.clip((p3_raw / p3_sum * target3) if p3_sum > 0 else np.full(n_runners, target3 / n_runners), 0, 1),
            place2), 0, 1)
        print("  連対率・複勝率を独立モデルで予想完了（正規化・包含関係保証）")
    else:
        place2, place3 = calc_place_probs_harvill(win_probs)
        place2 = np.maximum(place2, win_probs)
        place3 = np.maximum(place3, place2)
    pdf["連対確率"]  = place2
    pdf["複勝確率"]  = place3
    pdf["3着内確率"] = place3
    pdf["連対順位"]  = pd.Series(place2, index=pdf.index).rank(ascending=False)
    pdf["複勝順位"]  = pd.Series(place3, index=pdf.index).rank(ascending=False)

    pdf["単勝期待値"]    = pdf["勝ち確率"] * pdf["単勝オッズ"] - 1  # MFモデル取得後に上書き
    pdf["推奨賭け率"]    = pdf.apply(lambda r: kelly_fraction(r["勝ち確率"], r["単勝オッズ"]), axis=1)
    pdf["複勝推定オッズ"] = (pdf["単勝オッズ"] / 4).clip(lower=1.05)
    pdf["複勝期待値"]    = pdf["複勝確率"] * pdf["複勝推定オッズ"] - 1
    if "複勝オッズ_min" in pdf.columns:
        pdf["複勝期待値_実"] = pdf["複勝確率"] * pdf["複勝オッズ_min"] - 1
    else:
        pdf["複勝期待値_実"] = pdf["複勝期待値"]

    # ── MFモデル
    pdf["MF予測順位"] = np.nan
    pdf["乖離スコア"] = np.nan
    pdf["MF勝ち確率"] = np.nan
    if mf_models is not None and mf_cols is not None:
        try:
            X_mf     = pdf.reindex(columns=mf_cols)
            mf_preds = _wavg(mf_models, X_mf, mf_weights)
            pdf["MF予測順位"] = pd.Series(mf_preds).rank(ascending=False).values
            pdf["乖離スコア"] = pdf["予測順位"] - pdf["MF予測順位"]
            mf_raw = np.clip(np.nan_to_num(mf_preds, nan=0.0), 0, None)
            pdf["MF勝ち確率"] = mf_raw / mf_raw.sum() if mf_raw.sum() > 0 else np.ones(len(mf_raw)) / len(mf_raw)
            # EV を MF勝ち確率ベースで上書き（通常モデルの暫定値を置き換え）
            pdf["単勝期待値"] = pdf["MF勝ち確率"] * pdf["単勝オッズ"] - 1
            print("  市場フリー予測成功")
        except Exception as e:
            print(f"  市場フリー予測エラー（スキップ）: {e}")

    # ── MF複勝(place3)予測: place系買い目の「複勝妙」軸に使用
    #   2025BT: 複勝軸は勝率軸より馬券内率+8pt/複勝ROI114→124%・ワイド113→122%・3連複105→118%
    pdf["MF複勝率"]   = np.nan
    pdf["MF複勝順位"] = np.nan
    if mf_p3_models is not None and mf_p3_cols is not None:
        try:
            X_p3 = pdf.reindex(columns=mf_p3_cols)
            p3   = _wavg(mf_p3_models, X_p3, mf_p3_weights)
            pdf["MF複勝率"]   = np.clip(np.nan_to_num(p3, nan=0.0), 0, None)
            pdf["MF複勝順位"] = pd.Series(p3).rank(ascending=False).values
        except Exception as e:
            print(f"  MF複勝予測エラー（スキップ）: {e}")

    # ── 戦略判定（predict/auto 共通）
    # EV条件は実データでEV>=0.3の勝率が0%のため除外。AI予測順位とオッズ範囲を基準にする。
    def _check_strategy(row):
        s   = []
        jyo = int(str(row.get("race_id", "000000000000"))[4:6])
        r1  = row["予測順位"] == 1
        od  = row["単勝オッズ"]
        if r1 and 1.5 <= od <= 20:
            s.append("戦略A")
            if pd.notna(row.get("人気")) and row["人気"] != 1:
                s.append("戦略A-2")
        if pd.notna(row.get("人気")) and row["人気"] >= 3 and r1 and 2.0 <= od <= 20:
            s.append("戦略C")
        if pd.notna(row.get("前走間隔")) and 2 <= row["前走間隔"] <= 4 and r1:
            s.append("戦略D")
        if jyo in [5, 7] and r1 and 1.5 <= od <= 20:
            s.append("戦略F")
            if pd.notna(row.get("距離")) and row.get("距離", 9999) <= 1400:
                s.append("戦略FG")
        if jyo in [6, 10] and r1 and 1.5 <= od <= 20:
            s.append("戦略H")
        return " / ".join(s) if s else ""

    pdf["該当戦略"] = pdf.apply(_check_strategy, axis=1)

    # ── 券種推奨
    pdf["券種推奨"] = ""
    try:
        # 軸馬は MF勝ち確率（市場フリー）の1位。未取得なら通常モデルにフォールバック。
        _rank_for_axis = (
            pdf["MF勝ち確率"]
            if "MF勝ち確率" in pdf.columns and pdf["MF勝ち確率"].notna().any()
            else pdf["勝ち確率"]
        )
        axis_idx = _rank_for_axis.rank(ascending=False).idxmin()
        pdf.at[axis_idx, "券種推奨"] = (
            "軸◎" if pd.notna(pdf.at[axis_idx, "単勝期待値"]) and pdf.at[axis_idx, "単勝期待値"] >= 0
            else "軸(人気)"
        )
        aite = 0
        for idx in pdf["連対順位"].sort_values().index:
            if idx == axis_idx or pdf.at[idx, "券種推奨"] != "":
                continue
            pdf.at[idx, "券種推奨"] = "相手○"
            aite += 1
            if aite >= 2:
                break
        for idx in pdf["複勝順位"].sort_values().index:
            if pdf.at[idx, "券種推奨"] != "":
                continue
            if (pd.notna(pdf.at[idx, "単勝オッズ"]) and pdf.at[idx, "単勝オッズ"] >= 7
                    and pd.notna(pdf.at[idx, "複勝確率"]) and pdf.at[idx, "複勝確率"] >= 0.35) \
               or (pd.notna(pdf.at[idx, "単勝期待値"]) and pdf.at[idx, "単勝期待値"] >= 0.2):
                pdf.at[idx, "券種推奨"] = "穴▲"
                break
    except Exception as e:
        print(f"  券種推奨エラー（スキップ）: {e}")

    # ── 妙味判定
    pdf["妙味_単勝"] = ""
    pdf["妙味_複勝"] = ""
    for idx, row in pdf.iterrows():
        pop = row.get("人気", np.nan)
        ev  = row.get("単勝期待値", np.nan)
        fev = row.get("複勝期待値", np.nan)
        if pd.notna(pop) and pop >= 3 and pd.notna(ev) and ev >= 0:
            pdf.at[idx, "妙味_単勝"] = "妙味"
        if pd.notna(fev) and fev >= 0:
            pdf.at[idx, "妙味_複勝"] = "妙味"

    # ── 印割り当て
    # ◎○▲: MF 60% + 通常モデル 40% ブレンドスコア上位3頭 → 単勝・軸
    # △   : 連対確率（place2モデル）上位（◎○▲以外）  → 馬連・ワイドの相手
    # ×   : 複勝確率（place3モデル）上位（◎○▲△以外）→ 三連複・ヒモ
    n = len(pdf)
    has_mf = "MF勝ち確率" in pdf.columns and pdf["MF勝ち確率"].notna().any()

    # ── Phase 1: 軸馬精度アップ ──────────────────────────────────────────
    # MF+通常ブレンド (60:40) + EV>=0.1 ペナルティ
    # 実績: EV<0→勝率28.4%, EV>=0→勝率6.7% なのでEV>=0.1の馬を◎に選びにくくする
    if has_mf:
        mf_norm  = pdf["MF勝ち確率"] / pdf["MF勝ち確率"].sum()
        win_norm = pdf["勝ち確率"]    / pdf["勝ち確率"].sum()
        blend    = 0.6 * mf_norm + 0.4 * win_norm
        ev_val   = pdf["単勝期待値"].fillna(0.0)
        ev_adj   = np.where(ev_val >= 0.1,
                            np.maximum(0.55, 1.0 - ev_val.clip(0, 0.5)),
                            1.0)
        # 人気乖離補正: MF上位3頭 かつ 2-4番人気 → 市場が過小評価している馬に+5%
        # BT実績(2025 honest): 2-3番人気◎は回収率123.8%、1番人気は93.1%(赤字)
        # MFモデルが高評価しているのに市場人気が低い馬は狙い目として優先
        _pop = pd.to_numeric(pdf["人気"], errors="coerce").fillna(8)
        _mf_rank = pdf["MF勝ち確率"].rank(ascending=False)
        _ninki_bonus = ((_mf_rank <= 3) & (_pop >= 2) & (_pop <= 4)).astype(float) * 0.05
        blend_adj = (blend + _ninki_bonus)
        blend_adj = blend_adj / blend_adj.sum()  # 再正規化

        pdf["_ai_rank"] = (blend_adj * ev_adj).rank(ascending=False)
    else:
        pdf["_ai_rank"] = pdf["勝ち確率"].rank(ascending=False)
    pdf["総合スコア"] = (1 - (pdf["_ai_rank"] - 1) / n) * 80 + pdf["該当戦略"].apply(lambda s: 20 if s else 0)

    pdf["推奨ランク"] = ""
    pdf["印"]        = ""

    # ── 役割分離型の印割り当て（Bポリシー・2026-07-15確定）───────────────
    #   ◎ 本命      = 複勝確率(place3)1位 ＝ 馬券内軸
    #   ○ 対抗      = 複勝確率2位、▲=3位、△=4位
    #   × 穴        = 人気薄で複勝妙味のある馬（◎○▲△除く）＝一発・高配当のヒモ
    #   検証(2025 honest 3144R): ◎を総合スコア(blend)1位→place3-1位に変更で
    #   ◎複勝率54.5→64.2%(+10pt)・◎単勝的中29.2→33.6%。OP以上でも47.5→55.4%。
    #   さらに◎妙(MF1位)との分離が進み妙味軸の発生が547→1679R・妙単勝ROI156.5→167.1%。
    #   勝ち軸(旧blend◎)の妙味は「妙味軸=◎妙」が引き継ぐ役割分担。
    assigned = set()

    # ◎の選定方式をレース条件で切替（V3アダプティブ・2026-07-16確定）
    #   ① 1番人気オッズ<=2.0（市場が確信する堅いレース）→ ◎=人気1位
    #   ② OP以上（重賞はAI優位が消える市場）        → ◎=人気1位
    #   ③ それ以外（AIの得意ゾーン）               → ◎=複勝確率(place3)1位
    #   BT(2025): ◎複勝率64.5→65.9% / 妙発生1679→1855R / 妙単勝ROI167→172% / 馬単妙総流し148→157%
    if "複勝確率" in pdf.columns and pdf["複勝確率"].notna().any():
        fuku_sorted = pdf.sort_values("複勝確率", ascending=False)
    else:
        fuku_sorted = pdf.sort_values("総合スコア", ascending=False)
    _pop_all = pd.to_numeric(pdf["人気"], errors="coerce")
    _cls_v0 = pd.to_numeric(pdf["クラス_num"].iloc[0], errors="coerce") if "クラス_num" in pdf.columns else np.nan
    _fav_idx = _pop_all.idxmin() if _pop_all.notna().any() else None
    _fav_odds = pd.to_numeric(pdf.loc[_fav_idx, "単勝オッズ"], errors="coerce") if _fav_idx is not None else np.nan
    _use_fav = (pd.notna(_fav_odds) and float(_fav_odds) <= 2.0) or \
               (pd.notna(_cls_v0) and int(_cls_v0) >= 5)
    honmei_idx = _fav_idx if (_use_fav and _fav_idx is not None) else fuku_sorted.index[0]
    pdf.at[honmei_idx, "推奨ランク"] = "◎"
    pdf.at[honmei_idx, "印"] = "◎"
    assigned.add(honmei_idx)
    # ○▲△: ◎を除く複勝確率(place3)上位3頭
    _rest = [i for i in fuku_sorted.index if i != honmei_idx][:3]
    for mk, idx in zip(("○", "▲", "△"), _rest):
        pdf.at[idx, "推奨ランク"] = mk
        pdf.at[idx, "印"] = mk
        assigned.add(idx)

    # ×（穴）: 人気薄で複勝妙味のある馬。◎○▲△以外から。
    #   複勝妙味 = 複勝確率 × 推定複勝オッズ（人気薄ほど配当が大きく妙味）。
    #   まず人気薄(人気>=6)に限定、居なければ残り全体の複勝妙味最大にフォールバック。
    rest_x = pdf[~pdf.index.isin(assigned)].copy()
    if not rest_x.empty and "複勝確率" in rest_x.columns:
        if "複勝オッズ_min" in rest_x.columns and rest_x["複勝オッズ_min"].notna().any():
            fuku_odds = rest_x["複勝オッズ_min"].fillna(rest_x["単勝オッズ"] / 3.5)
        else:
            fuku_odds = (rest_x["単勝オッズ"] / 3.5).clip(lower=1.1)
        rest_x["_ana_score"] = rest_x["複勝確率"] * fuku_odds.clip(1.1, 8.0)
        _pop_x = pd.to_numeric(rest_x["人気"], errors="coerce").fillna(99)
        ana_pool = rest_x[_pop_x >= 6]           # 人気薄=穴の条件
        if ana_pool.empty:
            ana_pool = rest_x                     # 該当なければ残り全体から
        batu_idx = ana_pool["_ana_score"].idxmax()
        pdf.at[batu_idx, "推奨ランク"] = "×"
        pdf.at[batu_idx, "印"] = "×"

    # ── 妙味軸（回収率エンジン◎妙）: MF勝率が最上位の馬。◎(place3-1位=馬券内軸)と
    #   別馬のときだけ付与する。◎は堅い馬に寄る一方、MF最上位は市場と別の価値馬を指す。
    #   BT(2025 honest/3144R・B印): 発生1679R(53%)・単勝ROI167.1%・複勝107.8%・
    #   馬単妙→総流し148.0%・◎-妙ワイド114.7%。妙味買いの主役はこの◎妙。
    #   ◎と一致する場合は軸に一本化（=妙味印なし）。
    pdf["妙味軸"] = ""
    if has_mf:
        try:
            _mf_top_idx = pdf["MF勝ち確率"].idxmax()
            if _mf_top_idx != honmei_idx:
                pdf.at[_mf_top_idx, "妙味軸"] = "◎妙"
        except Exception as e:
            print(f"  妙味軸スキップ: {e}")

    # ── 買い指数（購入推奨度・レース単位）──────────────────────────────────
    # レース単位の買い目マトリクス(_race_bet_plan)に完全連動(2026-07-16再較正)。
    # 判定: 勝負(90+)/買い(75+)/少額(55)/見送り(25-40)。詳細は_race_bet_plan参照。
    pdf["買い指数"] = np.nan
    pdf["購入推奨"] = ""
    pdf["想定単回収"] = ""
    pdf["買いサイズ"] = ""
    try:
        _plan = _race_bet_plan(pdf)
        pdf["買い指数"] = _plan["指数"]
        pdf["購入推奨"] = _plan["判定"]
        pdf["想定単回収"] = _plan["理由"]
        pdf["買いサイズ"] = _plan["サイズ"]
    except Exception as e:
        print(f"  買い指数スキップ: {e}")

    # ── レース情報（PDF上に格納して呼び出し元でも使えるように）
    baba_inv = {1: "良", 2: "稍重", 3: "重", 4: "不良"}
    # クラス_num は keiba_auto/scraper と同じ 1-8 スケール（新馬・未勝利=1, 1勝=2, 2勝=3,
    # 3勝=4, OP=5, G3=6, G2=7, G1=8）。以前の cls_inv は 1→新馬,2→未勝利,3→1勝… と
    # 1つズレており、全クラスが1つ下に誤表示されていた。正しい逆マップに修正。
    # ※num=1 は 新馬/未勝利 が同値のため、実クラス名(レースクラス)を優先して区別する。
    cls_inv  = {1: "未勝利", 2: "1勝クラス", 3: "2勝クラス", 4: "3勝クラス",
                5: "オープン", 6: "G3", 7: "G2", 8: "G1"}
    pdf.attrs["jyo_name"] = jyo_name
    pdf.attrs["race_no"]  = race_no
    pdf.attrs["dist"]     = int(pdf["距離"].iloc[0])      if pd.notna(pdf["距離"].iloc[0])      else "不明"
    pdf.attrs["turf"]     = "芝" if pdf["is_turf"].iloc[0] == 1 else "ダート"

    # 馬場・クラスは get_race_data が取得した実文字列(race_df)を最優先で使う。
    # 数値からの逆マップは 新馬↔未勝利 の同値やスケールズレで誤るため。
    # race_df に文字列が無い/未発表(馬場が朝は未確定)のときのみ数値逆マップにフォールバック。
    def _race_str(col):
        if col in race_df.columns:
            v = race_df[col].dropna()
            if len(v) > 0 and str(v.iloc[0]).strip() and str(v.iloc[0]).strip().lower() != "nan":
                return str(v.iloc[0]).strip()
        return None
    _baba_str = _race_str("馬場状態")
    _cls_str  = _race_str("レースクラス")
    pdf.attrs["baba"] = _baba_str or baba_inv.get(
        int(pdf["馬場状態_num"].iloc[0]) if pd.notna(pdf["馬場状態_num"].iloc[0]) else 0, "不明")
    pdf.attrs["cls"]  = _cls_str or cls_inv.get(
        int(pdf["クラス_num"].iloc[0]) if pd.notna(pdf["クラス_num"].iloc[0]) else 0, "不明")

    # ── today_predictions.csv 保存
    _skip_save = False   # オッズ取得失敗時に保存を止めるフラグ（bets保存も連動）
    try:
        save_cols = [
            "race_id", "馬名", "馬番", "枠番",
            "単勝オッズ", "人気",
            "馬体重", "体重増減",
            "勝ち確率", "連対確率", "複勝確率", "3着内確率",
            "単勝期待値", "推奨賭け率",
            "乖離スコア", "MF予測順位", "MF勝ち確率", "MF複勝率", "MF複勝順位",
            "該当戦略", "推奨ランク", "総合スコア", "券種推奨", "妙味軸",
            "買い指数", "購入推奨", "想定単回収", "買いサイズ",
            "予測順位", "連対順位", "複勝順位",
            "過去勝率", "過去出走数", "前走間隔",
        ]
        save_cols = [c for c in save_cols if c in pdf.columns]
        save_df   = pdf[save_cols].copy()

        # ── 能力値パラメータ（レース内百分位 0-100・ダッシュボード表示用）──
        #   各馬の特徴を6軸で相対評価: 勝負力/安定感/末脚/先行力/距離適性/実績
        def _pct(col, invert=False):
            if col not in pdf.columns:
                return pd.Series(np.nan, index=pdf.index)
            s = pd.to_numeric(pdf[col], errors="coerce")
            if s.notna().sum() < 2:
                return pd.Series(np.nan, index=pdf.index)
            return (s.rank(pct=True, ascending=not invert) * 100).round(0)
        save_df["能力_勝負"]   = _pct("勝ち確率")                      # 勝ち切る力
        save_df["能力_安定"]   = _pct("複勝確率")                      # 馬券内の堅さ
        save_df["能力_末脚"]   = _pct("過去最速上り", invert=True)     # 上がりは小さいほど速い
        save_df["能力_先行"]   = _pct("過去平均先行指数")              # 前に行く力
        save_df["能力_距離"]   = _pct("同距離過去平均着順", invert=True)  # 今回距離への適性
        save_df["能力_実績"]   = _pct("過去勝率")                      # 地力・実績

        save_df["jyo"]      = jyo_name
        save_df["race_no"]  = race_no
        save_df["距離"]     = pdf.attrs["dist"]
        save_df["馬場"]     = pdf.attrs["turf"]
        save_df["馬場状態"] = pdf.attrs["baba"]
        save_df["クラス"]   = pdf.attrs["cls"]
        save_df["予想日時"] = datetime.now().strftime("%Y/%m/%d %H:%M")

        out_path = os.path.join(BASE_DIR, "today_predictions.csv")
        _skip_save = False
        if os.path.exists(out_path):
            existing = read_pred_csv(out_path)
            _same = existing[existing["race_id"] == str(race_id)]
            # 安全網(2026-07-20): 今回オッズが全馬NaN（取得失敗）で、既存にオッズ付き予想が
            # ある場合は上書きしない。オッズ無し判定は必ず「見送り」誤判定になるため。
            if save_df["単勝オッズ"].isna().all() and len(_same) and _same["単勝オッズ"].notna().any():
                print("  ⚠ オッズ取得失敗（全馬NaN）→ 既存のオッズ付き予想を保持し保存スキップ")
                _skip_save = True
            existing = existing[existing["race_id"] != str(race_id)]
            save_df  = pd.concat([existing, save_df], ignore_index=True)
        if not _skip_save:
            # 安全網: 同一レース×馬の重複行を除去（並行書き込み事故等でも最新のみ残す）
            save_df = save_df.drop_duplicates(subset=["race_id", "馬名"], keep="last")
            save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  予想データ保存完了 → {out_path}")
    except Exception as e:
        print(f"  予想データ保存エラー: {e}")

    # ── today_bets.csv 保存（推奨買い目を馬番展開して記録）─────────────────
    #   analyze_accuracy.py が実払戻(payout_scraper)と照合し、買い目単位のROIを日次検証する。
    try:
        bets_path = os.path.join(BASE_DIR, "today_bets.csv")
        if _skip_save:
            # オッズ取得失敗時は買い目も触らない（既存のオッズ付き買い目を保持）
            print("  ⚠ オッズ取得失敗 → today_bets.csv も更新スキップ")
        else:
            _bets = _build_bet_rows(pdf, race_id)
            if _bets:
                bets_df = pd.DataFrame(_bets)
                bets_df["jyo"] = jyo_name
                bets_df["race_no"] = race_no
                bets_df["予想日時"] = datetime.now().strftime("%Y/%m/%d %H:%M")
                if os.path.exists(bets_path):
                    _eb = read_pred_csv(bets_path)
                    _eb = _eb[_eb["race_id"] != str(race_id)]
                    bets_df = pd.concat([_eb, bets_df], ignore_index=True)
                # 安全網: 同一レース×同一買い目の重複を除去（最新のみ残す）
                bets_df = bets_df.drop_duplicates(subset=["race_id", "券種", "組み合わせ"], keep="last")
                bets_df.to_csv(bets_path, index=False, encoding="utf-8-sig")
                print(f"  買い目データ保存完了 → {bets_path}（{len(_bets)}点）")
            else:
                # 正当な見送り（オッズあり）→ このレースの既存買い目を削除して整合させる
                if os.path.exists(bets_path):
                    _eb = read_pred_csv(bets_path)
                    _before = len(_eb)
                    _eb = _eb[_eb["race_id"] != str(race_id)]
                    if len(_eb) != _before:
                        _eb.to_csv(bets_path, index=False, encoding="utf-8-sig")
                        print("  見送りに変更 → 既存買い目を削除")
                    else:
                        print("  買い目なし（見送りレース）→ today_bets.csv 追記なし")
                else:
                    print("  買い目なし（見送りレース）→ today_bets.csv 追記なし")
    except Exception as e:
        print(f"  買い目保存エラー: {e}")

    # オッズ変動特徴の材料を時系列で蓄積（朝/40分前/直前の各実行ぶんが貯まる）
    record_odds_snapshot(pdf, race_id)

    return pdf


# ── 購入しきい値（2026-07-17）─────────────────────────────────────────────
#   買い指数がこの値未満のレースは「買い目を出さない」（判定表示はするが購入対象外）。
#   目安: 55=少額も買う / 70=買い・勝負・堅実 / 85=勝負のみ
#   today_bets.csv・メール・ダッシュボードの買い目表示すべてに連動する。
BUY_INDEX_MIN = 70

# ── 資金設定（2026-07-17・自動投票対応の下地）────────────────────────────
#   today_bets.csv に1点ごとの「金額」列が出力される（将来の自動投票はこれを注文リストとして読む）。
#   照合(analyze_accuracy)も金額加重の実収支で集計される。
BET_UNIT = 100                    # 基本1点あたりの金額(円)・KIND_STAKE未定義券種のフォールバック
# ── 予算配分方式（2026-07-21）────────────────────────────────────────────
#   各レースで「予算内に収まるよう買い目を選び・券種ごとに掛け金を配分」する。
#   仕組み: 1点の金額 = KIND_STAKE[券種] × SIZE_WEIGHT[サイズ] → 100円丸め・最低100円。
#           合計が RACE_BUDGET[判定帯] を超えたらBT回収率の低い買い方から丸ごと削る。
#   例: 勝負帯で単勝500×1 + 馬単100×5 → 予算内なら両方、超えれば弱い連系から自動除外。
RACE_BUDGET = {                   # 帯別の1レース予算(円)。帯の強さ＝予算配分で表現(旧BAND_WEIGHT代替)
    "勝負": 2000, "買い": 1500, "堅実": 1000, "少額": 500,
}
# 予算超過時に「どの買い目を残すか」の優先度。2025BT(予算配分込み):
#   "balanced" = メニュー順(単勝→複勝→ワイド→…)。複勝/ワイドも買い分散・的中安定。回収198.5%・的中1113。
#   "ev"       = 単勝を軸に確保→残りBT回収率順。馬単/3連単に集中し高EVだが変動大。回収221.6%・的中591。
ALLOC_PRIORITY = "balanced"       # 高EV集中に切替えるなら "ev"
KIND_STAKE = {                    # 券種ごとの1点あたり基本額(円)。単勝を厚く・多点の連系を薄く
    "単勝": 500, "複勝": 300, "ワイド": 200,
    "馬連": 100, "馬単": 100, "3連複": 100, "3連単": 100,
}
SIZE_WEIGHT = {                   # MF自信度サイズの倍率（1点の掛け金を確信度で厚薄）
    # 確信度シグナルはここ(掛け金)で反映。予算側にも効かせる案は2025BTで効果ゼロと確認済み(2026-07-22)。
    "厚め": 1.6, "標準": 1.0, "薄め": 0.6,
}
BAND_WEIGHT = {}                  # 旧方式の残置(未使用)。帯別配分は RACE_BUDGET に移行済み
KIND_UNIT = {}                    # 券種別の金額“固定”上書き(円)。指定券種はSIZE倍率も無視して固定額
KIND_SKIP = []                    # 買わない券種 例: ["3連単"] で点数を大幅削減
RACE_BUDGET_MAX = None            # (旧)全帯共通の上限。設定時は RACE_BUDGET より優先
#   例: RACE_BUDGET_MAX=2000 → 1日約3.8万円上限 / KIND_SKIP=["3連単"] → 1日約270点に減


def _race_bet_plan(pdf):
    """レース条件（クラス×妙人気帯×MF自信度×頭数）から買い目プランを決める共有ロジック。
    2025実払戻BT(2026-07-16確定)。買い指数・レポート・today_bets・ダッシュボードが全てこれに連動。
    BUY_INDEX_MIN 未満の判定は menu を空にして購入対象から外す。

    判定マトリクス（妙=◎妙の人気帯 × クラス）:
      妙7-9人気 → 全クラスで最強帯(単BT175-315%) → 勝負
      妙4-6人気 → 下級/中級で買い(BT139-193%)。OP級はNG(59-94%)
      妙2-3人気 → 下級のみ少額(単BT126%)。中級/OP級はNG
      妙10人気-/妙なし/OP14頭以上 → 見送り
    サイズ: 妙のMF勝率 >=0.385(2025上位四分位)=厚め(単BT232%) / <0.20=薄め(105%)
    頭数: 17頭以上はワイド除外(BT84%) / 10頭以下は総流し系が特に強い(224%)
    """
    n = len(pdf)
    cls_v = pd.to_numeric(pdf["クラス_num"].iloc[0], errors="coerce") if "クラス_num" in pdf.columns else np.nan
    cls = int(cls_v) if pd.notna(cls_v) else 3
    is_op = cls >= 5
    plan = {"判定": "見送り", "指数": 25, "サイズ": "-", "理由": "", "menu": []}

    if is_op and n >= 14:
        plan["理由"] = "重賞/OP級14頭以上(AI優位なし・全買い目BT<100%)"
        return plan
    myo = pdf[pdf["妙味軸"] == "◎妙"] if "妙味軸" in pdf.columns else pdf.iloc[0:0]
    _pop_all2 = pd.to_numeric(pdf["人気"], errors="coerce")
    _fav_odds2 = np.nan
    if _pop_all2.notna().any():
        _fav_odds2 = pd.to_numeric(pdf.loc[_pop_all2.idxmin(), "単勝オッズ"], errors="coerce")
    if myo.empty:
        # ◎妙なし＝MF1位と◎が一致（両モデル合意）。市場が確信していない(1人気>2.0倍)
        # 非OPレースなら「堅実」帯: ◎軸で買う（BT: 単勝135%/馬単145%/3連複123%/3連単153%・887R）
        if pd.notna(_fav_odds2) and float(_fav_odds2) > 2.0 and cls <= 4:
            plan.update({"判定": "堅実", "指数": 75, "サイズ": "標準",
                         "理由": "モデル合意◎×市場不確実(1人気2倍超)＝堅実帯(BT≈140%)"})
            plan["menu"] = [("単勝", "◎単勝", 135),
                            ("馬単", "馬単 ◎→○▲△", 145),
                            ("3連複", "3連複 ◎○▲", 123),
                            ("3連単", "3連単 ◎→○▲→○▲△", 153)]
            return plan
        plan.update({"指数": 40, "理由": "◎妙なし×(1人気2倍以下orOP)=旨味なし(BT75-87%)"})
        return plan
    m = myo.iloc[0]
    pop = pd.to_numeric(m.get("人気"), errors="coerce")
    mfw = pd.to_numeric(m.get("MF勝ち確率"), errors="coerce")
    modds = pd.to_numeric(m.get("単勝オッズ"), errors="coerce")
    if pd.isna(pop) or pop >= 10:
        plan["理由"] = "妙10番人気以下(運任せ・BT99%)"
        return plan
    if is_op and pop <= 6:
        plan["理由"] = "OP級×妙6人気以内(BT59-94%)"
        return plan
    if 3 <= cls <= 4 and pop <= 3:
        plan["理由"] = "中級×妙2-3人気(BT86-109%)"
        return plan
    # 買い帯オッズゲート: 4-6人気でもオッズ8倍未満は市場評価と近く妙味なし(馬単BT59%)
    if 4 <= pop <= 6 and pd.notna(modds) and float(modds) < 8.0:
        plan.update({"指数": 50,
                     "理由": f"妙{int(pop)}人気だがオッズ{float(modds):.1f}倍<8倍=妙味薄(馬単BT59%)"})
        return plan

    size = "標準"
    if pd.notna(mfw) and float(mfw) >= 0.385:
        size = "厚め"
    elif pd.notna(mfw) and float(mfw) < 0.20:
        size = "薄め"
    wide_ok = n <= 16
    pop = int(pop)

    # メニューは betting_guide.md（2026-07-17確定・券種別最適構成）に完全準拠。
    # 相手は「人気」でレースごとに動的に決まる＝レース単位の最適化。
    if pop <= 3:      # ここに来るのは下級×2-3人気のみ
        plan.update({"判定": "少額", "指数": 55, "サイズ": "薄め",
                     "理由": f"下級×妙{pop}人気(単勝BT126%のみの薄利帯)"})
        plan["menu"] = [("単勝", "妙単勝", 126), ("馬単", "馬単 妙→◎○▲", 120)]
    elif pop <= 6:    # 買い帯: 下級/中級×妙4-6人気×オッズ8倍以上（ゲート後BT: 単勝183/馬単231）
        plan.update({"判定": "買い", "指数": 75 + (10 if size == "厚め" else 0), "サイズ": size,
                     "理由": f"{'下級' if cls <= 2 else '中級'}×妙{pop}人気×{float(modds):.0f}倍(買い帯)" if pd.notna(modds)
                              else f"{'下級' if cls <= 2 else '中級'}×妙{pop}人気(買い帯)"})
        plan["menu"] = [("単勝", "妙単勝", 183),
                        ("馬単", "馬単 妙→複勝上位5", 241),
                        ("馬連", "馬連 妙-複勝上位5", 171),
                        ("3連複", "3連複 妙◎軸-複勝上位5", 132),
                        ("3連単", "3連単 妙→複勝3→複勝5", 259)]
    else:             # 勝負帯: 妙7-9人気（BT: 単勝291/馬単345）
        plan.update({"判定": "勝負", "指数": 90 + (5 if size == "厚め" else 0), "サイズ": size,
                     "理由": f"妙{pop}人気(勝負帯・全クラス最強)"})
        plan["menu"] = [("単勝", "妙単勝", 291),
                        ("複勝", "妙複勝", 145),
                        ("ワイド", "ワイド 複妙-複勝上位3", 147),
                        ("馬単", "馬単 妙→複勝上位6", 337),
                        ("馬連", "馬連 妙-複勝上位6", 194),
                        ("3連複", "3連複 妙◎軸-複勝上位5", 131),
                        ("3連単", "3連単 妙◎軸マルチ上位5", 251)]

    # 購入しきい値: 指数がBUY_INDEX_MIN未満なら買い目を出さない（判定・理由は残す）
    if plan["menu"] and plan["指数"] < BUY_INDEX_MIN:
        plan["理由"] += f" → 指数{plan['指数']}<購入閾値{BUY_INDEX_MIN}のため購入対象外"
        plan["menu"] = []
    return plan


# ── 市場の歪み補正（2026-07-28導入）────────────────────────────────────────
# オッズとレース文脈だけから「そのレースで過剰人気になっている馬」を判定する。
# 2025検証(<=2024学習): 全馬の単勝ROI 72.25%→82.34% / 複勝 71.56%→81.40%
#   （補正係数の下位を除外した場合。固定の「100倍超を除外」78.83%より良い）
# 歪みは大穴に集中: 補正係数は30倍以下=0.94〜1.05 / 100-300倍=0.68 / 300倍超=0.32。
# ※逆（過小評価馬を買う）は期間を変えると再現しないので採用しない＝守りにのみ使う。
DEBIAS_MIN = 0.90        # これ未満の馬は買い目から除外。1.0に近づけるほど厳しい
_DEBIAS = None


def _get_debias():
    global _DEBIAS
    if _DEBIAS is None:
        p = os.path.join(BASE_DIR, "debias_model.pkl")
        try:
            with open(p, "rb") as f:
                _DEBIAS = pickle.load(f)
        except Exception as e:
            print(f"  [歪み補正] 読込不可のため除外なしで継続: {e}")
            _DEBIAS = {}
    return _DEBIAS


def excluded_horses(pdf, race_id):
    """買ってはいけない（過剰人気の）馬番の集合を返す。判定不能なら空集合。"""
    d = _get_debias()
    if not d:
        return set()
    try:
        from train_debias import apply_debias
        res = apply_debias(d["model"], pdf.assign(race_id=str(race_id)))
        bias = pd.to_numeric(res["bias"], errors="coerce")
        no = pd.to_numeric(pdf["馬番"], errors="coerce")
        ng = {int(n) for n, b in zip(no, bias)
              if pd.notna(n) and pd.notna(b) and b < DEBIAS_MIN}
        return ng
    except Exception as e:
        print(f"  [歪み補正] 判定スキップ: {e}")
        return set()


def _build_bet_rows(pdf, race_id):
    """_race_bet_plan のメニューを馬番展開して行リストで返す（today_bets.csv用）。
    馬単は着順順序つき(1着-2着)、馬連/ワイドはソート済み表記。
    2026-07-28〜: 市場の歪み補正で「過剰人気」と判定された馬を含む買い目は落とす。"""
    plan = _race_bet_plan(pdf)
    if not plan["menu"]:
        return []
    _ng = excluded_horses(pdf, race_id)

    def _no(row):
        v = pd.to_numeric(row["馬番"], errors="coerce")
        return int(v) if pd.notna(v) else None

    marks = {}
    for mk in ("◎", "○", "▲", "△"):
        r = pdf[pdf["印"] == mk]
        if len(r):
            no = _no(r.iloc[0])
            if no is not None:
                marks[mk] = no
    r = pdf[pdf["妙味軸"] == "◎妙"]
    if len(r):
        no = _no(r.iloc[0])
        if no is not None:
            marks["妙"] = no
    # 複勝妙軸(MF複勝順位1位): place系(複勝/ワイド/3連複)の軸に使う。
    # 判定帯は勝率妙のまま。勝率軸より馬券内率・ROIが高い(2025BT実証)。
    if "MF複勝順位" in pdf.columns and pdf["MF複勝順位"].notna().any():
        _rp = pdf[pd.to_numeric(pdf["MF複勝順位"], errors="coerce") == 1]
        if len(_rp):
            no = _no(_rp.iloc[0])
            if no is not None:
                marks["複妙"] = no
    if "◎" not in marks:
        return []
    hon = marks["◎"]
    myo = marks.get("妙")     # 堅実帯は妙なし（◎軸）
    myo_p = marks.get("複妙", myo)   # place系の軸（無ければ勝率妙にフォールバック）

    # 人気順の馬番リスト（妙を除く）・馬番→人気/オッズのマップ（相手の動的決定に使用）
    _pn = pdf.dropna(subset=["馬番"]).copy()
    _pn["_pop"] = pd.to_numeric(_pn["人気"], errors="coerce")
    _pn["_odds"] = pd.to_numeric(_pn["単勝オッズ"], errors="coerce")
    _pn = _pn.dropna(subset=["_pop"]).sort_values("_pop")
    pop_order = [int(x) for x in _pn["馬番"] if myo is None or int(x) != myo]
    pop_of = {int(r2["馬番"]): int(r2["_pop"]) for _, r2 in _pn.iterrows()}
    odds_of = {int(r2["馬番"]): (float(r2["_odds"]) if pd.notna(r2["_odds"]) else 999)
               for _, r2 in _pn.iterrows()}
    # MF複勝順位で並べた馬番リスト（相手選定用）。2025BTで「人気」より良い相手を選べる:
    #   馬連139.5→170% / 馬単221→261% / 3連単152→259% / 3連複113→131%（乱数2分割で頑健）。
    if "MF複勝順位" in _pn.columns and _pn["MF複勝順位"].notna().any():
        _mp = _pn.dropna(subset=["MF複勝順位"]).sort_values("MF複勝順位")
        mfp_order = [int(x) for x in _mp["馬番"]]
    else:
        mfp_order = pop_order   # MF複勝が無ければ人気順にフォールバック

    def _mf_partners(exclude, n):
        """MF複勝順位の上位から、exclude(軸等)を除いてn頭返す。"""
        return [t for t in mfp_order if t not in exclude][:n]

    def _tail_n(label, default=5):
        """ラベル末尾の数字を相手点数として取り出す（例 '…複勝上位6'→6）。"""
        tail = ""
        for ch in reversed(label):
            if ch.isdigit():
                tail = ch + tail
            else:
                break
        return int(tail) if tail else default

    rows = []

    # 1点の金額 = KIND_STAKE[券種] × SIZE_WEIGHT[サイズ]（100円丸め・最低100円）。
    # KIND_UNITに指定があればそれを固定額として最優先（サイズ倍率も無視）。
    _size_mult = SIZE_WEIGHT.get(plan["サイズ"], 1.0)

    def add(kind, name, combo, roi):
        if kind in KIND_SKIP:
            return
        if kind in KIND_UNIT:
            amt = KIND_UNIT[kind]
        else:
            amt = KIND_STAKE.get(kind, BET_UNIT) * _size_mult
        amt = max(100, int(round(amt / 100.0)) * 100)
        # 市場の歪み補正: 過剰人気と判定された馬を含む買い目は落とす
        if _ng and any(int(x) in _ng for x in str(combo).split("-") if x.isdigit()):
            return
        rows.append({"race_id": str(race_id), "券種": kind, "買い方": name,
                     "組み合わせ": combo, "BT回収率": roi,
                     "判定": plan["判定"], "サイズ": plan["サイズ"], "金額": amt})

    def s2(a, b):
        return f"{min(a, b):02d}-{max(a, b):02d}"

    for kind, name, roi in plan["menu"]:
        # ── 堅実帯（◎軸・妙なし）──
        if name == "◎単勝":
            add(kind, name, f"{hon:02d}", roi)
        elif name == "馬単 ◎→○▲△":
            for t in [marks[m] for m in ("○", "▲", "△") if m in marks and marks[m] != hon]:
                add(kind, name, f"{hon:02d}-{t:02d}", roi)
        elif name == "3連複 ◎○▲":
            tri = [marks[m] for m in ("◎", "○", "▲") if m in marks]
            if len(set(tri)) == 3:
                x = sorted(tri)
                add(kind, name, f"{x[0]:02d}-{x[1]:02d}-{x[2]:02d}", roi)
        elif name == "3連単 ◎→○▲→○▲△":
            a2 = [marks[m] for m in ("○", "▲") if m in marks and marks[m] != hon]
            b3 = [marks[m] for m in ("○", "▲", "△") if m in marks and marks[m] != hon]
            for a in a2:
                for b in b3:
                    if a != b:
                        add(kind, name, f"{hon:02d}-{a:02d}-{b:02d}", roi)
        # ── 妙軸帯 ──
        elif myo is None:
            continue
        elif name == "妙単勝":
            add(kind, name, f"{myo:02d}", roi)
        elif name == "妙複勝":            # place系: 複勝妙軸
            add(kind, name, f"{myo_p:02d}", roi)
        elif name == "ワイド 妙-◎":       # place系: 複勝妙軸（旧・1点）
            if hon != myo_p:
                add(kind, name, s2(myo_p, hon), roi)
        elif name.startswith("ワイド 複妙-複勝上位"):  # 複妙軸 + MF複勝上位N（勝負帯拡張）
            for t in _mf_partners({myo_p}, _tail_n(name)):
                add(kind, name, s2(myo_p, t), roi)
        elif name == "馬単 妙→◎○▲":
            for t in [marks[m] for m in ("◎", "○", "▲") if m in marks and marks[m] != myo]:
                add(kind, name, f"{myo:02d}-{t:02d}", roi)
        elif name.startswith("馬単 妙→複勝上位"):  # 相手=MF複勝上位N（帯別に点数可変）
            for t in _mf_partners({myo}, _tail_n(name)):
                add(kind, name, f"{myo:02d}-{t:02d}", roi)
        elif name.startswith("馬連 妙-複勝上位"):   # 相手=MF複勝上位N
            for t in _mf_partners({myo}, _tail_n(name)):
                add(kind, name, s2(myo, t), roi)
        elif name.startswith("3連複 妙◎軸-複勝上位"):  # place系: 複勝妙軸 + MF複勝相手N
            if myo_p != hon:
                for t in _mf_partners({myo_p, hon}, _tail_n(name)):
                    x = sorted((myo_p, hon, t))
                    add(kind, name, f"{x[0]:02d}-{x[1]:02d}-{x[2]:02d}", roi)
        elif name == "3連単 妙→複勝3→複勝5":     # 2着=MF複勝上位3 / 3着=MF複勝上位5
            cand = _mf_partners({myo}, 5)
            for a in cand[:3]:
                for b in cand[:5]:
                    if a != b:
                        add(kind, name, f"{myo:02d}-{a:02d}-{b:02d}", roi)
        elif name == "3連単 妙◎軸マルチ上位5":
            # 妙と◎の2頭軸マルチ（両頭が3着内の全着順×相手=人気上位5から1頭）
            from itertools import permutations as _perm
            seen = set()
            for t in pop_order[:5]:
                if t in (myo, hon):
                    continue
                for p3o in _perm((myo, hon, t)):
                    if p3o not in seen:
                        seen.add(p3o)
                        add(kind, name, f"{p3o[0]:02d}-{p3o[1]:02d}-{p3o[2]:02d}", roi)

    # ── 1レース予算に収める（ALLOC_PRIORITYで充当順を切替）──
    #   予算 = RACE_BUDGET_MAX(全帯共通・設定時優先) or RACE_BUDGET[判定帯]。
    #   "balanced": メニュー順(単勝→複勝→ワイド→…)のまま＝複勝/ワイドも残り分散・的中安定(2025 198.5%)。
    #   "ev"      : 単勝を軸に確保→残りBT回収率順＝馬単/3連単に集中し高EV・変動大(2025 221.6%)。
    #   どちらも各連系内は相手=MF複勝上位=良い順(安定ソートで保持)。予算超過分は落とす。
    _budget = RACE_BUDGET_MAX if RACE_BUDGET_MAX else RACE_BUDGET.get(plan["判定"])
    if _budget and rows:
        if ALLOC_PRIORITY == "ev":
            rows = sorted(rows, key=lambda r: (0 if r["券種"] == "単勝" else 1, -r["BT回収率"]))
        kept, total = [], 0
        for r in rows:
            if total + r["金額"] <= _budget:
                kept.append(r)
                total += r["金額"]
        rows = kept
    return rows


# ── メイン ────────────────────────────────────────────────────────────────
def predict_race(race_id: str, send_mail: bool = True):
    race_id = str(race_id).strip()
    if len(race_id) != 12 or not race_id.isdigit():
        print("エラー: race_id は12桁の数字で指定してください（例: 202606050811）")
        sys.exit(1)

    # ── 前日レースの再予想スキップ（スケジューラがschedule.every().day.at()で前日の
    #   各レース予想ジョブを翌日も再発火し、前日のrace_idを予想・メールしてしまう問題への防御）。
    #   today_predictions.csv 内で、この競馬場(race_id 5-6桁)の最新開催日(9-10桁)より
    #   古い開催日のレースは前日以前の残骸なので、予想・メール送信せずスキップする。
    try:
        _tp = os.path.join(BASE_DIR, "today_predictions.csv")
        if os.path.exists(_tp):
            _rid = read_pred_csv(_tp, usecols=["race_id"])["race_id"]
            _rid = _rid[_rid.str.fullmatch(r"\d{12}")]   # 'nan'等の不正値を除外
            _same = _rid[_rid.str[4:6] == race_id[4:6]]
            if len(_same) > 0:
                _maxday = _same.str[8:10].max()
                if race_id[8:10] < _maxday:
                    print(f"  [スキップ] {race_id} は前日以前のレース"
                          f"(開催{race_id[8:10]}日目 < 最新{_maxday}日目)。予想・メール送信せず終了。")
                    return
    except Exception:
        pass  # 判定不能時は通常通り予想する（安全側）

    # ── モデル読み込み → models_pack に統一
    print("モデル読み込み中...")
    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        saved = pickle.load(f)
    win_models = saved.get("win", {}).get("models", saved["models"])
    win_cols   = saved.get("win", {}).get("use_cols", saved["use_cols"])
    win_weights = saved.get("win", {}).get("weights", saved.get("weights"))
    is_multi   = saved.get("format") == "multi_v1" and saved.get("place2") and saved.get("place3")
    models_pack = {
        "win":    {"models": win_models, "use_cols": win_cols, "weights": win_weights},
        "place2": saved["place2"] if is_multi else None,
        "place3": saved["place3"] if is_multi else None,
    }
    print("  3モデル構成 → 独立予想を使用" if is_multi else "  旧モデル構成 → ハーヴィル変換を使用")

    # 2026-07-27〜: model_mf_parts/があれば逐次読込(ピークRAM激減)、無ければ従来pkl。
    import mf_model_io
    if mf_model_io.exists(BASE_DIR):
        try:
            mf_saved = mf_model_io.load_mf(BASE_DIR)
            models_pack["mf"] = mf_saved
            print("  市場フリーモデル読み込み完了")
        except Exception as e:
            print(f"  市場フリーモデルスキップ: {e}")
            models_pack["mf"] = None
    else:
        models_pack["mf"] = None

    # ── 履歴データ読み込み
    print("履歴データ読み込み中...")
    history_df = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"), low_memory=False)
    print(f"  読み込み完了: {len(history_df)}行")

    # ── 予測コア（共通エンジン）
    pdf = predict_race_pdf(race_id, history_df=history_df, models_pack=models_pack)
    if pdf is None:
        return

    # ── 詳細レポート生成・表示・送信
    jyo_name = pdf.attrs["jyo_name"]
    race_no  = pdf.attrs["race_no"]
    dist     = pdf.attrs["dist"]
    turf     = pdf.attrs["turf"]
    baba     = pdf.attrs["baba"]
    cls      = pdf.attrs["cls"]

    report = build_report(pdf, race_id, jyo_name, race_no, dist, turf, baba, cls, len(pdf))
    print("\n" + report)
    if send_mail:
        subject = f"【競馬AI詳細予想】{jyo_name} {race_no}R"
        print("\nメール送信中...")
        send_email(subject, report)
    else:
        print("\n（詳細予想メールは送信しません）")


def _run_predict_safe(race_id, send_mail=True):
    try:
        predict_race(race_id, send_mail=send_mail)
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
            _run_predict_safe(rid, send_mail=False)   # 朝一括はメール送信しない（36通の連投を防ぐ）
        print("\n全レース予想完了 → today_predictions.csv")
    else:
        # 個別レース予想（auto_predict_publish.py の発走40分前実行から呼び出される）
        # 引数に "nomail" があればメール送信しない（40分前ジョブが指定・手動実行は送る）。
        race_id = sys.argv[1] if len(sys.argv) > 1 else TARGET_RACE_ID
        predict_race(race_id, send_mail=("nomail" not in sys.argv))