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


def _mark_fallback(reason):
    """MFが読めず通常モデルに落ちたことを記録する（2026-08-11追加）。

    通常モデル(model.pkl・511MB)は現在フォールバック専用で、印にも買い判定にも
    使われていない。半年間このフラグが一度も立たなければ撤去できる、という
    判断材料にする。同時に、MFが壊れていることに気づく手段でもある
    （フォールバックしても予想自体は出てしまうので、黙って劣化する）。
    """
    try:
        with open(os.path.join(BASE_DIR, "fallback_triggered.flag"), "a",
                  encoding="utf-8") as f:
            f.write("{}\t{}\n".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), reason))
        print(f"  ⚠ 通常モデルにフォールバックしました: {reason}")
    except Exception:
        pass


def record_odds_snapshot(pdf, race_id):
    """予想実行の都度、現時点のオッズ/人気を odds_history.csv に追記（重複除去せず時系列蓄積）。
    朝・発走40分前・直前の各実行で自然に貯まり、後日「オッズ変動（金の流れ）」特徴の材料になる。
    新規スクレイピングはせず既に取得済みのpdfのオッズを使う（IP負荷ゼロ）。記録失敗で予想は止めない。"""
    try:
        # ⚠ 2026-08-15: 列を「あるものだけ」で組むと、複勝オッズ_minが取れない日に
        #   列数が9になり、既存の10列ヘッダとずれて全行が読めなくなった
        #   （記録時刻が複勝オッズ_minの位置に入り、pandasが解釈を誤る）。
        #   追記型のCSVなので、列は毎回同じ数・同じ順で書かなければならない。
        #   無い列は NaN で埋めて、必ず固定の並びにする。
        FIXED = ("馬番", "馬名", "単勝オッズ", "人気", "複勝オッズ_min")
        if "単勝オッズ" not in pdf.columns or pdf["単勝オッズ"].isna().all():
            return  # オッズ未取得なら記録しない（NaN行で汚さない）
        snap = pdf.reindex(columns=list(FIXED)).copy()
        snap.insert(0, "race_id", str(race_id))
        now = datetime.now()
        snap["記録時刻"] = now.strftime("%Y/%m/%d %H:%M:%S")

        # 発走時刻と「何分前か」を一緒に残す（2026-08-11追加）。
        #   これが無いと、後から「発走30分前→10分前で何%動いたか」を計算できない。
        #   発走時刻は today_race_times.json にあるが毎朝上書きされるため、
        #   ここで各行に埋め込んでおかないと時間軸の原点が永久に失われる。
        #   1年後にオッズ変動を特徴量にするとき、これが無いと下落スピードを作れない。
        mins = post = ""
        try:
            import json
            _p = os.path.join(BASE_DIR, "today_race_times.json")
            if os.path.exists(_p):
                with open(_p, encoding="utf-8") as f:
                    _t = json.load(f)
                v = _t.get(str(race_id))
                if v is not None:
                    post = f"{int(v) // 60:02d}:{int(v) % 60:02d}"
                    mins = int(v) - (now.hour * 60 + now.minute)
        except Exception:
            pass
        snap["発走時刻"] = post
        snap["分前"] = mins
        # どのジョブが記録したか。朝・40分前・7分前で性質が違うため区別する。
        # 分前が負（＝発走後）や不明のときは「不明」にする。誤ったラベルを
        # 付けると、後の分析で時間軸を取り違える。
        # 2026-08-14: 「締切前」を追加。7分前と締切直前を区別できないと、
        # 「投票を遅らせればスリッページがどれだけ縮むか」が測れない。
        # EV方式は確定オッズ基準119.6% / 7分前88.4%で、差の原因は賭ける時刻。
        if mins == "":
            snap["ジョブ"] = "不明"
        elif mins < 0:
            snap["ジョブ"] = "発走後"
        elif mins > 60:
            snap["ジョブ"] = "朝"
        elif mins > 20:
            snap["ジョブ"] = "40分前"
        elif mins > 4:
            snap["ジョブ"] = "直前"
        else:
            snap["ジョブ"] = "締切前"
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


# ── 確率の較正（2026-08-04追加）────────────────────────────────────────
#   MFモデルは正例重み(win 2.0 / place3 1.5)と時間重みをかけて学習しているため、
#   出てくる値は確率として過大。2025 OOSの実測:
#       勝率   予測11.0% → 実際 7.2%（1.5倍過大）
#       複勝率 予測26.8% → 実際21.5%（1.25倍過大）
#   期待値も推奨賭け率(ケリー)も「確率が正しい」ことが前提なので、
#   較正しないと表示も買い判断もすべてずれる。
#   較正器は build_calibrator.py が model_mf_result.csv（backtestモードの
#   正直なOOS出力）から作る。週次でモデルを更新したら作り直すこと。
_CALIBRATOR = None
_CALIBRATOR_LOADED = False


def _calibrate(values, target: str):
    """較正器があれば適用する。無ければ生値をそのまま返す（予測は止めない）。"""
    global _CALIBRATOR, _CALIBRATOR_LOADED
    if not _CALIBRATOR_LOADED:
        _CALIBRATOR_LOADED = True
        path = os.path.join(BASE_DIR, "mf_calibrator.pkl")
        try:
            import pickle
            with open(path, "rb") as fh:
                _CALIBRATOR = pickle.load(fh).get("cal", {})
            print(f"  確率較正器を読込: {len(_CALIBRATOR)}本")
        except FileNotFoundError:
            print("  確率較正器なし（生の確率を使用。build_calibrator.pyで作成できます）")
        except Exception as e:
            print(f"  確率較正器の読込に失敗（生の確率を使用）: {e}")
    if not _CALIBRATOR or target not in _CALIBRATOR:
        return values
    try:
        return np.clip(_CALIBRATOR[target].predict(np.asarray(values, dtype=float)),
                       0.0, 1.0)
    except Exception as e:
        print(f"  較正の適用に失敗（生の確率を使用）: {e}")
        return values


# ── 2次元較正（購入判定専用）──────────────────────────────────────────────
#   なぜ必要か（2026-08-12）
#     5年 walk-forward OOF 207,518頭で、EVと実払戻の順位相関が5年とも負だった。
#     EVが高い馬ほど損をする。EV = p × オッズ なので、オッズが大きい側でpが
#     過大だとEVは二重に膨らみ、そこだけを選ぶと誤差の上側の裾を集めてしまう。
#     本番ルール該当馬は 予測18.0% / 実勝率6.8%、実効EV 0.95で赤字が確定していた。
#     上の1次元Isotonicは全馬まとめた較正なのでこれを直せない。
#
#   ⚠ この値を印・★・順位・S〜D評価に使ってはいけない。
#     市場に約9〜33倍の重みが付くため、MF順位が人気順のコピーになる
#     （順位相関0.9924）。5年で★が4,507頭→19頭に消え、市場フリーという
#     設計そのものが壊れる。用途は「購入判定と実効EVの表示」だけ。
_CALIB2D = None
_CALIB2D_LOADED = False


def _calibrate2d(model_p, odds):
    """市場確率を織り込んだ勝率を返す。較正器もオッズも無ければ None。

    model_p はレース内で正規化済みのMF勝率、odds は単勝オッズ。
    戻り値もレース内で正規化して返す。
    """
    global _CALIB2D, _CALIB2D_LOADED
    if not _CALIB2D_LOADED:
        _CALIB2D_LOADED = True
        try:
            import pickle
            with open(os.path.join(BASE_DIR, "mf_calib2d.pkl"), "rb") as fh:
                _CALIB2D = pickle.load(fh)
            print("  2次元較正器を読込（{}〜{} / {:,}頭）".format(
                _CALIB2D["years"][0], _CALIB2D["years"][-1], _CALIB2D["n"]))
        except FileNotFoundError:
            print("  2次元較正器なし（実効EVは出しません。build_calib2d.pyで作成できます）")
        except Exception as e:
            print(f"  2次元較正器の読込に失敗（実効EVは出しません）: {e}")
    if not _CALIB2D:
        return None
    try:
        p = np.clip(np.asarray(model_p, dtype=float), 1e-6, 1 - 1e-6)
        o = np.asarray(odds, dtype=float)
        if not np.isfinite(o).all() or (o <= 1).any():
            return None            # オッズ未取得。市場確率が作れないので出さない
        m = (1.0 / o) / (1.0 / o).sum()
        m = np.clip(m, 1e-6, 1 - 1e-6)
        lp, lm = np.log(p / (1 - p)), np.log(m / (1 - m))
        c, b = _CALIB2D["coef"], _CALIB2D["intercept"]
        z = b + c[0] * lp + c[1] * lm + c[2] * lp * lm
        q = 1.0 / (1.0 + np.exp(-z))
        return q / q.sum() if q.sum() > 0 else None
    except Exception as e:
        print(f"  2次元較正の適用に失敗（実効EVは出しません）: {e}")
        return None


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

    # top5_ai: 実際に打った印(◎○▲△×)そのものを表示する。
    # 2026-07-31: 以前は総合スコア(MF60%+通常40%)順だったが、印の土台を
    #   市場フリーMFのplace3へ切替えたため基準がズレ、同じメール内で
    #   「表の◎」と「TOP5の◎」が別馬になる不整合が出ていた（デモで発覚）。
    _mk_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
    if "印" in pdf.columns and pdf["印"].isin(_mk_order).any():
        top5_ai = (pdf[pdf["印"].isin(_mk_order)]
                   .assign(_o=lambda x: x["印"].map(_mk_order))
                   .sort_values("_o").drop(columns="_o").head(5))
    else:
        top5_ai = pdf.sort_values("総合スコア", ascending=False).head(5)
    # top5_ev: 通常モデル高評価かつEV<0（市場評価と近い）→ 安定した信頼馬
    top5_ev = pdf[(pdf["単勝期待値"] < 0) & (pdf["勝ち確率"] >= 0.03)].sort_values("勝ち確率", ascending=False).head(5)
    if top5_ev.empty:
        top5_ev = pdf[pdf["勝ち確率"] >= 0.03].sort_values("勝ち確率", ascending=False).head(5)

    lines += rec_block(
        "AI予想 印（◎○▲△×）", "[AI]", top5_ai,
        "市場フリーモデルの3着内予測順。乖離=人気順位との差（参考値・買いには使わない）。"
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

        # 2026-08-13: ★表示を廃止（乖離が大きいほど外すと検証で判明したため）。
        # 乖離の数値だけは市場との評価差を見る参考として残す。
        _star = "　"
        _gap = pd.to_numeric(row.get("乖離"), errors="coerce")
        _gap_s = f" 乖離{_gap:+.0f}" if pd.notna(_gap) and abs(_gap) >= 1 else ""
        lines.append(
            f"  │{_star}{mk}【{lbl}】 馬番{int(row['馬番'])}番 {str(row['馬名']):<12}"
            f"  {odds_s} {pop_s}{_gap_s}"
            f"  勝率{wp*100:.1f}%  EV{_ev(ev)}"
            f"  総合{score:.0f}点"
        )
        # 実効EV＝市場を織り込んだ期待値。上のEVは市場フリーの見立てなので、
        # 両方を並べる。差が大きいほど市場と割れている＝危ない、と読む。
        _ev2 = pd.to_numeric(row.get("実効EV"), errors="coerce")
        if pd.notna(_ev2):
            _judge = "買い" if _ev2 >= EV_MIN_TOP else "見送り"
            lines.append(f"  │           実効EV {_ev2:.2f}（市場織込）→ {_judge}")
        if tag_s:
            lines.append(f"  │           {tag_s}")
        if i == 0:
            lines.append("  │  [見方] 乖離=人気順位−モデル複勝順位。市場との評価差（参考）")
            lines.append("  │         実効EV=市場を織り込んだ期待値。1.0未満は買うと損")
        if i < 4:
            lines.append("  ├─────────────────────────────────────────────────────────┤")
    lines.append("  └─────────────────────────────────────────────────────────┘")
    lines.append("")
    # 検証要約（2026-07-31改訂・市場フリーMF＋妙味方式・2023-2025の3年OOS）
    lines.append("  [検証] 複勝方式（2021-2025の5年OOS・実払戻）")
    lines.append("  1900m以上×MF複勝1位×20倍以下×4番人気以下 → 複勝1点")
    lines.append("  230点・的中103本(44.8%)・回収率105.2%・95%区間[89.8,120.7]")
    lines.append("  ※下限が100%を割っており、黒字が確定したわけではない")
    lines.append("")

    # ── 購入する馬（複勝方式）────────────────────────────────────────
    # 2026-08-13: ★方式・EV方式を廃止し、複勝方式に一本化。
    #   ★（乖離+3以上）は検証で否定された。乖離が大きいほどモデルの予測が外れ、
    #   実勝率/予測の比は 乖離0未満1.40 / 3-6で0.43 / 6以上で0.27。
    #   さらに2次元較正で正しく較正すると差自体が消えた（全帯0.99〜1.11）。
    #   ★の優位は実体ではなく、較正の歪みの裏返しだった。
    _dist_v = pd.to_numeric(pdf.get("距離"), errors="coerce")
    _dist_v = float(_dist_v.dropna().iloc[0]) if _dist_v.notna().any() else np.nan
    _buy = pdf[(pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce") == 1)
               & (pd.to_numeric(pdf.get("単勝オッズ"), errors="coerce") <= FUKU_ODDS_MAX)
               & (pd.to_numeric(pdf.get("人気"), errors="coerce") >= FUKU_POP_MIN)] \
        if pd.notna(_dist_v) and _dist_v >= FUKU_DIST_MIN else pdf.iloc[0:0]
    lines.append(header("[購入する馬]", "─"))
    if len(_buy) == 0:
        if pd.notna(_dist_v) and _dist_v < FUKU_DIST_MIN:
            lines.append(f"  見送り（{int(_dist_v)}m。複勝方式は{FUKU_DIST_MIN}m以上が対象）")
        else:
            lines.append("  見送り（MF複勝1位が20倍以下かつ4番人気以下に該当せず）")
        lines.append("  ※該当は年46点程度（週1点弱）。ほとんどのレースは見送りになる。")
    else:
        for _, row in _buy.iterrows():
            odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
            pop = pd.to_numeric(row.get("人気"), errors="coerce")
            lines.append(
                f"  複勝 馬番{int(row['馬番']):>2} {str(row['馬名']):<12} "
                f"{(f'{odds:.1f}倍' if pd.notna(odds) else '未確定'):>7} "
                f"{(f'{int(pop)}人気' if pd.notna(pop) else '-'):>5}"
                f"  {FUKU_STAKE:,}円")
            lines.append(f"           勝率{row['勝ち確率']*100:4.1f}% "
                         f"複勝率{row['複勝確率']*100:4.1f}%")
        lines.append("")
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
        lines.append("  このレースは購入非推奨（レース単位ゲートで除外）。資金は✅買いレースに温存。")
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
    # ── 距離で2モデルを切り替える（2026-08-14）────────────────────────
    #   長距離(>=dist_split)は全特徴、短距離は騎手厩舎31列を除いたモデルを使う。
    #   クリーンデータのwalk-forward検証で5条件すべて改善（全体で1.6倍）。
    #   騎手の勝率は「どの馬に乗るか」で決まる面が大きく、その情報は既にオッズに
    #   入っている。市場を条件に入れると短距離では残りがノイズになる。
    #   ⚠ 旧モデル（models_short を持たない pkl）でも落ちないよう、
    #     無ければ従来どおり単一モデルを使う。
    def _mf_pick(info, dist):
        """レースの距離に応じて (models, use_cols) を返す。距離不明なら全特徴。"""
        if not info:
            return None, None
        split, short = info.get("dist_split"), info.get("models_short")
        if split is None or not short or dist is None:
            return info.get("models"), info.get("use_cols")
        if float(dist) < float(split):
            return short, info.get("use_cols_short", info.get("use_cols"))
        return info.get("models"), info.get("use_cols")

    mf_weights    = mf_info.get("weights")  if mf_info else None
    # MF複勝(place3)モデル: place系買い目の妙軸に使う（勝率軸より馬券内率・ROIが高い）
    mf_p3_info    = mf_info.get("place3") if mf_info else None
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

    # 距離が確定したのでMFモデルを選ぶ（長距離=全特徴 / 短距離=騎手厩舎を除外）
    _dist_v = pd.to_numeric(pdf.get("距離"), errors="coerce")
    _dist_v = float(_dist_v.dropna().iloc[0]) if _dist_v.notna().any() else None
    mf_models, mf_cols = _mf_pick(mf_info, _dist_v)
    mf_p3_models, mf_p3_cols = _mf_pick(mf_p3_info, _dist_v)
    if mf_info and mf_info.get("models_short"):
        _sp = mf_info.get("dist_split")
        _side = "短距離(騎手厩舎なし)" if (_dist_v is not None and _dist_v < float(_sp)) \
            else "長距離(全特徴)"
        print(f"  MFモデル: {_side} を使用（{_dist_v}m / 閾値{_sp}m・{len(mf_cols or [])}列）")

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
            # 較正してから確率にする（2026-08-04追加）。
            # MFは正例重み2.0＋時間重みで学習しているため生の値は過大で、
            # 2025 OOSの実測では 勝率 予測11.0%に対し実際7.2%（1.5倍過大）だった。
            # 期待値も推奨賭け率も「確率が正しい」前提なので、ここを直さないと
            # 表示・買い判断のすべてがずれる。
            mf_raw = _calibrate(mf_raw, "win")
            pdf["MF勝ち確率"] = mf_raw / mf_raw.sum() if mf_raw.sum() > 0 else np.ones(len(mf_raw)) / len(mf_raw)
            # EV を MF勝ち確率ベースで上書き（通常モデルの暫定値を置き換え）
            pdf["単勝期待値"] = pdf["MF勝ち確率"] * pdf["単勝オッズ"] - 1
            # 2次元較正（購入判定専用の別列）。印・★・順位・評価には使わない。
            #   単勝期待値は「市場フリーの見立て」、実効EVは「市場を織り込んだ値」。
            #   両方を出して並べる。差が大きいほど市場と割れている＝危ない。
            pdf["較正後勝率"] = np.nan
            pdf["実効EV"] = np.nan
            _q2 = _calibrate2d(pdf["MF勝ち確率"].to_numpy(),
                               pd.to_numeric(pdf["単勝オッズ"], errors="coerce").to_numpy())
            if _q2 is not None:
                pdf["較正後勝率"] = _q2
                pdf["実効EV"] = _q2 * pd.to_numeric(pdf["単勝オッズ"], errors="coerce")
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
            # 複勝率も較正する（生値は 予測26.8%に対し実際21.5%＝1.25倍過大）。
            # 順位は較正しても変わらない（単調変換のため）ので印には影響しない。
            pdf["MF複勝率"]   = _calibrate(
                np.clip(np.nan_to_num(p3, nan=0.0), 0, None), "place3")
            pdf["MF複勝順位"] = pd.Series(p3).rank(ascending=False).values
        except Exception as e:
            print(f"  MF複勝予測エラー（スキップ）: {e}")

    # ── 表示用の確率をMFへ統一（2026-07-31）────────────────────────────
    # 印はMFのplace3で決めるのに、表示する勝率・連対率・複勝率は主モデル(市場込み)
    # の値だった。そのため「◎の複勝率18.7% < △の45.7%」という矛盾した表示が出ていた。
    # 印と数字の出所を揃える。主モデルの値は参照用に 主_ 接頭辞で残す。
    if mf_info and pdf["MF勝ち確率"].notna().any():
        for c in ("勝ち確率", "連対確率", "複勝確率", "3着内確率"):
            if c in pdf.columns:
                pdf[f"主_{c}"] = pdf[c]
        try:
            pdf["勝ち確率"] = pdf["MF勝ち確率"]
            # 連対はMFのplace2を使う。無ければ勝率と複勝率の間を取る。
            mf_p2 = mf_info.get("place2")
            if mf_p2 and mf_p2.get("models"):
                p2 = _wavg(mf_p2["models"], pdf.reindex(columns=mf_p2["use_cols"]),
                           mf_p2.get("weights"))
                p2 = np.clip(np.nan_to_num(p2, nan=0.0), 0, 1)
                s2 = p2.sum()
                pdf["連対確率"] = p2 * (2.0 / s2) if s2 > 0 else p2
            if pdf["MF複勝率"].notna().any():
                p3v = np.clip(pdf["MF複勝率"].to_numpy(), 0, 1)
                s3 = p3v.sum()
                pdf["複勝確率"] = p3v * (3.0 / s3) if s3 > 0 else p3v
                pdf["3着内確率"] = pdf["複勝確率"]
            # 確率の包含関係を保つ（勝率 <= 連対率 <= 複勝率）
            pdf["連対確率"] = np.maximum(pdf["連対確率"], pdf["勝ち確率"])
            pdf["複勝確率"] = np.maximum(pdf["複勝確率"], pdf["連対確率"])
            pdf["3着内確率"] = pdf["複勝確率"]
            # 順位もMF基準で作り直す（2026-08-16）。
            #   確率だけMFに差し替えて順位を主モデルのまま残すと、両者が食い違う。
            #   荒れR馬単裏方式は「MF連対順位」で軸と相手を決めるので、ここが要る。
            pdf["MF連対順位"] = pdf["連対確率"].rank(ascending=False, method="first")
            pdf["MF勝率順位"] = pdf["勝ち確率"].rank(ascending=False, method="first")
            pdf["連対順位"] = pdf["MF連対順位"]
            pdf["単勝期待値"] = pdf["勝ち確率"] * pdf["単勝オッズ"] - 1
            pdf["複勝期待値"] = pdf["複勝確率"] * (pdf["単勝オッズ"] / 3.5).clip(1.1, 8.0) - 1
            # 実オッズがあるならそちらで複勝EVを取り直す（確率を差し替えたので再計算）
            if "複勝オッズ_min" in pdf.columns and pdf["複勝オッズ_min"].notna().any():
                pdf["複勝期待値_実"] = pdf["複勝確率"] * pdf["複勝オッズ_min"] - 1
            else:
                pdf["複勝期待値_実"] = pdf["複勝期待値"]
            print("  表示確率をMFに統一")
        except Exception as e:
            print(f"  MF確率への統一エラー（主モデルのまま）: {e}")

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
    # 2026-07-31: 印の土台を「市場フリーのMFモデル(place3)」へ切替。
    #   主モデルは人気・オッズを特徴に持つため印が人気順の写しになっていた
    #   （◎の85.7%が1番人気）。MFは市場を一切見ないので、◎が1番人気になるのは
    #   約56%に下がり、印が人気帯に散る。3年検証で ◎複勝率 62.4/64.4/62.8%、
    #   1-3着カバー 2.07/2.10/2.06頭と安定。MFが無い時だけ従来の複勝確率に戻る。
    if "MF複勝順位" in pdf.columns and \
            pd.to_numeric(pdf["MF複勝順位"], errors="coerce").notna().any():
        fuku_sorted = pdf.assign(
            _mfr=pd.to_numeric(pdf["MF複勝順位"], errors="coerce")
        ).sort_values("_mfr", na_position="last")
    elif "複勝確率" in pdf.columns and pdf["複勝確率"].notna().any():
        fuku_sorted = pdf.sort_values("複勝確率", ascending=False)
    else:
        fuku_sorted = pdf.sort_values("総合スコア", ascending=False)
    _pop_all = pd.to_numeric(pdf["人気"], errors="coerce")
    _cls_v0 = pd.to_numeric(pdf["クラス_num"].iloc[0], errors="coerce") if "クラス_num" in pdf.columns else np.nan
    _fav_idx = _pop_all.idxmin() if _pop_all.notna().any() else None
    _fav_odds = pd.to_numeric(pdf.loc[_fav_idx, "単勝オッズ"], errors="coerce") if _fav_idx is not None else np.nan
    # 2026-07-31: MFを土台にする場合、◎を人気1位へ固定する分岐は使わない。
    #   市場評価で上書きすると、MFを市場フリーにした意味（印が人気順から離れる）が
    #   消えるため。MFが無い時だけ従来のアダプティブ◎を残す。
    _mf_base = "MF複勝順位" in pdf.columns and \
        pd.to_numeric(pdf["MF複勝順位"], errors="coerce").notna().any()
    _use_fav = (not _mf_base) and (
        (pd.notna(_fav_odds) and float(_fav_odds) <= 2.0)
        or (pd.notna(_cls_v0) and int(_cls_v0) >= 5))
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

    # ── 妙味判定（★）: モデル評価と市場評価の乖離を全頭に付ける ─────────────
    #   乖離 = 人気順位 − モデル順位（正＝モデルが市場より高く評価している）
    #   3年12.6万頭の実測（final_marks_2023/24/25）:
    #     乖離+3以上           単勝ROI 90.4%   （一致馬は65.9%）
    #     ＋モデル3位以内       97.3%（2023:97.8 / 2024:95.1 / 2025:98.9・振れ3.8pt）
    #     ＋オッズ20倍以下 が上記の条件。年997点。
    #   ◎に限れば単勝102.2%・複勝112.9%（★なしの◎は83.1%/92.2%）。
    #   ★は全頭に出す（平均0.42頭/レース・66%のレースは0頭）。買うのは★かつ◎○▲のみ。
    #   △×の★は参考表示（年244点で45.8〜126.5%と暴れるため購入対象外）。
    pdf["乖離"] = np.nan
    pdf["妙味"] = ""
    try:
        _mr = pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce")
        if _mr.isna().all() and "複勝確率" in pdf.columns:
            _mr = pdf["複勝確率"].rank(ascending=False, method="first")
        _pr = pd.to_numeric(pdf["人気"], errors="coerce").rank(method="first")
        _od = pd.to_numeric(pdf["単勝オッズ"], errors="coerce")
        pdf["乖離"] = _pr - _mr

        if MYOMI_MODE == "ratio":
            # 確率比方式: P_model / P_market。
            # 市場推定勝率は単勝オッズの逆数をレース内で正規化して控除率を割り戻す。
            # モデル側も同じくレース内で合計1に揃えてから比を取る（尺度を合わせる）。
            _inv = 1.0 / _od.where(_od > 0)
            _p_mkt = _inv / _inv.sum(skipna=True)
            _p_mdl = pd.to_numeric(pdf.get("MF勝ち確率"), errors="coerce")
            if _p_mdl.notna().sum() < 2 or not (_p_mdl.sum(skipna=True) > 0):
                _p_mdl = pd.to_numeric(pdf.get("勝ち確率"), errors="coerce")
            _p_mdl = _p_mdl / _p_mdl.sum(skipna=True)
            pdf["乖離比率"] = _p_mdl / _p_mkt
            _hit = (pdf["乖離比率"] >= MYOMI_RATIO_MIN) & (_od <= MYOMI_ODDS_MAX)
        else:
            _hit = (pdf["乖離"] >= MYOMI_GAP_MIN) & (_od <= MYOMI_ODDS_MAX)

        pdf.loc[_hit.fillna(False), "妙味"] = "★"
    except Exception as _e:
        print(f"  妙味判定スキップ: {_e}")

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
    # 判定: 買い(指数70-100)/見送り(25-45)。レース単位ゲート。詳細は_race_bet_plan参照。
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
            "較正後勝率", "実効EV",   # 2026-08-12: 市場を織り込んだ購入判定用の値
            "乖離スコア", "MF予測順位", "MF勝ち確率", "MF複勝率", "MF複勝順位",
            "該当戦略", "推奨ランク", "総合スコア", "券種推奨", "妙味軸",
            "妙味", "乖離",   # 2026-07-31: ★判定と市場との評価差（メール/ダッシュボード用）
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


# ── 購入しきい値（2026-07-30改訂）─────────────────────────────────────────
#   買い指数がこの値未満のレースは「買い目を出さない」（判定表示はするが購入対象外）。
#   新方式の指数は「ゲート通過=70」を起点に、優位が上乗せされる材料ごとに+10する。
#   目安: 70=ゲートを通った全レースを買う / 80=上乗せ1つ以上 / 90=上乗せ2つ以上
#   today_bets.csv・メール・ダッシュボードの買い目表示すべてに連動する。
BUY_INDEX_MIN = 70

# ── レース単位の購入ゲート（2026-07-30・帯を廃止して導入）──────────────────
#   2025全3130Rの実払戻で測定（race_eval.py → race_rule.py）。
#   対照実験・前後半分割・レース単位ブートストラップを通した材料だけを採用。
RACE_GATE_MIN_INTERVAL = 8.0      # 軸の前走間隔(週)。これ未満は見送り（63-71%）
RACE_GATE_SKIP_CLASS = (3, 4)     # 見送りにする クラス_num の範囲＝中級（67.2%）
RACE_GATE_ALLOW_UNKNOWN = True    # 前走間隔が不明（新馬・海外帰り等）は通す

# ── 妙味判定（2026-07-31）──────────────────────────────────────────────
#   市場フリーのMFモデルと市場評価が食い違う馬を「★」とする。
#   乖離 = 人気順位 − モデル順位。正なら市場より高く評価している。
#   3年実測: 乖離+3以上 × モデル3位以内 × 20倍以下 → 単勝97.3%（振れ3.8pt・997点/年）
#   ★かつ◎○▲だけを買う。△×の★は参考表示（小サンプルで年80pt振れるため）。
MYOMI_GAP_MIN = 3.0               # この順位以上、市場より高く評価していれば★
MYOMI_ODDS_MAX = 20.0             # 人気薄すぎる馬は対象外（20倍超は不安定）

# ── ★の測り方（2026-08-02追加・既定は従来のまま）────────────────────────
#   "rank"  : 乖離 = 人気順位 − モデル順位 が MYOMI_GAP_MIN 以上（従来方式）
#   "ratio" : 確率比 P_model / P_market が MYOMI_RATIO_MIN 以上
#
#   順位差は「3番人気と6番人気」も「10番人気と13番人気」も同じ+3として扱うが、
#   確率比ならその差を保てる。2025実測(gap_ratio.py・ワイド★軸×人気上位3頭):
#     順位差≥3 : 319R 的中40.8% 回収121.2% CI下102.4
#     比率≥2.5 : 335R 的中50.4% 回収133.1% CI下115.2  ← 対象が多く回収率も上
#   用量反応も単調(≥1.5:106.7 → ≥2.0:118.2 → ≥2.5:133.1 → ≥3.0:150.2)。
#
#   ⚠️既定を "rank" のままにしている理由:
#     ①検証が2025単年のみ。2023/2024での再確認が未了。
#     ②検証時の P_model は final_marks の score を softmax したもので、本番の
#       MF勝ち確率とはスケールが違う。閾値2.5をそのまま移植できない。
#     切り替える前に本番の確率分布で閾値を測り直すこと。
MYOMI_MODE = "rank"               # "rank"（従来）/ "ratio"（確率比）
MYOMI_RATIO_MIN = 2.5             # ratio方式のときの閾値（要・本番分布での再較正）

# ── 買い方: 較正済み期待値方式（2026-08-04・3年検証で確定）──────────────
#   2023/2024/2025の3年OOSで検証（bet_cache_*.csv / refine.py）。
#   従来の★方式（乖離≥3・20倍以下の◎を複勝）は3年で 81.4/73.6/82.0% と
#   控除率に負けていた。較正済み勝率×実オッズの期待値を使い、
#   「モデル順位ごとに要求する期待値を変える」ことで初めて3年とも100%を超えた。
#
#   条件: 乖離≥3 かつ 単勝20倍以下 かつ
#         モデル1位ならEV≥1.7 / モデル2〜5位ならEV≥2.2
#   買い方: 条件を満たす中で期待値が最大の1頭を、単勝で1点
#   成績: 114.6% / 114.8% / 111.5%（通算113.6%・年241レース・的中64本/3年）
#
#   ⚠️通算のCI下限は85前後で、統計的に確定はしていない（的中64本のため）。
#     年数を重ねる以外に確度を上げる手段はない。少額での運用が前提。
#   ※検証で分かったこと（変更する前に読むこと）:
#     ・乖離は0〜5で振って3が唯一3年100%超。0-2は2025が、4-5は2024が崩れる
#     ・クラス・頭数・距離での絞り込みは全て悪化（部分集合にすると必ずどこかの年が崩れる）
#     ・較正をオッズ帯別/頭数別に分けるのも悪化（サンプルが減り較正自体が不安定になる）
#     ・複勝・ワイド・馬連・馬単・3連複は22通り試して全滅。事前オッズが無い券種は
#       期待値を計算できず、単勝オッズからの推定では新しい情報が増えないため。
# ── 複勝方式（2026-08-13・既定）──────────────────────────────────────────
#   なぜこれにしたか
#     2,027構成を5つの独立した設計思想で総当たりし、生き残った唯一の構成。
#     従来のEV方式(A案)は紙で117.0%だったが、次の2つが未解決だった。
#       ① スリッページ: 確定オッズで選んだ数字。7分前で選び直すと88.4%（-28.7pt）
#          オッズ変動で直せないか検証したが、この帯では相関-0.13で先読み不能だった
#       ② オプティマイザの呪い: EVと実払戻の順位相関が5年とも負。実効EV 0.95
#     C案はどちらも回避する。選択条件の大半（距離・モデル順位・人気）がオッズに
#     依存せず、複勝は配当が着順で決まるため変動の影響が小さい。
#
#   検証（bet_cache 2021-2025・207,518頭・walk-forward OOS）
#     230点（年46点）・的中103本(44.8%)・複勝ROI 105.2%・95%区間[89.8, 120.7]
#     スリッページ模擬: 110.2% → 107.6%（-2.6pt）95%[102.7, 111.5]
#     順列検定: 単体 p=0.0000 / 探索81構成を織り込んだfamily-wise p=0.0725
#
#   1900mに断層がある（1800m以下にすると的中率44.8%→36.0%・ROI 105.2%→84.8%）。
#   滑らかに劣化せず崩れるので、ただの絞り込みではなく実体があると判断した。
#
#   ⚠ 黒字が確定したわけではない。下限89.8%で5年中2年は赤字（最悪年66%）。
#     p=0.0725は有意水準に届いていない。証明には約17年かかる。
#     「否定されずに残った唯一の候補」という位置づけ。
# 2026-08-14: 停止した。採用の根拠がすべて時系列リークの産物だったため。
#   騎手・調教師・馬主の累積勝率が race_id 順（＝場コード順）で集計されており、
#   場が切り替わるたびに日付が巻き戻って未来が混入していた。
#   修正して再構築したところ、この方式の数字は次のように崩れた。
#       的中率  44.8% → 34.8%
#       回収率 105.2% →  84.7%（95%区間 [68.8, 100.9]）
#       1900mの「断層」も消えた（1800mとの差 20.4pt → 4.7pt）
#   「2,027構成で唯一生き残った」という評価自体が汚染データ上のものだった。
#   クリーンデータでの再探索でも、的中100本以上で100%を超えた構成はゼロ。
# ── 荒れR馬単裏方式（2026-08-16採用）────────────────────────────────
#   条件: 1番人気のMF複勝順位が4位以下のレース（＝モデルが1番人気を信用していない）
#         連対順位1位を軸にし、連対順位4位以内の各馬から軸へ流す（馬単の裏）
#   買い方: 馬単 1点500円 × 相手の数（通常3点）
#
#   なぜ「裏」か
#     荒れRでは1番人気が崩れやすく、モデルの本命は勝ち切れないが2着には来る。
#     軸を2着に置く形（相手→軸）がその構造に合う。従来は表しか試していなかった。
#
#   検証（bet_cache 2021-2025・walk-forward OOS）
#     5年通算 112.6% / 5年中4年で100%超
#     直近2年 115.6%（2024・2025とも100%超）・的中46本
#     スリッページ模擬 115.6%（影響 0.0pt）
#       ※オッズ条件を付けないので、7分前で判定しても選ぶ馬が変わらない。
#         過去の候補が -20〜-27pt 落ちたのはオッズ帯を狭く切っていたため。
#
#   ⚠ 順列検定は通っていない（15,876構成から選んだ扱いで p=0.9067）。
#     過去データでは「探索が広いほど翌年に崩れる」ことも実測されている
#     （2,509構成から選ぶと 120.2% → 49.7%）。
#     したがってこれは**前向き検証**であって、黒字が確認された構成ではない。
#     実運用の成績を貯めて、半年後に判定する。
USE_ARE_UMATAN = True             # 荒れR方式（下の ARE_PLANS をまとめて実行）
ARE_FAV_MR_MIN = 4                # 1番人気のMF複勝順位がこれ以上なら「荒れR」
ARE_STAKE = 500                   # 1点あたりの金額（円）

# ⚠ 2026-08-16に2つの誤りを見つけたので記録する。
#   ① verify34.py の集計が誤っており、A構成を149.2%と報告していた。
#      5年分で正しく組み直すと 5年100.2% / 直近2年101.8% だった。
#      この誤った値をもとに7構成を採用してしまった。
#   ② 構成名の「1-20倍」は**軸のオッズ条件**なのに、実装で落としていた。
#      条件を外すと別の買い方になる（A構成は101.8%→96.5%に変わる）。
#
#   正しい軸の選び方（portfolio_fix.py で検証したものと同じ）:
#     軸   = 指定順位以内 かつ 単勝オッズが[下限,上限)の馬。その中で順位最小の1頭
#     相手 = 指定順位以内の馬（軸を除く）。相手にオッズ条件は付けない
#
#   採用は A構成のみ。7構成のうち5年通算と直近2年の両方で100%を超えたのが
#   これだけだったため（B は直近104.0%だが5年94.9%）。組み合わせても改善せず、
#   最良の BG でも直近99.5%だった。

# ── 採用する構成（2026-08-16）─────────────────────────────────────
#   verify34 でスリッページを通過した31件のうち、**影響が±3pt以内**の
#   7種類だけを採る。影響が大きいもの（-18〜-45pt）は狭いオッズ帯で切って
#   おり、実運用で必ず崩れるので入れない。
#
#   (名前, 券種, 基準列, 軸順位, 相手順位上限, レース条件)
#     券種  "馬単裏" = 相手→軸 / "馬単表" = 軸→相手 / "馬連" = 順不同
#     基準  win=MF勝率順位 / ren=MF連対順位
#     条件  are=荒れR（1番人気がMF複勝4位以下） / cls4=クラス4以上
#
#   各構成の検証値（直近2年ROI / 的中 / 7分前ROI / 影響pt）
#     荒れR 勝率1位軸x上位2 馬単裏   149.2% / 18 / 148.5% / -0.8
#     クラス4+ 勝率1位軸x上位2 馬単裏 132.6% / 34 / 132.8% / +0.2
#     荒れR 勝率1位軸x上位2 馬連     126.8% / 28 / 126.1% / -0.7
#     荒れR 連対1位軸x上位4 馬単裏   117.7% / 46 / 116.1% / -1.6
#     荒れR 勝率1位軸x上位3 馬単裏   114.1% / 31 / 115.3% / +1.2
#     荒れR 連対1位軸x上位5 馬単裏   110.6% / 55 / 109.5% / -1.1
#     荒れR 勝率1位軸x上位4 馬単表   108.0% / 41 / 106.8% / -1.2
#
#   ⚠ 6種類が同じ「荒れR」を買うので、1レースで複数の構成が同時に成立する。
#     組み合わせが重複したぶんは _build_bet_rows で1点にまとめる。
#   (名前, 券種, 基準, 軸順位, 相手順位, オッズ下限, オッズ上限, レース条件)
ARE_PLANS = [
    # 上限を20倍→10倍にした（2026-08-16・prereg.py / gradient.py）。
    # 軸が10倍を超える買い目は 277点で的中2本しかない。1-10倍と同じ的中率なら
    # 9.4本のはずで、そうならない確率は p=0.0044。市場が大穴を買いすぎるという
    # 独立に確かめた事実（60倍超で実現/理論比0.82）とも向きが一致する。
    #   1-10倍  5年110.0% 直近2年103.0% 7分前108.0% 100超年3/5
    #   1-20倍  5年100.2% 直近2年101.8% 7分前101.3% 100超年2/5
    # なお1-7倍は最終オッズで132.2%と最も高いが、7分前だと95.6%（-26.0pt）に
    # 落ちるので採らない。締切間際に人気が落ちた馬を拾っているだけ。
    ("荒れR勝率1位x2 馬単裏", "馬単裏", "win", 1, 2, 1.0, 10.0, "are"),
]

USE_FUKU_BETTING = False          # True で複勝方式を再開（非推奨）
# 併用モード（2026-08-14）: 複勝方式とEV方式を同時に走らせ、実測を並行して集める。
#   EV方式は確定オッズ基準119.6% / 7分前88.4%。差の原因は賭ける時刻であって
#   モデルではない。締切に近づけるほど縮むので、1分前オッズを半年貯めてから
#   「投票を遅らせる価値があるか」を検証する。それまでは両方の実測を取る。
#   ⚠ EV方式は5年OOFでEVと実払戻の順位相関が負（実効EV 0.95）。赤字の可能性が高い。
USE_DUAL_BETTING = True
FUKU_DIST_MIN = 1900              # 距離の下限（m）。1800にすると崩れる
FUKU_ODDS_MAX = 20.0              # 単勝オッズの上限
FUKU_POP_MIN = 4                  # 人気の下限（4番人気以下＝妙味のある側）
FUKU_STAKE = 1000                 # 複勝1点あたりの金額（円）

# 2026-08-14: 停止した。クリーンデータでスリッページを実測した結果による。
#   確定オッズで選ぶ  109.8%（928レース・的中73本）
#   7分前で選び直す   約82.8%（-27.0pt）
#   リーク修正前の -28.7pt とほぼ同じ幅で落ちた。
#
#   原因は実測したドリフトの構造。7分前→確定の中央倍率は
#     1番人気 0.846 / 2-3番 0.858 / 4-5番 1.013 / 6-7番 1.230 / 8番以下 1.907
#   EV方式は「20倍以下」を条件にするが、7分前に20倍だった人気薄は確定では38倍。
#   バックテストが条件を満たすと判定した馬は、7分前には条件を満たしていない。
#
#   これでクリーンデータにおいて100%を超える買い方は1つも残っていない。
#   締切2分前のオッズを半年貯めてから、投票時刻を遅らせれば回復するかを検証する。
#   それまでは購入せず、予想と記録だけを続ける。
#   2026-08-15: 実測データを取るため再開した。赤字が見込まれることは承知のうえで、
#   「締切に近づけて投票すれば回復するか」を半年後に判定する材料を集める。
#   紙(109.8%)と実運用(82.8%)の差が本当に賭ける時刻で説明できるのかは、
#   実際に買った記録と締切2分前オッズを突き合わせないと確かめられない。
#   2026-08-16: 荒れR馬単裏方式の採用に伴い停止した。両方走らせると同じレースで
#   重複購入になり、どちらの成績かも切り分けられなくなる。前向き検証を濁さない。
USE_EV_BETTING = False            # 実運用の期待値は82.8%。荒れR方式に置き換え
EV_GAP_MIN = 3.0                  # 乖離（人気順位 − モデル複勝順位）の下限
EV_ODDS_MAX = 20.0                # 単勝オッズの上限
EV_MIN_TOP = 1.7                  # モデル複勝1位に要求する期待値
EV_MIN_SUB = 2.2                  # モデル複勝2〜5位に要求する期待値
EV_MAX_PICKS = 1                  # 1レースに買う頭数（期待値の高い順）
EV_STAKE = 1000                   # 単勝1点あたりの金額（円）

# ── 馬単の併用（2026-08-05・3年検証で追加）────────────────────────────
#   軸（上のEV条件で選んだ1頭）を1着に固定し、相手は上位人気の印へ流す。
#   軸の優位は「勝つ力の過小評価」で、勝率は同人気平均の1.93倍あるが
#   連対率1.55倍・複勝率1.35倍と減衰する。だから軸は1着固定が最も効き、
#   2着は素直に強い馬（上位人気）を置くのが噛み合う。
#
#   3年OOS（2023/2024/2025・bet_cache_*.csv）:
#     単勝のみ            通算113.6% CI下87.5 最悪年111.5% 収支+9.8千
#     馬単 相手3番人気以内  通算134.1% CI下92.2 最悪年118.1% 収支+58.4千
#     単勝＋馬単（併用）    通算128.0% CI下97.9 最悪年117.2% 収支+68.2千 ← 採用
#   併用すると互いに独立に外れるため分散が下がり、CI下限が単体より上がる。
#
#   ⚠️CI下限97.9で統計的な確定には至っていない（3年723レース）。少額運用が前提。
#   ※検証で分かったこと（変更する前に読むこと）:
#     ・相手を穴にすると崩れる。軸が人気薄なので相手まで穴にすると当たらない
#     ・馬連・ワイド・3連複は同じ軸から流しても100%前後が上限（軸の優位が
#       2着以内・3着以内では薄まるうえ控除率が重い）
#     ・複勝は実オッズで測っても最良82.7%。当たりやすい分オッズが圧縮される
USE_UMATAN = True                 # False で単勝のみに戻す
UMATAN_MAX_POP = 3                # 相手にする印の人気上限（3番人気以内）
UMATAN_RANKS = (1, 2, 3, 4, 5)    # 相手にする印（モデル複勝順位）
UMATAN_STAKE = 500                # 馬単1点あたりの金額（円）
# 印ごとに買う券種を変える（2026-07-31・3年OOSの実測にもとづく）:
#   ★◎ 単勝102.2% / 複勝112.9%  → 3着内には来るが勝ち切れないので複勝
#   ★○ 単勝100.3% / 複勝 87.1%  → 勝ち切る側なので単勝
#   ★▲ 単勝 88.4% / 複勝 80.0%  → 3年とも劣る（2024は69.7%）ので買わない
#   ◎○▲を一律で単勝+複勝にすると97.3%だが、上記の使い分けなら3年平均101.4%。
MYOMI_BETS = {"◎": ("複勝",), "○": ("単勝",)}
MYOMI_MARKS = tuple(MYOMI_BETS)   # ★でも購入対象にする印（▲以下は参考表示のみ）
USE_MYOMI_BETTING = True          # False で従来のゲート方式に戻す
# 検証用の切り分けトグル（環境変数・既定は上の本番値のまま）:
#   KEIBA_GATE_INTERVAL=0 で間隔ゲートを無効 / KEIBA_GATE_CLASS=0 でクラスゲートを無効
if os.environ.get("KEIBA_GATE_INTERVAL") == "0":
    RACE_GATE_MIN_INTERVAL = None
if os.environ.get("KEIBA_GATE_CLASS") == "0":
    RACE_GATE_SKIP_CLASS = None

# ── 資金設定（2026-07-17・自動投票対応の下地）────────────────────────────
#   today_bets.csv に1点ごとの「金額」列が出力される（将来の自動投票はこれを注文リストとして読む）。
#   照合(analyze_accuracy)も金額加重の実収支で集計される。
BET_UNIT = 100                    # 基本1点あたりの金額(円)・KIND_STAKE未定義券種のフォールバック
# ── 予算配分方式（2026-07-21）────────────────────────────────────────────
#   各レースで「予算内に収まるよう買い目を選び・券種ごとに掛け金を配分」する。
#   仕組み: 1点の金額 = KIND_STAKE[券種] × SIZE_WEIGHT[サイズ] → 100円丸め・最低100円。
#           合計が RACE_BUDGET[判定帯] を超えたらBT回収率の低い買い方から丸ごと削る。
#   例: 勝負帯で単勝500×1 + 馬単100×5 → 予算内なら両方、超えれば弱い連系から自動除外。
RACE_BUDGET = {                   # 1レース予算(円)。帯を廃止したので「買い」の1本のみ。
    "買い": 2000,                 # 標準=1000円/R(単勝500+馬連100×5)・厚め=1800円/Rが収まる額
}
# 予算超過時に「どの買い目を残すか」の優先度。2025BT(予算配分込み):
#   "balanced" = メニュー順(単勝→複勝→ワイド→…)。複勝/ワイドも買い分散・的中安定。回収198.5%・的中1113。
#   "ev"       = 単勝を軸に確保→残りBT回収率順。馬単/3連単に集中し高EVだが変動大。回収221.6%・的中591。
ALLOC_PRIORITY = "balanced"       # 高EV集中に切替えるなら "ev"
KIND_STAKE = {                    # 券種ごとの1点あたり基本額(円)。単勝を厚く・多点の連系を薄く
    "単勝": 500, "複勝": 300, "ワイド": 200,
    "馬連": 100, "馬単": 100, "3連複": 100, "3連単": 100,
}
SIZE_WEIGHT = {                   # 確信度サイズの倍率（1点の掛け金を厚薄）
    # 確信度シグナルはここ(掛け金)で反映。予算側にも効かせる案は2025BTで効果ゼロと確認済み(2026-07-22)。
    "厚め": 1.6, "標準": 1.0, "薄め": 0.6,
}
if os.environ.get("KEIBA_FLAT_SIZE") == "1":   # 厚薄が効いているかの検証用（全て等倍）
    SIZE_WEIGHT = {k: 1.0 for k in SIZE_WEIGHT}
BAND_WEIGHT = {}                  # 旧方式の残置(未使用)。帯別配分は RACE_BUDGET に移行済み
KIND_UNIT = {}                    # 券種別の金額“固定”上書き(円)。指定券種はSIZE倍率も無視して固定額
KIND_SKIP = []                    # 買わない券種 例: ["3連単"] で点数を大幅削減
RACE_BUDGET_MAX = None            # (旧)全帯共通の上限。設定時は RACE_BUDGET より優先
#   例: RACE_BUDGET_MAX=2000 → 1日約3.8万円上限 / KIND_SKIP=["3連単"] → 1日約270点に減


def _race_bet_plan(pdf):
    """レース単位で「買う/買わない」を決める（2026-07-30・判定帯を廃止）。

    【なぜ変えたか】
    旧方式は妙の人気帯で 勝負/買い/堅実/少額 に分けていたが、その根拠のBT数値
    （勝負帯 単勝291%・買い帯183%）は時系列リーク・血統リーク下で出た幻だった。
    リーク修正後のクリーン実測では 勝負帯88.5% / 買い帯75.7% で、しかも勝負帯は
    年間の的中が27回しかなく信頼区間[53%,125%]＝帯そのものに識別力がない(band_clean.py)。

    【新方式】
    レースごとに測れる材料だけで「そのレースを買うか」を決める。採用材料は
    2025全3,130Rの実払戻で測定し、①対照実験(市場=1番人気にも同傾向が出ないか)
    ②前後半分割 ③レース単位ブートストラップ を通したもの(race_eval.py/race_rule.py)。

      ・軸の前走間隔 <8週  → 見送り。実測63-71%。市場側も同傾向だが我々の劣化幅が大きい
      ・クラス3-4級(中級)  → 見送り。実測67.2%(前70.1/後64.1)で前後半とも一貫して悪い
      → 通過は856R(全体の27%)。単勝+馬連で92.1%(前80.1/後106.0)

    【券種】単勝＋馬連の2本に絞る。同じ856Rでの実測が
      馬連93.2 / 単勝86.6 / 馬単89.7 / 複勝85.2 / 3連複83.2 / ワイド81.4% で、
      控除率の低い単勝(20%)・馬連(22.5%)に我々の優位が最も残るため
      （3連単は控除27.5%に対し優位ゼロと確認済み・exotic2.py）。

    【正直な限界】100%は超えない。これは「勝てるレースを選ぶ」施策ではなく
    「負けの大きいレースを買わない」施策。全体78.0%→92.1%＝損失を22%→8%に圧縮する。
    """
    plan = {"判定": "見送り", "指数": 25, "サイズ": "-", "理由": "", "menu": []}

    # ── 荒れR馬単裏方式（2026-08-16・既定）──────────────────────────
    #   1番人気をモデルが信用していないレースで、モデルの連対1位を「2着」に置く。
    #   根拠と検証値は ARE_* 定数のコメント参照。オッズ条件を付けないので
    #   7分前に判定しても選ぶ馬が変わらない（スリッページ影響 0.0pt）。
    if USE_ARE_UMATAN:
        _mr = pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce")
        _rank = {"win": pd.to_numeric(pdf.get("MF勝率順位"), errors="coerce"),
                 "ren": pd.to_numeric(pdf.get("MF連対順位"), errors="coerce")}
        _pop = pd.to_numeric(pdf.get("人気"), errors="coerce")
        _cls = pd.to_numeric(pdf.get("クラス_num"), errors="coerce")
        _fav = _mr[_pop == 1]
        _favv = float(_fav.iloc[0]) if len(_fav) and pd.notna(_fav.iloc[0]) else np.nan
        _clsv = float(_cls.dropna().iloc[0]) if _cls.notna().any() else np.nan
        _is_are = pd.notna(_favv) and _favv >= ARE_FAV_MR_MIN
        _is_cls4 = pd.notna(_clsv) and _clsv >= 4

        _od = pd.to_numeric(pdf.get("単勝オッズ"), errors="coerce")
        picks, names = [], []
        for nm, kind, bas, axr, mtr, olo, ohi, cond in ARE_PLANS:
            if cond == "are" and not _is_are:
                continue
            if cond == "cls4" and not _is_cls4:
                continue
            r = _rank.get(bas)
            if r is None or not r.notna().any():
                continue
            # 軸は「順位≦axr かつ オッズが[olo,ohi)」。その中で順位が最小の1頭。
            # このオッズ条件を落とすと検証と別の買い方になる（2026-08-16の反省）。
            _axm = (r <= axr) & (_od >= olo) & (_od < ohi)
            ax = pdf[_axm.fillna(False)]
            if ax.empty:
                continue
            ax = ax.loc[[r[ax.index].idxmin()]]
            mates = pdf[(r <= mtr) & (pdf["馬番"] != ax.iloc[0]["馬番"])]
            if mates.empty:
                continue
            picks.append({"名前": nm, "券種": kind,
                          "軸": ax.iloc[0]["馬番"],
                          "相手": mates["馬番"].tolist()})
            names.append(nm)
        if picks:
            _why = []
            if _is_are:
                _why.append(f"荒れR(1番人気=モデル{int(_favv)}位)")
            if _is_cls4:
                _why.append(f"クラス{int(_clsv)}")
            plan.update({"判定": "買い", "指数": 74, "サイズ": "標準",
                         "理由": f"{'・'.join(_why)} / {len(picks)}構成が成立",
                         "menu": [("馬単", "荒れR方式", 113)],
                         "are_picks": picks})
            return plan
        plan.update({"指数": 32,
                     "理由": (f"1番人気のモデル評価{int(_favv)}位（荒れR対象外）"
                            if pd.notna(_favv) and not _is_are
                            else "荒れR方式の条件に該当せず")})
        return plan
    cls_v = pd.to_numeric(pdf["クラス_num"].iloc[0], errors="coerce") \
        if "クラス_num" in pdf.columns else np.nan
    cls = float(cls_v) if pd.notna(cls_v) else np.nan

    # ── 複勝方式（2026-08-13・既定）────────────────────────────────────
    #   条件: 距離1900m以上・MF複勝1位・20倍以下・4番人気以下 → 複勝1点
    #   該当は年46点程度（週1点弱）。根拠は FUKU_* 定数のコメント参照。
    #   オッズに依存する条件が「20倍以下」だけなので、7分前に判定しても
    #   選ぶ馬がほぼ変わらない（模擬では124点中123点が同一）。
    if USE_FUKU_BETTING:
        _dist = pd.to_numeric(pdf.get("距離"), errors="coerce")
        _mr = pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce")
        _od = pd.to_numeric(pdf.get("単勝オッズ"), errors="coerce")
        _pop = pd.to_numeric(pdf.get("人気"), errors="coerce")
        # 距離が対象外・該当馬なしのときは複勝を見送るが、併用モードでは
        # EV方式の判定まで進む（EV方式に距離の縛りは無いため）。
        _skip = None
        if not (_dist.notna().any() and _mr.notna().any()):
            _skip = "距離またはモデル順位が取れず複勝方式は判定不可"
        elif float(_dist.dropna().iloc[0]) < FUKU_DIST_MIN:
            _skip = f"距離{FUKU_DIST_MIN}m未満（複勝方式の対象外）"
        if _skip is None:
            _hit = ((_mr == 1) & (_od <= FUKU_ODDS_MAX) & (_pop >= FUKU_POP_MIN))
            tgt = pdf[_hit.fillna(False)]
            if tgt.empty:
                _skip = "MF複勝1位が20倍以下かつ4番人気以下に該当せず"
        if _skip is not None:
            plan.update({"指数": 35, "理由": _skip})
            if not USE_DUAL_BETTING:
                return plan
        else:
            row = tgt.iloc[0]
            plan.update({"判定": "買い", "指数": 72, "サイズ": "標準",
                         "理由": (f"複勝方式（{int(_dist.dropna().iloc[0])}m・"
                                f"{row['馬名']} {int(row['人気'])}番人気 "
                                f"{float(row['単勝オッズ']):.1f}倍）"),
                         "menu": [("複勝", "複勝方式", 105)],
                         "fuku_picks": [row["馬番"]]})
            # 併用モード（2026-08-14）: EV方式も同時に記録する。
            #   EV方式は確定オッズ基準で119.6%だが、7分前で選ぶと88.4%に落ちる。
            #   この差は「賭ける時刻」の問題で、締切に近づけるほど縮む。
            #   1分前オッズを貯めて半年後に検証するため、それまで両方を走らせて
            #   実測を並行して集める。どちらが勝つかはデータに決めさせる。
            if not USE_DUAL_BETTING:
                return plan

    # ── 較正済み期待値方式（A案）──────────────────────────────────────
    #   条件: 乖離≥3・20倍以下・モデル複勝1位はEV≥1.7 / 2〜5位はEV≥2.2
    #   買い方: 該当馬のうち期待値が最大の1頭を単勝＋馬単。
    #
    #   ⚠ 判定に使うのは較正前のEV（＝バックテスト119.6%と同じ定義）。
    #     2次元較正した実効EVで判定すると該当が5年で0頭になり、実測が貯まらない。
    #     実効EVは併記して記録し、半年後に「どちらの判定が正しかったか」を見る。
    #
    #   ⚠ この方式は7分前で選ぶと88.4%（確定オッズ基準119.6%から-28.7pt）。
    #     差の原因は賭ける時刻で、締切に近づけるほど縮む。1分前オッズを
    #     半年貯めてから、投票時刻を早める価値があるかを検証する。
    if USE_EV_BETTING and "乖離" in pdf.columns and "単勝期待値" in pdf.columns:
        _mr = pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce")
        _gap = pd.to_numeric(pdf["乖離"], errors="coerce")
        _od = pd.to_numeric(pdf["単勝オッズ"], errors="coerce")
        _ev = pd.to_numeric(pdf["単勝期待値"], errors="coerce") + 1.0
        _hit = ((_gap >= EV_GAP_MIN) & (_od <= EV_ODDS_MAX) &
                (((_mr == 1) & (_ev >= EV_MIN_TOP)) |
                 (_mr.between(2, 5) & (_ev >= EV_MIN_SUB))))
        tgt = pdf[_hit.fillna(False)]
        if tgt.empty:
            if not plan.get("menu"):
                plan.update({"指数": 40, "理由": "期待値の条件を満たす馬なし"})
            return plan
        best = tgt.assign(_ev=_ev[tgt.index]).nlargest(EV_MAX_PICKS, "_ev")
        ev_max = float(_ev[best.index].max())
        _why = f"期待値{ev_max:.2f}（候補{len(tgt)}頭から1頭）"
        plan["判定"] = "買い"
        plan["サイズ"] = "標準"
        plan["指数"] = max(plan.get("指数", 0),
                          70 + min(30, int((ev_max - EV_MIN_TOP) * 20)))
        plan["理由"] = f"{plan['理由']} ＋ {_why}" if plan.get("menu") else _why
        plan["menu"] = list(plan.get("menu", [])) + [("単勝", "EV単勝", 113)]
        plan["ev_picks"] = best["馬番"].tolist()
        return plan

    # ── 妙味方式（2026-07-31・USE_EV_BETTING=False のとき）──────────────
    #   ★（モデルが市場より3順位以上高く評価・20倍以下）が付いた◎○▲だけを買う。
    #   3年実測: 単勝97.3%（97.8/95.1/98.9）・複勝93.1%・年997点。
    #   ★が無いレースは見送り（全体の66%）。従来のゲート方式は下に残してあり、
    #   USE_MYOMI_BETTING=False で戻せる。
    if USE_MYOMI_BETTING and "妙味" in pdf.columns:
        tgt = pdf[(pdf["妙味"] == "★") & (pdf["印"].isin(MYOMI_MARKS))]
        if tgt.empty:
            n_star = int((pdf["妙味"] == "★").sum())
            plan.update({"指数": 40,
                         "理由": f"妙味馬なし（★{n_star}頭・◎○▲に該当なし）"})
            return plan
        gap = pd.to_numeric(tgt["乖離"], errors="coerce").max()
        idx = 70 + min(30, int(max(0, gap - MYOMI_GAP_MIN) * 10))
        marks = "".join(tgt["印"].tolist())
        plan.update({"判定": "買い", "指数": idx, "サイズ": "標準",
                     "理由": f"妙味★{len(tgt)}頭({marks})・乖離最大{gap:.0f}"})
        # 印ごとに券種を変える（★◎は複勝・★○は単勝）。実際に居る印の分だけ組む。
        _roi = {"複勝": 113, "単勝": 100}
        plan["menu"] = [(k, f"★{mk}{k}", _roi.get(k, 100))
                        for mk in MYOMI_BETS if (tgt["印"] == mk).any()
                        for k in MYOMI_BETS[mk]]
        if plan["指数"] < BUY_INDEX_MIN:
            plan["menu"] = []
            plan["判定"] = "見送り"
        return plan

    # 軸を決める: 妙(MF勝率1位≠◎)があればそれ、無ければ◎
    myo = pdf[pdf["妙味軸"] == "◎妙"] if "妙味軸" in pdf.columns else pdf.iloc[0:0]
    hon = pdf[pdf["印"] == "◎"] if "印" in pdf.columns else pdf.iloc[0:0]
    has_myo = not myo.empty
    if has_myo:
        axis = myo.iloc[0]
    elif not hon.empty:
        axis = hon.iloc[0]
    else:
        plan["理由"] = "軸(◎)が決まらない"
        return plan

    # ── ゲート①: 軸の前走間隔 ──────────────────────────────────────────
    iv = pd.to_numeric(axis.get("前走間隔"), errors="coerce")
    if RACE_GATE_MIN_INTERVAL is None:
        pass                                   # 検証で無効化中
    elif pd.isna(iv):
        if not RACE_GATE_ALLOW_UNKNOWN:
            plan.update({"指数": 40, "理由": "軸の前走間隔が不明"})
            return plan
    elif float(iv) < RACE_GATE_MIN_INTERVAL:
        plan.update({"指数": 40,
                     "理由": f"軸の前走間隔{float(iv):.0f}週<{RACE_GATE_MIN_INTERVAL:.0f}週"
                             f"(実測63-71%・詰まったローテは市場も我々も沈む)"})
        return plan

    # ── ゲート②: クラス（中級は前後半とも一貫して悪い）────────────────────
    lo, hi = RACE_GATE_SKIP_CLASS if RACE_GATE_SKIP_CLASS else (None, None)
    if lo is not None and pd.notna(cls) and lo <= cls <= hi:
        plan.update({"指数": 45,
                     "理由": f"中級(クラス{int(cls)})=実測67.2%・前後半とも100%割れ"})
        return plan

    # ── 指数: ゲート通過70を起点に、優位が上乗せされる材料ごとに+10 ─────────
    #   妙あり     … 通過レースのうち妙ありは101.8%(妙なし込み全体92.1%)
    #   OP級       … 間隔ゲート通過のOP級は99.3%
    #   間隔15週以上 … 93.5%(≥8週全体は84.6%)
    idx, why = 70, []
    if has_myo:
        idx += 10
        why.append("妙あり")
    if pd.notna(cls) and cls >= 5:
        idx += 10
        why.append("OP級")
    if pd.notna(iv) and float(iv) >= 15:
        idx += 10
        why.append("間隔15週+")
    idx = min(idx, 100)
    size = "厚め" if idx >= 90 else "標準"

    iv_s = f"間隔{float(iv):.0f}週" if pd.notna(iv) else "間隔不明"
    plan.update({"判定": "買い", "指数": idx, "サイズ": size,
                 "理由": f"ゲート通過({iv_s}"
                         + (f"・クラス{int(cls)}" if pd.notna(cls) else "")
                         + ")" + ("＋" + "＋".join(why) if why else "")})

    # ── メニュー: 単勝＋馬連の2本のみ（軸が妙か◎かでラベルを切替）───────────
    if has_myo:
        plan["menu"] = [("単勝", "妙単勝", 87),
                        ("馬連", "馬連 妙-複勝上位5", 93)]
    else:
        plan["menu"] = [("単勝", "◎単勝", 87),
                        ("馬連", "馬連 ◎-複勝上位5", 93)]

    # 購入しきい値: 指数がBUY_INDEX_MIN未満なら買い目を出さない（判定・理由は残す）
    if plan["menu"] and plan["指数"] < BUY_INDEX_MIN:
        plan["理由"] += f" → 指数{plan['指数']}<購入閾値{BUY_INDEX_MIN}のため購入対象外"
        plan["menu"] = []
        plan["判定"] = "見送り"
    return plan


# ── 市場の歪み補正（2026-07-28導入）────────────────────────────────────────
# オッズとレース文脈だけから「そのレースで過剰人気になっている馬」を判定する。
# 2025検証(<=2024学習): 全馬の単勝ROI 72.25%→82.34% / 複勝 71.56%→81.40%
#   （補正係数の下位を除外した場合。固定の「100倍超を除外」78.83%より良い）
# 歪みは大穴に集中: 補正係数は30倍以下=0.94〜1.05 / 100-300倍=0.68 / 300倍超=0.32。
# ※逆（過小評価馬を買う）は期間を変えると再現しないので採用しない＝守りにのみ使う。
# 【2026-07-28 無効化】全馬購入なら72.25%→82.18%と効いたが、本番の買い方(妙=人気薄を
# 軸に高配当を狙う設計)に重ねると勝負帯80.9%→39.6%・全体72.2%と逆効果だった。
# 高オッズ馬を切ると「当たれば大きい買い目」だけが消え、外れ玉が残るため。
# 0で無効(除外しない)。全馬購入型の買い方に変える場合のみ0.90を検討する。
DEBIAS_MIN = 0.0
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

    # 期待値方式は買う馬が確定しているので、印の展開を通さず直接1点を組む
    # （2026-08-04）。印は表示用に残るが、買い目とは切り離す。
    # 荒れR方式（2026-08-16）: 採用した7構成をまとめて買い目にする。
    #   6構成が同じ「荒れR」を対象にするので、同じ組み合わせが重複しうる。
    #   同一の(券種,組み合わせ)は1点にまとめ、買い方の欄に全構成名を並べる。
    if plan.get("are_picks"):
        seen = {}
        for p in plan["are_picks"]:
            try:
                a = int(pd.to_numeric(p["軸"], errors="coerce"))
            except (TypeError, ValueError):
                continue
            for m in p["相手"]:
                try:
                    b = int(pd.to_numeric(m, errors="coerce"))
                except (TypeError, ValueError):
                    continue
                # 馬番は2桁ゼロ埋め。jv_payouts の組み合わせが "09-14" 形式なので、
                # ここを揃えないと1桁馬番の買い目が照合できず的中が消える
                # （2026-08-16に発生。的中54本が11本に見えていた）。
                if p["券種"] == "馬単裏":
                    kind, combo = "馬単", f"{b:02d}-{a:02d}"
                elif p["券種"] == "馬単表":
                    kind, combo = "馬単", f"{a:02d}-{b:02d}"
                else:
                    kind, combo = "馬連", f"{min(a,b):02d}-{max(a,b):02d}"
                key = (kind, combo)
                seen.setdefault(key, []).append(p["名前"])
        rows = []
        for (kind, combo), names in seen.items():
            rows.append({"race_id": race_id, "券種": kind,
                         "買い方": "荒れR方式", "組み合わせ": combo,
                         "BT回収率": 113, "判定": plan["判定"],
                         "サイズ": plan["サイズ"], "金額": ARE_STAKE,
                         "根拠構成": "/".join(sorted(set(names)))})
        return rows

    # 複勝方式は1頭を複勝で1点だけ買う（2026-08-13）。
    # 馬単の相手取りはしない。A案で相手を広げたのは単勝の的中率を補うためで、
    # 複勝は元々44.8%当たるので薄める意味がない。
    _fuku_rows = []
    for no in plan.get("fuku_picks", []):
        try:
            n = int(pd.to_numeric(no, errors="coerce"))
        except (TypeError, ValueError):
            continue
        _fuku_rows.append({"race_id": race_id, "券種": "複勝", "買い方": "複勝方式",
                           "組み合わせ": str(n), "BT回収率": 105,
                           "判定": plan["判定"], "サイズ": plan["サイズ"],
                           "金額": FUKU_STAKE})
    if _fuku_rows and not plan.get("ev_picks"):
        return _fuku_rows

    if plan.get("ev_picks"):
        rows = list(_fuku_rows)
        for no in plan["ev_picks"]:
            try:
                n = int(pd.to_numeric(no, errors="coerce"))
            except (TypeError, ValueError):
                continue
            r = pdf[pd.to_numeric(pdf["馬番"], errors="coerce") == n]
            ev = float(pd.to_numeric(r["単勝期待値"], errors="coerce").iloc[0]) + 1.0 \
                if len(r) else np.nan
            rows.append({"race_id": race_id, "券種": "単勝", "買い方": "EV単勝",
                         "組み合わせ": str(n), "BT回収率": 113,
                         "判定": plan["判定"], "サイズ": plan["サイズ"],
                         "金額": EV_STAKE,
                         "期待値": round(ev, 2) if pd.notna(ev) else ""})
            # 馬単: 軸を1着固定 → 相手は上位人気の印（2026-08-05）
            if USE_UMATAN:
                _mr = pd.to_numeric(pdf.get("MF複勝順位"), errors="coerce")
                _pr = pd.to_numeric(pdf["人気"], errors="coerce").rank(method="first")
                _bn = pd.to_numeric(pdf["馬番"], errors="coerce")
                _sub = pdf[_mr.isin(UMATAN_RANKS) & (_pr <= UMATAN_MAX_POP)
                           & (_bn != n)]
                for _, s in _sub.iterrows():
                    m = pd.to_numeric(s["馬番"], errors="coerce")
                    if pd.isna(m):
                        continue
                    rows.append({"race_id": race_id, "券種": "馬単",
                                 "買い方": "EV馬単", "組み合わせ": f"{n}-{int(m)}",
                                 "BT回収率": 134, "判定": plan["判定"],
                                 "サイズ": plan["サイズ"], "金額": UMATAN_STAKE,
                                 "期待値": ""})
        return rows

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
    myo = marks.get("妙")     # 妙が出ないレースは◎を軸にする
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

    # ★（妙味）が付いた馬を印ごとに引く。★◎は複勝、★○は単勝に使う。
    star_by_mark = {}
    if "妙味" in pdf.columns and "印" in pdf.columns:
        for mk in MYOMI_MARKS:
            _s = pdf[(pdf["妙味"] == "★") & (pdf["印"] == mk)]
            star_by_mark[mk] = [n for n in (_no(r) for _, r in _s.iterrows())
                                if n is not None]

    for kind, name, roi in plan["menu"]:
        # ── 妙味方式: ★◎複勝 / ★○単勝 のように印と券種が対になっている ──
        if name.startswith("★") and len(name) > 1 and name[1] in MYOMI_MARKS:
            for t in star_by_mark.get(name[1], []):
                add(kind, name, f"{t:02d}", roi)
        # ── ◎軸（妙が出ないレース）──
        elif name == "◎単勝":
            add(kind, name, f"{hon:02d}", roi)
        elif name == "馬単 ◎→○▲△":
            for t in [marks[m] for m in ("○", "▲", "△") if m in marks and marks[m] != hon]:
                add(kind, name, f"{hon:02d}-{t:02d}", roi)
        elif name == "3連複 ◎○▲":
            tri = [marks[m] for m in ("◎", "○", "▲") if m in marks]
            if len(set(tri)) == 3:
                x = sorted(tri)
                add(kind, name, f"{x[0]:02d}-{x[1]:02d}-{x[2]:02d}", roi)
        elif name.startswith("馬連 ◎-複勝上位"):   # 妙が出ないレースの◎軸馬連
            for t in _mf_partners({hon}, _tail_n(name)):
                add(kind, name, s2(hon, t), roi)
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
def predict_race(race_id: str, send_mail: bool = True, odds_only: bool = False):
    """odds_only=True なら締切直前のオッズだけ記録して抜ける（2026-08-14）。

    予想・メール・買い目保存・プッシュは一切しない。狙いは
    「7分前と締切直前でオッズがどれだけ動くか」の実測を貯めること。
    その差がEV方式の 119.6% → 88.4% を生んでいるので、
    投票を遅らせる価値があるかを半年後に判定する材料になる。
    """
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
            _mark_fallback(f"MFの読込に失敗: {str(e)[:120]}")
    else:
        models_pack["mf"] = None
        _mark_fallback("model_mf_parts/ も model_mf.pkl も見つからない")

    # ── 履歴データ読み込み
    print("履歴データ読み込み中...")
    history_df = pd.read_csv(os.path.join(BASE_DIR, "race_data_clean.csv"), low_memory=False)
    print(f"  読み込み完了: {len(history_df)}行")

    # ── 予測コア（共通エンジン）
    pdf = predict_race_pdf(race_id, history_df=history_df, models_pack=models_pack)
    if pdf is None:
        return

    # 締切直前ジョブはここで終わる。オッズは predict_race_pdf の中で
    # record_odds_snapshot により既に記録済みなので、追加の処理は要らない。
    if odds_only:
        print(f"  [オッズのみ] {race_id} を記録して終了（予想・買い目は出さない）")
        return pdf

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
        # "oddsonly" は締切直前ジョブ（オッズだけ記録して予想は出さない）
        predict_race(race_id, send_mail=("nomail" not in sys.argv),
                     odds_only=("oddsonly" in sys.argv))