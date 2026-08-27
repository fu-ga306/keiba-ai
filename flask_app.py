"""
Flask Dashboard - 競馬AI予想サイト
GitHub の today_predictions.csv / prediction_record_v2.csv を読み込んで表示する。
"""
import os
import pickle
import time
from functools import lru_cache
from io import StringIO

import numpy as np
import pandas as pd
import requests
from flask import (Flask, render_template, redirect, url_for, request,
                   jsonify, make_response)

app = Flask(__name__)


# ── 閲覧の制限（2026-08-27追加）────────────────────────────────────────
#   売り物を有料にしたのに、**同じ内容が別ページで無料公開されていた。**
#     /sale/<id>  合言葉が要る（有料）
#     /race/<id>  誰でも見られる ← 能力・乖離・印・gap が全部出ていた
#   URLを知っていれば合言葉を買う必要がない状態だった。認証は元々ゼロ。
#   自分専用のうちは問題なかったが、売り物にした時点で穴になった。
#
#   ⚠ これは強固な認証ではない。合言葉が共有されれば誰でも見られる。
#     ただし「無料で全部見える」よりはるかにマシで、規模に見合っている。
#     購読者が増えて実害が出たら本物の認証に移す。
_GATE_COOKIE = "keiba_k"
_OPEN_PREFIX = ("/sale", "/static", "/api", "/favicon")


def _gate_ok():
    """合言葉が通っているか。クエリ ?k= か、前に通したときのクッキー。"""
    import sale_gate
    k = request.args.get("k", "")
    if k and sale_gate.check(k):
        return True, k
    c = request.cookies.get(_GATE_COOKIE, "")
    if c and sale_gate.check(c):
        return True, c
    return False, k


@app.before_request
def _require_passphrase():
    path = request.path or "/"
    if path.startswith(_OPEN_PREFIX):
        return None
    # ⚠ remote_addr だけで判定してはいけない（2026-08-28に事故）
    #   ngrok は localhost:5000 に転送するので、**外部からのアクセスも
    #   remote_addr は 127.0.0.1 になる。** 「ローカルは通す」だけだと
    #   全員が素通りし、閲覧制限が完全に無効化される。実際にそうなっていた。
    #   転送されてきた要求には X-Forwarded-For が付くので、そこで見分ける。
    fwd = request.headers.get("X-Forwarded-For", "")
    local = (not fwd) and (request.remote_addr or "") in ("127.0.0.1", "::1", "localhost")
    if local:
        return None
    ok, k = _gate_ok()
    if ok:
        return None
    return make_response(render_template("gate.html", tried=bool(k)), 401)


@app.after_request
def _remember_passphrase(resp):
    """合言葉で通ったらクッキーに覚える。毎回入力させないため。"""
    k = request.args.get("k", "")
    if k and resp.status_code == 200:
        try:
            import sale_gate
            if sale_gate.check(k):
                # 8日。週が変わっても前週分は通るので、体感で切れない
                resp.set_cookie(_GATE_COOKIE, k, max_age=8 * 24 * 3600,
                                samesite="Lax")
        except Exception:
            pass
    return resp
app.config["TEMPLATES_AUTO_RELOAD"] = True

# ── 評価グレードのしきい値（2026-08-16・馬券内率そのもので定義）────────
#   これまで2回しきい値の引き方を間違えている。記録しておく。
#     ①「馬券内率70/50/30%」で切ったつもりが、実際は**上位からの累積平均**で
#       切っていた。累積平均はゆっくりしか下がらないので、各評価の実際の
#       馬券内率とは一致しない。Cが作れなかったのもこれが原因。
#     ②「1レースに何頭出すか」で切り直したが、これは頭数の話であって
#       「来るか来ないか」を表していない。
#
#   正しくは「スコア帯ごとの実際の馬券内率」を測り、目標の率になる境界で切る。
#   クリーンデータ5年OOS 207,518頭・14,972レースで、スコアを300分位に刻んで
#   各帯の馬券内率を実測し、70/45/12%を跨ぐ点をしきい値にした。
#
#   段階数は4つ（C廃止）。5段階だとCがDの手前の層を吸収してしまい、
#   Dの馬券内率が6.7%までしか下がらなかった。
#
#   BとDは1レース13.9頭を分け合う関係なので、片方を絞ると片方が太る。
#   Bが7.30頭では「押さえ」として広すぎたので、Bを約4頭に絞った。
#   その分Dは4.67→7.53頭になり、馬券内率は5.3%→8.9%に上がるが、
#   11回に1回しか来ないので「来ない馬」としては十分機能する。
#
#     評価  しきい値   馬券内率  勝率    1R当り  複勝ROI
#     S     ≥1.555     75.8%  42.0%   0.23頭   90.5%
#     A     ≥0.821     49.9%  20.0%   2.12頭   81.8%
#     B     ≥0.395     27.5%   8.3%   3.97頭   78.9%
#     D     それ未満     8.9%   2.0%   7.53頭   66.4%
#
#   Sは馬券内率75.8%で1番人気(64.7%)を大きく上回る。1レースに0.23頭しか出ない。
#   Dは8.9%＝11回に1回。明確に「来ない馬」として買わない判定に使う。
#   ⚠ しきい値を変えたら、この表も実測で更新すること。
#
# ── 2026-08-16 改定：市場（オッズ）を評価に入れた ────────────────────
#   上の表は「モデルだけ」で決めていたときのもの。モデルは市場をわざと見ない
#   設計だが、市場の重みはモデルの8〜9倍ある。市場を無視して評価を切ると、
#   市場が「来る」と言っている馬を取りこぼす。
#
#   複勝確率と勝率を、市場とモデルの2次元ロジスティックで出し直し、
#       score = P(3着以内) + P(1着)      （0〜2の範囲）
#   として固定しきい値で切る。固定なのは、少頭数レースで実力のない馬に
#   Sが付くのを防ぐため。較正器は grade_calib.pkl（build_grade.py で作成）。
#
#   walk-forward 5年OOS 207,518頭での実測:
#
#     評価  しきい値   馬券内率  連対率  勝率    1R当り  中央オッズ
#     S     ≥1.276     87.0%   77.5%  57.5%   0.11頭    1.5
#     A     ≥0.791     65.2%   51.7%  31.2%   1.00頭    2.7
#     B     ≥0.437     43.3%   29.8%  14.9%   2.35頭    5.5
#     D     それ未満    11.9%    6.7%   2.7%  10.39頭   45.1
#
#   Sは馬券内87.0%・勝率57.5%（現行は74.6%/41.0%）。9レースに1頭なので
#   1日36レースなら4頭ほど。年ごとの馬券内率は84.6〜89.2%で安定している。
#
#   ⚠ 正直に書いておくと、この改善はほぼ全部が「市場を入れたこと」による。
#     市場のみで切っても同じ人数で馬券内80.8%になり、モデルを足した効果は
#     +0.4pt程度しかない（係数も市場0.829に対しモデル0.117）。
#     評価は「当たる表示」としては良くなったが、モデルの手柄ではない。
#   ⚠ しきい値を変えたら build_grade.py を回し、この表も実測で更新すること。
GRADE_TH = [("S", 1.276), ("A", 0.791), ("B", 0.437)]
GRADE_NOBUY = "D"        # この評価は購入対象外として画面に明示する

_GCAL = None
_GCAL_LOADED = False
_G_EPS = 1e-6


def _gcal():
    """評価用の2次元較正器を読む。無ければ None（そのときはモデルのみで切る）。"""
    global _GCAL, _GCAL_LOADED
    if not _GCAL_LOADED:
        _GCAL_LOADED = True
        try:
            with open(os.path.join(BASE_DIR, "grade_calib.pkl"), "rb") as fh:
                _GCAL = pickle.load(fh)
        except FileNotFoundError:
            print("  評価用較正器なし（build_grade.py で作成できます）")
        except Exception as e:
            print(f"  評価用較正器の読込に失敗: {e}")
    return _GCAL


def _logit(v):
    v = np.clip(np.asarray(v, dtype=float), _G_EPS, 1 - _G_EPS)
    return np.log(v / (1 - v))


def grade_scores(odds, p_win, p_top3):
    """レース1つぶんの評価スコアを返す。市場とモデルを合わせた P(複勝)+P(1着)。

    odds/p_win/p_top3 は同じレースの全馬ぶんの配列。市場確率はレース内で
    正規化するので、レース単位で呼ぶこと。較正器が無ければ None を返す。
    """
    c = _gcal()
    if not c:
        return None
    o = pd.to_numeric(pd.Series(odds), errors="coerce")
    w = pd.to_numeric(pd.Series(p_win), errors="coerce")
    t = pd.to_numeric(pd.Series(p_top3), errors="coerce")
    if o.isna().all() or w.isna().all() or t.isna().all():
        return None
    imp = 1.0 / o.clip(lower=1.01)
    s = imp.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    lm = _logit(imp / s)
    X3 = pd.DataFrame({"lm": lm, "l3": _logit(t)})
    X3["i3"] = X3.lm * X3.l3
    X1 = pd.DataFrame({"lm": lm, "l1": _logit(w)})
    X1["i1"] = X1.lm * X1.l1
    try:
        p3 = c["m3"].predict_proba(X3[c["F3"]])[:, 1]
        p1 = c["m1"].predict_proba(X1[c["F1"]])[:, 1]
    except Exception:
        return None
    return pd.Series(p3 + p1, index=o.index)


# ── 残差モデルの印（2026-08-17）──────────────────────────────────────
#   gap = モデルの予測確率 ÷ 市場の確率。市場が見落としている度合い。
#   軸(1.5以上でレース内最大) → 単勝。相手(1.3以上) → 軸とのワイド・馬連。
#   数字の根拠は resid_io.py と SYSTEM.md「運用の現在形」を参照。
RESID_AX = 1.5          # 軸のしきい値（resid_io.AX_GAP と揃える）
RESID_MATE = 1.3        # 相手のしきい値（resid_io.MATE_GAP と揃える）
RESID_SUB = 1.1         # 押さえ


def resid_marks(horses):
    """各馬に残差モデルの印を付ける。gap が無ければ何もしない。

    h["r_mark"]  : "★軸" / "○" / "△" / ""
    h["r_gap"]   : gap（小数2桁）
    h["r_main"]  : True ならメインで買う馬（単勝の対象）
    """
    # ⚠ 列名は resid_gap。"gap" は既に「乖離（人気順位−モデル順位）」で
    #   使われているので衝突する（2026-08-17に実際に衝突した）。
    gs = [pd.to_numeric(h.get("resid_gap"), errors="coerce") for h in horses]
    if not any(pd.notna(g) for g in gs):
        for h in horses:
            h["r_mark"], h["r_gap"], h["r_main"] = "", None, False
        return
    top = max((g for g in gs if pd.notna(g)), default=None)
    for h, g in zip(horses, gs):
        h["r_gap"] = round(float(g), 2) if pd.notna(g) else None
        h["r_main"] = bool(pd.notna(g) and g == top and g >= RESID_AX)
        if h["r_main"]:
            h["r_mark"] = "★軸"
        elif pd.notna(g) and g >= RESID_MATE:
            h["r_mark"] = "○"
        elif pd.notna(g) and g >= RESID_SUB:
            h["r_mark"] = "△"
        else:
            h["r_mark"] = ""


def resid_bets(horses, baba=None):
    """残差モデルの買い目を作る（2026-08-22）。

    画面の「推奨買い目」を実態に合わせるための関数。旧方式（購入推奨・買い指数）は
    購入停止に伴い常に空になり、画面が何も出さなくなっていた。
    いま実際に記録しているのは残差モデルの買い目なので、それを出す。

    買い方は resid_io.pick_bets と同じ:
      軸  gapが最大かつ RESID_AX 以上           → 単勝1点
      相手 ダートのみ。gapが RESID_MATE 以上・最大3頭 → ワイド追加
      芝は単勝のみ

    返り値: (買い目リスト, 軸dict または None, 相手リスト)
    ⚠ 実際には購入していない。表示と記録だけ。
    """
    resid_marks(horses)
    ax = next((h for h in horses if h.get("r_main")), None)
    if ax is None:
        return [], None, []
    an = pd.to_numeric(ax.get("馬番"), errors="coerce")
    if pd.isna(an):
        return [], None, []
    an = str(int(an)).zfill(2)
    is_dirt = str(baba or "").startswith(("ダ", "ダート"))
    mates = []
    if is_dirt:
        cand = [h for h in horses
                if h is not ax and (h.get("r_gap") or 0) >= RESID_MATE]
        mates = sorted(cand, key=lambda h: -(h.get("r_gap") or 0))[:3]
    bets = [{"kind": "単勝", "combo": an, "points": 1, "role": "軸",
             "馬番": ax.get("馬番"), "馬名": ax.get("馬名"),
             "odds": ax.get("単勝オッズ"), "gap": ax.get("r_gap")}]
    for m in mates:
        bn = pd.to_numeric(m.get("馬番"), errors="coerce")
        if pd.isna(bn):
            continue
        bn = str(int(bn)).zfill(2)
        bets.append({"kind": "ワイド", "combo": f"{min(an,bn)}-{max(an,bn)}",
                     "points": 1, "role": "相手",
                     "馬番": m.get("馬番"), "馬名": m.get("馬名"),
                     "odds": m.get("単勝オッズ"), "gap": m.get("r_gap")})
    return bets, ax, mates


def grade_of(score):
    """スコアを評価に変換する。しきい値は GRADE_TH。"""
    if score is None or not np.isfinite(score):
        return "D"
    for g, th in GRADE_TH:
        if score >= th:
            return g
    return "D"


# 穴注目（2026-08-15追加）。10-12番人気でモデル複勝5位以内の馬。
#   5年OOSで勝率2.49%（同帯全体1.25%の2.00倍）・単勝ROI 99.0%。
#   13番人気以下は230頭で1着ゼロだったので対象外にしている。
#   「めったに出ないが、出たら本命級の価値」という位置づけ。
ANA_POP_MIN, ANA_POP_MAX, ANA_MF_MAX = 10, 12, 5

GITHUB_RAW = "https://raw.githubusercontent.com/fu-ga306/keiba-ai/main"
TODAY_PRED_URL = f"{GITHUB_RAW}/today_predictions.csv"
RECORD_URL = f"{GITHUB_RAW}/prediction_record_v2.csv"
TODAY_BETS_URL = f"{GITHUB_RAW}/today_bets.csv"
# 同じ競馬場の終了レースの着順・払戻（2026-08-07追加）。40分前ジョブが逐次取得する。
TODAY_RESULTS_URL = f"{GITHUB_RAW}/today_results.csv"

# flaskは予想を生成するのと同じPCで動くので、ローカルにファイルがあれば直接読む。
# → GitHub raw のCDNキャッシュ(max-age=300)＋自前キャッシュによる「判定は買いなのに
#   買い目が数分出ない」desyncを解消。ローカルが無い環境ではGitHub rawへフォールバック。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MAP = {
    TODAY_PRED_URL: "today_predictions.csv",
    RECORD_URL: "prediction_record_v2.csv",
    TODAY_BETS_URL: "today_bets.csv",
    TODAY_RESULTS_URL: "today_results.csv",
}

_cache = {}
CACHE_TTL = 300       # リモート取得時のキャッシュ(5分)
LOCAL_CACHE_TTL = 15  # ローカル読込時のキャッシュ(15秒・ほぼ即時反映)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    # race_idのfloat化('202602011201.0')はURL照合を外し詳細ページを404にする。除去。
    if "race_id" in df.columns:
        df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def _fill_resid_gap(df: pd.DataFrame) -> pd.DataFrame:
    """today_predictions.csv に resid_gap が無いとき、記録から軸だけ補う。

    2026-08-22まで、予想の保存より後に resid_gap を付けていたため、
    保存済みCSVに列が入らず画面に買い印が出なかった。予想側は直したが、
    その日のうちに出た古いCSVは列を持たないままなので、ここで補完する。

    ⚠ 補えるのは paper_resid.csv に載っている馬（軸と相手）だけ。
      他の馬の gap は分からないので空のままにする。
      次回の予想からは全馬に付くので、この補完は使われなくなる。
    """
    if "race_id" not in df.columns:
        return df
    # 列があっても「値が入っていない行」は埋める。
    # ⚠ 列の有無だけで判定すると、一部のレースにしか値が無いときに補完を
    #   飛ばしてしまい、残りが全部「見送り」になる（2026-08-22に実際に起きた）。
    if "resid_gap" in df.columns:
        if pd.to_numeric(df["resid_gap"], errors="coerce").notna().all():
            return df
    p = os.path.join(BASE_DIR, "paper_resid.csv")
    if not os.path.exists(p):
        return df
    try:
        r = pd.read_csv(p, dtype={"race_id": str})
        r = r[r["判定"].isin(("買い", "候補"))]
        if r.empty or "馬名" not in df.columns:
            return df
        key = r.drop_duplicates(["race_id", "馬名"]).set_index(["race_id", "馬名"])["gap"]
        fill = [key.get((a, b)) for a, b in zip(df["race_id"], df["馬名"])]
        if "resid_gap" in df.columns:
            cur = pd.to_numeric(df["resid_gap"], errors="coerce")
            df["resid_gap"] = [c if pd.notna(c) else f for c, f in zip(cur, fill)]
        else:
            df["resid_gap"] = fill
    except Exception:
        pass
    return df


def fetch_csv(url: str) -> pd.DataFrame:
    now = time.time()
    local_path = os.path.join(BASE_DIR, LOCAL_MAP[url]) if url in LOCAL_MAP else None
    use_local = local_path is not None and os.path.exists(local_path)
    ttl = LOCAL_CACHE_TTL if use_local else CACHE_TTL
    if url in _cache and now - _cache[url]["ts"] < ttl:
        return _cache[url]["df"].copy()
    try:
        if use_local:
            df = pd.read_csv(local_path, low_memory=False, dtype={"race_id": str})
        else:
            r = requests.get(url, timeout=15)
            r.encoding = "utf-8"
            df = pd.read_csv(StringIO(r.text), low_memory=False, dtype={"race_id": str})
        df = _finalize(df)
        _cache[url] = {"df": df, "ts": now}
        return df.copy()
    except Exception:
        return _cache.get(url, {}).get("df", pd.DataFrame()).copy()


RANK_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
SIGNAL_MAP = {
    "◎": ("軸", "axis"),
    "○": ("複勝圏", "fuku"),
    "▲": ("相手候補", "aite"),
    "△": ("3着候補", "chui"),
    "×": ("穴・ヒモ", "yosomi"),
}
TRACK_EMO = {"芝": "🌿", "ダ": "🟤", "ダート": "🟤"}
VALID_RANKS = set(RANK_ORDER.keys())  # {'◎', '○', '▲', '△', '×'}

# 買い対象の判定は keiba_predict を単一の情報源にする。
# ここに定数を書くと、片方だけ直して表示と実際の買い目がズレる。
# 2026-08-04: 買い方を較正済み期待値方式に変更。★や印はもう買いの根拠ではなく、
#   「乖離≥3・20倍以下・モデル1位ならEV≥1.7/2〜5位ならEV≥2.2」を満たす馬のうち
#   期待値最大の1頭だけを単勝で買う。印基準のままだと実際に買う馬と表示がズレる。
try:
    from keiba_predict import (MYOMI_BETS as _BUY_BETS, MYOMI_MARKS as _BUY_MARKS,
                               USE_EV_BETTING as _USE_EV, EV_GAP_MIN as _EV_GAP,
                               EV_ODDS_MAX as _EV_ODDS, EV_MIN_TOP as _EV_TOP,
                               EV_MIN_SUB as _EV_SUB)
except Exception:
    _BUY_BETS, _BUY_MARKS = {"◎": ("複勝",), "○": ("単勝",)}, ("◎", "○")
    _USE_EV, _EV_GAP, _EV_ODDS, _EV_TOP, _EV_SUB = False, 3.0, 20.0, 1.7, 2.2
try:
    from keiba_predict import (USE_UMATAN as _USE_UMATAN,
                               UMATAN_MAX_POP as _UM_MAX_POP,
                               UMATAN_RANKS as _UM_RANKS)
except Exception:
    _USE_UMATAN, _UM_MAX_POP, _UM_RANKS = False, 3, (1, 2, 3, 4, 5)


def _ev_candidates(group: pd.DataFrame) -> set:
    """期待値方式で買う馬の馬番を返す（keiba_predict の条件と同じ）。"""
    if not _USE_EV:
        return set()
    g = group.copy()
    mr = pd.to_numeric(g.get("MF複勝順位"), errors="coerce")
    gap = pd.to_numeric(g.get("乖離"), errors="coerce")
    od = pd.to_numeric(g.get("単勝オッズ"), errors="coerce")
    ev = pd.to_numeric(g.get("単勝期待値"), errors="coerce") + 1.0
    hit = ((gap >= _EV_GAP) & (od <= _EV_ODDS) &
           (((mr == 1) & (ev >= _EV_TOP)) | (mr.between(2, 5) & (ev >= _EV_SUB))))
    t = g[hit.fillna(False)]
    if t.empty:
        return set()
    best = t.assign(_ev=ev[t.index]).nlargest(1, "_ev")
    return {str(int(v)) for v in pd.to_numeric(best["馬番"], errors="coerce").dropna()}


def _result_map(race_id: str) -> dict:
    """このレースの確定結果を {馬番: {着順, 単勝, 複勝}} で返す。

    40分前ジョブとレース後の後片付けが today_results.csv に貯めたものを読むだけ。
    ダッシュボードから直接スクレイピングはしない（ブロック回避）。
    まだ確定していなければ空の辞書。
    """
    try:
        d = fetch_csv(TODAY_RESULTS_URL)
    except Exception:
        return {}
    if d is None or d.empty or "race_id" not in d.columns:
        return {}
    g = d[d["race_id"].astype(str) == str(race_id)]
    if g.empty:
        return {}
    out = {}
    for _, h in g.iterrows():
        bn = pd.to_numeric(h.get("馬番"), errors="coerce")
        pos = pd.to_numeric(h.get("着順"), errors="coerce")
        if pd.isna(bn) or pd.isna(pos):
            continue
        tan = pd.to_numeric(h.get("単勝"), errors="coerce")
        fuku = pd.to_numeric(h.get("複勝"), errors="coerce")
        out[int(bn)] = {
            "pos": int(pos),
            "tan": f"{int(tan):,}" if pd.notna(tan) and tan > 0 else "",
            "fuku": f"{int(fuku):,}" if pd.notna(fuku) and fuku > 0 else "",
        }
    return out


def _umatan_partners(group: pd.DataFrame, ax: set) -> set:
    """馬単の相手（2着に置く馬）を返す。keiba_predict の条件と同じ。"""
    if not _USE_EV or not _USE_UMATAN or not ax:
        return set()
    g = group.copy()
    mr = pd.to_numeric(g.get("MF複勝順位"), errors="coerce")
    pr = pd.to_numeric(g.get("人気"), errors="coerce").rank(method="first")
    bn = pd.to_numeric(g.get("馬番"), errors="coerce")
    sub = g[mr.isin(_UM_RANKS) & (pr <= _UM_MAX_POP)
            & (~bn.astype("Int64").astype(str).isin(ax))]
    return {str(int(v)) for v in pd.to_numeric(sub["馬番"], errors="coerce").dropna()}


def rank_sort_key(s):
    return s.map(lambda x: RANK_ORDER.get(str(x) if pd.notna(x) else "", 9))


def enrich_group(group: pd.DataFrame) -> list[dict]:
    rows = []
    _ev_buy = _ev_candidates(group)
    _ev_sub = _umatan_partners(group, _ev_buy)
    for _, r in group.iterrows():
        d = r.to_dict()
        raw_rank = d.get("推奨ランク", "")
        rank = str(raw_rank) if pd.notna(raw_rank) and str(raw_rank) in VALID_RANKS else ""
        d["推奨ランク"] = rank  # NaN → 空文字に正規化
        sig_label, sig_cls = SIGNAL_MAP.get(rank, ("穴・ヒモ", "yosomi"))
        d["signal_label"] = sig_label
        d["signal_cls"] = sig_cls
        # 2026-08-13: ★表示を廃止。乖離が大きいほどモデルが外すと検証で判明した
        # （実勝率/予測の比は 乖離0未満1.40 / 3-6で0.43 / 6以上で0.27）。
        # 較正を正すと差自体が消えるので、★は実体ではなく較正の歪みの裏返しだった。
        # 列は蓄積の連続性のため残すが、画面では強調しない。
        d["is_star"] = False
        _gap = pd.to_numeric(d.get("乖離"), errors="coerce")
        d["gap"] = f"{_gap:+.0f}" if pd.notna(_gap) else ""
        d["gap_val"] = float(_gap) if pd.notna(_gap) else -99
        # 買い対象＝★かつ◎○▲。△×の★は参考表示に留める。
        # 買い対象の印は keiba_predict 側の設定を正とする（ここに書くと二重管理になる。
        # 2026-08-01: ▲を除外したのにダッシュボードだけ緑のままだった）。
        if _USE_EV:
            _bn = pd.to_numeric(d.get("馬番"), errors="coerce")
            _k = str(int(_bn)) if pd.notna(_bn) else ""
            # 軸＝単勝を買う馬。相手＝馬単の2着に置く馬（上位人気の印）。
            # どちらも「買い対象」として色を付けるが、役割が分かるよう券種を書き分ける。
            if _k in _ev_buy:
                d["is_buy"], d["buy_kinds"] = True, ("単勝・馬単軸" if _ev_sub
                                                     else "単勝")
            elif _k in _ev_sub:
                d["is_buy"], d["buy_kinds"] = True, "馬単の相手"
            else:
                d["is_buy"], d["buy_kinds"] = False, ""
        else:
            d["is_buy"] = d["is_star"] and rank in _BUY_MARKS
            d["buy_kinds"] = "・".join(_BUY_BETS.get(rank, ())) if d["is_buy"] else ""
        _ev_v = pd.to_numeric(d.get("単勝期待値"), errors="coerce")
        d["ev_val"] = round(float(_ev_v) + 1.0, 2) if pd.notna(_ev_v) else None
        # AI予想順位＝市場フリーモデルの3着内予測順（印の根拠そのもの）。
        # 印は上位5頭にしか付かないので、6位以下の序列を見るためにも列で出す。
        _ai = pd.to_numeric(d.get("MF複勝順位"), errors="coerce")
        d["ai_rank"] = int(_ai) if pd.notna(_ai) else None
        d["ai_rank_val"] = float(_ai) if pd.notna(_ai) else 999
        d["win_pct"] = f"{float(d.get('勝ち確率', 0)) * 100:.1f}"
        d["ren_pct"] = f"{float(d.get('連対確率', 0)) * 100:.1f}"
        d["fuku_pct"] = f"{float(d.get('複勝確率', 0)) * 100:.1f}"
        # 2026-08-05: ここで ev_val を上書きしていたため、上で計算した正しい値
        #   （単勝期待値+1.0＝確率×オッズ）が「確率×オッズ−1」に戻っていた。
        #   買い判定の閾値(EV_MIN_TOP=1.7等)は「確率×オッズ」基準なので、
        #   表示も同じ土俵に揃える。1.0が損益分岐。
        d["ev_cls"] = ("ev-good" if (d["ev_val"] or 0) >= 1.3
                       else "ev-ok" if (d["ev_val"] or 0) >= 1.0 else "ev-bad")
        rows.append(d)
    return rows


def calc_are_score(group: pd.DataFrame) -> tuple[int, int]:
    """荒れスコア: 単勝向け / 連系向け (0-100, 高=荒れやすい)"""
    try:
        top1_win = group.nlargest(1, "勝ち確率")["勝ち確率"].iloc[0]
        tansho_are = max(0, min(100, int(100 - top1_win * 180)))
    except Exception:
        tansho_are = 50
    try:
        top2_ren = group.nlargest(2, "連対確率")["連対確率"].sum()
        rentan_are = max(0, min(100, int(100 - top2_ren * 80)))
    except Exception:
        rentan_are = 50
    return tansho_are, rentan_are


def are_label(score: int) -> str:
    if score >= 70:
        return ("高", "are-high")
    elif score >= 40:
        return ("中", "are-mid")
    else:
        return ("低", "are-low")


def build_bet_recs(group: pd.DataFrame, are_tan: int = 50, are_ren: int = 50) -> list[dict]:
    """荒れスコア・EV・MF穴馬の有無に応じて馬券種を動的に決定する"""
    recs = []

    def hl(r):
        _u = pd.to_numeric(r.get("馬番"), errors="coerce")
        return f"馬番{int(_u) if pd.notna(_u) else '-'} {r.get('馬名', '')}"

    # 基本情報
    axis = group[group["推奨ランク"] == "◎"]
    axis_ev = float(axis.iloc[0].get("単勝期待値", 0) or 0) if len(axis) > 0 else 0
    axis_odds = axis.iloc[0].get("単勝オッズ", "-") if len(axis) > 0 else "-"

    top2_win  = group.nlargest(2, "勝ち確率")
    top3_win  = group.nlargest(3, "勝ち確率")
    top3_fuku = group.nlargest(3, "複勝確率")

    # MF穴馬: AI推奨(◎○▲)かつ4番人気以下
    try:
        mf_ana = group[
            group["推奨ランク"].isin(["◎", "○", "▲"]) &
            (pd.to_numeric(group["人気"], errors="coerce") >= 4)
        ]
        has_ana = len(mf_ana) > 0
    except Exception:
        mf_ana = pd.DataFrame()
        has_ana = False

    # 荒れ判定
    low_are  = are_tan < 35 and are_ren < 35   # 安定
    high_are = are_tan >= 65 or are_ren >= 65   # 荒れ
    # mid_are = それ以外

    if low_are:
        # ── 安定型: 本命中心 ──────────────────────────
        strategy = "🔒 安定型"
        if axis_ev >= 1.2 and len(axis) > 0:
            recs.append({"type": "単勝◎", "cls": "bet-tansho",
                         "horses": [hl(axis.iloc[0])],
                         "note": f"EV:{axis_ev:.2f} / {axis_odds}倍",
                         "strategy": strategy})
        if len(axis) > 0:
            recs.append({"type": "複勝◎", "cls": "bet-fuku",
                         "horses": [hl(axis.iloc[0])],
                         "note": "軸1頭固め", "strategy": strategy})
        if len(top2_win) >= 2:
            recs.append({"type": "馬連", "cls": "bet-uren",
                         "horses": [hl(r) for _, r in top2_win.iterrows()],
                         "note": "◎○固め", "strategy": strategy})

    elif high_are:
        # ── 荒れ型: 穴狙い or 見送り ─────────────────
        strategy = "💥 荒れ型"
        if has_ana:
            recs.append({"type": "ワイド穴", "cls": "bet-wide",
                         "horses": [hl(r) for _, r in mf_ana.head(2).iterrows()],
                         "note": "MF穴馬軸・高配当狙い", "strategy": strategy})
            if len(top3_win) >= 3:
                recs.append({"type": "三連複", "cls": "bet-sanrenfuku",
                             "horses": [hl(r) for _, r in top3_win.iterrows()],
                             "note": "穴含みBOX", "strategy": strategy})
        else:
            recs.append({"type": "見送り", "cls": "bet-pass",
                         "horses": [],
                         "note": "荒れ度高・明確な穴候補なし", "strategy": strategy})

    else:
        # ── バランス型: 状況に応じて ──────────────────
        strategy = "⚖️ バランス型"
        if axis_ev >= 1.0 and len(axis) > 0:
            recs.append({"type": "単勝◎", "cls": "bet-tansho",
                         "horses": [hl(axis.iloc[0])],
                         "note": f"EV:{axis_ev:.2f} / {axis_odds}倍",
                         "strategy": strategy})
        if len(top3_fuku) >= 3:
            recs.append({"type": "複勝3点", "cls": "bet-fuku",
                         "horses": [hl(r) for _, r in top3_fuku.iterrows()],
                         "note": "", "strategy": strategy})
        if len(top2_win) >= 2:
            note = "穴含み2頭" if has_ana else "2頭流し"
            recs.append({"type": "ワイド", "cls": "bet-wide",
                         "horses": [hl(r) for _, r in top2_win.iterrows()],
                         "note": note, "strategy": strategy})
        if has_ana and len(top3_win) >= 3:
            recs.append({"type": "三連複", "cls": "bet-sanrenfuku",
                         "horses": [hl(r) for _, r in top3_win.iterrows()],
                         "note": "穴馬含みBOX", "strategy": strategy})

    return recs


def calc_rec_level(group: pd.DataFrame, are_tan: int, are_ren: int, bet_recs: list) -> tuple:
    """レース推奨レベルを返す (label, css_class, score)。

    2026-08-22改定: 残差モデルの判定を最優先にする。
      いま実際に見ているのは残差モデル（gap＝モデルの予測確率÷市場の確率）で、
      旧方式（購入推奨・買い指数）は購入停止に伴い常に「見送り」になる。
      画面が全レース見送りに見えるのに記録では32レースが「買い」という
      食い違いが起きていたので、画面を実態に合わせる。

      ★軸あり（gap>=1.5）→「★軸」
      それ未満          →「見送り」
    """
    g = pd.to_numeric(group.get("resid_gap"), errors="coerce")
    if g is not None and g.notna().any():
        mx = float(g.max())
        if mx >= RESID_AX:
            # gapが大きいほど強い。1.5→60, 2.0→75, 3.0→90 くらいの目盛り
            sc = int(min(99, 45 + (mx - 1.0) * 30))
            return ("★軸", "rec-hot" if mx >= 2.0 else "rec-good", sc)
        return "見送り", "rec-pass", int(max(0, mx * 30))

    # 見送り判定
    if bet_recs and bet_recs[0]["type"] == "見送り":
        return "見送り", "rec-pass", 0

    # ── レース判定ベース（2026-07-16: 購入推奨=勝負/買い/少額/見送り に連動）──
    if "購入推奨" in group.columns:
        lab_s = group["購入推奨"].dropna().astype(str)
        lab = lab_s.iloc[0] if len(lab_s) else ""
        kai_s = pd.to_numeric(group.get("買い指数"), errors="coerce").dropna()
        kai = int(kai_s.iloc[0]) if len(kai_s) else 0
        if lab == "勝負":
            return "🔥勝負", "rec-hot", kai
        elif lab == "買い":
            return "✅買い", "rec-good", kai
        elif lab == "堅実":
            return "🟢堅実", "rec-good", kai
        elif lab == "少額":
            return "⚠少額", "rec-mid", kai
        elif lab == "見送り":
            return "見送り", "rec-pass", kai
    # ── 旧・買い指数ベース（85+積極/70-84買い/55-69様子見/<55単勝見送り）──
    if "買い指数" in group.columns:
        kai_s = pd.to_numeric(group["買い指数"], errors="coerce").dropna()
        if len(kai_s) > 0:
            kai = int(kai_s.iloc[0])
            if kai >= 85:
                return "積極", "rec-good", kai
            elif kai >= 70:
                return "買い", "rec-good", kai
            elif kai >= 55:
                return "様子見", "rec-mid", kai
            else:
                return "単勝見送り", "rec-watch", kai

    # ── フォールバック（旧データ・EVベース）──
    axis = group[group["推奨ランク"] == "◎"]
    axis_ev = float(axis.iloc[0].get("単勝期待値", 0) or 0) if len(axis) > 0 else 0
    score = axis_ev * 100
    if are_tan < 35 and are_ren < 35:
        score += 15
    if axis_ev >= 1.2:
        return "推奨", "rec-good", score
    elif axis_ev >= 1.0:
        return "参考", "rec-mid", score
    else:
        return "様子見", "rec-watch", score


def _fmt_axis_partners(kind: str, combos: list[str]) -> str:
    """組み合わせ群を「軸→相手」の形で表示。相手(複勝上位)の馬番が全部見えるようにする。
    例: 馬単 09-04,09-01,... → "9→4・1・7・10・3・2" / 3連複 01-03-04,... → "1・3-4・6・…"。"""
    lists = [str(c).split("-") for c in combos]
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

    if 1 <= len(common) <= 2 and len(combos) >= 2:
        axis = "・".join(_i(x) for x in lists[0] if x in common)   # 軸(元の並び順)
        partners = []
        for p in lists:
            for x in p:
                if x not in common and x not in partners:
                    partners.append(x)
        arrow = "→" if kind in ("馬単", "3連単") else "-"
        return f"{axis}{arrow}" + "・".join(_i(x) for x in partners)
    if len(combos) <= 6:
        return " / ".join(combos)
    return f"{combos[0]} 他{len(combos) - 1}点"


def build_my_bets(race_id: str, bets_df: pd.DataFrame) -> list[dict]:
    """today_bets.csv から当該レースの推奨買い目を表示用に集約する。
    買い方ごとに1行（点数・BT回収率・組み合わせ要約）。"""
    if bets_df is None or bets_df.empty or "race_id" not in bets_df.columns:
        return []
    sub = bets_df[bets_df["race_id"].astype(str) == str(race_id)]
    if sub.empty:
        return []
    out = []
    order = {"単勝": 0, "複勝": 1, "馬単": 2, "馬連": 3, "ワイド": 4, "3連複": 5, "3連単": 6}
    for (kind, name), g in sub.groupby(["券種", "買い方"], sort=False):
        combos = [str(c) for c in g["組み合わせ"]]
        if "総流し" in str(name):
            axis = combos[0].split("-")[0]
            combo_s = f"{int(axis)} → 総流し"
        else:
            combo_s = _fmt_axis_partners(kind, combos)   # 軸→相手(馬番)を全部表示
        bt = g["BT回収率"].iloc[0] if "BT回収率" in g.columns else ""
        amt = int(pd.to_numeric(g["金額"], errors="coerce").fillna(100).sum()) if "金額" in g.columns else len(g) * 100
        out.append({"kind": kind, "name": str(name), "combo": combo_s,
                    "points": len(g), "bt": bt, "amount": amt, "ord": order.get(kind, 9)})
    return sorted(out, key=lambda x: x["ord"])


def build_race_card(jyo, race_no, group: pd.DataFrame, bets_df: pd.DataFrame = None) -> dict:
    group = group.copy()
    are_tan, are_ren = calc_are_score(group)
    bet_recs = build_bet_recs(group, are_tan, are_ren)
    sorted_g = group.sort_values("推奨ランク", key=rank_sort_key)
    # 印あり馬のみカード表示（◎○▲△×のみ）
    marked = sorted_g[sorted_g["推奨ランク"].isin(VALID_RANKS)]
    top5 = enrich_group(marked.head(5))

    first = group.iloc[0]
    dist = first.get("距離", "")
    baba = first.get("馬場", "")
    baba_emo = TRACK_EMO.get(str(baba), "")
    cond = first.get("馬場状態", "")
    klass = first.get("クラス", "")
    race_id = str(first.get("race_id", ""))

    are_tan_lbl, are_tan_cls = are_label(are_tan)
    are_ren_lbl, are_ren_cls = are_label(are_ren)

    # 激熱バッジは廃止（2026-08-14）。
    #   根拠にしていた2つの指標が、どちらも検証で否定されたため。
    #     EV>=1.2      : EVと実払戻の順位相関は5年OOFすべて負。実効EVは0.95
    #     MF上位×人気薄 : 順列検定で否定（探索を織り込んだ family-wise p=0.598）
    #   さらにこれらの根拠値は時系列リーク混入前のバックテストで、
    #   クリーンデータでは100%を超える構成が1つも残っていない。
    #   購入判定を全停止したのに画面だけが「買え」と言い続ける状態も避ける。
    hot_badges = []

    # レース推奨レベルとスコアを算出
    rec_level, rec_cls, race_score = calc_rec_level(group, are_tan, are_ren, bet_recs)

    # レース単位の買い目プラン（today_bets.csv連動・2026-07-16）
    my_bets = build_my_bets(race_id, bets_df)
    plan_reason = str(first.get("想定単回収", "") or "")
    plan_size = str(first.get("買いサイズ", "") or "")
    myo_row = group[group.get("妙味軸", pd.Series(dtype=str)) == "◎妙"] if "妙味軸" in group.columns else group.iloc[0:0]
    myo_info = ""
    if len(myo_row):
        _m = myo_row.iloc[0]
        _pop = pd.to_numeric(_m.get("人気"), errors="coerce")
        _ban = pd.to_numeric(_m.get("馬番"), errors="coerce")
        if pd.notna(_ban):
            myo_info = f"◎妙 {int(_ban)}番 {_m.get('馬名', '')}"
            if pd.notna(_pop):
                myo_info += f"（{int(_pop)}人気）"

    return {
        "jyo": jyo,
        "race_no": race_no,
        "race_id": race_id,
        "my_bets": my_bets,
        "plan_reason": plan_reason,
        "plan_size": plan_size,
        "myo_info": myo_info,
        "dist": dist,
        "baba": baba,
        "baba_emo": baba_emo,
        "cond": cond,
        "klass": klass,
        "are_tan": are_tan,
        "are_ren": are_ren,
        "are_tan_lbl": are_tan_lbl,
        "are_tan_cls": are_tan_cls,
        "are_ren_lbl": are_ren_lbl,
        "are_ren_cls": are_ren_cls,
        "top5": top5,
        "bet_recs": bet_recs,
        "hot_badges": hot_badges,
        "rec_level": rec_level,
        "rec_cls": rec_cls,
        "race_score": race_score,
    }


@app.route("/")
def index():
    return redirect(url_for("races"))


def _keep_latest_meet_day(df):
    """前日データ混入除去: 各競馬場(race_id 5-6桁)で最新の開催日(9-10桁)のレースのみ残す。
    スケジューラが前日ジョブを再発火して古いrace_idが混ざるための表示側フィルタ。"""
    if df.empty or "race_id" not in df.columns:
        return df
    rid = df["race_id"].astype(str)
    jyo_cd = rid.str[4:6]
    day = rid.str[8:10]
    return df[day == day.groupby(jyo_cd).transform("max")].copy()


@app.route("/races")
def races():
    df = _fill_resid_gap(fetch_csv(TODAY_PRED_URL))
    if df.empty:
        return render_template("error.html", msg="予想データが取得できませんでした。しばらく後に再度お試しください。")
    df = _keep_latest_meet_day(df)  # 前日データ混入を除去

    # jyo, race_no が存在するか確認
    if "jyo" not in df.columns:
        df["jyo"] = df["race_id"].astype(str).str[4:6]
    if "race_no" not in df.columns:
        df["race_no"] = df["race_id"].astype(str).str[10:12].astype(int)

    date_str = ""
    if "予想日時" in df.columns:
        date_str = str(df["予想日時"].iloc[0])[:10]

    bets_df = fetch_csv(TODAY_BETS_URL)

    cards = []
    for (jyo, race_no), group in df.groupby(["jyo", "race_no"], sort=True):
        cards.append(build_race_card(jyo, race_no, group, bets_df))

    # 推奨レース（🔥勝負・✅買い）を指数順にハイライト抽出
    highlights = sorted(
        [c for c in cards if c["rec_cls"] in ("rec-hot", "rec-good")],
        key=lambda c: c["race_score"], reverse=True
    )[:8]

    return render_template("races.html", cards=cards, highlights=highlights, date_str=date_str)


def _race_context(race_id, limit=None):
    """レース詳細の表示に必要なものを全部そろえる。

    ⚠ /race と /sale で**同じものを見せる**ためにここに集約した（2026-08-28）。
      販売用に別UIを作ったが失敗した。確率順に並べると1位は平均1.1番人気で、
      「AIの推奨＝1番人気」となり、買う人が既に知っていることしか出せない。
      乖離を前面に出す案も測ったが、実測で
        AIが人気より高く見た馬 馬券内17.6% / 市場と同じ 25.4%
      と**差が出るほど成績が悪く**、売り物にならなかった。
      的中率と妙味は必ず逆を向くので、片方を前面に出す設計は成立しない。
      元の「全部見せて買う人が選ぶ」形に戻し、limit で頭数だけ絞る。
    """
    return _race_detail_impl(race_id, limit)


@app.route("/race/<race_id>")
def race_detail(race_id):
    return _race_detail_impl(race_id, None)


def _race_detail_impl(race_id, limit=None, template="race_detail.html", extra=None):
    df = _fill_resid_gap(fetch_csv(TODAY_PRED_URL))
    if df.empty:
        return render_template("error.html", msg="データ取得失敗")

    group = df[df["race_id"].astype(str) == str(race_id)].copy()
    if group.empty:
        return render_template("error.html", msg=f"レース {race_id} が見つかりません")

    sorted_g = group.sort_values("推奨ランク", key=rank_sort_key)
    horses = enrich_group(sorted_g)
    n_all = len(horses)
    if limit is not None:
        horses = horses[:limit]      # 販売の無料枠。上位だけ見せる
    are_tan, are_ren = calc_are_score(group)
    bet_recs = build_bet_recs(group, are_tan, are_ren)

    # レース判定＋推奨買い目（today_bets.csv連動）
    bets_df = fetch_csv(TODAY_BETS_URL)
    my_bets = build_my_bets(race_id, bets_df)
    rec_level, rec_cls, _ = calc_rec_level(group, are_tan, are_ren, bet_recs)

    first = group.iloc[0]
    plan_reason = str(first.get("想定単回収", "") or "")
    plan_size = str(first.get("買いサイズ", "") or "")
    jyo = first.get("jyo", "")
    race_no = first.get("race_no", "")
    dist = first.get("距離", "")
    baba = first.get("馬場", "")
    baba_emo = TRACK_EMO.get(str(baba), "")
    cond = first.get("馬場状態", "")
    klass = first.get("クラス", "")
    date_str = str(first.get("予想日時", ""))[:10]

    are_tan_lbl, are_tan_cls = are_label(are_tan)
    are_ren_lbl, are_ren_cls = are_label(are_ren)

    # 能力値パラメータ（印付き馬＋◎妙、レース内百分位0-100を6軸バー表示）
    ABILITY_AXES = [("能力_勝負", "勝負力"), ("能力_安定", "安定感"), ("能力_末脚", "末脚"),
                    ("能力_先行", "先行力"), ("能力_距離", "距離適性"), ("能力_実績", "実績")]
    # 2026-08-06: 印付き馬＋◎妙だけに絞っていたが、全頭を出すよう変更。
    #   このモデルはオッズ・人気を一切見ない能力評価なので、印が付かなかった馬の
    #   能力値にも意味がある（なぜ選ばれなかったかが読める）。
    #   ただし18頭ぶん並べると見づらいので、印付きを先頭にしてそれ以外は
    #   テンプレート側で折りたたむ（is_sub で判別）。
    param_horses = []
    if "能力_勝負" in group.columns:
        for h in horses:
            is_myo = str(h.get("妙味軸", "")) == "◎妙"
            is_marked = (h.get("推奨ランク") in VALID_RANKS) or is_myo
            abilities = []
            for col, label in ABILITY_AXES:
                v = pd.to_numeric(h.get(col), errors="coerce")
                abilities.append({"label": label, "val": int(v) if pd.notna(v) else None})
            ban = pd.to_numeric(h.get("馬番"), errors="coerce")
            param_horses.append({
                "mark": h.get("推奨ランク") or ("◎妙" if is_myo else ""),
                "is_myo": is_myo,
                "is_sub": not is_marked,      # 印なし＝折りたたみ側
                "ban": int(ban) if pd.notna(ban) else "-",
                "name": h.get("馬名", ""),
                "signal_cls": h.get("signal_cls", ""),
                "abilities": abilities,
            })
        # 印付きを先に、その中はAI順位順。印なしもAI順位順で後ろに続ける。
        param_horses.sort(key=lambda p: (p["is_sub"], p.get("ban") if isinstance(p.get("ban"), int) else 99))
        _ai = {}
        for h in horses:
            b = pd.to_numeric(h.get("馬番"), errors="coerce")
            if pd.notna(b):
                _ai[int(b)] = h.get("ai_rank_val", 999)
        param_horses.sort(key=lambda p: (p["is_sub"],
                                         _ai.get(p.get("ban"), 999)))

    # 穴馬候補: AI高推奨だが人気薄の馬
    ana_candidates = [
        h for h in horses
        if h.get("signal_cls") in ("axis", "fuku", "aite")
        and pd.notna(pd.to_numeric(h.get("人気"), errors="coerce"))
        and pd.to_numeric(h.get("人気"), errors="coerce") >= 4
    ]

    # 確定していれば各馬に着順・払戻を付ける（2026-08-07）。
    # 結果は別枠にせず、その馬の行に横並びで出す。
    rmap = _result_map(race_id)
    # 評価スコアはレース単位で作る（市場確率をレース内で正規化するため）。
    # 較正器が無い環境では None が返るので、その場合は旧方式（モデルのみ）に落ちる。
    _gsc = grade_scores([h.get("単勝オッズ") for h in horses],
                        [h.get("勝ち確率") for h in horses],
                        [h.get("複勝確率") for h in horses])
    for _i, h in enumerate(horses):
        bn = pd.to_numeric(h.get("馬番"), errors="coerce")
        r = rmap.get(int(bn)) if pd.notna(bn) else None
        h["res_pos"] = r["pos"] if r else None
        h["res_tan"] = r["tan"] if r else ""
        h["res_fuku"] = r["fuku"] if r else ""

        # 評価グレード（2026-08-16改定）。しきい値と根拠は GRADE_TH のコメント参照。
        #   市場（オッズ）とモデルを合わせた P(3着以内)+P(1着) で切る。
        #   較正器が読めないときだけ、旧方式（モデルの勝率+連対率+複勝率）に落ちる。
        #   旧方式は市場を見ないぶん精度が落ちる（Sの馬券内 87.0%→74.6%）ので、
        #   あくまで非常用。grade_calib.pkl が無ければ build_grade.py で作ること。
        if _gsc is not None and pd.notna(_gsc.iloc[_i]):
            score = float(_gsc.iloc[_i])
        else:
            _p = [pd.to_numeric(h.get(c), errors="coerce")
                  for c in ("勝ち確率", "連対確率", "複勝確率")]
            score = float(sum(float(v) for v in _p if pd.notna(v)))
        h["grade_score"] = round(score, 2)
        h["grade"] = grade_of(score)
        # 穴注目（根拠は ANA_POP_MIN 付近のコメント）。人気薄でモデルだけが推す馬。
        _pop = pd.to_numeric(h.get("人気"), errors="coerce")
        _mfr = pd.to_numeric(h.get("MF複勝順位"), errors="coerce")
        h["is_ana"] = bool(
            pd.notna(_pop) and pd.notna(_mfr)
            and ANA_POP_MIN <= _pop <= ANA_POP_MAX and _mfr <= ANA_MF_MAX)
        # 能力総合点＝6軸（レース内百分位0-100）の平均。オッズを一切見ない評価。
        vals = [pd.to_numeric(h.get(c), errors="coerce") for c, _ in ABILITY_AXES]
        vals = [float(v) for v in vals if pd.notna(v)]
        h["abil_avg"] = round(sum(vals) / len(vals), 1) if vals else None

        # ── 馬体重（2026-08-22追加）──────────────────────────────
        #   馬体重はレースの約50分前に発表される。朝の一括予想（6:55〜）の
        #   時点では全馬が未発表なので、値が無いことは異常ではない。
        #   「まだ出ていない」のか「取得に失敗した」のかを画面で区別できるよう、
        #   未発表は "―" と出し、発表済みなら増減も併せて出す。
        _w = pd.to_numeric(h.get("馬体重"), errors="coerce")
        _dw = pd.to_numeric(h.get("体重増減"), errors="coerce")
        h["w_val"] = int(_w) if pd.notna(_w) else None
        h["w_diff"] = int(_dw) if pd.notna(_dw) else None
        # 増減の大きさで色を変える。±20kg以上は明らかな変化として目立たせる。
        h["w_cls"] = ("" if pd.isna(_dw) else
                      "w-big" if abs(_dw) >= 20 else
                      "w-mid" if abs(_dw) >= 10 else "")

    # 馬体重が発表済みか（レース単位）。1頭でも入っていれば発表済みとみなす。
    _w_done = sum(1 for h in horses if h.get("w_val") is not None)
    weight_status = {
        "done": _w_done > 0,
        "n": _w_done,
        "total": len(horses),
        "label": (f"発表済み（{_w_done}/{len(horses)}頭）" if _w_done
                  else "未発表（発走50分前ごろに出ます）"),
    }

    # ── 残差モデルの印（2026-08-17）────────────────────────────────
    #   このモデルは「市場（オッズ）が何を見落としているか」を学ぶ。
    #   gap = モデルの予測確率 ÷ 市場の確率。2.0なら市場の2倍強いと見ている。
    #
    #   印の意味
    #     ★軸 … gapが最大かつ 1.5以上。**メインで買う馬（単勝）**
    #     ○  … gap 1.3以上。軸との組み合わせ（ワイド・馬連）で狙える相手
    #     △  … gap 1.1以上。押さえ
    #   軸と○を組み合わせれば、単勝が外れてもワイド・馬連で拾える形になる。
    #
    #   ⚠ 表示のみ。購入は BETTING_ENABLED=False で停止中。
    #   ⚠ 検証値は「軸の単勝＋ダートならワイド」で 5年120.6%（軸gap>=1.5）。
    #     馬連・3連系は検証で単勝に届かなかったので、印はあくまで目安。
    resid_marks(horses)
    # 残差モデルの買い目（2026-08-22）。旧方式(my_bets)は購入停止で常に空なので、
    # 実際に記録している買い目をこちらで作って画面に出す。
    r_bets, r_ax, r_mates = resid_bets(horses, baba)

    ctx = dict(extra or {})
    ctx.update({"n_all": n_all, "limited": limit is not None})
    return render_template(
        template,
        **ctx,
        race_id=race_id,
        horses=horses,
        has_result=bool(rmap),
        jyo=jyo,
        race_no=race_no,
        dist=dist,
        baba=baba,
        baba_emo=baba_emo,
        cond=cond,
        klass=klass,
        date_str=date_str,
        my_bets=my_bets,
        r_bets=r_bets, r_ax=r_ax, r_mates=r_mates, weight_status=weight_status,
        r_is_dirt=str(baba or "").startswith(("ダ", "ダート")),
        rec_level=rec_level,
        rec_cls=rec_cls,
        plan_reason=plan_reason,
        plan_size=plan_size,
        param_horses=param_horses,
        are_tan=are_tan,
        are_ren=are_ren,
        are_tan_lbl=are_tan_lbl,
        are_tan_cls=are_tan_cls,
        are_ren_lbl=are_ren_lbl,
        are_ren_cls=are_ren_cls,
        bet_recs=bet_recs,
        ana_candidates=ana_candidates,
    )


@app.route("/results")
def results():
    df = fetch_csv(RECORD_URL)
    if df.empty:
        return render_template("error.html", msg="成績データが取得できませんでした")

    # 成績サマリー
    summary = {}
    if "hit" in df.columns:
        recent = df.tail(100)
        hits = recent["hit"].sum()
        total = len(recent)
        summary["recent_hit"] = f"{hits}/{total} ({hits/total*100:.1f}%)" if total > 0 else "-"

    if "return_rate" in df.columns:
        rr = df["return_rate"].dropna()
        summary["avg_return"] = f"{rr.mean()*100:.1f}%" if len(rr) > 0 else "-"

    records = df.tail(50).iloc[::-1].to_dict("records")
    return render_template("results.html", records=records, summary=summary)


def _stats_bar(rows, key, maxv=None):
    """棒グラフ用に幅(%)を付ける。値が全て0でも落ちないようにする。"""
    vals = [r[key] for r in rows if r[key] is not None]
    m = maxv or (max(vals) if vals else 0)
    for r in rows:
        r[key + "_w"] = round((r[key] / m * 100), 1) if (m and r[key]) else 0
    return rows


@app.route("/stats")
def stats():
    """蓄積データの集計。history_marks.csv（1行1頭）を土台にする。

    このファイルは開催日ごとに追記されるので、日が経つほど精度が上がる。
    まだ数日分しか無い時期は「参考値」と分かるように件数を必ず添える。
    """
    path = os.path.join(BASE_DIR, "history_marks.csv")
    if not os.path.exists(path):
        return render_template("error.html",
                               msg="まだ蓄積データがありません（開催日の21:10に作られます）")
    try:
        d = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    except Exception as e:
        return render_template("error.html", msg=f"蓄積データを読めません: {e}")
    if d.empty:
        return render_template("error.html", msg="蓄積データが空です")

    n_race = d.groupby(["日付", "race_id"]).ngroups
    days = sorted(d["日付"].astype(str).unique())

    # ── 印別 ──────────────────────────────────────────────
    marks = []
    for m in ("◎", "○", "▲", "△", "×"):
        s = d[d["推奨ランク"] == m]
        if not len(s):
            continue
        tan = pd.to_numeric(s.loc[s["1着"] == 1, "単勝"], errors="coerce").fillna(0).sum()
        marks.append({"mark": m, "n": len(s),
                      "win": round(s["1着"].mean() * 100, 1),
                      "ren": round(s["2着内"].mean() * 100, 1),
                      "fuku": round(s["3着内"].mean() * 100, 1),
                      "roi": round(tan / (len(s) * 100) * 100, 1) if len(s) else 0})
    _stats_bar(marks, "fuku", 100)

    # ── 評価グレード別（予測と実測の突き合わせ）──────────────
    grades = []
    if {"勝ち確率", "連対確率", "複勝確率"} <= set(d.columns):
        sc = None
        # レースごとに market+model のスコアを作る（詳細ページと同じ計算）。
        if {"race_id", "単勝オッズ"} <= set(d.columns) and _gcal():
            parts = []
            for _rid, _g in d.groupby("race_id", sort=False):
                _s = grade_scores(_g["単勝オッズ"], _g["勝ち確率"], _g["複勝確率"])
                parts.append(_s if _s is not None
                             else pd.Series(np.nan, index=_g.index))
            sc = pd.concat(parts).reindex(d.index)
        if sc is None or sc.isna().all():
            sc = (pd.to_numeric(d["勝ち確率"], errors="coerce")
                  + pd.to_numeric(d["連対確率"], errors="coerce")
                  + pd.to_numeric(d["複勝確率"], errors="coerce"))
        d["_score"] = sc
        d["_grade"] = np.select([sc >= th for _, th in GRADE_TH],
                                [g for g, _ in GRADE_TH], GRADE_NOBUY)
        for g in "SABD":
            s = d[d["_grade"] == g]
            if not len(s):
                continue
            grades.append({"g": g, "n": len(s),
                           "act": round(s["3着内"].mean() * 100, 1),
                           "pred": round(pd.to_numeric(s["複勝確率"],
                                                       errors="coerce").mean() * 100, 1),
                           "win": round(s["1着"].mean() * 100, 1)})
        _stats_bar(grades, "act", 100)

    # ── 人気帯別 ──────────────────────────────────────────
    pops, pv = [], pd.to_numeric(d.get("人気"), errors="coerce")
    for lo, hi, lbl in [(1, 1, "1番人気"), (2, 3, "2-3"), (4, 5, "4-5"),
                        (6, 7, "6-7"), (8, 10, "8-10"), (11, 99, "11番人気〜")]:
        s = d[(pv >= lo) & (pv <= hi)]
        if not len(s):
            continue
        pops.append({"lbl": lbl, "n": len(s),
                     "win": round(s["1着"].mean() * 100, 1),
                     "fuku": round(s["3着内"].mean() * 100, 1)})
    _stats_bar(pops, "fuku", 100)

    # ── 買い目の収支（bets_result_log.csv）──────────────────
    bets, cum = [], []
    bp = os.path.join(BASE_DIR, "bets_result_log.csv")
    if os.path.exists(bp):
        try:
            b = pd.read_csv(bp)
            dc = next((c for c in b.columns if "日" in c), None)
            tot = b[b["買い方"].astype(str).str.contains("合計")] if "買い方" in b else b
            if dc and len(tot):
                tot = tot.sort_values(dc)
                run_in = run_out = 0.0
                for _, r in tot.iterrows():
                    run_in += float(r.get("購入額", 0) or 0)
                    run_out += float(r.get("購入額", 0) or 0) + float(r.get("収支", 0) or 0)
                    cum.append({"date": str(r[dc])[-5:],
                                "roi": round(run_out / run_in * 100, 1) if run_in else 0,
                                "pl": int(run_out - run_in)})
                for _, r in tot.tail(12).iterrows():
                    bets.append({"date": str(r[dc])[-5:],
                                 "n": int(r.get("点数", 0) or 0),
                                 "hit": int(r.get("的中数", 0) or 0),
                                 "amt": int(r.get("購入額", 0) or 0),
                                 "pl": int(r.get("収支", 0) or 0),
                                 "roi": round(float(r.get("回収率", 0) or 0), 1)})
        except Exception:
            pass

    # ── オッズの動き（予想時 → 確定）────────────────────────
    drift = None
    if "オッズ変化率" in d.columns:
        v = pd.to_numeric(d["オッズ変化率"], errors="coerce").dropna()
        if len(v):
            w = pd.to_numeric(d.loc[d["1着"] == 1, "オッズ変化率"], errors="coerce").dropna()
            bins = [(-999, -20, "−20%超 下落"), (-20, -5, "−20〜−5%"),
                    (-5, 5, "ほぼ変わらず"), (5, 20, "+5〜20%"), (20, 999, "+20%超 上昇")]
            dist = []
            for lo, hi, lbl in bins:
                s = v[(v >= lo) & (v < hi)]
                dist.append({"lbl": lbl, "n": len(s),
                             "pct": round(len(s) / len(v) * 100, 1)})
            _stats_bar(dist, "pct")
            drift = {"n": len(v), "med": round(v.median(), 1),
                     "win_med": round(w.median(), 1) if len(w) else None,
                     "dist": dist}

    return render_template("stats.html", n_race=n_race, n_horse=len(d),
                           days=len(days), first=days[0], last=days[-1],
                           marks=marks, grades=grades, pops=pops,
                           bets=list(reversed(bets)), cum=cum, drift=drift)


@app.route("/api/refresh")
def api_refresh():
    _cache.clear()
    return jsonify({"status": "ok", "msg": "キャッシュをクリアしました"})


# ── 販売用ページ（2026-08-27追加）──────────────────────────────────────
#   既存のページ（/races /race/<id> /results /stats）には一切触っていない。
#   売り物は「当たる予想」ではなく **「確率が正しい表」**。
#   複勝確率は較正済みの値（複勝確率_較正）を使う。元の列は書き換えない。
#   課金は note に任せ、ここは合言葉で有料部分を隠すだけ（sale_gate.py 参照）。
_SALE_ABI = ["能力_勝負", "能力_安定", "能力_末脚", "能力_先行", "能力_距離", "能力_実績"]
_SALE_LAB = ["勝負", "安定", "末脚", "先行", "距離", "実績"]


def _sale_bar(v, w=5):
    if pd.isna(v):
        return "―"
    n = int(round(float(v) / 100 * w))
    return "■" * n + "□" * (w - n) + f" {int(v)}"


def _sale_rows(g):
    """販売表示用の1レース分。確率の高い順。

    ⚠ 辞書のキーに "pop" を使ってはいけない（2026-08-28に事故）
      Jinja の {{ h.pop }} は**辞書の pop メソッド**を先に拾うため、
      画面に「<built-in method pop of dict>」と出た。
      dict のメソッド名（pop/keys/items/values/get/copy/update/clear）は
      キーに使わない。ここでは ninki にしている。
    """
    import sale_view
    g = g.copy()
    for c in ("複勝確率", "人気", "単勝オッズ", "勝ち確率"):
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
    g = sale_view.apply_calib(g)

    # 評価ランク（S/A/B/D）。市場とモデルを合わせたスコアから決まる
    grades = None
    try:
        sc = grade_scores(g.get("単勝オッズ"), g.get("勝ち確率"), g.get("複勝確率"))
        if sc is not None:
            grades = [grade_of(v) for v in pd.to_numeric(sc, errors="coerce")]
    except Exception:
        grades = None
    # ⚠ 列名をアンダースコアで始めない（2026-08-28に事故）
    #   itertuples() はアンダースコア始まりの列を _1 などへ改名するため、
    #   getattr(r, "_grade") が取れずランクが全部「―」になっていた。
    g["grade"] = grades if grades is not None else "―"

    g = g.sort_values("複勝確率_較正", ascending=False).reset_index(drop=True)
    out = []
    for i, r in enumerate(g.itertuples(), start=1):
        pv = float(r.複勝確率_較正) * 100
        cls = "p5" if pv >= 50 else "p4" if pv >= 35 else "p3" if pv >= 20 else "p0"
        gap = getattr(r, "resid_gap", np.nan)
        out.append({
            "rank": i,                      # AI順位（確率の高い順）
            "grade": getattr(r, "grade", "―"),
            "num": r.馬番, "name": r.馬名,
            "prob": f"{pv:.1f}", "cls": cls,
            "ninki": f"{int(r.人気)}" if pd.notna(r.人気) else "―",
            "odds": f"{r.単勝オッズ:.1f}" if pd.notna(r.単勝オッズ) else "―",
            # 市場より高く見ている馬にだけ印。全部に出すと読めない
            "gapup": "妙" if (pd.notna(gap) and gap >= 1.3) else "",
            "abil": [_sale_bar(getattr(r, c, np.nan)) for c in _SALE_ABI],
        })
    return out


@app.route("/sale")
def sale_index():
    """販売ページの入口。note からはここに来る。

    ⚠ これが無いと、note のリンクから飛んだ人の着地点が無い（2026-08-27に気づいた）。
      /sale/<race_id> はレース単位なので、一覧が要る。
    """
    import datetime as _dt
    import sale_gate
    df = _fill_resid_gap(fetch_csv(TODAY_PRED_URL))
    if df.empty:
        return render_template("sale_index.html", races=[], unlocked=False,
                               tried=False, date_str="")
    df = _keep_latest_meet_day(df)
    if "jyo" not in df.columns:
        df["jyo"] = df["race_id"].astype(str).str[4:6]
    if "race_no" not in df.columns:
        df["race_no"] = df["race_id"].astype(str).str[10:12].astype(int)
    k = request.args.get("k", "")
    unlocked = sale_gate.check(k)
    races = []
    for rid, g in df.groupby("race_id", sort=True):
        rows = _sale_rows(g)
        if not rows:
            continue
        races.append({"id": rid, "jyo": str(g["jyo"].iloc[0]),
                      "no": int(g["race_no"].iloc[0]),
                      "top": rows[0], "n": len(rows),
                      "cls": rows[0]["cls"]})
    races.sort(key=lambda x: (x["jyo"], x["no"]))
    venues = sorted({r["jyo"] for r in races})
    return render_template("sale_index.html", races=races, venues=venues,
                           unlocked=unlocked, tried=bool(k), k=k,
                           date_str=_dt.datetime.now().strftime("%m/%d"))


@app.route("/sale/<race_id>")
def sale(race_id):
    """販売用のレース詳細。**中身は元のレース詳細と同じ。**無料は上位3頭まで。"""
    import sale_gate
    k = request.args.get("k", "")
    unlocked = sale_gate.check(k)
    return _race_detail_impl(
        race_id, limit=None if unlocked else 3, template="sale.html",
        extra={"unlocked": unlocked, "tried": bool(k), "k": k})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
