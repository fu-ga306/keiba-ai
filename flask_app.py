"""
Flask Dashboard - 競馬AI予想サイト
GitHub の today_predictions.csv / prediction_record_v2.csv を読み込んで表示する。
"""
import os
import time
from functools import lru_cache
from io import StringIO

import pandas as pd
import requests
from flask import Flask, render_template, redirect, url_for, request, jsonify

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

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
        # ★＝市場より高く評価している馬（乖離+3以上・20倍以下）。買いの対象。
        # 2026-07-31: 3年OOSで ★の◎○▲ は単勝97.3%／★なしの◎は83.1%。
        d["is_star"] = str(d.get("妙味", "")) == "★"
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
    買い指数(honest backtestで単勝回収率に較正)を最優先で使う。
    無い旧データはEVベースにフォールバック。"""
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

    # 激熱バッジ
    hot_badges = []
    for _, r in sorted_g.head(3).iterrows():
        ev = float(r.get("単勝期待値", 0) or 0)
        mf_rank = r.get("MF予測順位", None)
        pop = r.get("人気", None)
        if ev >= 1.2:
            hot_badges.append({"label": f"EV{ev:.1f}", "cls": "badge-ev"})
        if pd.notna(mf_rank) and pd.notna(pop) and int(mf_rank) <= 2 and int(pop) >= 4:
            hot_badges.append({"label": "MF穴", "cls": "badge-ana"})

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
    df = fetch_csv(TODAY_PRED_URL)
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


@app.route("/race/<race_id>")
def race_detail(race_id):
    df = fetch_csv(TODAY_PRED_URL)
    if df.empty:
        return render_template("error.html", msg="データ取得失敗")

    group = df[df["race_id"].astype(str) == str(race_id)].copy()
    if group.empty:
        return render_template("error.html", msg=f"レース {race_id} が見つかりません")

    sorted_g = group.sort_values("推奨ランク", key=rank_sort_key)
    horses = enrich_group(sorted_g)
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
    for h in horses:
        bn = pd.to_numeric(h.get("馬番"), errors="coerce")
        r = rmap.get(int(bn)) if pd.notna(bn) else None
        h["res_pos"] = r["pos"] if r else None
        h["res_tan"] = r["tan"] if r else ""
        h["res_fuku"] = r["fuku"] if r else ""

        # 評価グレード（2026-08-07）。複勝確率＝キャリブレーション済みの絶対値なので、
        # レース内の相対順ではなく固定のしきい値で切る。頭数の少ないレースで
        # 実力がないのにAが付く、といったことを防ぐ。
        fp = pd.to_numeric(h.get("fuku_pct"), errors="coerce")
        fp = float(fp) if pd.notna(fp) else 0.0
        h["grade"] = ("S" if fp >= 50 else "A" if fp >= 40 else
                      "B" if fp >= 30 else "C" if fp >= 20 else "D")
        # 能力総合点＝6軸（レース内百分位0-100）の平均。オッズを一切見ない評価。
        vals = [pd.to_numeric(h.get(c), errors="coerce") for c, _ in ABILITY_AXES]
        vals = [float(v) for v in vals if pd.notna(v)]
        h["abil_avg"] = round(sum(vals) / len(vals), 1) if vals else None

    return render_template(
        "race_detail.html",
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


@app.route("/api/refresh")
def api_refresh():
    _cache.clear()
    return jsonify({"status": "ok", "msg": "キャッシュをクリアしました"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
