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

_cache = {}
CACHE_TTL = 300  # 5分キャッシュ


def fetch_csv(url: str) -> pd.DataFrame:
    now = time.time()
    if url in _cache and now - _cache[url]["ts"] < CACHE_TTL:
        return _cache[url]["df"].copy()
    try:
        r = requests.get(url, timeout=15)
        r.encoding = "utf-8"
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        _cache[url] = {"df": df, "ts": now}
        return df.copy()
    except Exception:
        return _cache.get(url, {}).get("df", pd.DataFrame()).copy()


RANK_ORDER = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
SIGNAL_MAP = {
    "◎": ("軸", "axis"),
    "○": ("複勝圏", "fuku"),
    "▲": ("相手候補", "aite"),
    "△": ("注意", "chui"),
    "×": ("様子見", "yosomi"),
}
TRACK_EMO = {"芝": "🌿", "ダ": "🟤", "ダート": "🟤"}
VALID_RANKS = set(RANK_ORDER.keys())  # {'◎', '○', '▲', '△', '×'}


def rank_sort_key(s):
    return s.map(lambda x: RANK_ORDER.get(str(x) if pd.notna(x) else "", 9))


def enrich_group(group: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in group.iterrows():
        d = r.to_dict()
        raw_rank = d.get("推奨ランク", "")
        rank = str(raw_rank) if pd.notna(raw_rank) and str(raw_rank) in VALID_RANKS else ""
        d["推奨ランク"] = rank  # NaN → 空文字に正規化
        sig_label, sig_cls = SIGNAL_MAP.get(rank, ("様子見", "yosomi"))
        d["signal_label"] = sig_label
        d["signal_cls"] = sig_cls
        d["win_pct"] = f"{float(d.get('勝ち確率', 0)) * 100:.1f}"
        d["ren_pct"] = f"{float(d.get('連対確率', 0)) * 100:.1f}"
        d["fuku_pct"] = f"{float(d.get('複勝確率', 0)) * 100:.1f}"
        d["ev_val"] = float(d.get("単勝期待値", 0) or 0)
        d["ev_cls"] = "ev-good" if d["ev_val"] >= 1.0 else ("ev-ok" if d["ev_val"] >= 0.7 else "ev-bad")
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
        return f"馬番{int(r.get('馬番', 0) or 0)} {r.get('馬名', '')}"

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
    """レース推奨レベルを返す (label, css_class, score)"""
    # 見送り判定
    if bet_recs and bet_recs[0]["type"] == "見送り":
        return "見送り", "rec-pass", 0

    # ◎のEV
    axis = group[group["推奨ランク"] == "◎"]
    axis_ev = float(axis.iloc[0].get("単勝期待値", 0) or 0) if len(axis) > 0 else 0

    # スコア: EV × 100 + 安定ボーナス
    score = axis_ev * 100
    if are_tan < 35 and are_ren < 35:
        score += 15  # 安定レースボーナス

    if axis_ev >= 1.2:
        return "推奨", "rec-good", score
    elif axis_ev >= 1.0:
        return "参考", "rec-mid", score
    else:
        return "様子見", "rec-watch", score


def build_race_card(jyo, race_no, group: pd.DataFrame) -> dict:
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
        if mf_rank and pop and int(mf_rank) <= 2 and int(pop) >= 4:
            hot_badges.append({"label": "MF穴", "cls": "badge-ana"})

    # レース推奨レベルとスコアを算出
    rec_level, rec_cls, race_score = calc_rec_level(group, are_tan, are_ren, bet_recs)

    return {
        "jyo": jyo,
        "race_no": race_no,
        "race_id": race_id,
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


@app.route("/races")
def races():
    df = fetch_csv(TODAY_PRED_URL)
    if df.empty:
        return render_template("error.html", msg="予想データが取得できませんでした。しばらく後に再度お試しください。")

    # jyo, race_no が存在するか確認
    if "jyo" not in df.columns:
        df["jyo"] = df["race_id"].astype(str).str[4:6]
    if "race_no" not in df.columns:
        df["race_no"] = df["race_id"].astype(str).str[10:12].astype(int)

    date_str = ""
    if "予想日時" in df.columns:
        date_str = str(df["予想日時"].iloc[0])[:10]

    cards = []
    for (jyo, race_no), group in df.groupby(["jyo", "race_no"], sort=True):
        cards.append(build_race_card(jyo, race_no, group))

    # 推奨レースをスコア順に並べてハイライト抽出
    highlights = sorted(
        [c for c in cards if c["rec_cls"] == "rec-good"],
        key=lambda c: c["race_score"], reverse=True
    )[:5]

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

    first = group.iloc[0]
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

    # 穴馬候補: AI高推奨だが人気薄の馬
    ana_candidates = [
        h for h in horses
        if h.get("signal_cls") in ("axis", "fuku", "aite")
        and int(h.get("人気", 0) or 0) >= 4
    ]

    return render_template(
        "race_detail.html",
        race_id=race_id,
        horses=horses,
        jyo=jyo,
        race_no=race_no,
        dist=dist,
        baba=baba,
        baba_emo=baba_emo,
        cond=cond,
        klass=klass,
        date_str=date_str,
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
