"""
dashboard.py  ―  競馬AI ダッシュボード
起動: python -m streamlit run dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

GITHUB_USER   = "fu-ga306"
GITHUB_REPO   = "keiba-ai"
GITHUB_BRANCH = "main"
RECORD_FILE_URL  = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/prediction_record_v2.csv"
TODAY_PRED_URL   = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/today_predictions.csv"

BASE_DIR    = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
RECORD_FILE = os.path.join(BASE_DIR, "prediction_record_v2.csv")

st.set_page_config(
    page_title="競馬AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ベース */
.race-header{background:var(--color-background-secondary);border-radius:12px;padding:14px 18px;margin-bottom:12px;border:0.5px solid var(--color-border-tertiary)}
.kpi-card{background:var(--color-background-secondary);border-radius:8px;padding:14px;text-align:center;border:0.5px solid var(--color-border-tertiary)}
.kpi-val{font-size:1.8rem;font-weight:500;color:var(--color-text-primary)}
.kpi-lbl{font-size:0.78rem;color:var(--color-text-secondary);margin-top:2px}
.mark-honmei{border-left-color:#f0b429;background:rgba(240,180,41,0.06);border-left:3px solid #f0b429;border-radius:8px;padding:10px 14px;margin-bottom:6px}
.mark-taiko{border-left-color:#3b82f6;background:rgba(59,130,246,0.06);border-left:3px solid #3b82f6;border-radius:8px;padding:10px 14px;margin-bottom:6px}
.mark-ana{border-left-color:#10b981;background:rgba(16,185,129,0.06);border-left:3px solid #10b981;border-radius:8px;padding:10px 14px;margin-bottom:6px}
.mark-name{font-size:1.05rem;font-weight:500;color:var(--color-text-primary)}
.mark-sub{font-size:0.8rem;color:var(--color-text-secondary);margin-top:2px}
.strat-badge{display:inline-block;background:rgba(124,58,237,0.15);color:#7c3aed;border-radius:4px;padding:1px 7px;font-size:0.72rem;margin-top:3px}
.hit-green td{background:rgba(16,185,129,0.08)!important}
.miss-row td{background:transparent!important}
.bar-wrap{background:var(--color-border-tertiary);border-radius:4px;height:6px;width:100%;display:inline-block;vertical-align:middle}
.bar-inner{height:6px;border-radius:4px}
section[data-testid="stSidebar"]{background:var(--color-background-secondary)}

/* モバイル対応 */
@media (max-width: 768px) {
    .kpi-val{font-size:1.3rem}
    .kpi-card{padding:10px 8px}
    .mark-name{font-size:0.95rem}
    .mark-sub{font-size:0.75rem}
    h1{font-size:1.4rem!important}
    h2{font-size:1.2rem!important}
    h3{font-size:1.0rem!important}
    .block-container{padding-left:0.5rem!important;padding-right:0.5rem!important}
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data():
    try:
        return pd.read_csv(RECORD_FILE_URL)
    except Exception:
        pass
    if os.path.exists(RECORD_FILE):
        return pd.read_csv(RECORD_FILE)
    return pd.DataFrame()

@st.cache_data(ttl=120)
def load_today():
    try:
        return pd.read_csv(TODAY_PRED_URL)
    except Exception:
        pass
    local = os.path.join(BASE_DIR, "today_predictions.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    return pd.DataFrame()

def get_result_df(df):
    if df.empty or "hit" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["hit"]).copy()

def prob_bar_html(val, color, width=60):
    if pd.isna(val):
        return "-"
    pct = min(float(val)*100, 100)
    w = int(pct/100*width)
    return (f"<span style='color:var(--color-text-primary);font-size:13px'>{pct:.1f}%</span>"
            f"<span class='bar-wrap' style='width:{width}px;margin-left:6px'>"
            f"<span class='bar-inner' style='width:{w}px;background:{color}'></span></span>")

st.sidebar.markdown("### 🏇 競馬AI")
st.sidebar.markdown("---")
# session_stateでページ管理
if "page" not in st.session_state:
    st.session_state.page = "🏇 当日予想"

_PAGES = ["🏇 当日予想", "📊 成績サマリー", "📋 レース結果", "🏆 戦略分析", "📈 精度分析"]
page = st.sidebar.radio(
    "ページ",
    _PAGES,
    index=_PAGES.index(st.session_state.page) if st.session_state.page in _PAGES else 0
)
st.session_state.page = page
st.sidebar.markdown("---")
if st.sidebar.button("🔄 更新"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"更新: {datetime.now().strftime('%H:%M')}")

# モバイル用ナビゲーションボタン（メイン画面上部）
nav_cols = st.columns(5)
nav_items = [
    ("🏇 当日", "🏇 当日予想"),
    ("📊 成績", "📊 成績サマリー"),
    ("📋 結果", "📋 レース結果"),
    ("🏆 戦略", "🏆 戦略分析"),
    ("📈 精度", "📈 精度分析"),
]
for col, (label, full_page) in zip(nav_cols, nav_items):
    is_active = full_page == page
    if col.button(
        f"**{label}**" if is_active else label,
        key=f"nav_{label}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.page = full_page
        st.rerun()
st.markdown("---")

df_all    = load_data()
df_result = get_result_df(df_all)

# ════════════════════════════════════════════════════════════════════════
if st.session_state.page == "🏇 当日予想":
    st.markdown("## 🏇 当日予想")
    df_today = load_today()

    if df_today.empty:
        st.warning("本日の予想データがありません。`keiba_predict.py today` を実行してください。")
        st.info("実行コマンド: `python keiba_predict.py today`")
        st.stop()

    if "jyo" in df_today.columns and "race_no" in df_today.columns:
        races = df_today[["race_id","jyo","race_no"]].drop_duplicates()
        races["label"] = races["jyo"].astype(str) + " " + races["race_no"].astype(str) + "R"
        sel = st.selectbox("レース選択", races["label"].tolist())
        sel_id = races[races["label"]==sel]["race_id"].iloc[0]
        rdf = df_today[df_today["race_id"].astype(str)==str(sel_id)].copy()
    else:
        rdf = df_today.copy()

    if rdf.empty:
        st.warning("選択レースのデータなし")
        st.stop()

    if len(rdf) == 0:
        st.warning("選択レースのデータが空です")
        st.stop()

    r0 = rdf.iloc[0]
    dist_s   = f"{int(r0.get('距離',0))}m" if pd.notna(r0.get('距離')) else ""
    turf_s   = r0.get('馬場','')
    n_horse  = len(rdf)
    jyo_s    = r0.get('jyo','')
    race_no  = int(r0.get('race_no', 0)) if pd.notna(r0.get('race_no')) else 0

    # 推奨馬の最強ピック（ヘッダーバッジ用）
    best_roi = ""
    if "推奨ランク" in rdf.columns:
        honmei = rdf[rdf["推奨ランク"]=="◎"]
        if not honmei.empty:
            ev = honmei.iloc[0].get("単勝期待値", np.nan)
            if pd.notna(ev) and ev > 0:
                odds = honmei.iloc[0].get("単勝オッズ", np.nan)
                roi = (1+ev) * 100
                best_roi = f"<span style='background:rgba(231,126,34,0.15);color:#e67e22;border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:600'>🔥 期待値+{ev:.2f} 推定回収{roi:.0f}%</span>"

    # ── レースヘッダーバー ──
    st.markdown(
        f"<div class='race-header' style='display:flex;align-items:center;gap:16px'>"
        f"<div style='background:#3b3f5c;color:#fff;border-radius:10px;padding:8px 14px;text-align:center;min-width:54px'>"
        f"<div style='font-size:1.3rem;font-weight:700;line-height:1'>{race_no}</div>"
        f"<div style='font-size:0.65rem;opacity:0.8'>{jyo_s}</div></div>"
        f"<div style='flex:1'>"
        f"<div style='font-size:1.1rem;font-weight:600;color:var(--color-text-primary)'>{jyo_s} {race_no}R</div>"
        f"<div style='font-size:0.8rem;color:var(--color-text-secondary);margin-top:2px'>"
        f"{turf_s} {dist_s} / {n_horse}頭 &nbsp;|&nbsp; 馬場:{r0.get('馬場状態','-')} &nbsp;|&nbsp; クラス:{r0.get('クラス','-')}</div>"
        f"</div>"
        f"{best_roi}"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── 全頭テーブル（画像2スタイル） ──
    WAKU_COLORS = {
        1: "#ffffff", 2: "#000000", 3: "#dc2626", 4: "#3b82f6",
        5: "#fbbf24", 6: "#10b981", 7: "#f97316", 8: "#ec4899",
    }
    WAKU_TEXT = {1: "#000000", 5: "#000000"}  # 白・黄枠は文字を黒に

    sort_options = {
        "AI予測順位": ("勝ち確率", False),
        "オッズ順": ("単勝オッズ", True),
        "人気順": ("人気", True),
        "期待値順": ("単勝期待値", False),
        "純粋AI評価順": ("MF勝ち確率", False),
        "連対率順": ("連対確率", False),
        "複勝率順": ("複勝確率", False),
    }
    sort_options = {k: v for k, v in sort_options.items() if v[0] in rdf.columns}
    sort_label = st.radio("並び替え", list(sort_options.keys()), horizontal=True, key=f"sort_{rdf['race_id'].iloc[0]}")
    sort_col, sort_asc = sort_options.get(sort_label, ("勝ち確率", False))
    tdf = rdf.sort_values(sort_col, ascending=sort_asc, na_position="last").copy()

    rows_html = ""
    for _, row in tdf.iterrows():
        waku   = int(row.get("枠番", 0)) if pd.notna(row.get("枠番")) else 0
        umaban = int(row.get("馬番", 0)) if pd.notna(row.get("馬番")) else 0
        name   = row.get("馬名", "")
        odds   = row.get("単勝オッズ", np.nan)
        pop    = row.get("人気", np.nan)
        wp     = row.get("MF勝ち確率", row.get("勝ち確率", np.nan))  # 市場フリー優先
        mfp    = wp  # 後方互換用
        p2     = row.get("連対確率", np.nan)
        pp     = row.get("複勝確率", np.nan)
        # EV も MF勝ち確率ベースで再計算（オッズが取れる場合）
        _stored_ev = row.get("単勝期待値", np.nan)
        ev = (wp * odds - 1) if (pd.notna(wp) and pd.notna(odds)) else _stored_ev
        mark   = row.get("推奨ランク", "")
        if pd.isna(mark):
            mark = ""
        strat  = row.get("該当戦略", "")
        if pd.isna(strat):
            strat = ""
        ken    = row.get("券種推奨", "")
        if pd.isna(ken):
            ken = ""

        odds_s = f"{odds:.1f}" if pd.notna(odds) else "-"
        pop_s  = f"{int(pop)}人気" if pd.notna(pop) else "-"
        wp_s   = f"{wp*100:.1f}%" if pd.notna(wp) else "-"
        p2_s   = f"{p2*100:.1f}%" if pd.notna(p2) else "-"
        pp_s   = f"{pp*100:.1f}%" if pd.notna(pp) else "-"

        # 枠番の色付き丸
        wcolor = WAKU_COLORS.get(waku, "#888888")
        wtext  = WAKU_TEXT.get(waku, "#ffffff")
        waku_html = f"<span style='display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:{wcolor};color:{wtext};font-size:0.75rem;font-weight:700;border:1px solid rgba(128,128,128,0.3)'>{waku}</span>"

        # 印
        mark_color = {"◎":"#f0b429","○":"#3b82f6","▲":"#10b981","△":"#8892a4","×":"#e74c3c"}.get(mark,"")
        mark_html  = f"<span style='color:{mark_color};font-weight:700;font-size:1.05rem'>{mark}</span>" if mark else ""

        # 券種推奨バッジ（軸◎/軸(人気)/相手○/穴▲）
        ken_styles = {
            "軸◎":     ("#dc2626", "#fff"),
            "軸(人気)": ("#f59e0b", "#1a1a1a"),
            "相手○":   ("#3b82f6", "#fff"),
            "穴▲":     ("#8b5cf6", "#fff"),
        }
        if ken and ken in ken_styles:
            bg, fg = ken_styles[ken]
            ken_html = f"<span style='background:{bg};color:{fg};border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;white-space:nowrap'>{ken}</span>"
        else:
            ken_html = "<span style='color:var(--color-text-secondary);font-size:0.7rem'>-</span>"

        # 買い推奨（案B）: 🔥買い=戦略該当 / ◯検討=印◎○▲ かつ EV>=0 / それ以外は空欄
        is_top_mark = mark in ("◎", "○", "▲")
        if strat:
            sig_html = "<span style='background:#dc2626;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>🔥買い</span>"
        elif is_top_mark and pd.notna(ev) and ev >= 0:
            sig_html = "<span style='background:#3b82f6;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>◯検討</span>"
        else:
            sig_html = "<span style='color:var(--color-text-secondary);font-size:0.7rem'>-</span>"

        row_bg = "background:rgba(240,180,41,0.06);" if mark == "◎" else ""
        ev_color = "#10b981" if pd.notna(ev) and ev >= 0 else "var(--color-text-secondary)"
        ev_s = f"<span style='color:{ev_color};font-weight:600'>{ev:+.2f}</span>" if pd.notna(ev) else "-"

        rows_html += (
            f"<tr style='{row_bg}'>"
            f"<td style='text-align:center'>{odds_s}</td>"
            f"<td style='text-align:center'>{pop_s}</td>"
            f"<td style='text-align:center'>{waku_html}</td>"
            f"<td style='text-align:center'>{umaban}</td>"
            f"<td style='text-align:center'>{mark_html}</td>"
            f"<td>{name}</td>"
            f"<td style='text-align:center'>{ken_html}</td>"
            f"<td style='text-align:center'>{sig_html}</td>"
            f"<td style='text-align:center'>{wp_s}</td>"
            f"<td style='text-align:center'>{p2_s}</td>"
            f"<td style='text-align:center'>{pp_s}</td>"
            f"<td style='text-align:center'>{ev_s}</td>"
            f"</tr>"
        )

    st.markdown(f"""
    <style>
    .race-table {{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}}
    .race-table th {{background:var(--color-background-secondary);color:var(--color-text-secondary);
        padding:8px 6px;text-align:center;border-bottom:1px solid var(--color-border-tertiary);
        white-space:nowrap;font-weight:500;font-size:0.75rem}}
    .race-table td {{padding:7px 6px;border-bottom:0.5px solid var(--color-border-tertiary);
        white-space:nowrap;color:var(--color-text-primary);vertical-align:middle}}
    .race-table tr:hover td {{background:var(--color-background-secondary)}}
    </style>
    <table class='race-table'>
    <thead><tr>
        <th>オッズ</th><th>人気</th><th>枠</th><th>馬番</th><th>印</th><th>馬名</th>
        <th>券種推奨</th><th>買い推奨</th><th>勝率(AI)</th><th>連対率</th><th>複勝率</th><th>期待値(AI)</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.caption(f"予想日時: {rdf['予想日時'].iloc[0] if '予想日時' in rdf.columns else '-'}")
    st.caption("◎○▲ = MF勝ち確率順（単勝・軸）　△ = 連対率上位（馬連・ワイド相手）　× = 複勝率上位（三連複・ヒモ）")

    # ── 推奨馬の詳細カード ──
    st.markdown("---")
    st.subheader("推奨馬 詳細")

    marks = [("◎","単勝軸 MF1位","#f0b429","mark-honmei"),
             ("○","単勝相手 MF2位","#3b82f6","mark-taiko"),
             ("▲","単勝穴 MF3位","#10b981","mark-ana"),
             ("△","連対候補 連対率上位","#8892a4","mark-ana"),
             ("×","複勝候補 複勝率上位","#e74c3c","mark-ana")]

    if "推奨ランク" in rdf.columns:
        for mk,lbl,color,cls in marks:
            rows = rdf[rdf["推奨ランク"]==mk]
            if rows.empty:
                continue
            row = rows.iloc[0]
            odds  = row.get("単勝オッズ",np.nan)
            pop   = row.get("人気",np.nan)
            wp    = row.get("MF勝ち確率", row.get("勝ち確率", np.nan))
            p2    = row.get("連対確率",np.nan)
            pp    = row.get("複勝確率",np.nan)
            _odds = row.get("単勝オッズ", np.nan)
            ev    = (wp * _odds - 1) if (pd.notna(wp) and pd.notna(_odds)) else row.get("単勝期待値", np.nan)
            strat = row.get("該当戦略","")
            if pd.isna(strat):
                strat = ""
            odds_s = f"{odds:.1f}倍" if pd.notna(odds) else "-"
            pop_s  = f"{int(pop)}番人気" if pd.notna(pop) else "-"
            wp_s   = f"{wp*100:.1f}%" if pd.notna(wp) else "-"
            p2_s   = f"{p2*100:.1f}%" if pd.notna(p2) else "-"
            pp_s   = f"{pp*100:.1f}%" if pd.notna(pp) else "-"
            ev_s   = f"{ev:+.2f}" if pd.notna(ev) else "-"
            badge  = f"<span class='strat-badge'>{strat}</span>" if strat else ""
            st.markdown(
                f"<div class='{cls}'>"
                f"<span style='color:{color};font-size:1.2rem;font-weight:500'>{mk}</span>"
                f"<span class='mark-name' style='margin-left:8px'>【{lbl}】馬番{int(row['馬番'])}番 {row['馬名']}</span><br>"
                f"<span class='mark-sub'>{odds_s} / {pop_s} &nbsp;|&nbsp; 勝率{wp_s} &nbsp;連対率{p2_s} &nbsp;複勝率{pp_s} &nbsp;期待値{ev_s}</span>"
                f"{'<br>'+badge if badge else ''}</div>",
                unsafe_allow_html=True
            )

# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 成績サマリー":
    st.markdown("## 📊 成績サマリー")

    if df_result.empty:
        st.warning("結果データがありません。`result_tracker.py update` を実行してください。")
        st.stop()

    # ── 日付フィルタ（モデル更新前後などで絞り込み） ──
    if "日付" in df_result.columns:
        dates = pd.to_datetime(df_result["日付"], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input(
                "対象期間（モデル更新日以降などで絞り込み）",
                value=(min_d, max_d),
                min_value=min_d, max_value=max_d,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                mask = (dates.dt.date >= start_d) & (dates.dt.date <= end_d)
                n_nodate = dates.isna().sum()
                if n_nodate > 0:
                    st.caption(f"※ 日付不明のデータ{n_nodate}件は期間絞り込みの対象外（除外）です")
                df_result = df_result[mask].reset_index(drop=True)

    if df_result.empty:
        st.info("選択した期間にデータがありません。")
        st.stop()

    total = len(df_result)
    hits  = int(df_result["hit"].sum())
    rate  = hits/total*100 if total > 0 else 0
    avg_h = pd.to_numeric(df_result.get("honmei_actual"),errors="coerce").mean()
    avg_t = pd.to_numeric(df_result.get("taiko_actual"), errors="coerce").mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl in [
        (c1,f"{total}回","総ベット数"),
        (c2,f"{hits}回","◎本命的中"),
        (c3,f"{rate:.1f}%","的中率"),
        (c4,f"{avg_h:.1f}着" if pd.notna(avg_h) else "-","◎平均着順"),
        (c5,f"{avg_t:.1f}着" if pd.notna(avg_t) else "-","○平均着順"),
    ]:
        col.markdown(f"<div class='kpi-card'><div class='kpi-val'>{val}</div><div class='kpi-lbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        st.markdown("#### 競馬場別成績")
        rows=[]
        for jyo,g in df_result.groupby("jyo"):
            gh=int(g["hit"].sum())
            rows.append({"競馬場":jyo,"ベット":len(g),"的中":gh,"的中率":f"{gh/len(g)*100:.1f}%"})
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("的中",ascending=False),hide_index=True,use_container_width=True)

    with col2:
        st.markdown("#### 累積的中率")
        dr = df_result.copy().reset_index(drop=True)
        dr["cum_hit"]  = dr["hit"].cumsum()
        dr["cum_n"]    = range(1,len(dr)+1)
        dr["cum_rate"] = dr["cum_hit"]/dr["cum_n"]*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dr["cum_n"],y=dr["cum_rate"],
            fill="tozeroy",fillcolor="rgba(240,180,41,0.1)",
            line=dict(color="#f0b429",width=2)))
        fig.add_hline(y=20,line_dash="dash",line_color="#3b82f6",annotation_text="目標20%")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)",range=[0,50]),
            margin=dict(l=0,r=0,t=10,b=0),showlegend=False,height=240)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### 印別着順分布")
    c1,c2,c3 = st.columns(3)
    for col,cname,lbl,color in [(c1,"honmei_actual","◎本命","#f0b429"),(c2,"taiko_actual","○対抗","#3b82f6"),(c3,"ana_actual","▲穴馬","#10b981")]:
        if cname in df_result.columns:
            vals = pd.to_numeric(df_result[cname],errors="coerce").dropna()
            if len(vals)>0:
                bins=[0.5,1.5,2.5,3.5,5.5,8.5,99]
                labels=["1着","2着","3着","4-5着","6-8着","9着以下"]
                counts=pd.cut(vals,bins=bins,labels=labels).value_counts().sort_index()
                fig=px.bar(x=counts.index,y=counts.values,color_discrete_sequence=[color])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="rgba(128,128,128,0.8)"),
                    yaxis=dict(color="rgba(128,128,128,0.8)",gridcolor="rgba(128,128,128,0.2)"),
                    margin=dict(l=0,r=0,t=20,b=0),height=200,showlegend=False,
                    title=dict(text=lbl,font=dict(size=14)))
                col.plotly_chart(fig,use_container_width=True)

    # ── 戦略該当のみ購入時の回収率 ──────────────────────────────
    st.markdown("#### 🔥 戦略該当のみ購入時の回収率")
    if "honmei_strat" in df_result.columns and "honmei_odds" in df_result.columns:
        strat_df = df_result[
            df_result["honmei_strat"].notna()
            & (df_result["honmei_strat"].astype(str) != "")
            & (df_result["honmei_strat"].astype(str) != "nan")
        ].copy()
        if len(strat_df) > 0:
            strat_df["honmei_odds"] = pd.to_numeric(strat_df["honmei_odds"], errors="coerce")
            n_bet = len(strat_df)
            n_hit = int(strat_df["hit"].sum())
            # 的中時のみオッズ分の払い戻し（単勝100円ベース）
            payout = strat_df[strat_df["hit"] == 1]["honmei_odds"].sum() * 100
            invest = n_bet * 100
            roi = payout / invest * 100 if invest > 0 else 0
            cc1, cc2, cc3, cc4 = st.columns(4)
            for col, val, lbl in [
                (cc1, f"{n_bet}回", "戦略該当ベット"),
                (cc2, f"{n_hit}回", "的中"),
                (cc3, f"{n_hit/n_bet*100:.1f}%" if n_bet else "-", "的中率"),
                (cc4, f"{roi:.1f}%", "単勝回収率"),
            ]:
                col.markdown(f"<div class='kpi-card'><div class='kpi-val'>{val}</div><div class='kpi-lbl'>{lbl}</div></div>", unsafe_allow_html=True)
            # 戦略別内訳
            st.markdown("###### 戦略別内訳")
            rows = []
            for strat_name, g in strat_df.groupby("honmei_strat"):
                gh = int(g["hit"].sum())
                gp = g[g["hit"] == 1]["honmei_odds"].sum() * 100
                gi = len(g) * 100
                rows.append({
                    "戦略": strat_name,
                    "ベット": len(g),
                    "的中": gh,
                    "的中率": f"{gh/len(g)*100:.1f}%",
                    "回収率": f"{gp/gi*100:.1f}%" if gi > 0 else "-",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("ベット", ascending=False),
                             hide_index=True, use_container_width=True)
            st.caption("※ 単勝100円固定で計算。複数戦略該当馬は戦略名の組み合わせで集計。")
        else:
            st.info("選択期間に戦略該当馬の記録がありません（来週以降データが蓄積されます）。")
    else:
        st.info("戦略フラグの記録がありません。新しい予想が記録されると表示されます。")

# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📋 レース結果":
    st.markdown("## 📋 レース結果")

    if df_all.empty:
        st.warning("データなし")
        st.stop()

    col1,_ = st.columns([1,3])
    with col1:
        jyo_list = ["全て"] + sorted(df_all["jyo"].dropna().unique().tolist())
        sel_jyo = st.selectbox("競馬場",jyo_list)

    vdf = df_all[df_all["jyo"]==sel_jyo].copy() if sel_jyo!="全て" else df_all.copy()
    has_result = "hit" in vdf.columns

    rows_html=""
    for _,row in vdf.iterrows():
        hit=row.get("hit",np.nan)
        cls="hit-green" if hit==1 else "miss-row"
        h_a=row.get("honmei_actual","-");t_a=row.get("taiko_actual","-");a_a=row.get("ana_actual","-")
        hp=row.get("honmei_win_p",np.nan);tp=row.get("taiko_win_p",np.nan);ap=row.get("ana_win_p",np.nan)
        icon="✅" if hit==1 else ("❌" if hit==0 else "⏳")
        hp_s=f"{hp*100:.0f}%" if pd.notna(hp) else "-"
        tp_s=f"{tp*100:.0f}%" if pd.notna(tp) else "-"
        ap_s=f"{ap*100:.0f}%" if pd.notna(ap) else "-"
        rows_html+=f"<tr class='{cls}'><td>{row.get('race_id','')}</td><td>{row.get('jyo','')}</td><td>{row.get('race','')}R</td><td><b style='color:#f0b429'>◎</b> {row.get('honmei','')} <small style='color:var(--color-text-secondary)'>({hp_s})</small></td><td>{h_a}着</td><td><b style='color:#3b82f6'>○</b> {row.get('taiko','')} <small style='color:var(--color-text-secondary)'>({tp_s})</small></td><td>{t_a}着</td><td><b style='color:#10b981'>▲</b> {row.get('ana','')} <small style='color:var(--color-text-secondary)'>({ap_s})</small></td><td>{a_a}着</td><td style='font-size:1.1rem'>{icon}</td></tr>"

    st.markdown(f"""
    <style>
    .rt{{width:100%;border-collapse:collapse;font-size:13px}}
    .rt th{{background:var(--color-background-secondary);color:var(--color-text-secondary);padding:7px 8px;text-align:left;border-bottom:0.5px solid var(--color-border-tertiary);white-space:nowrap;font-weight:400}}
    .rt td{{padding:7px 8px;border-bottom:0.5px solid var(--color-border-tertiary);white-space:nowrap;color:var(--color-text-primary)}}
    .rt tr:hover td{{background:var(--color-background-secondary)}}
    .hit-green td{{background:rgba(16,185,129,0.07)!important}}
    </style>
    <table class='rt'><thead><tr>
    <th>レースID</th><th>競馬場</th><th>R</th>
    <th>◎本命</th><th>結果</th>
    <th>○対抗</th><th>結果</th>
    <th>▲穴馬</th><th>結果</th>
    <th>判定</th>
    </tr></thead><tbody>{rows_html}</tbody></table>
    """,unsafe_allow_html=True)
    st.caption("🟢 緑行=的中  ⏳=結果未更新")

# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🏆 戦略分析":
    st.markdown("## 🏆 バックテスト戦略分析")

    bt = pd.DataFrame([
        {"戦略":"戦略A",  "説明":"予測1位×期待値≥0.3×1.5〜20倍",        "ベット":188,"的中率":22.3,"回収率":114.9,"信頼度":"高"},
        {"戦略":"戦略A-2","説明":"戦略Aから1番人気を除外",                "ベット":122,"的中率":20.5,"回収率":123.7,"信頼度":"高"},
        {"戦略":"戦略C",  "説明":"人気3番手以下×予測1位×期待値≥0.3",     "ベット": 45,"的中率":22.2,"回収率":176.4,"信頼度":"中"},
        {"戦略":"戦略D",  "説明":"前走間隔2〜4週×予測1位×期待値≥0.2",    "ベット": 22,"的中率":50.0,"回収率":240.9,"信頼度":"低(要観察)"},
        {"戦略":"戦略F",  "説明":"中京・東京×予測1位×期待値≥0.3",        "ベット": 39,"的中率":33.3,"回収率":197.2,"信頼度":"中"},
        {"戦略":"戦略FG", "説明":"中京・東京×短距離×予測1位×期待値≥0.3", "ベット": 10,"的中率":40.0,"回収率":265.0,"信頼度":"低(要観察)"},
        {"戦略":"戦略H",  "説明":"中山・小倉×予測1位×期待値≥0.3",        "ベット": 47,"的中率":27.7,"回収率":128.9,"信頼度":"中"},
    ])

    cols = st.columns(len(bt))
    for col,(_, row) in zip(cols, bt.iterrows()):
        roi = row["回収率"]
        color = "#f0b429" if roi>=180 else "#10b981" if roi>=120 else "#3b82f6" if roi>=100 else "#e74c3c"
        col.markdown(
            f"<div class='kpi-card'><div class='kpi-val' style='color:{color};font-size:1.4rem'>{roi}%</div>"
            f"<div class='kpi-lbl'>{row['戦略']}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        st.markdown("#### 戦略詳細")
        st.dataframe(bt[["戦略","ベット","的中率","回収率","信頼度"]],hide_index=True,use_container_width=True)

    with col2:
        st.markdown("#### 回収率比較")
        colors=["#f0b429" if r>=180 else "#10b981" if r>=120 else "#3b82f6" if r>=100 else "#e74c3c" for r in bt["回収率"]]
        fig=go.Figure(go.Bar(x=bt["戦略"],y=bt["回収率"],marker_color=colors,
            text=[f"{r}%" for r in bt["回収率"]],textposition="outside"))
        fig.add_hline(y=100,line_dash="dash",line_color="#e74c3c",annotation_text="損益分岐100%")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="rgba(128,128,128,0.8)"),
            yaxis=dict(color="rgba(128,128,128,0.8)",gridcolor="rgba(128,128,128,0.2)",range=[0,300]),
            margin=dict(l=0,r=0,t=30,b=0),showlegend=False,height=280,
            font=dict(color="var(--color-text-primary)"))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("---")
    st.markdown("#### 信頼性評価")
    for _,row in bt.iterrows():
        n=row["ベット"]
        icon="🟢" if n>=200 else "🟡" if n>=100 else "🟠" if n>=30 else "🔴"
        st.markdown(f"{icon} **{row['戦略']}** ({n}回) ─ {row['説明']} ─ 信頼度: **{row['信頼度']}**")

elif st.session_state.page == "📈 精度分析":
    st.markdown("## 📈 精度分析 ─ 月曜改修インプット")

    rec = load_data()
    confirmed = rec[rec["hit"].notna()].copy()

    if confirmed.empty:
        st.info("照合済みレコードがありません。週次更新後に表示されます。")
    else:
        confirmed["hit"]            = pd.to_numeric(confirmed["hit"], errors="coerce")
        confirmed["honmei_actual"]  = pd.to_numeric(confirmed["honmei_actual"], errors="coerce")
        confirmed["taiko_actual"]   = pd.to_numeric(confirmed["taiko_actual"], errors="coerce")
        confirmed["ana_actual"]     = pd.to_numeric(confirmed["ana_actual"], errors="coerce")
        confirmed["honmei_ninki"]   = pd.to_numeric(confirmed["honmei_ninki"], errors="coerce")
        confirmed["honmei_ev"]      = pd.to_numeric(confirmed["honmei_ev"], errors="coerce")

        honmei_win  = (confirmed["honmei_actual"] == 1)
        honmei_fuku = (confirmed["honmei_actual"] <= 3)
        taiko_fuku  = (confirmed["taiko_actual"].notna() & (confirmed["taiko_actual"] <= 3))
        ana_fuku    = (confirmed["ana_actual"].notna() & (confirmed["ana_actual"] <= 3))
        any_fuku    = honmei_fuku | taiko_fuku | ana_fuku
        n = len(confirmed)

        # ── KPI ──
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("◎勝率",   f"{honmei_win.mean()*100:.1f}%", f"/{n}件")
        k2.metric("◎複勝率", f"{honmei_fuku.mean()*100:.1f}%")
        k3.metric("◎○▲いずれか複勝", f"{any_fuku.mean()*100:.1f}%")
        k4.metric("照合件数", f"{n}件")

        st.markdown("---")

        # ── 競馬場別 ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 競馬場別 ◎勝率・複勝率")
            by_jyo = confirmed.groupby("jyo").agg(
                件数=("hit", "count"),
                勝率=("honmei_actual", lambda x: (x == 1).mean() * 100),
                複勝率=("honmei_actual", lambda x: (x <= 3).mean() * 100),
            ).reset_index().sort_values("勝率")
            by_jyo["勝率"]  = by_jyo["勝率"].round(1)
            by_jyo["複勝率"] = by_jyo["複勝率"].round(1)
            st.dataframe(by_jyo, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("#### 人気帯別 ◎勝率")
            def ninki_band(n):
                if pd.isna(n): return "不明"
                n = int(n)
                if n == 1: return "1番人気"
                if n <= 3: return "2-3番人気"
                if n <= 5: return "4-5番人気"
                return "6番人気以下"
            confirmed["人気帯"] = confirmed["honmei_ninki"].apply(ninki_band)
            by_ninki = confirmed.groupby("人気帯").agg(
                件数=("hit","count"),
                勝率=("honmei_actual", lambda x: (x==1).mean()*100),
                複勝率=("honmei_actual", lambda x: (x<=3).mean()*100),
            ).reset_index()
            by_ninki["勝率"]  = by_ninki["勝率"].round(1)
            by_ninki["複勝率"] = by_ninki["複勝率"].round(1)
            order = ["1番人気","2-3番人気","4-5番人気","6番人気以下","不明"]
            by_ninki["_o"] = by_ninki["人気帯"].map({v:i for i,v in enumerate(order)})
            st.dataframe(by_ninki.sort_values("_o").drop(columns="_o"), hide_index=True, use_container_width=True)

        st.markdown("---")

        # ── EVフィルター効果 ──
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### EV別 ◎勝率")
            def ev_band(v):
                if pd.isna(v): return "不明"
                if v >= 0.3: return "EV≥0.3"
                if v >= 0.0: return "0≤EV<0.3"
                return "EV<0"
            confirmed["EV帯"] = confirmed["honmei_ev"].apply(ev_band)
            by_ev = confirmed.groupby("EV帯").agg(
                件数=("hit","count"),
                勝率=("honmei_actual", lambda x: (x==1).mean()*100),
                複勝率=("honmei_actual", lambda x: (x<=3).mean()*100),
            ).reset_index()
            by_ev["勝率"]  = by_ev["勝率"].round(1)
            by_ev["複勝率"] = by_ev["複勝率"].round(1)
            st.dataframe(by_ev, hide_index=True, use_container_width=True)

        with col_d:
            st.markdown("#### 距離・クラス別（新規データから蓄積）")
            has_dist  = "距離" in confirmed.columns and confirmed["距離"].notna().any()
            has_class = "クラス" in confirmed.columns and confirmed["クラス"].notna().any()
            if has_dist:
                def dist_band(v):
                    try:
                        v = int(v)
                        if v <= 1400: return "短距離(〜1400)"
                        if v <= 1800: return "マイル(〜1800)"
                        if v <= 2200: return "中距離(〜2200)"
                        return "長距離(2200〜)"
                    except: return "不明"
                confirmed["距離帯"] = confirmed["距離"].apply(dist_band)
                by_dist = confirmed.groupby("距離帯").agg(
                    件数=("hit","count"),
                    勝率=("honmei_actual", lambda x: (x==1).mean()*100),
                ).reset_index()
                by_dist["勝率"] = by_dist["勝率"].round(1)
                st.dataframe(by_dist, hide_index=True, use_container_width=True)
            else:
                st.info("距離・クラスデータは6/27以降の照合から蓄積されます")

        st.markdown("---")

        # ── 月曜改善ポイント自動生成 ──
        st.markdown("#### 月曜改修インプット")
        issues = []

        # 全体精度チェック
        overall_win = honmei_win.mean() * 100
        if overall_win < 25:
            issues.append(f"**◎勝率が低い ({overall_win:.1f}%)** — MFモデルの上位予測精度を要確認。コース形態特徴量(B-2)の効果が出ているか確認。")

        # 競馬場別弱点
        weak_jyo = by_jyo[by_jyo["勝率"] < 20]
        if not weak_jyo.empty:
            jyo_list = "、".join(weak_jyo["jyo"].tolist())
            issues.append(f"**競馬場別弱点: {jyo_list}** — 該当競馬場のコース特性が特徴量に反映されているか確認。直線長_m・坂ありの値を要チェック。")

        # 人気帯別チェック：1番人気◎が多すぎる
        nb1 = confirmed[confirmed["honmei_ninki"] == 1]
        if len(nb1) > 0:
            pct_1pop = len(nb1) / n * 100
            if pct_1pop > 70:
                issues.append(f"**◎の{pct_1pop:.0f}%が1番人気** — MF化後も1番人気に引っ張られている。B-1血統データ追加でMFモデルの識別力向上を検討。")

        # EVフィルター効果確認
        ev_pos = confirmed[confirmed["honmei_ev"] >= 0]
        ev_neg = confirmed[confirmed["honmei_ev"] < 0]
        if len(ev_pos) > 5 and len(ev_neg) > 5:
            win_pos = (ev_pos["honmei_actual"] == 1).mean() * 100
            win_neg = (ev_neg["honmei_actual"] == 1).mean() * 100
            if win_pos < win_neg:
                issues.append(f"**EV≥0の勝率({win_pos:.1f}%)がEV<0({win_neg:.1f}%)を下回っている** — EVフィルターが逆効果の可能性。閾値再検討を推奨。")
            elif win_pos - win_neg > 10:
                issues.append(f"**EV≥0フィルターが有効** (勝率差+{win_pos-win_neg:.1f}pt) — 戦略の絞り込み条件として積極活用を推奨。")

        if not issues:
            issues.append("現時点で明確な弱点は検出されていません。モデル再学習後に再確認してください。")

        # サマリーファイルに保存
        summary_path = os.path.join(BASE_DIR, "accuracy_summary.txt")
        summary_lines = [f"=== 精度分析サマリー {datetime.now().strftime('%Y/%m/%d')} ===", ""]
        summary_lines += [f"◎勝率: {overall_win:.1f}%  複勝率: {honmei_fuku.mean()*100:.1f}%  照合件数: {n}件", ""]
        summary_lines += ["【月曜改修ポイント】"] + [f"- {i}" for i in issues]
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        for issue in issues:
            st.warning(issue)

        st.caption(f"このサマリーは {summary_path} にも保存されます")