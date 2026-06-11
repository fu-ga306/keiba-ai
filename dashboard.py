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

page = st.sidebar.radio(
    "ページ",
    ["🏇 当日予想", "📊 成績サマリー", "📋 レース結果", "🏆 戦略分析"],
    index=["🏇 当日予想", "📊 成績サマリー", "📋 レース結果", "🏆 戦略分析"].index(st.session_state.page)
)
st.session_state.page = page
st.sidebar.markdown("---")
if st.sidebar.button("🔄 更新"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"更新: {datetime.now().strftime('%H:%M')}")

# モバイル用ナビゲーションボタン（メイン画面上部）
nav_cols = st.columns(4)
nav_items = [
    ("🏇 当日", "🏇 当日予想"),
    ("📊 成績", "📊 成績サマリー"),
    ("📋 結果", "📋 レース結果"),
    ("🏆 戦略", "🏆 戦略分析"),
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

    sort_col = "予測順位" if "予測順位" in rdf.columns else ("勝ち確率" if "勝ち確率" in rdf.columns else "単勝オッズ")
    sort_asc = sort_col != "勝ち確率"
    tdf = rdf.sort_values(sort_col, ascending=sort_asc).copy()

    rows_html = ""
    for _, row in tdf.iterrows():
        waku   = int(row.get("枠番", 0)) if pd.notna(row.get("枠番")) else 0
        umaban = int(row.get("馬番", 0)) if pd.notna(row.get("馬番")) else 0
        name   = row.get("馬名", "")
        odds   = row.get("単勝オッズ", np.nan)
        pop    = row.get("人気", np.nan)
        wp     = row.get("勝ち確率", np.nan)
        mfp    = row.get("MF勝ち確率", np.nan)
        pp     = row.get("複勝確率", np.nan)
        ev     = row.get("単勝期待値", np.nan)
        mark   = row.get("推奨ランク", "")
        strat  = row.get("該当戦略", "")

        odds_s = f"{odds:.1f}" if pd.notna(odds) else "-"
        pop_s  = f"{int(pop)}人気" if pd.notna(pop) else "-"
        wp_s   = f"{wp*100:.1f}%" if pd.notna(wp) else "-"
        mfp_s  = f"{mfp*100:.1f}%" if pd.notna(mfp) else "-"
        pp_s   = f"{pp*100:.1f}%" if pd.notna(pp) else "-"

        # 枠番の色付き丸
        wcolor = WAKU_COLORS.get(waku, "#888888")
        wtext  = WAKU_TEXT.get(waku, "#ffffff")
        waku_html = f"<span style='display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:{wcolor};color:{wtext};font-size:0.75rem;font-weight:700;border:1px solid rgba(128,128,128,0.3)'>{waku}</span>"

        # 印
        mark_color = {"◎":"#f0b429","○":"#3b82f6","▲":"#10b981","△":"#8892a4","×":"#e74c3c"}.get(mark,"")
        mark_html  = f"<span style='color:{mark_color};font-weight:700;font-size:1.05rem'>{mark}</span>" if mark else ""

        # シグナル（戦略該当 / 期待値プラス で色分け）
        if strat:
            sig_html = "<span style='background:#dc2626;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>戦略該当</span>"
        elif pd.notna(ev) and ev >= 0.3:
            sig_html = "<span style='background:#f97316;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem;font-weight:600'>注目</span>"
        elif pd.notna(ev) and ev >= 0:
            sig_html = "<span style='background:#6b7280;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.7rem'>様子見</span>"
        else:
            sig_html = "<span style='color:var(--color-text-secondary);font-size:0.7rem'>-</span>"

        # 純粋AI能力値バー（0-100スケール想定でMF勝ち確率を強調表示）
        ability_pct = mfp*100 if pd.notna(mfp) else 0
        bar_color = "#dc2626" if ability_pct >= 15 else ("#f97316" if ability_pct >= 8 else "#3b82f6")
        bar_w = min(ability_pct * 4, 100)  # スケール調整
        ability_html = (
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<span style='font-size:0.78rem;min-width:42px'>{mfp_s}</span>"
            f"<span class='bar-wrap' style='width:50px'><span class='bar-inner' style='width:{bar_w:.0f}%;background:{bar_color}'></span></span>"
            f"</div>"
        )

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
            f"<td style='text-align:center'>{sig_html}</td>"
            f"<td>{ability_html}</td>"
            f"<td style='text-align:center'>{wp_s}</td>"
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
        <th>シグナル</th><th>純粋AI評価</th><th>勝率</th><th>複勝率</th><th>期待値</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.caption(f"予想日時: {rdf['予想日時'].iloc[0] if '予想日時' in rdf.columns else '-'}")
    st.caption("純粋AI評価=人気・オッズを見ない市場フリーモデルの勝率 / 勝率・複勝率は通常モデル（人気・オッズ考慮）")

    # ── 推奨馬の詳細カード ──
    st.markdown("---")
    st.subheader("推奨馬 詳細")

    marks = [("◎","最強推奨","#f0b429","mark-honmei"),
             ("○","強く推奨","#3b82f6","mark-taiko"),
             ("▲","推奨","#10b981","mark-ana"),
             ("△","穴候補","#8892a4","mark-ana"),
             ("×","注目","#e74c3c","mark-ana")]

    if "推奨ランク" in rdf.columns:
        for mk,lbl,color,cls in marks:
            rows = rdf[rdf["推奨ランク"]==mk]
            if rows.empty:
                continue
            row = rows.iloc[0]
            odds  = row.get("単勝オッズ",np.nan)
            pop   = row.get("人気",np.nan)
            wp    = row.get("勝ち確率",np.nan)
            pp    = row.get("複勝確率",np.nan)
            ev    = row.get("単勝期待値",np.nan)
            strat = row.get("該当戦略","")
            odds_s = f"{odds:.1f}倍" if pd.notna(odds) else "-"
            pop_s  = f"{int(pop)}番人気" if pd.notna(pop) else "-"
            wp_s   = f"{wp*100:.1f}%" if pd.notna(wp) else "-"
            pp_s   = f"{pp*100:.1f}%" if pd.notna(pp) else "-"
            ev_s   = f"{ev:+.2f}" if pd.notna(ev) else "-"
            badge  = f"<span class='strat-badge'>{strat}</span>" if strat else ""
            st.markdown(
                f"<div class='{cls}'>"
                f"<span style='color:{color};font-size:1.2rem;font-weight:500'>{mk}</span>"
                f"<span class='mark-name' style='margin-left:8px'>【{lbl}】馬番{int(row['馬番'])}番 {row['馬名']}</span><br>"
                f"<span class='mark-sub'>{odds_s} / {pop_s} &nbsp;|&nbsp; 勝率{wp_s} &nbsp;複勝率{pp_s} &nbsp;期待値{ev_s}</span>"
                f"{'<br>'+badge if badge else ''}</div>",
                unsafe_allow_html=True
            )

# ════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 成績サマリー":
    st.markdown("## 📊 成績サマリー")

    if df_result.empty:
        st.warning("結果データがありません。`result_tracker.py update` を実行してください。")
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