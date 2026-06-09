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
)

st.markdown("""
<style>
.race-header{background:var(--color-background-secondary);border-radius:12px;padding:14px 18px;margin-bottom:12px;border:0.5px solid var(--color-border-tertiary)}
.kpi-card{background:var(--color-background-secondary);border-radius:8px;padding:14px;text-align:center;border:0.5px solid var(--color-border-tertiary)}
.kpi-val{font-size:1.8rem;font-weight:500;color:var(--color-text-primary)}
.kpi-lbl{font-size:0.78rem;color:var(--color-text-secondary);margin-top:2px}
.mark-row{border-radius:8px;padding:10px 14px;margin-bottom:6px;border-left:3px solid}
.mark-honmei{border-left-color:#f0b429;background:rgba(240,180,41,0.06)}
.mark-taiko{border-left-color:#3b82f6;background:rgba(59,130,246,0.06)}
.mark-ana{border-left-color:#10b981;background:rgba(16,185,129,0.06)}
.mark-name{font-size:1.05rem;font-weight:500;color:var(--color-text-primary)}
.mark-sub{font-size:0.8rem;color:var(--color-text-secondary);margin-top:2px}
.strat-badge{display:inline-block;background:rgba(124,58,237,0.15);color:#7c3aed;border-radius:4px;padding:1px 7px;font-size:0.72rem;margin-top:3px}
.hit-green td{background:rgba(16,185,129,0.08)!important}
.miss-row td{background:transparent!important}
.bar-wrap{background:var(--color-border-tertiary);border-radius:4px;height:6px;width:100%;display:inline-block;vertical-align:middle}
.bar-inner{height:6px;border-radius:4px}
section[data-testid="stSidebar"]{background:var(--color-background-secondary)}
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
page = st.sidebar.radio("ページ", ["🏇 当日予想", "📊 成績サマリー", "📋 レース結果", "🏆 戦略分析"])
st.sidebar.markdown("---")
if st.sidebar.button("🔄 更新"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"更新: {datetime.now().strftime('%H:%M')}")

df_all    = load_data()
df_result = get_result_df(df_all)

# ════════════════════════════════════════════════════════════════════════
if page == "🏇 当日予想":
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

    r0 = rdf.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in [(c1,f"{r0.get('馬場','')}{r0.get('距離','')}m","コース"),
                        (c2,str(r0.get('馬場状態','-')),"馬場状態"),
                        (c3,str(r0.get('クラス','-')),"クラス"),
                        (c4,f"{len(rdf)}頭","出走頭数")]:
        col.markdown(f"<div class='kpi-card'><div class='kpi-val'>{val}</div><div class='kpi-lbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("推奨馬")

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

    st.markdown("---")
    st.subheader("全馬評価")

    show_cols = ["推奨ランク","馬番","馬名","単勝オッズ","人気",
                 "勝ち確率","複勝確率","単勝期待値","乖離スコア","該当戦略"]
    show_cols = [c for c in show_cols if c in rdf.columns]
    disp = rdf[show_cols].sort_values("予測順位" if "予測順位" in rdf.columns else show_cols[0]).copy()

    for col in ["勝ち確率","複勝確率"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
    if "単勝期待値" in disp.columns:
        disp["単勝期待値"] = disp["単勝期待値"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "-")
    if "乖離スコア" in disp.columns:
        disp["乖離スコア"] = disp["乖離スコア"].apply(lambda x: f"{x:+.0f}" if pd.notna(x) else "-")

    st.dataframe(disp, hide_index=True, use_container_width=True)
    st.caption(f"予想日時: {rdf['予想日時'].iloc[0] if '予想日時' in rdf.columns else '-'}")

# ════════════════════════════════════════════════════════════════════════
elif page == "📊 成績サマリー":
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
elif page == "📋 レース結果":
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
elif page == "🏆 戦略分析":
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