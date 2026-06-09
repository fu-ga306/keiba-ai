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

BASE_DIR    = r"c:\Users\別府飛河\OneDrive\デスクトップ\keiba_ai"
RECORD_FILE = os.path.join(BASE_DIR, "prediction_record_v2.csv")

st.set_page_config(
    page_title="競馬AI ダッシュボード",
    page_icon="🏇",
    layout="wide",
)

# ── カスタムCSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ベース */
body, .stApp { background: #0f1117; color: #e8eaf0; }

/* レースヘッダーカード */
.race-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #252d42 100%);
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.race-title { font-size: 1.4rem; font-weight: 700; color: #fff; margin: 0 0 6px; }
.race-meta  { font-size: 0.85rem; color: #8892a4; }
.badge {
    display: inline-block; border-radius: 6px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
    margin-right: 6px;
}
.badge-g1   { background: #c0392b; color: #fff; }
.badge-open { background: #2980b9; color: #fff; }
.badge-hit  { background: #27ae60; color: #fff; }

/* KPIカード */
.kpi-card {
    background: #1a1f2e;
    border: 1px solid #2d3548;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #f0b429; }
.kpi-label { font-size: 0.8rem; color: #8892a4; margin-top: 4px; }

/* 馬テーブル */
.horse-table { width: 100%; border-collapse: collapse; }
.horse-table th {
    background: #1a1f2e;
    color: #8892a4;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 8px 10px;
    text-align: center;
    border-bottom: 1px solid #2d3548;
    white-space: nowrap;
}
.horse-table td {
    padding: 8px 10px;
    text-align: center;
    font-size: 0.85rem;
    border-bottom: 1px solid #1e2436;
    white-space: nowrap;
}
.horse-table tr:hover td { background: #1e2436; }

/* 着順バッジ */
.rank-1 { background:#f0b429; color:#000; border-radius:50%; width:24px; height:24px;
           display:inline-flex; align-items:center; justify-content:center;
           font-weight:800; font-size:0.8rem; }
.rank-2 { background:#94a3b8; color:#000; border-radius:50%; width:24px; height:24px;
           display:inline-flex; align-items:center; justify-content:center;
           font-weight:700; font-size:0.8rem; }
.rank-3 { background:#cd7f32; color:#fff; border-radius:50%; width:24px; height:24px;
           display:inline-flex; align-items:center; justify-content:center;
           font-weight:700; font-size:0.8rem; }

/* 確率バー */
.prob-bar-wrap { width: 80px; background: #1e2436; border-radius: 4px; height: 8px; display:inline-block; vertical-align:middle; margin-left:6px; }
.prob-bar { height: 8px; border-radius: 4px; }
.bar-win   { background: #f0b429; }
.bar-place { background: #3b82f6; }
.bar-trio  { background: #10b981; }

/* 印 */
.mark-honmei { color:#f0b429; font-weight:900; font-size:1.1rem; }
.mark-taiko  { color:#3b82f6; font-weight:700; }
.mark-ana    { color:#10b981; font-weight:700; }

/* 行ハイライト */
.hit-row  td { background: rgba(39,174,96,0.12) !important; }
.miss-row td { background: transparent; }
.pending-row td { background: rgba(240,180,41,0.05); }

/* 戦略バッジ */
.strat-a  { background:#7c3aed; color:#fff; border-radius:4px; padding:2px 6px; font-size:0.7rem; }
.strat-c  { background:#0891b2; color:#fff; border-radius:4px; padding:2px 6px; font-size:0.7rem; }
.strat-d  { background:#c2410c; color:#fff; border-radius:4px; padding:2px 6px; font-size:0.7rem; }

/* サイドバー */
section[data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)


# ── データ読み込み ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(RECORD_FILE):
        return pd.DataFrame()
    df = pd.read_csv(RECORD_FILE)
    return df

def get_result_df(df):
    if df.empty or "hit" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["hit"]).copy()


# ── ユーティリティ ────────────────────────────────────────────────────────
def prob_bar(val, cls, width=60):
    if pd.isna(val):
        return "-"
    pct = min(val * 100, 100)
    bar_w = int(pct / 100 * width)
    return (
        f"<span style='color:#e8eaf0'>{pct:.1f}%</span>"
        f"<span class='prob-bar-wrap' style='width:{width}px'>"
        f"<span class='prob-bar {cls}' style='width:{bar_w}px'></span></span>"
    )

def rank_badge(rank):
    if pd.isna(rank):
        return "-"
    r = int(rank)
    if r == 1:   return f"<span class='rank-1'>1</span>"
    elif r == 2: return f"<span class='rank-2'>2</span>"
    elif r == 3: return f"<span class='rank-3'>3</span>"
    else:        return f"<span style='color:#8892a4'>{r}着</span>"

def strat_badges(s):
    if not s or pd.isna(s):
        return ""
    out = []
    for st in str(s).split("/"):
        st = st.strip()
        if "A" in st: out.append(f"<span class='strat-a'>{st}</span>")
        elif "C" in st: out.append(f"<span class='strat-c'>{st}</span>")
        elif "D" in st: out.append(f"<span class='strat-d'>{st}</span>")
    return " ".join(out)


# ── サイドバー ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏇 競馬AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["📊 成績サマリー", "📋 レース結果", "🏆 戦略分析"])
st.sidebar.markdown("---")
if st.sidebar.button("🔄 更新"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"更新: {datetime.now().strftime('%H:%M')}")


df_all    = load_data()
df_result = get_result_df(df_all)


# ════════════════════════════════════════════════════════════════════════
# ページ① 成績サマリー
# ════════════════════════════════════════════════════════════════════════
if page == "📊 成績サマリー":
    st.markdown("## 📊 成績サマリー")

    if df_result.empty:
        st.warning("結果データがありません。`result_tracker.py update` を実行してください。")
        st.stop()

    total = len(df_result)
    hits  = int(df_result["hit"].sum())
    rate  = hits / total * 100 if total > 0 else 0

    avg_h = pd.to_numeric(df_result.get("honmei_actual"), errors="coerce").mean()
    avg_t = pd.to_numeric(df_result.get("taiko_actual"),  errors="coerce").mean()
    avg_a = pd.to_numeric(df_result.get("ana_actual"),    errors="coerce").mean()

    # KPIカード
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{total}回",      "総ベット数"),
        (c2, f"{hits}回",       "◎本命的中"),
        (c3, f"{rate:.1f}%",    "的中率"),
        (c4, f"{avg_h:.1f}着" if pd.notna(avg_h) else "-", "◎平均着順"),
        (c5, f"{avg_t:.1f}着" if pd.notna(avg_t) else "-", "○平均着順"),
    ]:
        col.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value'>{val}</div>"
            f"<div class='kpi-label'>{label}</div>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🏟 競馬場別成績")
        rows = []
        for jyo, g in df_result.groupby("jyo"):
            gh = int(g["hit"].sum())
            rows.append({
                "競馬場": jyo,
                "ベット": len(g),
                "的中": gh,
                "的中率": f"{gh/len(g)*100:.1f}%",
            })
        if rows:
            jdf = pd.DataFrame(rows).sort_values("的中", ascending=False)
            st.dataframe(jdf, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 📈 累積的中率")
        df_result2 = df_result.copy().reset_index(drop=True)
        df_result2["cum_hit"]  = df_result2["hit"].cumsum()
        df_result2["cum_n"]    = range(1, len(df_result2) + 1)
        df_result2["cum_rate"] = df_result2["cum_hit"] / df_result2["cum_n"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_result2["cum_n"], y=df_result2["cum_rate"],
            fill="tozeroy", fillcolor="rgba(240,180,41,0.15)",
            line=dict(color="#f0b429", width=2), name="的中率"
        ))
        fig.add_hline(y=20, line_dash="dash", line_color="#3b82f6",
                      annotation_text="目標20%", annotation_font_color="#3b82f6")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8892a4", gridcolor="#1e2436"),
            yaxis=dict(color="#8892a4", gridcolor="#1e2436", range=[0, 50]),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, height=260,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 印別着順分布
    st.markdown("#### 🐴 印別着順分布")
    c1, c2, c3 = st.columns(3)
    for col, col_name, label, color in [
        (c1, "honmei_actual", "◎本命", "#f0b429"),
        (c2, "taiko_actual",  "○対抗", "#3b82f6"),
        (c3, "ana_actual",    "▲穴馬", "#10b981"),
    ]:
        if col_name in df_result.columns:
            vals = pd.to_numeric(df_result[col_name], errors="coerce").dropna()
            if len(vals) > 0:
                bins   = [0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 99]
                labels = ["1着", "2着", "3着", "4-5着", "6-8着", "9着以下"]
                counts = pd.cut(vals, bins=bins, labels=labels).value_counts().sort_index()
                fig = px.bar(x=counts.index, y=counts.values,
                             labels={"x": "", "y": "件数"},
                             color_discrete_sequence=[color])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#8892a4"), yaxis=dict(color="#8892a4", gridcolor="#1e2436"),
                    margin=dict(l=0, r=0, t=10, b=0), height=200, showlegend=False,
                    title=dict(text=label, font=dict(color="#e8eaf0")),
                )
                col.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# ページ② レース結果
# ════════════════════════════════════════════════════════════════════════
elif page == "📋 レース結果":
    st.markdown("## 📋 レース結果")

    if df_all.empty:
        st.warning("データがありません。")
        st.stop()

    # フィルター
    col1, col2 = st.columns([1, 3])
    with col1:
        jyo_list = ["全て"] + sorted(df_all["jyo"].dropna().unique().tolist())
        sel_jyo = st.selectbox("競馬場", jyo_list)

    view_df = df_all.copy()
    if sel_jyo != "全て":
        view_df = view_df[view_df["jyo"] == sel_jyo]

    # テーブル生成
    has_result = "hit" in view_df.columns

    rows_html = ""
    for _, row in view_df.iterrows():
        hit = row.get("hit", np.nan)
        if has_result and pd.notna(hit):
            row_cls = "hit-row" if hit == 1 else "miss-row"
        else:
            row_cls = "pending-row"

        h_actual = rank_badge(row.get("honmei_actual", np.nan))
        t_actual = rank_badge(row.get("taiko_actual", np.nan))
        a_actual = rank_badge(row.get("ana_actual",   np.nan))

        hp = row.get("honmei_win_p", np.nan)
        tp = row.get("taiko_win_p",  np.nan)
        ap = row.get("ana_win_p",    np.nan)

        hit_icon = "✅" if hit == 1 else ("❌" if hit == 0 else "⏳")

        rows_html += f"""
        <tr class='{row_cls}'>
            <td>{row.get('race_id','')}</td>
            <td>{row.get('jyo','')}</td>
            <td>{row.get('race','')}R</td>
            <td><span class='mark-honmei'>◎</span> {row.get('honmei','')}</td>
            <td>{prob_bar(hp, 'bar-win', 50)}</td>
            <td>{h_actual}</td>
            <td><span class='mark-taiko'>○</span> {row.get('taiko','')}</td>
            <td>{prob_bar(tp, 'bar-place', 50)}</td>
            <td>{t_actual}</td>
            <td><span class='mark-ana'>▲</span> {row.get('ana','')}</td>
            <td>{prob_bar(ap, 'bar-trio', 50)}</td>
            <td>{a_actual}</td>
            <td style='font-size:1.1rem'>{hit_icon}</td>
        </tr>"""

    table_html = f"""
    <table class='horse-table'>
        <thead><tr>
            <th>レースID</th><th>競馬場</th><th>R</th>
            <th>◎本命</th><th>勝率</th><th>結果</th>
            <th>○対抗</th><th>勝率</th><th>結果</th>
            <th>▲穴馬</th><th>勝率</th><th>結果</th>
            <th>判定</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>"""

    st.markdown(table_html, unsafe_allow_html=True)
    st.caption("🟢 緑行=的中  ⏳=結果未更新")


# ════════════════════════════════════════════════════════════════════════
# ページ③ 戦略分析
# ════════════════════════════════════════════════════════════════════════
elif page == "🏆 戦略分析":
    st.markdown("## 🏆 バックテスト戦略分析")

    backtest = pd.DataFrame([
        {"戦略": "戦略A",   "説明": "予測1位 × 期待値≥0.3 × 1.5〜20倍",    "ベット": 254, "的中率": 20.5, "回収率": 122.1, "信頼度": "高"},
        {"戦略": "戦略A-2", "説明": "戦略A から1番人気を除外",               "ベット": 225, "的中率": 17.8, "回収率": 117.2, "信頼度": "高"},
        {"戦略": "戦略C",   "説明": "人気3番手以下 × 予測1位 × 期待値≥0.3", "ベット": 152, "的中率": 15.8, "回収率": 119.1, "信頼度": "中"},
        {"戦略": "戦略D",   "説明": "前走間隔2〜4週 × 予測1位 × 期待値≥0.2","ベット":  28, "的中率": 35.7, "回収率": 182.9, "信頼度": "低(要観察)"},
    ])

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    for col, row in zip([c1, c2, c3, c4], backtest.itertuples()):
        color = "#f0b429" if row.回収率 >= 150 else ("#10b981" if row.回収率 >= 110 else "#e74c3c")
        col.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-value' style='color:{color}'>{row.回収率}%</div>"
            f"<div class='kpi-label'>{row.戦略} 回収率</div>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 戦略別詳細")
        st.dataframe(backtest, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 回収率比較")
        colors = ["#f0b429" if r >= 150 else "#10b981" if r >= 110 else "#e74c3c"
                  for r in backtest["回収率"]]
        fig = go.Figure(go.Bar(
            x=backtest["戦略"], y=backtest["回収率"],
            marker_color=colors,
            text=[f"{r}%" for r in backtest["回収率"]],
            textposition="outside",
        ))
        fig.add_hline(y=100, line_dash="dash", line_color="#e74c3c",
                      annotation_text="損益分岐100%", annotation_font_color="#e74c3c")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8892a4"),
            yaxis=dict(color="#8892a4", gridcolor="#1e2436", range=[0, 220]),
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False, height=280,
            font=dict(color="#e8eaf0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 信頼性注意書き
    st.markdown("---")
    st.markdown("#### ⚠️ 戦略信頼性")
    for _, row in backtest.iterrows():
        n = row["ベット"]
        icon = "🟢" if n >= 200 else "🟡" if n >= 100 else "🔴"
        st.markdown(
            f"{icon} **{row['戦略']}** ({n}回) ─ {row['説明']} ─ 信頼度: **{row['信頼度']}**"
        )