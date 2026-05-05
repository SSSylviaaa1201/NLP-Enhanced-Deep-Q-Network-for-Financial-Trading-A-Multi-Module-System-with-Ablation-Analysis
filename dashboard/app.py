"""Streamlit dashboard — NLP-RL Trading Platform. Premium fintech UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, TICKERS, INITIAL_CAPITAL, REFRESH_INTERVAL_SECONDS
from data_storage.db_manager import DatabaseManager

# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantumTrade · NLP-RL Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #0f1419 40%, #0d1520 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #0e131c 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stButton button {
    font-family: 'Inter', sans-serif;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(17, 25, 46, 0.6) 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 12px;
    padding: 16px !important;
    backdrop-filter: blur(10px);
}
[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15)) !important;
    color: #38bdf8 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    transition: all 0.3s !important;
}
.stButton button:hover {
    border-color: rgba(56, 189, 248, 0.5) !important;
    color: #38bdf8 !important;
}

/* Select box */
.stSelectbox [data-baseweb="select"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
}

/* Info / Warning / Success boxes */
.stSuccess {
    background: rgba(34, 197, 94, 0.08) !important;
    border: 1px solid rgba(34, 197, 94, 0.2) !important;
    border-radius: 10px !important;
}
.stWarning {
    background: rgba(251, 191, 36, 0.08) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    border-radius: 10px !important;
}
.stInfo {
    background: rgba(56, 189, 248, 0.06) !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: rgba(56, 189, 248, 0.08) !important;
}

/* Code block */
code, .stCode {
    color: #38bdf8 !important;
    background: rgba(15, 23, 42, 0.6) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Plotly Dark Template ──────────────────────────────────────────────

PLOTLY_DARK = go.layout.Template()
PLOTLY_DARK.layout.update({
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": "#94a3b8", "size": 12},
    "title": {"font": {"color": "#e2e8f0", "size": 14}},
    "xaxis": {
        "gridcolor": "rgba(56,189,248,0.06)",
        "zerolinecolor": "rgba(56,189,248,0.12)",
        "linecolor": "rgba(56,189,248,0.15)",
    },
    "yaxis": {
        "gridcolor": "rgba(56,189,248,0.06)",
        "zerolinecolor": "rgba(56,189,248,0.12)",
        "linecolor": "rgba(56,189,248,0.15)",
    },
    "legend": {
        "font": {"color": "#94a3b8"},
        "bgcolor": "rgba(0,0,0,0)",
    },
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
})

COLORS = {
    "primary": "#38bdf8",
    "secondary": "#818cf8",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#fbbf24",
    "surface": "#0f172a",
    "text": "#e2e8f0",
    "muted": "#64748b",
}

# ── DB ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return DatabaseManager(DB_PATH)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_market_data(_db, ticker):
    return _db.get_market_data(ticker)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_sentiment_data(_db, ticker):
    return _db.get_sentiment(ticker)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_trading_logs(_db):
    return _db.get_trading_logs()


db = get_db()

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:24px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#38bdf8,#818cf8);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-weight:700;color:#0a0e17;font-size:14px;">Q</div>
        <div>
            <div style="font-weight:600;font-size:15px;color:#e2e8f0;">QuantumTrade</div>
            <div style="font-size:11px;color:#64748b;">NLP-RL Platform v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Controls")
    ticker = st.selectbox("Ticker", TICKERS, key="ticker_select")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        refresh = st.button("🔄 Refresh")
    with col_b:
        st.button("⚡ Live", disabled=True)

    if refresh:
        st.cache_data.clear()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#64748b;line-height:1.6;">
        DB: <code style="font-size:10px;">{Path(DB_PATH).name}</code><br>
        Capital: <span style="color:#38bdf8;">\${INITIAL_CAPITAL:,.0f}</span>
    </div>
    """, unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────

market_df = load_market_data(db, ticker)
sentiment_df = load_sentiment_data(db, ticker)
logs_df = load_trading_logs(db)

db_exists = Path(DB_PATH).exists()
has_market = not market_df.empty
has_sentiment = not sentiment_df.empty
has_logs = not logs_df.empty

# ── Header ────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <h1 style="margin:0;">◆ QuantumTrade</h1>
    <div style="display:flex;gap:16px;">
        <div style="text-align:center;">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Model</div>
            <div style="font-size:13px;color:#38bdf8;font-weight:600;">DQN + FinBERT</div>
        </div>
        <div style="text-align:center;">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Status</div>
            <div style="font-size:13px;color:{'#22c55e' if has_logs else '#fbbf24'};font-weight:600;">
                {'● LIVE' if has_logs else '○ IDLE'}
            </div>
        </div>
    </div>
</div>
<div style="color:#64748b;font-size:13px;margin-bottom:24px;">
    End-to-end pipeline: news → NLP sentiment → RL agent → trade execution
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────

if has_logs:
    best_ep = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
    best = logs_df[logs_df["episode"] == best_ep]
    final_val = best["portfolio_value"].iloc[-1]
    pnl = final_val - INITIAL_CAPITAL
    pnl_pct = (pnl / INITIAL_CAPITAL) * 100
    returns = best["portfolio_value"].pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    cummax_v = best["portfolio_value"].cummax()
    drawdown = (best["portfolio_value"] - cummax_v) / cummax_v
    mdd = drawdown.min() * 100
    n_trades = int((best["action"] != 0).sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Portfolio Value", f"${final_val:,.0f}",
                  delta=f"{pnl_pct:+.1f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    with col3:
        st.metric("Max Drawdown", f"{mdd:.1f}%",
                  delta=f"{mdd:.1f}%", delta_color="inverse")
    with col4:
        st.metric("Total Trades", str(n_trades))
    with col5:
        win_rate = (best["portfolio_value"].diff() > 0).mean() * 100
        st.metric("Win Days", f"{win_rate:.0f}%")
elif has_market:
    last_price = market_df["close"].iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(f"{ticker} Price", f"${last_price:.2f}")
    with col2:
        st.metric("Data Points", f"{len(market_df):,}")
    with col3:
        st.metric("Sentiment Records",
                  f"{len(sentiment_df):,}" if has_sentiment else "—")
    with col4:
        st.metric("Status", "Data Ready")
    with col5:
        st.metric("Next Step", "Train Agent")

# ── Tabs ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Intelligence",
    "🤖 Agent Lab",
    "💰 Portfolio Analytics",
    "🔧 System Core",
])

# ── Tab 1: Market Intelligence ───────────────────────────────────────

with tab1:
    if not has_market:
        st.info("No market data yet. Run `python main.py` to collect data.")
    else:
        left, right = st.columns([2, 1])

        with left:
            st.markdown("#### Price Action & Moving Averages")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=market_df["date"], y=market_df["close"],
                mode="lines", name="Close",
                line=dict(color=COLORS["primary"], width=1.8),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.04)",
            ))
            if "MA50" in market_df.columns:
                fig.add_trace(go.Scatter(
                    x=market_df["date"], y=market_df["MA50"],
                    mode="lines", name="MA50",
                    line=dict(color="#818cf8", width=1, dash="dash"),
                ))
            if "MA200" in market_df.columns:
                fig.add_trace(go.Scatter(
                    x=market_df["date"], y=market_df["MA200"],
                    mode="lines", name="MA200",
                    line=dict(color="#94a3b8", width=1, dash="dot"),
                ))
            fig.update_layout(
                template=PLOTLY_DARK, height=380,
                hovermode="x unified",
                showlegend=True, legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.markdown("#### Sentiment Signals")
            if has_sentiment:
                methods = sorted(sentiment_df["method"].unique())
                method_colors = {"vader": "#22c55e", "lr": "#fbbf24", "finbert": "#38bdf8"}
                fig = go.Figure()
                for m in methods:
                    sub = sentiment_df[sentiment_df["method"] == m]
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=sub["sentiment_score"],
                        mode="lines+markers", name=m.upper(),
                        line=dict(color=method_colors.get(m, COLORS["muted"]), width=1.5),
                        marker=dict(size=4),
                    ))
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,116,139,0.4)")
                fig.add_hrect(y0=0.05, y1=1.0, fillcolor="rgba(34,197,94,0.04)", line_width=0)
                fig.add_hrect(y0=-1.0, y1=-0.05, fillcolor="rgba(239,68,68,0.04)", line_width=0)
                fig.update_layout(
                    template=PLOTLY_DARK, height=380,
                    yaxis=dict(range=[-1, 1]),
                    showlegend=True, legend=dict(orientation="h", y=1.02, x=0),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Run NLP pipeline to generate sentiment signals.")

        col_ta1, col_ta2 = st.columns(2)

        with col_ta1:
            st.markdown("#### RSI (14)")
            if "RSI" in market_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_df["date"], y=market_df["RSI"],
                    mode="lines", name="RSI",
                    line=dict(color="#818cf8", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(129,140,248,0.06)",
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)", line_width=1)
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)", line_width=1)
                fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.06)", line_width=0)
                fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.06)", line_width=0)
                fig.update_layout(template=PLOTLY_DARK, height=280)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_ta2:
            st.markdown("#### MACD")
            if "MACD" in market_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_df["date"], y=market_df["MACD"],
                    mode="lines", name="MACD",
                    line=dict(color=COLORS["primary"], width=1.5),
                ))
                if "MACD_signal" in market_df.columns:
                    fig.add_trace(go.Scatter(
                        x=market_df["date"], y=market_df["MACD_signal"],
                        mode="lines", name="Signal",
                        line=dict(color="#fbbf24", width=1.2),
                    ))
                fig.update_layout(template=PLOTLY_DARK, height=280)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Tab 2: Agent Lab ──────────────────────────────────────────────────

with tab2:
    if not has_logs:
        st.info("No trading data yet. Train the RL agent first.")
    else:
        st.markdown("#### Agent Trade Decisions")
        episodes = sorted(logs_df["episode"].unique(), reverse=True)
        episode = st.selectbox("Episode", episodes, key="ep_select")

        ep_logs = logs_df[logs_df["episode"] == episode]

        buys = ep_logs[ep_logs["action"] == 1]
        sells = ep_logs[ep_logs["action"] == 2]
        holds = ep_logs[ep_logs["action"] == 0]

        fig = go.Figure()

        # Portfolio equity area
        fig.add_trace(go.Scatter(
            x=ep_logs["step"], y=ep_logs["portfolio_value"],
            mode="lines", name="Equity",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.08)",
        ))

        # Buy markers
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["step"], y=buys["portfolio_value"],
                mode="markers", name="Buy",
                marker=dict(
                    color=COLORS["positive"], symbol="triangle-up",
                    size=14, line=dict(color="#fff", width=1.5),
                ),
            ))

        # Sell markers
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["step"], y=sells["portfolio_value"],
                mode="markers", name="Sell",
                marker=dict(
                    color=COLORS["negative"], symbol="triangle-down",
                    size=14, line=dict(color="#fff", width=1.5),
                ),
            ))

        fig.add_hline(
            y=INITIAL_CAPITAL, line_dash="dash",
            line_color="rgba(100,116,139,0.4)", line_width=1,
            annotation_text="Initial",
        )

        fig.update_layout(
            template=PLOTLY_DARK, height=420,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Action distribution
        st.markdown("#### Action Distribution")
        action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
        action_colors = {0: COLORS["muted"], 1: COLORS["positive"], 2: COLORS["negative"]}
        action_counts = ep_logs["action"].value_counts()

        cols = st.columns(3)
        for a_idx, a_name in action_names.items():
            with cols[a_idx]:
                count = action_counts.get(a_idx, 0)
                pct = count / len(ep_logs) * 100
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                            border:1px solid {action_colors[a_idx]}33;border-radius:12px;padding:20px;text-align:center;">
                    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">{a_name}</div>
                    <div style="font-size:28px;font-weight:700;color:{action_colors[a_idx]};margin:8px 0;">{count}</div>
                    <div style="font-size:12px;color:#64748b;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# ── Tab 3: Portfolio Analytics ───────────────────────────────────────

with tab3:
    if not has_logs:
        st.info("Train the RL agent and run backtest to see portfolio analytics.")
    else:
        st.markdown("#### Performance Overview")

        # Best episode
        best_ep = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
        best = logs_df[logs_df["episode"] == best_ep]
        final_val = best["portfolio_value"].iloc[-1]
        pnl = final_val - INITIAL_CAPITAL
        pnl_pct = (pnl / INITIAL_CAPITAL) * 100
        returns = best["portfolio_value"].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        cummax_v = best["portfolio_value"].cummax()
        dd_series = (best["portfolio_value"] - cummax_v) / cummax_v
        mdd = dd_series.min() * 100

        cols = st.columns(5)
        with cols[0]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Total Return</div>
                <div style="font-size:22px;font-weight:700;color:{COLORS['positive'] if pnl>=0 else COLORS['negative']};margin:6px 0;">
                    {pnl_pct:+.2f}%
                </div>
                <div style="font-size:11px;color:#64748b;">${pnl:+,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Sharpe Ratio</div>
                <div style="font-size:22px;font-weight:700;color:{COLORS['primary']};margin:6px 0;">
                    {sharpe:.4f}
                </div>
                <div style="font-size:11px;color:#64748b;">Risk-Adj.</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Max Drawdown</div>
                <div style="font-size:22px;font-weight:700;color:{COLORS['negative']};margin:6px 0;">
                    {mdd:.1f}%
                </div>
                <div style="font-size:11px;color:#64748b;">Peak-to-Trough</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[3]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Final Value</div>
                <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin:6px 0;">
                    ${final_val:,.0f}
                </div>
                <div style="font-size:11px;color:#64748b;">Ep #{best_ep}</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[4]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Win Days</div>
                <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin:6px 0;">
                    {(returns > 0).mean()*100:.0f}%
                </div>
                <div style="font-size:11px;color:#64748b;">Daily</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=best["step"], y=best["portfolio_value"],
                mode="lines", name="RL Agent",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.08)",
            ))
            fig.add_hline(
                y=INITIAL_CAPITAL, line_dash="dash",
                line_color="rgba(100,116,139,0.3)", line_width=1,
            )
            fig.update_layout(template=PLOTLY_DARK, height=340)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            st.markdown("#### Drawdown Profile")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=best["step"], y=dd_series * 100,
                mode="lines", name="Drawdown",
                line=dict(color=COLORS["negative"], width=1.5),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.12)",
            ))
            fig.update_layout(
                template=PLOTLY_DARK, height=340,
                yaxis=dict(title="Drawdown (%)", tickformat=".1f"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Episode Performance Comparison")
        ep_summary = logs_df.groupby("episode").agg(
            final_value=("portfolio_value", "last"),
            max_value=("portfolio_value", "max"),
        ).reset_index()
        ep_summary["return"] = (ep_summary["final_value"] / INITIAL_CAPITAL - 1) * 100

        fig = go.Figure()
        colors_ep = [COLORS["positive"] if r >= 0 else COLORS["negative"] for r in ep_summary["return"]]
        fig.add_trace(go.Bar(
            x=ep_summary["episode"], y=ep_summary["return"],
            marker=dict(color=colors_ep, opacity=0.7),
            name="Return %",
        ))
        fig.update_layout(
            template=PLOTLY_DARK, height=300,
            yaxis=dict(title="Return (%)"),
            xaxis=dict(title="Episode"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Tab 4: System Core ────────────────────────────────────────────────

with tab4:
    st.markdown("#### Pipeline Status")

    news_count = len(db.get_news(ticker)) if db_exists else 0
    mkt_rows = len(market_df)
    sent_rows = len(sentiment_df)
    log_rows = len(logs_df)

    status_items = [
        ("Data Ingestion", news_count > 0 and mkt_rows > 0,
         f"{news_count} news · {mkt_rows} mkt rows"),
        ("NLP Pipeline", sent_rows > 0,
         f"{sent_rows} sentiment records · {len(sentiment_df['method'].unique()) if has_sentiment else 0} methods"),
        ("RL Engine", log_rows > 0,
         f"{log_rows} trading logs · {logs_df['episode'].nunique() if has_logs else 0} episodes"),
        ("Dashboard", True, "Operational"),
    ]

    cols = st.columns(4)
    for idx, (name, ok, detail) in enumerate(status_items):
        with cols[idx]:
            color = COLORS["positive"] if ok else COLORS["neutral"]
            icon = "●" if ok else "○"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                        border:1px solid {color}22;border-radius:12px;padding:18px;text-align:center;">
                <div style="font-size:20px;color:{color};">{icon}</div>
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;margin:6px 0;">{name}</div>
                <div style="font-size:10px;color:#64748b;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_db1, col_db2 = st.columns(2)

    with col_db1:
        st.markdown("#### Database Statistics")
        db_path = Path(DB_PATH)
        db_size = db_path.stat().st_size / 1024 if db_path.exists() else 0
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                    border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:20px;font-family:monospace;">
            <div style="color:#64748b;">Path: <span style="color:#94a3b8;">{db_path}</span></div>
            <div style="color:#64748b;margin-top:8px;">Size: <span style="color:#38bdf8;">{db_size:.1f} KB</span></div>
            <div style="color:#64748b;margin-top:8px;">Tables:</div>
            <div style="color:#94a3b8;margin-left:12px;">· news</div>
            <div style="color:#94a3b8;margin-left:12px;">· market_data</div>
            <div style="color:#94a3b8;margin-left:12px;">· sentiment_signals</div>
            <div style="color:#94a3b8;margin-left:12px;">· trading_logs</div>
            <div style="color:#94a3b8;margin-left:12px;">· trade_orders</div>
        </div>
        """, unsafe_allow_html=True)

    with col_db2:
        st.markdown("#### Configuration")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                    border:1px solid rgba(56,189,248,0.12);border-radius:12px;padding:20px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">Tickers</span>
                <span style="color:#94a3b8;">{', '.join(TICKERS)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">RL Algorithm</span>
                <span style="color:#38bdf8;">DQN (from scratch)</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">NLP Models</span>
                <span style="color:#38bdf8;">VADER + LR + FinBERT</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">State Dim</span>
                <span style="color:#94a3b8;">8 (price + indicators + sentiment)</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">Initial Capital</span>
                <span style="color:#94a3b8;">${INITIAL_CAPITAL:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">Episodes</span>
                <span style="color:#94a3b8;">200</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#64748b;">Validation</span>
                <span style="color:#94a3b8;">Walk-Forward (no look-ahead)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    pipeline_ok = all([news_count > 0, mkt_rows > 0, sent_rows > 0, log_rows > 0])
    if pipeline_ok:
        st.success("◆ All modules operational — trading pipeline is healthy")
    else:
        missing = []
        if news_count == 0: missing.append("Data Ingestion")
        if mkt_rows == 0: missing.append("Market Data")
        if sent_rows == 0: missing.append("NLP Pipeline")
        if log_rows == 0: missing.append("RL Engine")
        st.warning(f"○ Incomplete: {', '.join(missing)}")

# ── Footer ────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:24px 0 8px 0;border-top:1px solid rgba(56,189,248,0.06);margin-top:24px;">
    <span style="color:#64748b;font-size:11px;">
        QuantumTrade v1.0 · NLP-Driven RL Trading Platform · Fintech Group Project
    </span>
</div>
""", unsafe_allow_html=True)
