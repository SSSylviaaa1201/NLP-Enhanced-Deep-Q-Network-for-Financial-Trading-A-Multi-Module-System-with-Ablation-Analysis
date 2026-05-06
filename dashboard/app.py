"""QuantumTrade v2.0 Dashboard — Professional Trading Terminal UI.

Layout:
  Header (search + ticker tape)
  ├── Main Area (70%)
  │   ├── Row 1: Candlestick (OHLCV) + Volume + Trade markers
  │   ├── Row 2: RSI | MACD
  │   ├── Row 3: Sentiment signals (4 methods)
  │   ├── Row 4: Ablation comparison
  │   └── Row 5: Market heatmap
  └── Side Panel (30%)
      ├── News feed (RSS cards)
      ├── AI Chat (RAG)
      └── System status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, TICKERS, INITIAL_CAPITAL, REFRESH_INTERVAL_SECONDS
from data_storage.db_manager import DatabaseManager
from dashboard.components.charts import (
    COLORS, DARK_TEMPLATE,
    create_candlestick_chart, create_sentiment_quad, create_heatmap,
    create_ablation_chart, create_rsi_chart, create_macd_chart,
    create_equity_curve,
)

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantumTrade v2 · NLP-RL Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e17 0%, #0f1419 40%, #0d1520 100%); }

/* Ticker tape animation */
@keyframes ticker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-wrap {
    overflow: hidden; white-space: nowrap;
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,25,46,0.8));
    border: 1px solid rgba(56,189,248,0.1); border-radius: 8px;
    padding: 8px 0; margin-bottom: 16px;
}
.ticker-track {
    display: inline-block; animation: ticker 40s linear infinite;
}
.ticker-item {
    display: inline-block; padding: 0 24px; font-size: 13px; font-weight: 500;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1018 0%, #0e131c 100%);
    border-right: 1px solid rgba(56,189,248,0.08);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(17,25,46,0.6));
    border: 1px solid rgba(56,189,248,0.12); border-radius: 12px; padding: 16px !important;
}
[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.5rem !important; font-weight: 600 !important; }

/* News cards */
.news-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(17,25,46,0.6));
    border: 1px solid rgba(56,189,248,0.08); border-radius: 8px;
    padding: 10px 12px; margin-bottom: 6px; font-size: 12px;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: rgba(56,189,248,0.25); }
.news-source { color: #38bdf8; font-size: 10px; font-weight: 600; }
.news-title { color: #e2e8f0; margin: 4px 0; }
.news-date { color: #64748b; font-size: 10px; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border: 1px solid rgba(56,189,248,0.2) !important; color: #38bdf8 !important;
    border-radius: 8px !important; font-weight: 500 !important;
}
.stButton button:hover { border-color: #38bdf8 !important; }

/* Chat */
.stChatMessage { background: rgba(15,23,42,0.6) !important; border-radius: 10px !important; }

/* Code */
code, .stCode { color: #38bdf8 !important; background: rgba(15,23,42,0.6) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── DB + Cache ───────────────────────────────────────────────────────

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

# ── Session State ─────────────────────────────────────────────────────

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = list(TICKERS[:8])
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "AAPL"

ticker = st.session_state.current_ticker

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#38bdf8,#818cf8);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-weight:700;color:#0a0e17;font-size:14px;">Q</div>
        <div>
            <div style="font-weight:600;font-size:15px;color:#e2e8f0;">QuantumTrade</div>
            <div style="font-size:11px;color:#64748b;">NLP-RL Terminal v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search
    st.markdown("### 🔍 Ticker Search")
    from data_ingestion.ticker_lookup import search_tickers

    search_query = st.text_input("Enter ticker symbol", placeholder="e.g. AAPL, NVDA...", label_visibility="collapsed")
    if search_query:
        matches = search_tickers(search_query)
        if matches:
            cols = st.columns(min(len(matches), 3))
            for i, m in enumerate(matches[:9]):
                with cols[i % 3]:
                    if st.button(m["symbol"], key=f"search_{m['symbol']}", use_container_width=True):
                        st.session_state.current_ticker = m["symbol"]
                        st.cache_data.clear()
                        st.rerun()
        else:
            if st.button(f"🔎 Lookup '{search_query.upper()}' live", use_container_width=True):
                from data_ingestion.ticker_lookup import lookup_ticker
                result = lookup_ticker(search_query.upper(), db)
                if result["has_data"]:
                    st.session_state.current_ticker = search_query.upper()
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result.get("error", "No data found"))

    st.markdown("---")

    # Watchlist
    st.markdown("### 📋 Watchlist")
    watchlist = st.multiselect(
        "Select tickers", TICKERS,
        default=st.session_state.watchlist,
        label_visibility="collapsed",
    )
    if watchlist:
        st.session_state.watchlist = watchlist

    # Quick ticker select
    for t in watchlist[:12]:
        is_active = t == ticker
        bg = "rgba(56,189,248,0.12)" if is_active else "transparent"
        border = "1px solid rgba(56,189,248,0.3)" if is_active else "1px solid transparent"
        if st.button(
            f"{'●' if is_active else '○'}  {t}",
            key=f"wl_{t}", use_container_width=True,
        ):
            st.session_state.current_ticker = t
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # RSS News Feed
    st.markdown("### 📰 Live News Feed")
    st.caption(f"Latest headlines for {ticker}")

    try:
        from data_ingestion.rss_fetcher import fetch_news_rss
        with st.spinner("Fetching news..."):
            rss_news = fetch_news_rss(ticker, max_per_source=3)
        if rss_news:
            for article in rss_news[:8]:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-source">{article['source']}</div>
                    <div class="news-title">{article['title'][:80]}</div>
                    <div class="news-date">{article['published_at'][:10]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No RSS news available")
    except Exception:
        st.caption("News feed temporarily unavailable")

    st.markdown("---")

    # AI Chat
    st.markdown("### 🤖 AI Assistant")
    with st.expander("Ask about this ticker", expanded=False):
        for msg in st.session_state.chat_messages[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if chat_input := st.chat_input(f"Ask about {ticker}...", key="chat_input"):
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from agents.research_agent import ask
                        response = ask(question=chat_input, ticker=ticker)
                    except Exception as e:
                        response = f"AI assistant not available: {e}"
                st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # System status
    st.markdown("---")
    db_path = Path(DB_PATH)
    db_size = db_path.stat().st_size / 1024 if db_path.exists() else 0
    st.caption(f"DB: {db_size:.1f} KB | Capital: ${INITIAL_CAPITAL:,.0f}")

# ── Data Loading ──────────────────────────────────────────────────────

market_df = load_market_data(db, ticker)
sentiment_df = load_sentiment_data(db, ticker)
logs_df = load_trading_logs(db)

has_market = not market_df.empty
has_sentiment = not sentiment_df.empty
has_logs = not logs_df.empty

# ── HEADER: Title + Ticker Tape ──────────────────────────────────────

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
    <h1 style="margin:0;font-size:1.8rem;">◆ QuantumTrade <span style="color:#64748b;font-size:14px;">v2.0</span></h1>
    <div style="display:flex;gap:20px;font-size:12px;color:#64748b;">
        <span>◆ DQN + FinBERT + LLM</span>
        <span style="color:{'#22c55e' if has_logs else '#fbbf24'};">{'● Live' if has_logs else '○ Idle'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Ticker Tape
tape_prices = []
for t in st.session_state.watchlist[:12]:
    mkt = load_market_data(db, t)
    if not mkt.empty:
        latest = mkt["close"].iloc[-1]
        prev = mkt["close"].iloc[-2] if len(mkt) > 1 else latest
        chg = ((latest - prev) / prev * 100) if prev != 0 else 0
        color = COLORS["up"] if chg >= 0 else COLORS["down"]
        arrow = "▲" if chg >= 0 else "▼"
        tape_prices.append(f'<span class="ticker-item"><strong>{t}</strong> ${latest:.2f} <span style="color:{color};">{arrow} {abs(chg):.2f}%</span></span>')

if tape_prices:
    tape_html = "".join(tape_prices)
    st.markdown(f"""
    <div class="ticker-wrap">
        <div class="ticker-track">{tape_html}{tape_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────

if has_market:
    latest_price = market_df["close"].iloc[-1]
    prev_price = market_df["close"].iloc[-2] if len(market_df) > 1 else latest_price
    day_chg = (latest_price - prev_price) / prev_price * 100 if prev_price != 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(f"{ticker} Price", f"${latest_price:.2f}", delta=f"{day_chg:+.2f}%")
    with c2:
        st.metric("Market Data", f"{len(market_df):,} rows")
    with c3:
        st.metric("Sentiment Records", f"{len(sentiment_df):,}" if has_sentiment else "—")
    with c4:
        st.metric("Trading Logs", f"{len(logs_df):,}" if has_logs else "—")
    with c5:
        if has_logs:
            best = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
            ep_logs = logs_df[logs_df["episode"] == best]
            final_val = ep_logs["portfolio_value"].iloc[-1]
            pnl_pct = (final_val / INITIAL_CAPITAL - 1) * 100
            st.metric(f"Best Ep #{best}", f"${final_val:,.0f}", delta=f"{pnl_pct:+.1f}%")
        else:
            st.metric("Status", "Ready")

# ── MAIN AREA ─────────────────────────────────────────────────────────

if not has_market:
    st.info("No market data yet. Run `python main.py` or `python collect_data.py` first.")
    st.stop()

main_col, side_col = st.columns([0.7, 0.3], gap="medium")

with main_col:
    # Row 1: Candlestick + Volume
    st.markdown("#### 📈 Candlestick Chart")
    if has_logs:
        best_ep = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
        ep_logs = logs_df[logs_df["episode"] == best_ep]
        trades = ep_logs[ep_logs["action"] != 0][["step", "action", "price"]].copy()
    else:
        trades = None

    fig = create_candlestick_chart(market_df.tail(200), trades)
    st.plotly_chart(fig, key="candlestick", use_container_width=True, config={"displayModeBar": False})

    # Row 2: RSI | MACD
    c_ta1, c_ta2 = st.columns(2)
    with c_ta1:
        st.markdown("#### 📊 RSI (14)")
        fig_rsi = create_rsi_chart(market_df.tail(200))
        st.plotly_chart(fig_rsi, key="rsi", use_container_width=True, config={"displayModeBar": False})
    with c_ta2:
        st.markdown("#### 📉 MACD")
        fig_macd = create_macd_chart(market_df.tail(200))
        st.plotly_chart(fig_macd, key="macd", use_container_width=True, config={"displayModeBar": False})

    # Row 3: Sentiment
    st.markdown("#### 🧠 NLP Sentiment Signals (4 Methods)")
    if has_sentiment:
        fig_sent = create_sentiment_quad(sentiment_df)
        st.plotly_chart(fig_sent, key="sentiment", use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Run `python main.py --skip-collect` to generate sentiment data.")

    # Row 4: Portfolio / Ablation
    if has_logs:
        st.markdown("#### 💰 Portfolio Performance")
        best_ep = logs_df.groupby("episode")["portfolio_value"].last().idxmax()
        ep_logs = logs_df[logs_df["episode"] == best_ep]

        c_perf1, c_perf2 = st.columns(2)
        with c_perf1:
            fig_eq = create_equity_curve(ep_logs, INITIAL_CAPITAL)
            st.plotly_chart(fig_eq, key="equity", use_container_width=True, config={"displayModeBar": False})
        with c_perf2:
            # Drawdown
            cummax = ep_logs["portfolio_value"].cummax()
            dd = (ep_logs["portfolio_value"] - cummax) / cummax * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=ep_logs["step"], y=dd, mode="lines", name="Drawdown",
                line=dict(color=COLORS["down"], width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            ))
            fig_dd.update_layout(template=DARK_TEMPLATE, height=300, yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, key="drawdown", use_container_width=True, config={"displayModeBar": False})

    # Row 5: Market Heatmap
    st.markdown("#### 🔥 Market Heatmap")
    pct_changes = {}
    for t in st.session_state.watchlist[:15]:
        mkt = load_market_data(db, t)
        if not mkt.empty and len(mkt) > 1:
            pct_changes[t] = (mkt["close"].iloc[-1] - mkt["close"].iloc[-2]) / mkt["close"].iloc[-2] * 100
    if pct_changes:
        fig_hm = create_heatmap(pct_changes)
        st.plotly_chart(fig_hm, key="heatmap", use_container_width=True, config={"displayModeBar": False})

# ── Side Panel ────────────────────────────────────────────────────────

with side_col:
    # Stats card
    st.markdown("#### 📊 Quick Stats")
    if has_market:
        high_52 = market_df["high"].tail(252).max() if len(market_df) >= 252 else market_df["high"].max()
        low_52 = market_df["low"].tail(252).min() if len(market_df) >= 252 else market_df["low"].min()
        vol_avg = market_df["volume"].tail(20).mean()

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(15,23,42,0.9),rgba(17,25,46,0.6));
                    border:1px solid rgba(56,189,248,0.1);border-radius:12px;padding:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">52-Week High</span>
                <span style="color:#ef4444;">${high_52:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">52-Week Low</span>
                <span style="color:#22c55e;">${low_52:.2f}</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">Avg Vol (20d)</span>
                <span style="color:#38bdf8;">{vol_avg/1e6:.1f}M</span></div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#64748b;">Data Points</span>
                <span style="color:#e2e8f0;">{len(market_df):,}</span></div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#64748b;">Sentiment Methods</span>
                <span style="color:#f472b6;">{sentiment_df['method'].nunique() if has_sentiment else 0}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Sentiment summary
    if has_sentiment:
        st.markdown("#### 🧠 Sentiment Summary")
        for m in sorted(sentiment_df["method"].unique()):
            sub = sentiment_df[sentiment_df["method"] == m]
            if not sub.empty:
                avg = sub["sentiment_score"].mean()
                color = COLORS["up"] if avg > 0.05 else (COLORS["down"] if avg < -0.05 else COLORS["lr"])
                emoji = "🟢" if avg > 0.05 else ("🔴" if avg < -0.05 else "🟡")
                st.markdown(f"{emoji} **{m.upper()}**: {avg:+.4f}")

    # Ablation summary
    if has_logs:
        st.markdown("#### 🔬 Ablation")
        st.caption("NLP signal impact on Sharpe Ratio")
        st.info("Run `python main.py --ablate` for full results")

# ── Footer ────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:16px 0 4px 0;border-top:1px solid rgba(56,189,248,0.06);margin-top:20px;">
    <span style="color:#64748b;font-size:10px;">
        QuantumTrade v2.0 · NLP-Driven RL Trading Platform · Fintech Group Project 2026
    </span>
</div>
""", unsafe_allow_html=True)
