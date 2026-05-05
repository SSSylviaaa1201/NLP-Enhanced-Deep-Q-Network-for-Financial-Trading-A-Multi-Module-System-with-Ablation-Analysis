"""Shared chart components for QuantumTrade v2.0 dashboard.

Color convention: China-style (红涨绿跌).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "up": "#ef4444",
    "down": "#22c55e",
    "primary": "#38bdf8",
    "secondary": "#818cf8",
    "vader": "#22c55e",
    "lr": "#fbbf24",
    "finbert": "#38bdf8",
    "llm": "#f472b6",
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(56,189,248,0.06)",
    "text": "#94a3b8",
    "muted": "#64748b",
}

DARK_TEMPLATE = go.layout.Template()
DARK_TEMPLATE.layout.update({
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, sans-serif", "color": COLORS["text"], "size": 12},
    "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": "rgba(56,189,248,0.12)", "linecolor": "rgba(56,189,248,0.15)"},
    "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": "rgba(56,189,248,0.12)", "linecolor": "rgba(56,189,248,0.15)"},
    "legend": {"font": {"color": COLORS["text"]}, "bgcolor": "rgba(0,0,0,0)"},
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
})


def create_candlestick_chart(market_df: pd.DataFrame, trades_df=None) -> go.Figure:
    """Professional candlestick chart with volume subplot and trade markers."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.03,
    )

    # K-line
    fig.add_trace(go.Candlestick(
        x=market_df["date"], open=market_df["open"],
        high=market_df["high"], low=market_df["low"], close=market_df["close"],
        name="Price",
        increasing_line_color=COLORS["up"], decreasing_line_color=COLORS["down"],
        increasing_fillcolor=COLORS["up"], decreasing_fillcolor=COLORS["down"],
    ), row=1, col=1)

    # MAs
    for col_name, color, dash in [("MA50", COLORS["secondary"], "dash"), ("MA200", COLORS["muted"], "dot")]:
        if col_name in market_df.columns:
            fig.add_trace(go.Scatter(
                x=market_df["date"], y=market_df[col_name],
                mode="lines", name=col_name,
                line=dict(color=color, width=1, dash=dash), opacity=0.7,
            ), row=1, col=1)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        buys = trades_df[trades_df["action"] == 1]
        sells = trades_df[trades_df["action"] == 2]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["step"], y=buys["price"],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=12, color=COLORS["up"], line=dict(color="white", width=1)),
            ), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["step"], y=sells["price"],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=12, color=COLORS["down"], line=dict(color="white", width=1)),
            ), row=1, col=1)

    # Volume
    vol_colors = [COLORS["up"] if c >= o else COLORS["down"]
                  for c, o in zip(market_df["close"], market_df["open"])]
    fig.add_trace(go.Bar(
        x=market_df["date"], y=market_df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        template=DARK_TEMPLATE, height=500,
        xaxis_rangeslider_visible=False,
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def create_sentiment_quad(sentiment_df: pd.DataFrame) -> go.Figure:
    """Four-method sentiment overlay chart."""
    fig = go.Figure()
    method_cfg = {
        "vader": {"color": COLORS["vader"], "dash": "solid"},
        "lr": {"color": COLORS["lr"], "dash": "dot"},
        "finbert": {"color": COLORS["finbert"], "dash": "dash"},
        "llm": {"color": COLORS["llm"], "dash": "longdash"},
    }
    for method, cfg in method_cfg.items():
        sub = sentiment_df[sentiment_df["method"] == method]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["sentiment_score"],
                mode="lines+markers", name=method.upper(),
                line=dict(color=cfg["color"], width=1.5, dash=cfg["dash"]),
                marker=dict(size=3),
            ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_hrect(y0=0.05, y1=1.0, fillcolor="rgba(239,68,68,0.04)", line_width=0)
    fig.add_hrect(y0=-1.0, y1=-0.05, fillcolor="rgba(34,197,94,0.04)", line_width=0)
    fig.update_layout(template=DARK_TEMPLATE, height=280, yaxis=dict(range=[-1.1, 1.1], title="Sentiment Score"), legend=dict(orientation="h"))
    return fig


def create_heatmap(pct_changes: dict[str, float]) -> go.Figure:
    """Multi-ticker heatmap: bar chart of daily % changes."""
    symbols = list(pct_changes.keys())
    values = list(pct_changes.values())
    colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in values]

    fig = go.Figure(go.Bar(
        x=symbols, y=values, marker_color=colors,
        text=[f"{v:+.2f}%" for v in values], textposition="outside",
        textfont=dict(size=10), hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(template=DARK_TEMPLATE, height=250, yaxis_title="Daily Change (%)", margin=dict(l=0, r=0, t=10, b=40))
    return fig


def create_ablation_chart(ablation_results: dict) -> go.Figure:
    """Ablation comparison: with-NLP vs without-NLP Sharpe ratios per ticker."""
    if not ablation_results:
        return go.Figure()

    tickers_list = list(ablation_results.keys())
    with_nlp = [ablation_results[t].get("with_nlp", {}).get("sharpe_ratio", 0) for t in tickers_list]
    without_nlp = [ablation_results[t].get("without_nlp", {}).get("sharpe_ratio", 0) for t in tickers_list]

    fig = go.Figure()
    x_pos = list(range(len(tickers_list)))
    fig.add_trace(go.Bar(x=x_pos, y=with_nlp, name="With NLP", marker_color=COLORS["primary"], marker_opacity=0.8))
    fig.add_trace(go.Bar(x=x_pos, y=without_nlp, name="Without NLP", marker_color=COLORS["muted"], marker_opacity=0.5))

    fig.update_layout(
        template=DARK_TEMPLATE, height=300,
        xaxis=dict(tickmode="array", tickvals=x_pos, ticktext=tickers_list),
        yaxis_title="Sharpe Ratio", legend=dict(orientation="h"),
    )
    return fig


def create_rsi_chart(market_df: pd.DataFrame) -> go.Figure:
    """RSI indicator with overbought/oversold zones."""
    fig = go.Figure()
    if "RSI" not in market_df.columns:
        return fig
    fig.add_trace(go.Scatter(
        x=market_df["date"], y=market_df["RSI"], mode="lines", name="RSI",
        line=dict(color=COLORS["secondary"], width=1.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.06)",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.4)")
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)")
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.06)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.06)", line_width=0)
    fig.update_layout(template=DARK_TEMPLATE, height=250, yaxis=dict(range=[0, 100], title="RSI"))
    return fig


def create_macd_chart(market_df: pd.DataFrame) -> go.Figure:
    """MACD indicator chart."""
    fig = go.Figure()
    if "MACD" not in market_df.columns:
        return fig
    fig.add_trace(go.Scatter(
        x=market_df["date"], y=market_df["MACD"], mode="lines", name="MACD",
        line=dict(color=COLORS["primary"], width=1.5),
    ))
    if "MACD_signal" in market_df.columns:
        fig.add_trace(go.Scatter(
            x=market_df["date"], y=market_df["MACD_signal"], mode="lines", name="Signal",
            line=dict(color=COLORS["lr"], width=1.2),
        ))
    fig.update_layout(template=DARK_TEMPLATE, height=250, yaxis_title="MACD")
    return fig


def create_equity_curve(portfolio_logs: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Portfolio equity curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_logs["step"], y=portfolio_logs["portfolio_value"],
        mode="lines", name="RL Agent", line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="rgba(100,116,139,0.3)")
    fig.update_layout(template=DARK_TEMPLATE, height=300, yaxis_title="Portfolio Value")
    return fig
