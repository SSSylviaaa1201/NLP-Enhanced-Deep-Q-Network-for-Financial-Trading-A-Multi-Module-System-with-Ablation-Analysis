# QuantumTrade v2.0 — Dashboard & Data Upgrade Plan

> **Project Root**: `C:\Users\21423\Desktop\STUDY\Y3_S2_group\Fintech_group`
> **For**: Claude Code 执行（读取本文件并按 Phase 顺序完成全部任务）
> **Date**: 2026-05-05
>
> **当前状态**：
> - DB: market_data=8275行(GBM假数据), news=187条, sentiment=153条
> - DQN: 已修复(reward/epsilon/state归一化/惩罚), 训练跑通
> - Dashboard: localhost:8501 运行中, 4-tab 布局, 视觉偏简单
> - API Keys: NewsAPI ✅ / AlphaVantage ✅ / Volcano LLM ✅ (MODEL_ID=`ep-20260505223049-dk5ss`)
> - 独立采集脚本: `collect_data.py` 已创建但未执行过
>
> **本次升级目标**：
> 1. 新闻源从 100次/天 → 无限（RSS 聚合）
> 2. 股票范围从 5 只 → 可搜索任意 ticker
> 3. Dashboard 从 "作业 Demo" → 专业交易平台视觉
> 4. 新增: 蜡烛图 / Ticker Tape / 热力图 / AI Chat 面板

---

## PHASE 1: RSS 新闻聚合引擎（无限新闻源）

### 目的：解决 NewsAPI 每天 100 次限制，实现免费无限量新闻获取。

### Task 1.1 — 创建 `data_ingestion/rss_fetcher.py`（新文件）

RSS 数据源列表：

```python
"""RSS feed aggregator — free unlimited financial news."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd

from config import TICKERS, NEWS_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

# ── RSS Sources ──────────────────────────────────────────────

RSS_SOURCES = {
    # Yahoo Finance RSS — 按股票搜索，最稳定
    "yahoo_finance": {
        "url_template": "https://finance.yahoo.com/news/rss/{ticker}",
        "parser": "_parse_yahoo_rss",
    },
    # Google News RSS — 覆盖面广
    "google_news": {
        "url_template": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        "parser": "_parse_generic_rss",
    },
    # Seeking Alpha — 专业财经深度
    "seeking_alpha": {
        "url_template": "https://seekingalpha.com/symbol/{ticker}/news/feed",
        "parser": "_parse_generic_rss",
    },
    # MarketWatch — 市场动态
    "marketwatch": {
        "url_template": "https://www.marketwatch.com/investing/stock/{ticker}/news/rss",
        "parser": "_parse_marketwatch_rss",
    },
}

# User-Agent 必须设置，否则部分 RSS 源会拒绝请求
HEADERS = {
    "User-Agent": "Mozilla/5.0 (QuantumTrade/1.0; Research Bot) "
                 "(+https://github.com/example/nlp-rl-trading)"
}


def _parse_yahoo_rss(entry: dict, ticker: str) -> Optional[dict]:
    """解析 Yahoo Finance RSS 条目。"""
    title = entry.get("title", "").strip()
    if not title:
        return None
    return {
        "ticker": ticker,
        "source": "Yahoo Finance",
        "title": title,
        "content": entry.get("summary", "") or entry.get("description", ""),
        "url": entry.get("link", ""),
        "published_at": _parse_date(entry),
    }


def _parse_generic_rss(entry: dict, ticker: str, source_name: str) -> Optional[dict]:
    """通用 RSS 解析器（Google News / Seeking Alpha）。"""
    title = entry.get("title", "").strip()
    if not title:
        return None
    return {
        "ticker": ticker,
        "source": source_name,
        "title": title,
        "content": entry.get("summary", "") or entry.get("description", "") or "",
        "url": entry.get("link", ""),
        "published_at": _parse_date(entry),
    }


def _parse_marketwatch_rss(entry: dict, ticker: str) -> Optional[dict]:
    """MarketWatch 特殊解析（有时内容在 description 里带 HTML）。"""
    import re
    title = entry.get("title", "").strip()
    if not title:
        return None
    raw_content = entry.get("summary", "") or entry.get("description", "") or ""
    # 清理 HTML 标签
    clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()
    return {
        "ticker": ticker,
        "source": "MarketWatch",
        "title": title,
        "content": clean_text[:1000],  # 截断过长内容
        "url": entry.get("link", ""),
        "published_at": _parse_date(entry),
    }


def _parse_date(entry: dict) -> str:
    """统一日期解析，返回 ISO 格式字符串。"""
    date_str = ""
    # 尝试多个可能的字段
    for field in ["published_parsed", "updated_parsed"]:
        parsed = entry.get(field)
        if parsed:
            try:
                dt = datetime(*parsed[:6])
                return dt.isoformat()
            except (TypeError, ValueError):
                continue
    # fallback: 取原始字符串
    return entry.get("published") or entry.get("updated") or datetime.now().isoformat()


def fetch_news_rss(
    ticker: str,
    max_per_source: int = 20,
    sources: Optional[list[str]] = None,
) -> list[dict]:
    """
    从所有 RSS 源获取指定股票的新闻。
    
    Args:
        ticker: 股票代码 (如 "AAPL")
        max_per_source: 每个 RSS 源最多取多少条
        sources: 要使用的源名称列表，None 表示使用全部
    
    Returns:
        list[dict]: 新闻记录列表
    """
    records = []
    active_sources = sources or list(RSS_SOURCES.keys())
    
    for source_name in active_sources:
        if source_name not in RSS_SOURCES:
            logger.warning("Unknown RSS source: %s", source_name)
            continue
            
        config = RSS_SOURCES[source_name]
        url = config["url_template"].format(ticker=ticker.lower())
        
        try:
            logger.info("Fetching %s RSS for %s...", source_name, ticker)
            # feedparser 内部处理 HTTP，加 User-Agent
            feed = feedparser.parse(url, request_headers=HEADERS)
            
            if not feed.entries:
                logger.info("  No entries from %s for %s", source_name, ticker)
                continue
                
            parser_func_name = config["parser"]
            # 动态获取解析函数
            parser_func = globals().get(parser_func_name, _parse_generic_rss)
            
            count = 0
            for entry in feed.entries[:max_per_source]:
                if parser_func_name == "_parse_generic_rss":
                    result = parser_func(entry, ticker, source_name)
                else:
                    result = parser_func(entry, ticker)
                
                if result:
                    records.append(result)
                    count += 1
                    
            logger.info("  Got %d articles from %s for %s", count, source_name, ticker)
            
        except Exception as e:
            logger.warning("Failed to fetch %s RSS for %s: %s", source_name, ticker, e)
    
    # 去重（按标题 + 来源）
    seen = set()
    unique_records = []
    for r in records:
        key = (r["title"][:80], r["source"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
    
    logger.info("RSS total for %s: %d unique articles (%d before dedup)",
                ticker, len(unique_records), len(records))
    return unique_records


def fetch_news_rss_all_tickers(
    tickers: Optional[list[str]] = None,
    max_per_source: int = 15,
) -> list[dict]:
    """批量获取多只股票的 RSS 新闻。"""
    tickers = tickers or TICKERS
    all_records = []
    
    for t in tickers:
        records = fetch_news_rss(t, max_per_source=max_per_source)
        all_records.extend(records)
    
    return all_records


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 单只测试
    print("\n=== AAPL RSS Test ===")
    results = fetch_news_rss("AAPL", max_per_source=5)
    for r in results[:5]:
        print(f"  [{r['source']}] {r['title'][:70]} | {r['published_at'][:10]}")
    print(f"Total: {len(results)} articles\n")
```

### Task 1.2 — 更新 `data_ingestion/news_fetcher.py` — 加入 RSS 作为二级数据源

在现有文件末尾追加以下函数，并修改 `fetch_news_for_all_tickers()` 的 fallback chain:

```python
# 在文件顶部导入区添加:
from data_ingestion.rss_fetcher import fetch_news_rss

# 在 fetch_news_for_all_tickers() 中修改 fallback 逻辑:
# 原来: NewsAPI → sample template
# 改为: NewsAPI → RSS (free/unlimited) → sample template (last resort)

# 具体改动: 当 NewsAPI 返回空或失败时:
if not records:
    logger.info("NewsAPI exhausted/failed, trying RSS feeds...")
    rss_records = fetch_news_rss(ticker, max_per_source=30)
    all_records.extend(rss_records)
    logger.info("RSS contributed %d articles for %s", len(rss_records), ticker)

if not all_records:
    # 只有 RSS 也失败时才用模板兜底
    ...
```

### Task 1.3 — 更新 `config.py` — RSS 配置

添加:
```python
# --- RSS Configuration ---
RSS_ENABLED = True
RSS_MAX_PER_SOURCE = 20       # 每个 RSS 源每只股票最多拉取数量
RSS_SOURCES = ["yahoo_finance", "google_news"]  # 默认启用的源（可扩展）
RSS_REQUEST_DELAY = 1         # 秒（礼貌性延迟）
```

---

## PHASE 2: 扩展股票池 + 任意 Ticker 搜索

### 目的：从固定 5 只股票扩展到 ~30 只 S&P 成分股，同时支持用户搜索任意不在列表中的 ticker。

### Task 2.1 — 更新 `config.py` — 扩充 TICKERS 列表

将 `TICKERS` 从 5 只扩展到 30 只（S&P 主要成分股，覆盖科技、金融、医疗、消费、能源）:

```python
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Financial  
    "JPM", "BAC", "V", "MA", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "KO", "PEP", "WMT", "COST", "NKE",
    # Energy / Industrial
    "XOM", "CVX", "CAT", "BA",
    # Communication / Utilities
    "DIS", "NFLX", "NEE",
]
```

### Task 2.2 — 创建 `data_ingestion/ticker_lookup.py`（新文件）

支持任意 ticker 的实时数据查询（不限于预配置列表）:

```python
"""Ticker lookup service — supports searching ANY stock, not just pre-configured ones."""

import logging
from typing import Optional

import pandas as pd

from config import START_DATE
from data_ingestion.market_data import fetch_ohlcv, fetch_ohlcv_alpha_vantage
from data_ingestion.rss_fetcher import fetch_news_rss
from data_storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


def is_valid_us_stock(ticker: str) -> bool:
    """
    Basic validation: US stock ticker is 1-5 uppercase letters.
    不做严格校验（因为有些特殊符号如 BRK.B），只做基本格式检查。
    """
    ticker = ticker.strip().upper()
    if len(ticker) < 1 or len(ticker) > 5:
        return False
    return ticker.replace(".", "").isalpha()


def lookup_ticker(ticker: str, db: DatabaseManager) -> dict:
    """
    查询任意 ticker 的完整信息。
    先查本地缓存（DB），没有则实时拉取。
    
    Returns dict:
    {
        "ticker": str,
        "has_data": bool,
        "market_rows": int,
        "news_count": int,
        "sentiment_available": bool,
        "is_realtime": bool,   # True=刚从API拉取, False=DB已有
        "price_latest": float | None,
        "error": str | None,
    }
    """
    ticker = ticker.strip().upper()
    
    if not is_valid_us_stock(ticker):
        return {"ticker": ticker, "has_data": False, "error": f"Invalid ticker format: {ticker}"}
    
    result = {
        "ticker": ticker,
        "has_data": False,
        "market_rows": 0,
        "news_count": 0,
        "sentiment_available": False,
        "is_realtime": False,
        "price_latest": None,
        "error": None,
    }
    
    # Step 1: Check local cache first
    market_df = db.get_market_data(ticker)
    news_df = db.get_news(ticker, limit=10)
    sent_df = db.get_sentiment(ticker)
    
    if not market_df.empty:
        result["has_data"] = True
        result["market_rows"] = len(market_df)
        result["price_latest"] = float(market_df["close"].iloc[-1])
        
    if not news_df.empty:
        result["news_count"] = len(db.get_news(ticker))  # full count
        
    if not sent_df.empty:
        result["sentiment_available"] = True
    
    # Step 2: If missing market data, fetch live
    if market_df.empty:
        logger.info("Ticker %s not in cache, fetching live...", ticker)
        try:
            df = fetch_ohlcv(ticker, start=START_DATE)
            if not df.empty:
                db.insert_market_data(ticker, df)
                result["has_data"] = True
                result["market_rows"] = len(df)
                result["price_latest"] = float(df["close"].iloc[-1])
                result["is_realtime"] = True
                logger.info("Fetched %d rows for %s from API", len(df), ticker)
        except Exception as e:
            result["error"] = f"Market data fetch failed: {e}"
    
    # Step 3: If missing news, fetch via RSS (free, unlimited)
    if news_df.empty:
        try:
            rss_news = fetch_news_rss(ticker, max_per_source=15)
            if rss_news:
                db.insert_news(rss_news)
                result["news_count"] = len(rss_news)
                result["is_realtime"] = True
                logger.info("Fetched %d RSS news for %s", len(rss_news), ticker)
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", ticker, e)
    
    return result


def search_tickers(query: str, limit: int = 10) -> list[dict]:
    """
    Fuzzy ticker search against known list.
    返回匹配结果供下拉框自动补全使用。
    
    Args:
        query: 用户输入的部分代码（如 "ap" 匹配 AAPL, ABBV 等）
        limit: 最大返回数量
    
    Returns:
        [{"symbol": "AAPL", "name": "Apple Inc."}, ...]
    """
    # 简单前缀匹配 + 子串匹配
    query_upper = query.upper().strip()
    from config import TICKERS
    
    matches = []
    for t in TICKERS:
        if t.startswith(query_upper) or query_upper in t:
            matches.append({"symbol": t, "name": t})  # 后期可接 Alpha Vantage 的 company info API
            if len(matches) >= limit:
                break
    
    return matches
```

### Task 2.3 — 更新 `collect_data.py` — 支持新增的 30 只股票

无需大改，只需确认 `collect_data.py` 会自动遍历新的 `config.TICKERS` 列表即可。如果它直接读 `config.TICKERS` 则自动生效；如果硬编码了旧列表则需要更新。

---

## PHASE 3: Dashboard v2.0 — 专业交易终端级 UI 重写

### 目的：从当前 4-tab 简单布局升级为 Bloomberg 终端风格的多面板界面。

### Task 3.1 — 完全重写 `dashboard/app.py`

**新布局结构**:

```
┌──────────────────────────────────────────────────────────────────┐
│ Header Bar: [◆ QuantumTrade v2] [Search: ______] [🔄] [⚙️]     │
├─────────┬────────────────────────────────────────────────────────┤
│ Sidebar │ Ticker Tape (滚动行情条)                                │
│         │ AAPL $178 ▲0.8% │ MSFT $380 ▼0.3% │ NVDA $890 ▲2.1%..│
│ ◆ Logo  ├────────────────────────────────────────────────────────┤
│ 🔍Search│                                                        │
│ [______]│ ┌─ Main Chart (蜡烛图 K-line + MA + 交易信号标记) ───┐ │
│         │ │                                                      │ │
│ 📋 Watch│ │          📈 Candlestick Chart                       │ │
│ AAPL ●  │ │              + Buy/Sell markers                     │ │
│ NVDA    │ │                                                      │ │
│ ...     │ └──────────────────────────────────────────────────────┘ │
│         │                                                        │
│ 📰 News  │ ┌─ Indicators ───┬─ Volume ─────┬─ Sentiment ───────┐ │
│ Feed(RSS)│ │ RSI(14) / MACD  │  成交量柱状图 │ NLP情感曲线        │ │
│ ·标题..  │ │                │              │ V/L/F/LLM 4线      │ │
│ ·标题..  │ └────────────────┴──────────────┴────────────────────┘ │
│         │                                                        │
│ 🤖 AI    │ ┌─ Ablation Results ──┬─ AI Assistant (RAG Chat) ──┐  │
│ Chat     │ │ Sharpe对比柱状图      │  输入问题 → 分析回答        │  │
│         │ │ NLP vs No-NLP        │                              │  │
│         │ └──────────────────────┴──────────────────────────────┘  │
│         │                                                        │
│ ────────│ ┌─ Market Heatmap (多股票涨跌热力图) ─────────────────┐ │
│ ⚙️ Sys   │ │  红/绿矩阵: 各股票当日涨跌幅                         │ │
│         │ └──────────────────────────────────────────────────────┘ │
└─────────┴────────────────────────────────────────────────────────┘
```

**核心功能清单**:

| # | 功能组件 | 技术实现 | 说明 |
|---:|---------|---------|------|
| 1 | **Ticker 搜索框** | `st.text_input` + 自动补全 | 替代原来的 `st.selectbox(TICKERS)` |
| 2 | **Ticker Tape** | `st.markdown` + CSS marquee 动画 | 顶部滚动显示多只股票实时价格 |
| 3 | **K线蜡烛图** | `plotly.graph_objects.Candlestick` | Open/High/Low/Close 四价蜡烛图，替代折线图 |
| 4 | **成交量子图** | `plotly` bar chart 共享 x-axis | 在蜡烛图下方叠加成交量 |
| 5 | **交易信号标记** | `scatter` mode='markers' on candlestick | Buy 三角向上▲ / Sell 三角向下▼ |
| 6 | **RSI/MACD 双面板** | 保持现有逻辑，优化配色 | 加超买超卖区域填充色 |
| 7 | **NLP 情感四线图** | VADER/LR/FinBERT/LLM 四条线 | 新增 LLM 第4条线（蓝色） |
| 8 | **消融实验对比** | with-NLP vs without-NLP 并排 equity curve | 新增图表类型 |
| 9 | **AI Chat 面板** | `st.chat_input` + `agents/research_agent.ask` | RAG 问答界面 |
| 10 | **市场热力图** | `plotly.imshow` 或自定义 heatmap | 多股票涨跌颜色矩阵 |
| 11 | **Sidebar Watchlist** | `st.multiselect` 或 checkbox 列表 | 用户自选关注列表 |
| 12 | **RSS 新闻流** | 滚动卡片式展示最新新闻 | 替代当前简单的 sentiment 图 |

**CSS 暗主题增强要求**:
- 保持现有的深蓝黑渐变背景 (`#0a0e17`)
- 卡片增加微妙的玻璃态效果 (`backdrop-filter: blur`)
- 数值变化时加入颜色过渡动画（绿色上涨/红色下跌，遵循中国股市惯例）
- K线图阳线红色、阴线绿色（中国惯例）

**完整的 app.py 重写代码**（约 800-1000 行）应包含以上所有面板。由于篇幅原因，此处给出结构框架，Claude Code 需要基于此框架生成完整实现:

```python
"""QuantumTrade v2.0 Dashboard — Professional Trading Terminal UI.

Layout:
  Header (search + ticker tape)
  ├── Main Area (70% width)
  │   ├── Row 1: Candlestick chart (OHLCV) + Volume
  │   ├── Row 2: RSI (left) | MACD (right)
  │   ├── Row 3: Sentiment signals (4 methods)
  │   ├── Row 4: Ablation comparison
  │   └── Row 5: Market heatmap
  └── Side Panel (30% width)
      ├── News feed (RSS cards)
      ├── AI Chat (RAG)
      └── System status
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports...
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import DB_PATH, TICKERS, INITIAL_CAPITAL, REFRESH_INTERVAL_SECONDS
from data_storage.db_manager import DatabaseManager
from data_ingestion.ticker_lookup import lookup_ticker, search_tickers

# === PAGE CONFIG ===
st.set_page_config(
    page_title="QuantumTrade v2 · NLP-RL Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === DARK THEME CSS (enhanced) ===
# [保持现有暗主题 + 新增: 蜡烛图配色、动画、玻璃态效果]

# === STATE MANAGEMENT ===
@st.cache_resource
def get_db(): ...

@st.cache_data(ttl=300)  # 5 min cache for fetched ticker data
def load_ticker_data(_db, ticker): ...

db = get_db()

# === SIDEBAR: Search + Watchlist + News Feed + AI Chat ===
with st.sidebar:
    # Logo
    # Search box (text input with autocomplete)
    # Watchlist (multiselect from TICKERS)
    # RSS News scrollable cards
    # AI Chat panel (collapsible)
    # System status at bottom
    pass

# === HEADER: Title + Ticker Tape ===
# Marquee-style scrolling price strip

# === MAIN AREA ===
# Row 1: Candlestick + Volume (make_subplots(rows=2, shared_x=True))
# Row 2: RSI | MACD (columns)
# Row 3: Sentiment 4-method overlay
# Row 4: Ablation study results
# Row 5: Multi-ticker heatmap

# === FOOTER ===
```

### Task 3.2 — 创建 `dashboard/components/charts.py`（新文件）

提取可复用的绘图函数:

```python
"""Shared chart components for QuantumTrade dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color scheme (中国股市: 红=涨, 绿=跌)
COLORS = {
    "up": "#ef4444",       # Red = up (China convention)
    "down": "#22c55e",     # Green = down
    "primary": "#38bdf8",
    "secondary": "#818cf8",
    "vader": "#22c55e",
    "lr": "#fbbf24",
    "finbert": "#38bdf8",
    "llm": "#f472b6",      # Pink for LLM
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(56,189,248,0.06)",
    "text": "#94a3b8",
}

DARK_TEMPLATE = go.layout.Template()  # 复用现有 dark template


def create_candlestick_chart(market_df: pd.DataFrame, trades_df=None) -> go.Figure:
    """创建专业 K 线蜡烛图。
    
    Args:
        market_df: 含 open/high/low/close/volume/date 列的 DataFrame
        trades_df: 可选，含 step/action/price 的交易记录，用于标记买卖点
    """
    fig = make_subplots(
        rows=2, cols=1, 
        shared_x=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )
    
    # K线 (Row 1)
    fig.add_trace(go.Candlestick(
        x=market_df["date"],
        open=market_df["open"],
        high=market_df["high"],
        low=market_df["low"],
        close=market_df["close"],
        name="Price",
        increasing_line_color=COLORS["up"],
        decreasing_line_color=COLORS["down"],
        increasing_fillcolor=COLORS["up"],
        decreasing_fillcolor=COLORS["down"],
    ), row=1, col=1)
    
    # MA 叠加
    for ma_col, color in [("MA50", COLORS["secondary"]), ("MA200", COLORS["muted"])]:
        if ma_col in market_df.columns:
            fig.add_trace(go.Scatter(
                x=market_df["date"], y=market_df[ma_col],
                mode="lines", name=ma_col,
                line=dict(color=color, width=1, dash="dash"),
                opacity=0.7,
            ), row=1, col=1)
    
    # 买卖标记
    if trades_df is not None and not trades_df.empty:
        buys = trades_df[trades_df["action"] == 1]
        sells = trades_df[trades_df["action"] == 2]
        
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["price"],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=12, 
                           color=COLORS["up"], line=dict(color="white", width=1)),
            ), row=1, col=1)
        
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["price"],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=12,
                           color=COLORS["down"], line=dict(color="white", width=1)),
            ), row=1, col=1)
    
    # 成交量 (Row 2)
    colors_vol = [COLORS["up"] if c >= o else COLORS["down"] 
                  for c, o in zip(market_df["close"], market_df["open"])]
    fig.add_trace(go.Bar(
        x=market_df["date"], y=market_df["volume"],
        name="Volume", marker_color=colors_vol, opacity=0.6,
    ), row=2, col=1)
    
    fig.update_layout(template=DARK_TEMPLATE, height=500, 
                      xaxis_rangeslider_visible=False,
                      showlegend=True, legend_orientation="h",
                      legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def create_heatmap(tickers_data: dict[str, float]) -> go.Figure:
    """创建市场热力图: 各股票当日涨跌。
    
    Args:
        tickers_data: {ticker: pct_change} 字典
    """
    # 将字典转为网格形式（适合方形热力图）
    symbols = list(tickers_data.keys())
    values = list(tickers_data.values())
    colors_map = [COLORS["up"] if v >= 0 else COLORS["down"] for v in values]
    
    # 使用 bar 替代 heatmap 以获得更好的标签显示
    fig = go.Figure(data=[go.Bar(
        x=symbols,
        y=values,
        marker_color=colors_map,
        text=[f"{v:+.2f}%" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    )])
    fig.update_layout(template=DARK_TEMPLATE, height=250,
                      yaxis_title="Daily Change (%)",
                      margin=dict(l=0, r=0, t=10, b=40))
    return fig


def create_sentiment_quad(sentiment_df: pd.DataFrame) -> go.Figure:
    """四路情感对比图 (VADER/LR/FinBERT/LLM)。"""
    fig = go.Figure()
    method_config = {
        "vader": {"color": COLORS["vader"], "dash": "solid"},
        "lr":    {"color": COLORS["lr"],    "dash": "dot"},
        "finbert": {"color": COLORS["finbert"], "dash": "dash"},
        "llm":   {"color": COLORS["llm"],   "dash": "longdash"},
    }
    
    for method, cfg in method_config.items():
        sub = sentiment_df[sentiment_df["method"] == method]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["sentiment_score"],
                mode="lines", name=method.upper(),
                line=dict(color=cfg["color"], width=1.5, dash=cfg["dash"]),
            ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(template=DARK_TEMPLATE, height=280,
                      yaxis_range=[-1.1, 1.1], yaxis_title="Sentiment Score",
                      legend_orientation="h")
    return fig
```

### Task 3.3 — 更新 `.env` 和 `requirements.txt`

`.env` 无需更改（已有全部 key）。

`requirements.txt` 追加:
```
feedparser>=6.0.10    # RSS 解析（PHASE 1 需要）
```

---

## PHASE 4: 验证与测试

### Task 4.1 — 验证 RSS 新闻抓取

```bash
cd C:\Users\21423\Desktop\STUDY\Y3_S2_group\Fintech_group
python -c "
from data_ingestion.rss_fetcher import fetch_news_rss
results = fetch_news_rss('AAPL', max_per_source=5)
print(f'Total: {len(results)} articles')
for r in results[:3]:
    print(f'  [{r[\"source\"]}] {r[\"title\"][:60]}')
"
```

预期输出: 至少 10+ 条来自 Yahoo Finance / Google News / MarketWatch 的真实新闻。

### Task 4.2 — 验证任意 Ticker 搜索

```bash
python -c "
from data_ingestion.ticker_lookup import lookup_ticker, is_valid_us_stock
from data_storage.db_manager import DatabaseManager
db = DatabaseManager()

# 搜索预配置列表外的 ticker
result = lookup_ticker('NVDA', db)
print(f'NVDA: has_data={result[\"has_data\"]}, rows={result[\"market_rows\"]}, realtime={result[\"is_realtime\"]}')

# 搜索一个完全随机的
result2 = lookup_ticker('AMD', db)
print(f'AMD: has_data={result2[\"has_data\"]}, error={result2.get(\"error\", \"none\")}')
"
```

### Task 4.3 — 启动 Dashboard v2

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

验证清单:
- [ ] 搜索框能输入任意 ticker（如 NVDA、AMD）
- [ ] 蜡烛图正确显示 OHLCV（红涨绿跌）
- [ ] Sidebar 有 RSS 新闻卡片滚动
- [ ] AI Chat 面板可用
- [ ] 市场热力图显示多股票涨跌

---

## 执行顺序总览

```
Phase 1: RSS 引擎 (30 min)
  1.1 创建 rss_fetcher.py
  1.2 修改 news_fetcher.py fallback chain
  1.3 config.py 加 RSS 配置
  
Phase 2: 搜索与扩展 (20 min)
  2.1 config.py TICKERS 扩到 30 只
  2.2 创建 ticker_lookup.py
  2.3 验证 collect_data.py 兼容性
  
Phase 3: Dashboard v2 (60-90 min)
  3.1 完全重写 app.py (最大工作量)
  3.2 创建 components/charts.py
  3.3 requirements.txt + feedparser
  
Phase 4: 验证 (15 min)
  4.1 RSS 抓取测试
  4.2 任意 ticker 搜索测试
  4.3 Dashboard 启动验证
```

---

## 给 Claude Code 的注意事项

1. **先读后改**: 每个文件修改前先 read_file 了解当前代码结构
2. **保留现有功能**: 所有改动是增量式的，不删除任何已工作的代码
3. **中文注释可以**: 项目允许中文注释，但变量名/函数名必须英文
4. **错误处理**: RSS/网络调用必须有 try/except + logging，不能用 bare except
5. **Dashboard 重写是最大的任务**: `app.py` 约 900 行，建议一次性完整写出而非多次小改
6. **不要在 Dashboard 里启动训练**: Dashboard 是纯展示层，训练通过 main.py 或 collect_data.py 完成
7. **图片占位符**: 如果需要架构图等静态资源，用文字占位符描述位置和内容，不实际生成图片
