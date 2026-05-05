# FinTech Project Upgrade Plan — For Claude Code Execution

> **Project Root**: `C:\Users\21423\Desktop\STUDY\Y3_S2_group\Fintech_group`
> **API Keys Available**:
> - NewsAPI: `d822bb76b1d84972a405166e9b3d23e6` (already in `.env`)
> - Alpha Vantage: `RPU1OXX1YUUB5XKF`
> - Volcano Engine (ByteDance/Doubao) LLM: `ff4fac01-ba31-41bf-a397-da8249d7cbe5`
>
> **Current DB Status** (from last check):
> - market_data: 8,275 rows (real yfinance data, 5 tickers, 2020-2024)
> - news: 187 rows (real NewsAPI data from 37 sources)
> - sentiment_signals: 153 rows (VADER=51, LR=51, FinBERT=51)
> - trading_logs: 0 rows (training not completed)
> - trade_orders: 0 rows
>
> **Execution Order**: Complete ALL tasks below in Phase order. Each Phase must be fully completed before moving to the next.

---

## PHASE 0: API Keys & Environment Setup

### Task 0.1 — Update `.env` with all keys

**File**: `.env` (currently only has NEWSAPI_KEY)

Add these lines to `.env`:

```
NEWSAPI_KEY=d822bb76b1d84972a405166e9b3d23e6
ALPHA_VANTAGE_KEY=RPU1OXX1YUUB5XKF
VOLCANO_API_KEY=ff4fac01-ba31-41bf-a397-da8249d7cbe5
VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCANO_MODEL_ID=  # User will fill in their deployed model ID from Volcano Engine console
```

### Task 0.2 — Update `requirements.txt`

Append these dependencies to the END of existing `requirements.txt`:

```
# === Phase A: Multi-source data & scheduling ===
alpha-vantage>=1.0.0
feedparser>=6.0.10
apscheduler>=3.10.4

# === Phase B: LLM Sentiment (4th method) ===
openai>=1.12.0

# === Phase C: RAG + Vector Store ===
chromadb>=0.4.22
sentence-transformers>=2.5.0

# === Phase D: SHAP Explainability ===
shap>=0.45.0

# === Phase E: Engineering ===
pytest>=8.0.0
ruff>=0.3.0
pyyaml>=6.0.1
docker-compose>=2.20.0

# === Dashboard enhancement ===
streamlit-extras>=0.43.0
```

---

## PHASE A: Multi-Source Data Fusion + Scheduled Collection

### Goal: Fix the "no real-time update" problem. Make data ingestion robust with dual sources and auto-scheduling.

### Task A.1 — Create `data_ingestion/scheduler.py` (NEW FILE)

Create APScheduler-based scheduler that implements the dormant `COLLECTION_INTERVAL_MINUTES=60` config.

**Requirements**:
- Use `apscheduler.schedulers.background.BackgroundScheduler`
- Expose a function `start_scheduler(db: DatabaseManager)` that:
  - Schedules `step_ingest(db)` every `config.COLLECTION_INTERVAL_MINUTES` minutes
  - Schedules `step_nlp(db)` after each successful ingestion
  - Logs each run with timestamp
  - Handles errors gracefully (one failure doesn't stop subsequent runs)
- Expose a function `run_once(db)` that runs one full ingest+nlp cycle (useful for manual trigger)
- Add CLI entry point: when running `python -m data_ingestion.scheduler`, it starts the scheduler and keeps running
- The scheduler should be importable and start/stop controllable

```python
"""APScheduler-based automated data collection for the trading platform."""

import logging
import signal
import sys

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from data_storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

_scheduler = None


def _ingest_cycle():
    """Run one full ingest → NLP cycle."""
    # Import here to avoid circular imports at module level
    from main import step_ingest, step_nlp
    db = DatabaseManager()
    try:
        step_ingest(db)
        step_nlp(db)
        logger.info("✅ Scheduled pipeline cycle completed")
    except Exception:
        logger.exception("❌ Scheduled pipeline cycle failed")


def start_scheduler(db: DatabaseManager | None = None):
    """Start the background scheduler for periodic data collection."""
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.warning("Scheduler already running")
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _ingest_cycle,
        trigger=IntervalTrigger(minutes=config.COLLECTION_INTERVAL_MINUTES),
        id="pipeline_cycle",
        name="Full Pipeline Cycle (Ingest + NLP)",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "📅 Scheduler started — pipeline runs every %d min",
        config.COLLECTION_INTERVAL_MINUTES,
    )
    return _scheduler


def stop_scheduler():
    """Stop the background scheduler if running."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("⏹ Scheduler stopped")


def run_once(db: DatabaseManager | None = None):
    """Manually trigger one pipeline cycle immediately."""
    db = db or DatabaseManager()
    _ingest_cycle()


def main():
    """CLI entry point: start scheduler and run until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Start scheduler on boot
    scheduler = start_scheduler()

    # Graceful shutdown on Ctrl+C
    def _shutdown(signum, frame):
        logger.info("Received shutdown signal...")
        stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("🚀 Data scheduler running. Press Ctrl+C to stop.")

    # Run one cycle immediately on start
    logger.info("Running initial pipeline cycle...")
    run_once()

    # Keep process alive
    try:
        import time
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        stop_scheduler()


if __name__ == "__main__":
    main()
```

### Task A.2 — Upgrade `data_ingestion/market_data.py` — Add Alpha Vantage Dual Source

**Current file**: Has yfinance primary + GBM synthetic fallback. MAX_RETRIES=1 (too low).

**Changes needed**:

1. Add Alpha Vantage as SECONDARY data source (after yfinance fails):

```python
def fetch_ohlcv_alpha_vantage(ticker, start="2020-01-01", end=None):
    """Fetch OHLCV from Alpha Vantage as fallback."""
    from alpha_vantage.timeseries import TimeSeries
    key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if not key:
        return None
    ts = TimeSeries(key=key, output_format="pandas")
    df, meta = ts.get_daily(symbol=ticker, outputsize="full")
    # Rename columns to standard format
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    col_map = {
        "date": "date", "1. open": "open", "2. high": "high",
        "3. low": "low", "4. close": "close", "6. volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["adjusted_close"] = df.get("close")
    return df[["date","open","high","low","close","volume","adjusted_close"]]
```

2. Change `MAX_RETRIES` from 1 to 3

3. Update `fetch_ohlcv()` fallback chain:
   ```
   Try yfinance (3 retries) 
     ↓ fail
   Try Alpha Vantage (1 retry)
     ↓ fail  
   Generate synthetic GBM data (last resort)
   ```

4. Add incremental fetch support — new function `fetch_incremental(ticker, since_date)` that only returns data newer than `since_date`. Check DB for latest date first.

### Task A.3 — Upgrade `data_ingestion/news_fetcher.py` — RSS Fallback + Incremental

**Changes needed**:

1. Add RSS feed fetching as secondary source (when NewsAPI fails or runs out of quota). Use free RSS feeds:
   - Yahoo Finance RSS: `https://finance.yahoo.com/news/rss/{ticker}`
   - Seeking Alpha: `https://seekingalpha.com/symbol/{ticker}/news/feed`
   - Google News: `https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en`

```python
import feedparser

RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/news/rss/{ticker}",
    "google_news": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
}

def fetch_news_rss(ticker: str, max_articles: int = 30) -> list[dict]:
    """Fetch news via RSS feeds as free fallback."""
    records = []
    for source_name, url_template in RSS_FEEDS.items():
        try:
            url = url_template.format(ticker=ticker)
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                published = entry.get("published") or entry.get("updated", "")
                records.append({
                    "ticker": ticker,
                    "source": f"rss_{source_name}",
                    "title": entry.get("title", ""),
                    "content": entry.get("description") or entry.get("summary", ""),
                    "url": entry.get("link", ""),
                    "published_at": published,
                })
        except Exception:
            continue
    return records
```

2. Update `fetch_news_for_all_tickers()` fallback chain:
   ```
   Try NewsAPI (with existing error handling)
     ↓ no results / failed
   Try RSS feeds (free, unlimited)
     ↓ still nothing
   Sample news (hardcoded, last resort)
   ```

3. Add `get_latest_news_date(ticker)` function that queries DB for most recent article date per ticker, so we only fetch newer articles.

### Task A.4 — Update `main.py` Step Ingestion for Incremental Mode

In `step_ingest()`:
- Before fetching, query DB for latest market data date per ticker using `db.get_market_data(ticker)`
- Pass `since_date` to fetch functions so they only get new data
- Log how many NEW rows were inserted vs skipped (duplicates)

### Task A.5 — Update `config.py`

Add:
```python
# --- Scheduler ---
SCHEDULER_ENABLED = True  # Set False to disable auto-scheduling

# --- Data Sources Priority ---
DATA_SOURCE_PRIORITY = ["yfinance", "alpha_vantage"]  # Market data fallback order
NEWS_SOURCE_PRIORITY = ["newsapi", "rss"]              # News fallback order
```

---

## PHASE B: LLM Sentiment Analysis (4th Method)

### Goal: Add Volcano Engine LLM as the 4th sentiment method, making the NLP pipeline truly impressive.

### Task B.1 — Create `nlp_pipeline/sentiment_llm.py` (NEW FILE)

Implement LLM-based sentiment analysis using Volcano Engine (Doubao) API via OpenAI-compatible SDK.

**Requirements**:
- Read API key and endpoint from environment variables (`VOLCANO_API_KEY`, `VOLCANO_BASE_URL`, `VOLCANO_MODEL_ID`)
- Function signature MUST match other sentiment modules for consistency:
  ```python
  def llm_sentiment_batch(news_df: pd.DataFrame) -> pd.DataFrame
  ```
- Returns DataFrame with columns: `[date, title, sentiment_score, confidence, label, reasoning]`
- The extra `reasoning` column is KEY — it stores the LLM's explanation like "Bullish because Q3 revenue beat expectations by 15%"
- Prompt engineering: send batched requests (5 articles at a time to manage cost/rate limits), ask for structured JSON response:
  - score: float in [-1, +1]
  - label: "positive" / "negative" / "neutral"
  - confidence: float in [0, 1]
  - reasoning: string (1 sentence explanation)
- Handle API errors gracefully: retry once, then fall back to neutral (0.0) score with warning log
- Cache results locally to avoid re-calling API for same articles (check DB before calling)

**Implementation template**:

```python
"""LLM-based sentiment analysis using Volcano Engine (Doubao) via OpenAI-compatible API."""

import json
import logging

import numpy as np
import pandas as pd
from openai import OpenAI

from config import VOLCANO_API_KEY, VOLCANO_BASE_URL, VOLCANO_MODEL_ID

logger = __import__("logging").getLogger(__name__)

SYSTEM_PROMPT = """You are a financial sentiment analysis expert. Analyze the given news headline and content.
Return a JSON object with exactly these fields:
- "score": a float from -1.0 (very bearish) to +1.0 (very bullish), 0.0 is neutral
- "label": one of "positive", "negative", or "neutral"
- "confidence": a float from 0.0 to 1.0 indicating your certainty
- "reasoning": a brief one-sentence explanation of your judgment

Respond ONLY with valid JSON, no markdown, no extra text."""

BATCH_SIZE = 5  # Articles per API call


def _call_llm(articles: list[dict]) -> list[dict]:
    """Send batch of articles to LLM, return sentiment results."""
    client = OpenAI(
        api_key=VOLCANO_API_KEY,
        base_url=VOLCANO_BASE_URL,
    )
    results = []
    for article in articles:
        user_content = f"Title: {article['title']}\nContent: {article.get('content', '')[:500]}"
        try:
            resp = client.chat.completions.create(
                model=VOLCANO_MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=150,
            )
            text = resp.choices[0].message.content.strip()
            # Clean potential markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(text)
            results.append({
                "title": article["title"],
                "sentiment_score": float(result.get("score", 0.0)),
                "label": result.get("label", "neutral"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", ""),
            })
        except Exception:
            logger.warning("LLM analysis failed for: %.60s...", article["title"])
            results.append({
                "title": article["title"],
                "sentiment_score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
                "reasoning": "Analysis unavailable (error)",
            })
    return results


def llm_sentiment_batch(news_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment using LLM. Returns DataFrame matching aggregator interface."""
    if news_df.empty:
        return pd.DataFrame(columns=["date", "sentiment_score", "confidence", "label", "reasoning"])

    articles = news_df[["title", "content", "published_at"]].to_dict("records")
    all_results = []

    # Process in batches
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i : i + BATCH_SIZE]
        logger.info("LLM analyzing articles %d-%d/%d...", i + 1, min(i + BATCH_SIZE, len(articles)), len(articles))
        batch_results = _call_llm(batch)
        all_results.extend(batch_results)

    # Build result DataFrame
    records = []
    for r in all_results:
        # Find matching original article for date
        match = next((a for a in articles if a["title"] == r["title"]), None)
        pub_date = match.get("published_at", "") if match else ""
        # Parse date safely
        try:
            dt = pd.to_datetime(pub_date).date()
        except Exception:
            dt = None
        records.append({
            "date": dt,
            "sentiment_score": np.clip(r["sentiment_score"], -1.0, 1.0),
            "confidence": np.clip(r["confidence"], 0.0, 1.0),
            "label": r["label"],
            "method": "llm",
            "reasoning": r["reasoning"],
        })

    df = pd.DataFrame(records)
    logger.info("LLM sentiment complete: %d articles analyzed", len(df))
    return df
```

### Task B.2 — Update `nlp_pipeline/aggregator.py` — Add Consensus & Agreement Metrics

**Current behavior**: Simple merge of three methods.

**Upgrades needed**:

1. Accept 4 methods now (add LLM column)
2. Compute **Fleiss' Kappa** across all 4 methods' labels (categorical agreement):
   ```python
   from statsmodels.stats.inter_rater import fleiss_kappa
   # Convert label → numeric: positive=2, neutral=1, negative=0
   # Build contingency matrix per day, compute Kappa
   ```

3. Compute **inter-method correlation matrix** (Pearson r between each pair's sentiment scores)

4. Weighted average aggregation option:
   - If Fleiss' Kappa > 0.6 (substantial agreement): use simple mean
   - If Kappa < 0.6 (poor disagreement): weight FinBERT and LLM higher (they're more sophisticated)

5. Output an aggregated daily DataFrame with extra metadata columns:
   ```python
   # New columns in aggregated result:
   - vader_score, lr_score, finbert_score, llm_score  (individual scores)
   - consensus_score          (final weighted avg)
   - fleiss_kappa             (agreement metric)
   - agreement_level         ("high"/"moderate"/"low")
   - llm_reasoning_sample    (pick 1 LLM reason per day for display)
   ```

### Task B.3 — Update `data_storage/schema.py` — Add LLM Reasoning Column

Add to `CREATE_SENTIMENT_TABLE`: add `reasoning TEXT` column after `label`.

Also add a new table for caching LLM results (avoid re-calling API):

```sql
CREATE TABLE IF NOT EXISTS llm_analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title_hash TEXT UNIQUE,  -- SHA256 of title for dedup
    title TEXT,
    sentiment_score REAL,
    confidence REAL,
    label TEXT,
    reasoning TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Update `ALL_TABLES` list accordingly.

### Task B.4 — Update `data_storage/db_manager.py` — Add LLM Methods

Add:
```python
def upsert_llm_cache(self, records: list[dict]):
    """Cache LLM analysis results to avoid re-calling API."""
    
def get_llm_cache(self, title_hashes: list[str]) -> dict:
    """Check cache before making API calls. Returns {hash: record}."""
```

### Task B.5 — Update `main.py` Step NLP — Include LLM

In `step_nlp()`, add after FinBERT:
```python
# 4) LLM (Volcano Engine Doubao)
logger.info("    - LLM sentiment...")
df_llm = llm_sentiment_batch(news_df)
```

And pass `df_llm` to `get_merged_sentiment()` along with the other 3.

### Task B.6 — Update `config.py` — Add LLM Config

```python
# --- LLM (Volcano Engine / Doubao) ---
VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "")
VOLCANO_BASE_URL = os.getenv("VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
VOLCANO_MODEL_ID = os.getenv("VOLCANO_MODEL_ID", "")  # Fill your model endpoint ID
LLM_BATCH_SIZE = 5
LLM_TEMPERATURE = 0.1
LLM_ENABLED = bool(VOLCANO_API_KEY and VOLCANO_MODEL_ID)
```

---

## PHASE C: RAG-Powered AI Chat Interface

### Goal: Let users ask natural language questions about stocks/news/sentiments and get intelligent answers backed by retrieved context.

### Task C.1 — Create `vector_store/` directory + `__init__.py`

### Task C.2 — Create `vector_store/chroma_store.py` (NEW FILE)

ChromaDB-backed vector store for semantic search over news articles.

```python
"""ChromaDB vector store for news article retrieval (RAG)."""

import hashlib
import logging

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import DB_PATH

logger = logging.getLogger(__name__)

# Lightweight embedding model (runs locally, no API needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, fast, good quality

_collection = None
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        logger.info("Loading sentence transformer model: %s", EMBEDDING_MODEL)
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_PATH.parent / "chroma_db"))
        _collection = client.get_or_create_collection(
            name="news_articles",
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def index_news(news_df: pd.DataFrame, batch_size: int = 50) -> int:
    """Embed and store news articles in ChromaDB for retrieval."""
    encoder = _get_encoder()
    collection = _get_collection()

    indexed = 0
    for i in range(0, len(news_df), batch_size):
        batch = news_df.iloc[i : i + batch_size]
        texts = (batch["title"].fillna("") + ". " + batch["content"].fillna("")).tolist()
        ids = [_text_hash(t) for t in texts]
        metadatas = [
            {
                "ticker": row["ticker"],
                "source": row.get("source", ""),
                "date": str(row.get("published_at", "")),
                "title": row["title"],
            }
            for _, row in batch.iterrows()
        ]
        embeddings = encoder.encode(texts).tolist()
        
        # Upsert (update if exists, insert if not)
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        indexed += len(batch)
    
    logger.info("Indexed %d news articles in vector store", indexed)
    return indexed


def search(query: str, top_k: int = 5, ticker_filter: str | None = None) -> list[dict]:
    """Semantic search over news articles. Returns relevant articles with scores."""
    encoder = _get_encoder()
    collection = _get_collection()
    
    query_embedding = encoder.encode([query]).tolist()[0]
    
    where = {"ticker": ticker_filter} if ticker_filter else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    
    articles = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"][0]):
            articles.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "relevance_score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
            })
    return articles


def reset_store():
    """Clear all data from the vector store (for testing)."""
    client = chromadb.PersistentClient(path=str(DB_PATH.parent / "chroma_db"))
    client.delete_collection("news_articles")
    global _collection
    _collection = None
```

### Task C.3 — Create `agents/research_agent.py` (NEW FILE)

RAG-powered research agent that answers questions using retrieved news + DB data + LLM.

```python
"""RAG-powered research agent: retrieves context, queries LLM, returns natural-language answers."""

import logging

import pandas as pd
from openai import OpenAI

from config import VOLCANO_API_KEY, VOLCANO_BASE_URL, VOLCANO_MODEL_ID
from data_storage.db_manager import DatabaseManager
from vector_store.chroma_store import search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial research assistant for an AI-driven trading platform.
You have access to:
1. Relevant news articles (retrieved via semantic search)
2. Current sentiment data for the stock
3. Recent RL agent trading decisions

Answer the user's question based on the provided context. Be specific, cite sources, and note uncertainty.
If the context doesn't contain enough information, say so honestly.
Keep answers concise but informative (2-4 sentences unless asked for detail)."""


def _build_context(question: str, ticker: str, db: DatabaseManager) -> str:
    """Gather relevant context from vector store and database."""
    parts = []

    # 1. Semantic search for related news
    news_results =.search(question, top_k=5, ticker_filter=ticker)
    if news_results:
        parts.append("=== RELEVANT NEWS ARTICLES ===")
        for i, article in enumerate(news_results, 1):
            meta = article["metadata"]
            parts.append(
                f"[{i}] {meta['title']} ({meta.get('ticker', '?')}, {meta.get('date', '?')}) "
                f"[Relevance: {article['relevance_score']:.2f}]\n"
                f"    {article['text'][:300]}"
            )

    # 2. Recent sentiment data
    sent_df = db.get_sentiment(ticker)
    if not sent_df.empty:
        latest = sent_df.sort_values("date").tail(7)  # Last 7 days
        parts.append(f"\n=== RECENT SENTIMENT ({ticker}, last 7 days) ===")
        for _, row in latest.iterrows():
            parts.append(
                f"  {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']} "
                f"| method={row['method']} | score={row['sentiment_score']:.3f}"
            )

    # 3. Latest agent decisions
    logs_df = db.get_trading_logs()
    if not logs_df.empty:
        ticker_logs = logs_df[logs_df["ticker"] == ticker] if "ticker" in logs_df.columns else logs_df
        if not ticker_logs.empty:
            latest_log = ticker_logs.iloc[-1]
            parts.append(f"\n=== LATEST AGENT STATE ({ticker}) ===")
            parts.append(
                f"  portfolio_value={latest_log.get('portfolio_value', 'N/A')}, "
                f"position={latest_log.get('position', 'N/A')}, "
                f"cash={latest_log.get('cash', 'N/A')}, "
                f"last_action={latest_log.get('action', 'N/A')}"
            )

    return "\n\n".join(parts) if parts else "No context data available."


def ask(question: str, ticker: str = "AAPL") -> str:
    """
    Main entry point: user asks a question, agent retrieves context and generates answer.
    
    Examples of questions this should handle:
    - "Why did the agent buy TSLA on March 15?"
    - "What's the recent sentiment trend for AAPL?"
    - "Should I be concerned about TSLA right now?"
    - "Summarize the biggest news affecting GOOGL this week"
    """
    if not VOLCANO_API_KEY or not VOLCANO_MODEL_ID:
        return "⚠️ LLM service is not configured. Please set VOLCANO_API_KEY and VOLCANO_MODEL_ID."

    db = DatabaseManager()
    context = _build_context(question, ticker, db)

    client = OpenAI(api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL)

    resp = client.chat.completions.create(
        model=VOLCANO_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    answer = resp.choices[0].message.content.strip()
    logger.info("Q: %s | A: %s", question[:80], answer[:80])
    return answer
```

### Task C.4 — Upgrade Dashboard to Multi-Page with Chat Tab

#### C.4.a — Create `dashboard/pages/` directory structure

Create these files (Streamlit multi-page convention — any Python file in `pages/` automatically becomes a page tab):

**File**: `dashboard/pages/1_📊_Market.py`
- Move current Tab 1 (Market & Sentiment) content here
- Enhance: add candlestick chart using `plotly.graph_objects.Candlestick`

**File**: `dashboard/pages/2_📰_Sentiment.py`
- Move sentiment visualization here
- ADD: inter-method correlation heatmap (4x4 matrix including LLM)
- ADD: Fleiss' Kappa gauge/chart
- ADD: LLM reasoning display panel (show the "why" behind scores)
- ADD: method comparison bar chart (scores per method per day)

**File**: `dashboard/pages/3_🤖_Agent.py`
- Move Agent Decisions tab here
- ADD: SHAP waterfall chart for latest decision explanation (see Phase D)

**File**: `dashboard/pages/4_💰_Performance.py`
- Move Portfolio Performance here
- ADD: side-by-side equity curves (With-NLP vs Without-NLP vs Buy-and-Hold)
- ADD: rolling Sharpe ratio chart
- ADD: underwater/drawdown chart (professional quant style)

**File**: `dashboard/pages/5_🔬_Ablation.py` (NEW PAGE — dedicated ablation study view)
- Full ablation study results table (all tickers)
- Interactive: click a ticker → see its with/without NLP comparison charts
- Statistical significance test results display (t-test p-value)
- Conclusion summary: "NLP helps for X out of Y tickers"

**File**: `dashboard/pages/6_💬_Chat.py` (★ THE SHOWSTOPPER — AI Chat Interface)
This is the most visually impressive feature. Implement:

```python
"""AI Chat Interface — RAG-powered conversational analysis."""

import streamlit as st
from agents.research_agent import ask

st.title("💬 AI Research Assistant")
st.caption("Ask questions about any ticker. Answers are powered by retrieved news + sentiment data + LLM.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about any ticker..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ticker = st.sidebar.selectbox("Select ticker context", TICKERS, key="chat_ticker")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(question=prompt, ticker=ticker)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar: suggested questions
st.sidebar.header("💡 Try asking:")
suggestions = [
    f"What's the recent sentiment trend for {ticker}?",
    f"Why should I buy or sell {ticker} right now?",
    f"Summarize the biggest news about {ticker} this month.",
    f"How confident is the NLP model about {ticker}'s outlook?",
    "Compare sentiment across all 5 tickers.",
]
for s in suggestions:
    if st.sidebar.button(s):
        st.session_state.messages.append({"role": "user", "content": s})
        st.rerun()
```

#### C.4.b — Simplify `dashboard/app.py` to landing page only

After extracting pages, `app.py` becomes a clean landing/overview page with:
- System status overview (metrics cards)
- Quick links to each page
- Architecture diagram (using `st.markdown` with Mermaid or image placeholder)

---

## PHASE D: SHAP Explainability for RL Decisions

### Goal: Make the DQN's "black box" decisions interpretable. This addresses the biggest criticism of RL in finance.

### Task D.1 — Create `rl_engine/explainer.py` (NEW FILE)

```python
"""SHAP-based explainer for DQN trading decisions."""

import logging

import numpy as np
import pandas as pd
import shap
import torch

from rl_engine.dqn import DQNAgent, QNetwork
from rl_engine.env import FinancialTradingEnv, STATE_DIM

logger = logging.getLogger(__name__)

STATE_FEATURE_NAMES = [
    "price", "MA50", "MA200", "RSI", "MACD",
    "position_shares", "cash", "sentiment_score",
]


class TradingExplainer:
    """Use SHAP to explain which features drove the DQN's Q-value predictions."""
    
    def __init__(self, agent: DQNAgent):
        self.agent = agent
        self.explainer = None
        self._background_data = None
        
    def fit(self, env: FinancialTradingEnv, n_background: int = 100):
        """
        Prepare explainer with background data from the environment.
        Should be called before explain().
        """
        # Sample random states from the environment as background
        states = []
        for _ in range(n_background):
            state, _ = env.reset()
            # Random walk through env to get diverse states
            for _ in range(np.random.randint(10, 50)):
                action = env.action_space.sample()
                state, _, terminated, _, _ = env.step(action)
                if terminated:
                    break
            states.append(state)
        
        self._background_data = np.array(states)
        
        # Use DeepExplainer for neural networks (fast and accurate for DNNs)
        self.explainer = shap.DeepExplainer(self.agent.q_network, self._background_data)
        logger.info("SHAP explainer fitted with %d background samples", n_background)
        
    def explain_state(self, state: np.ndarray) -> dict:
        """
        Explain why DQN chose action for given state.
        
        Returns:
            dict with:
            - 'shap_values': array of SHAP values per feature
            - 'feature_importance': sorted list of (feature_name, shap_value)
            - 'base_value': expected SHAP value (baseline)
            - 'explanation_text': human-readable summary
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain()")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(state_tensor)
        
        # shap_values shape: [n_actions, 1, state_dim] → flatten
        if isinstance(shap_values, list):
            # One set of SHAP values per action
            action_shap = {i: sv.flatten() for i, sv in enumerate(shap_values)}
        else:
            action_shap = {i: shap_values[i].flatten() for i in range(3)}
        
        # Get the chosen action's SHAP values
        chosen_action = int(self.agent.select_action(state, evaluate=True))
        sv = action_shap[chosen_action]
        
        # Feature importance ranking
        importance = sorted(
            zip(STATE_FEATURE_NAMES, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        # Generate human-readable explanation
        pos_drivers = [(f, v) for f, v in importance if v > 0.01]
        neg_drivers = [(f, v) for f, v in importance if v < -0.01]
        
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        explanation_parts = [f"DQN chose **{action_names[chosen_action]}** because:"]
        
        if pos_drivers:
            drivers_str = ", ".join([f"{f} (+{v:.3f})" for f, v in pos_drivers[:3]])
            explanation_parts.append(f"- Positive signals: {drivers_str}")
        if neg_drivers:
            drivers_str = ", ".join([f"{f} ({v:.3f})" for f, v in neg_drivers[:3]])
            explanation_parts.append(f"- Negative signals: {drivers_str}")
        if not pos_drivers and not neg_drivers:
            explanation_parts.append("- All features near neutral (weak signal)")
        
        return {
            "shap_values": sv.tolist(),
            "feature_importance": [(f, round(v, 4)) for f, v in importance],
            "base_value": float(self.explainer.expected_value.data.cpu().numpy()) if hasattr(self.explainer.expected_value, 'data') else 0.0,
            "chosen_action": chosen_action,
            "action_name": action_names[chosen_action],
            "explanation_text": "\n".join(explanation_parts),
        }
    
    def generate_summary_report(self, df: pd.DataFrame, agent: DQNAgent, n_samples: int = 50) -> pd.DataFrame:
        """
        Run explainer on multiple states to build a global feature importance report.
        Useful for the dashboard's "What drives the agent?" section.
        """
        env = FinancialTradingEnv(df)
        self.fit(env)
        
        records = []
        for _ in range(n_samples):
            state, _ = env.reset()
            for _ in range(np.random.randint(10, min(len(df) - env.current_step, 100))):
                exp = self.explain_state(state)
                record = {f"feat_{f}": v for f, v in exp["feature_importance"]}
                record["action"] = exp["chosen_action"]
                records.append(record)
                action = env.action_space.sample()
                state, _, term, _, _ = env.step(action)
                if term:
                    break
        
        report_df = pd.DataFrame(records)
        # Average absolute SHAP value per feature = global importance
        feat_cols = [c for c in report_df.columns if c.startswith("feat_")]
        global_importance = report_df[feat_cols].abs().mean().sort_values(ascending=False)
        logger.info("Global feature importance computed")
        
        return global_importance
```

### Task D.2 — Integrate Explainer into Training/Evaluation Flow

In `rl_engine/train.py` or `evaluation.py`, add an optional step after training:
- Create `TradingExplainer`, fit it on the training environment
- Run `explain_state()` on a few key decision points (e.g., largest trade, worst drawdown moment)
- Log explanations to `trading_logs` or a separate `explanations` table

### Task D.3 — Add SHAP Chart to Dashboard Agent Page

In `dashboard/pages/3_🤖_Agent.py`:
- Add a SHAP waterfall chart section using `shap.plots.waterfall()` rendered via `plotly`
- Show top 3 features driving the latest decision with bar chart

---

## PHASE E: Engineering — CI/CD, Docker, Config

### Task E.1 — Create `.github/workflows/ci.yml` (NEW FILE)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest ruff
      
      - name: Lint with ruff
        run: ruff check . --output-format=github
      
      - name: Type check (basic)
        run: ruff check --select I .
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
      
      - name: Test imports
        run: |
          python -c "from data_ingestion.market_data import fetch_all_tickers; print('✓ market_data OK')"
          python -c "from nlp_pipeline.sentiment_finbert import finbert_sentiment_batch; print('✓ finbert OK')"
          python -c "from rl_engine.dqn import DQNAgent; print('✓ dqn OK')"
          python -c "from rl_engine.env import FinancialTradingEnv; print('✓ env OK')"
          python -c "from dashboard.app import *; print('✓ dashboard OK')"
```

### Task E.2 — Create `tests/` Directory + Basic Tests

**File**: `tests/__init__.py` — empty

**File**: `tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

**File**: `tests/test_market_data.py`:
```python
"""Test market data fetching functions."""
import pandas as pd
import pytest
from data_ingestion.market_data import _generate_synthetic_ohlcv, fetch_ohlcv

def test_synthetic_data_shape():
    df = _generate_synthetic_ohlcv("AAPL", start="2024-01-01", end="2024-01-31")
    assert not df.empty
    assert "close" in df.columns
    assert "volume" in df.columns
    assert len(df) > 15  # At least 20 trading days in Jan

def test_synthetic_data_positive_prices():
    df = _generate_synthetic_ohlcv("TSLA")
    assert (df["close"] > 0).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
```

**File**: `tests/test_nlp.py`:
```python
"""Test NLP pipeline components."""
import pandas as pd
import pytest
from nlp_pipeline.aggregator import get_merged_sentiment

def test_aggregator_empty_input():
    empty_df = pd.DataFrame()
    result = get merged_sentiment(empty_df, empty_df, empty_df, empty_df)
    assert result.empty

def test_aggregator_basic_merge():
    # Create minimal input DataFrames
    for method in ["vader"]:
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "sentiment_score": [0.1, -0.2, 0.3, 0.0, -0.1],
        })
    # Verify aggregation produces expected output
    assert isinstance(result, pd.DataFrame)
```

**File**: `tests/test_rl_env.py`:
```python
"""Test RL trading environment."""
import numpy as np
import pandas as pd
import pytest
from rl_engine.env import FinancialTradingEnv

def _make_test_df(n=100):
    dates = pd.date_range("2023-01-01", periods=n)
    prices = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "date": dates, "close": prices, "open": prices * 0.99,
        "high": prices * 1.02, "low": prices * 0.98, "volume": 1000000,
        "MA50": prices, "MA200": prices, "RSI": 50.0, "MACD": 0.0, "sentiment_score": 0.0,
    })

def test_env_init():
    df = _make_test_df()
    env = FinancialTradingEnv(df)
    assert env.action_space.n == 3
    state, info = env.reset()
    assert len(state) == 8  # STATE_DIM

def test_env_step():
    df = _make_test_df()
    env = FinancialTradingEnv(df)
    env.reset()
    obs, reward, term, trunc, info = env.step(1)  # Buy
    assert isinstance(reward, float)
    assert "portfolio_value" in info

def test_env_full_episode():
    df = _make_test_df(n=50)
    env = FinancialTradingEnv(df)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
    assert steps > 0
```

**File**: `tests/test_db.py`:
```python
"""Test database operations."""
import tempfile
import os
from pathlib import Path
from data_storage.db_manager import DatabaseManager
from data_storage.schema import ALL_TABLES

def test_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    # Check tables exist
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    expected = {"news", "market_data", "sentiment_signals", "trading_logs", "trade_orders"}
    assert expected.issubset(tables)

def test_insert_and_query_news(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    count = db.insert_news([{
        "ticker": "TEST", "source": "test", "title": "Hello",
        "content": "World", "url": "http://t.co", "published_at": "2024-01-01"
    }])
    assert count == 1
    news = db.get_news("TEST")
    assert len(news) == 1
    assert news.iloc[0]["title"] == "Hello"
```

### Task E.3 — Create `Dockerfile` (NEW FILE, project root)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python deps — copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download models (so container starts faster on first run)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')" || true
RUN pip install sentence-transformers --quiet && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Default: run full pipeline then launch dashboard
CMD ["bash", "-c", "python main.py --ablate && streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
```

### Task E.4 — Create `docker-compose.yml` (NEW FILE, project root)

```yaml
version: "3.8"

services:
  trading-platform:
    build: .
    container_name: nlp-rl-trading
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data          # Persist SQLite DB
      - ./models:/app/models       # Persist trained DQN models
      - ./.env:/app/.env:ro       # Mount API keys (read-only)
    environment:
      - TZ=Asia/Shanghai
    restart: unless-stopped
```

### Task E.5 — Create `pyproject.toml` (NEW FILE)

```toml
[project]
name = "nlp-rl-trading-platform"
version = "1.0.0"
description = "End-to-end NLP-Driven Reinforcement Learning Trading Platform"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "FinTech Group", email = ""}]
keywords = ["nlp", "reinforcement-learning", "trading", "finbert", "dqn", "quantitative-finance"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.scripts]
nlp-rl-run = "main:main"
nlp-rl-scheduler = "data_ingestion.scheduler:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### Task E.6 — Update `.gitignore`

Ensure these entries exist:
```
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Project-specific
data/trading.db
data/chroma_db/
models/*.pt
.ipynb_checkpoints/

# Secrets
.env
*.key

# IDE
.vscode/
.idea/
*.swp
*

# OS
.DS_Store
Thumbs.db
```

---

## PHASE F: README & Documentation Overhaul

### Task F.1 — Rewrite `README.md` (COMPLETE REPLACEMENT)

Write a professional GitHub README with:

1. **Header section**: Project name + badges (Python, PyTorch, Docker, CI, License)
2. **Features table**: 8 feature rows with emoji icons
3. **Architecture diagram**: Using Mermaid syntax (renders on GitHub)
4. **Quick Start**: Docker (primary) + Local (secondary) — both 3-line commands
5. **Results snapshot table**: With-NLP vs Without-NLP comparison
6. **Tech Stack grid**: Visual component breakdown
7. **Project structure tree**: Collapsed directory view
8. **Module documentation links**
9. **Contributing guide** (brief)
10. **License**

**CRITICAL**: The README must make an impact in under 5 seconds of scrolling. Use emojis, badges, tables, and clear visual hierarchy.

### Task F.2 — Create `docs/architecture.md`

Detailed system architecture document covering:
- Data flow diagram (text/Mermaid)
- Module responsibilities
- API interfaces between modules
- Database schema diagram (text)
- Configuration reference

### Task F.3 — Update `config.py` — Externalize Parameters to YAML

Create `config.yaml` (NEW FILE):
```yaml
# === Tickers ===
tickers: [AAPL, MSFT, GOOGL, AMZN, TSLA]
start_date: "2020-01-01"
end_date: "2024-12-31"

# === Data Sources ===
market_data_sources: [yfinance, alpha_vantage]
news_sources: [newsapi, rss]
collection_interval_minutes: 60

# === NLP ===
finbert_model: "ProsusAI/finbert"
sentiment_methods: [vader, logistic_regression, finbert, llm]
llm_enabled: true
llm_batch_size: 5
llm_temperature: 0.1

# === RL ===
lookback_window: 200
initial_capital: 100000.0
transaction_cost_pct: 0.001
episodes: 200
gamma: 0.99
epsilon_start: 1.0
epsilon_min: 0.01
epsilon_decay: 0.995
learning_rate: 0.001
batch_size: 64
replay_buffer_size: 10000
target_update_freq: 10
train_split: 0.6
val_split: 0.2

# === Dashboard ===
dashboard_port: 8501
refresh_interval_seconds: 5
```

Then update `config.py` to load from YAML with env var overrides:
```python
import yaml
# Load YAML config
_config_path = Path(__file__).parent / "config.yaml"
if _config_path.exists():
    with open(_config_path) as f:
        _yaml_cfg = yaml.safe_load(f)
    # Override with YAML values (keep env vars as highest priority)
    ...
```

---

## EXECUTION CHECKLIST (for Claude Code to verify completion)

After completing ALL phases above, verify each item:

### Functionality Checklist
- [ ] `python main.py --skip-collect --skip-nlp` runs without errors (uses cached data)
- [ ] `python -m data_ingestion.scheduler` starts and runs initial cycle
- [ ] All 4 sentiment methods execute in `step_nlp()` (VADER, LR, FinBERT, LLM)
- [ ] Aggregator computes Fleiss' Kappa and weighted consensus
- [ ] DQN trains 200 episodes per ticker and saves model to `models/dqn_model.pt`
- [ ] Ablation study runs and prints comparison table
- [ ] `streamlit run dashboard/app.py` launches with 6 pages
- [ ] Chat page (Page 6) responds to questions using RAG + LLM
- [ ] SHAP explainer generates decision explanations
- [ ] `pytest tests/ -v` passes all tests
- [ ] `docker compose up --build` builds and starts successfully

### Code Quality Checklist
- [ ] No hardcoded API keys (all in `.env`)
- [ ] All new functions have docstrings
- [ ] Error handling with try/except + logging (never bare except)
- [ ] Type hints on function signatures
- [ ] `.gitignore` covers all generated files

### Files That Should Exist After Completion
```
data_ingestion/scheduler.py           ← NEW
data_ingestion/market_data.py         ← MODIFIED (dual-source)
data_ingestion/news_fetcher.py        ← MODIFIED (RSS fallback)
nlp_pipeline/sentiment_llm.py         ← NEW
nlp_pipeline/aggregator.py            ← MODIFIED (4-method + kappa)
vector_store/__init__.py              ← NEW
vector_store/chroma_store.py          ← NEW
agents/__init__.py                    ← NEW
agents/research_agent.py              ← NEW
rl_engine/explainer.py                ← NEW
dashboard/pages/                     ← NEW DIRECTORY (6 files)
dashboard/app.py                     ← MODIFIED (landing page)
tests/                               ← NEW DIRECTORY (4 test files)
.github/workflows/ci.yml             ← NEW
Dockerfile                           ← NEW
docker-compose.yml                   ← NEW
pyproject.toml                       ← NEW
config.yaml                          ← NEW
config.py                            ← MODIFIED (YAML loading)
data_storage/schema.py               ← MODIFIED (llm cache table)
data_storage/db_manager.py           ← MODIFIED (llm cache methods)
requirements.txt                     ← MODIFIED (new deps)
README.md                            ← COMPLETE REWRITE
.gitignore                           ← MODIFIED
docs/architecture.md                  ← NEW
UPGRADE_PLAN.md                      ← THIS FILE (can delete after completion)
```

---

## IMPORTANT NOTES FOR CLAUDE CODE

1. **Read existing code before modifying** — understand patterns first, then change
2. **Preserve all existing functionality** — every upgrade is additive, never remove working code
3. **Test incrementally** — after each phase, verify the pipeline still runs end-to-end
4. **The Volcano Engine MODEL_ID is not yet confirmed** — user needs to fill this in from their console. If empty, make LLM features gracefully skip with a clear log message
5. **Chinese comments are acceptable** in this project (BNBU course), but keep function/variable names in English
6. **Database schema changes need migration handling** — since `trading.db` may already exist, use `ALTER TABLE` or recreate-safe DDL
7. **The `main.py` entry point must remain unchanged** — `python main.py --ablate` must still work as the single command to run everything

---

## PHASE G: Dashboard UX Overhaul & Prompt Engineering Enhancement

### Goal: Make the dashboard intuitive at first glance, design for graceful degradation without API keys, and implement production-quality prompt/RAG strategies.

### Task G.1 — Redesign Dashboard Layout: Progressive Disclosure

**Current problem**: 6 separate tabs are too deep; users can't see important info quickly.

**New layout**: Single-page app with collapsible sections, not tabs.

**File**: `dashboard/app.py` — COMPLETE RESTRUCTURE

```
NEW LAYOUT STRUCTURE (replace tab-based with section-based):
┌─────────────────────────────────────────────────────────────┐
│  Header: title + ticker selector + refresh button           │
├─────────────────────────────────────────────────────────────┤
│  ROW 1: Metric Cards (always visible, 4 columns)            │
│  [Return] [Sharpe] [MDD] [Sentiment]                       │
├──────────────────────────┬──────────────────────────────────┤
│  ROW 2: Main Chart Area   │  ROW 2 Right: Analysis Panel    │
│  (70% width)              │  (30% width)                    │
│                           │                                  │
│  [View Switcher]:         │  ┌─ NLP Sentiment Panel ──┐     │
│  • Price (K-line + MA)    │  │ 4-method gauge/meter    │     │
│  • Equity Curve (3 lines) │  | Method comparison bars │     │
│  • Ablation Comparison    │  | LLM reasoning sample    │     │
│  • Agent Decision Trace   │  | Agreement level (kappa) │     │
│                           │  └────────────────────────┘     │
│                           │  ┌─ Agent Status Panel ────┐     │
│                           │  │ Latest action + price   │     │
│                           │  │ SHAP top drivers        │     │
│                           │  │ Confidence meter       │     │
│                           │  └────────────────────────┘     │
│                           │  ┌─ AI Chat (collapsible)──┐    │
│                           │  │ Floating chat widget    │    │
│                           │  └────────────────────────┘    │
├──────────────────────────┴──────────────────────────────────┤
│  ROW 3: Expandable Detail Sections (accordion/collapsible)  │
│  [▶ Technical Indicators] [▶ Ablation Results] [▶ SHAP]    │
│  (click to expand, collapsed by default)                     │
└─────────────────────────────────────────────────────────────┘
```

Implementation requirements:
- Use `st.columns` with ratios `[0.7, 0.3]` for main + side panel
- Use `st.expander()` for detail sections (collapsed by default)
- View switcher: `st.segmented_control()` or radio buttons for chart type selection
- Chat widget: use `st.chat_input()` positioned in sidebar or as expandable panel
- Add "narrative flow" arrows between sections using st.markdown with emojis

### Task G.2 — Implement Graceful Degradation (No-API-Key Mode)

**Goal**: Users WITHOUT API keys can still experience a convincing demo.

Create `utils/capability_manager.py` (NEW FILE):

```python
"""Detect available capabilities and provide graceful degradation.

When API keys are missing, fall back to local-only mode 
with sample/synthetic data so the platform always demonstrates value.
"""

import os
from dataclasses import dataclass
from enum import Enum


class CapabilityLevel(Enum):
    FULL = "full"          # All APIs working, all methods available
    ENHANCED = "enhanced"  # Local models work, no LLM/API data sources
    BASIC = "basic"        # Only VADER + LR (purely local)
    DEMO = "demo"          # Sample data only, no real data


@dataclass
class CapabilityReport:
    level: CapabilityLevel
    sentiment_methods: list[str]      # Available method names
    data_sources: list[str]           # Available data source names
    llm_enabled: bool
    rag_enabled: bool
    chat_enabled: bool
    warnings: list[str]


def detect_capabilities() -> CapabilityReport:
    """Check environment and report what's available."""
    
    # Always available (no external deps)
    methods = ["vader", "logistic_regression"]
    sources = ["synthetic"]
    warnings = []
    
    # Check FinBERT model (downloaded locally)
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("ProsusAI/finbert")
        methods.append("finbert")
    except Exception:
        warnings.append("FinBERT model not cached. Run pipeline once to download, or set TRANSFORMERS_OFFLINE=1")
    
    # Check LLM
    llm_ok = bool(os.getenv("VOLCANO_API_KEY") and os.getenv("VOLCANO_MODEL_ID"))
    if llm_ok:
        methods.append("llm")
    else:
        warnings.append("LLM not configured. Set VOLCANO_API_KEY and VOLCANO_MODEL_ID in .env for 4th sentiment method.")
    
    # Check data source APIs
    if os.getenv("ALPHA_VANTAGE_KEY"):
        sources.append("alpha_vantage")
    if os.getenv("NEWSAPI_KEY"):
        sources.append("newsapi")
    # RSS always works (free, no key needed)
    sources.append("rss")
    
    # Determine level
    if llm_ok and "newsapi" in sources:
        level = CapabilityLevel.FULL
    elif "finbert" in methods and ("newsapi" in sources or "rss" in sources):
        level = CapabilityLevel.ENHANCED
    elif len(methods) >= 2:
        level = CapabilityLevel.BASIC
    else:
        level = CapabilityLevel.DEMO
    
    return CapabilityReport(
        level=level,
        sentiment_methods=methods,
        data_sources=sources,
        llm_enabled=llm_ok,
        rag_enabled=llm_ok,  # RAG depends on LLM
        chat_enabled=llm_ok,
        warnings=warnings,
    )
```

Then in `dashboard/app.py`, show capability banner at top:

```python
from utils.capability_manager import detect_capabilities

cap = detect_capabilities()

# Show capability status bar
if cap.level != CapabilityLevel.FULL:
    emoji_map = {
        CapabilityLevel.ENHANCED: "🟡",
        CapabilityLevel.BASIC: "🟠",
        CapabilityLevel.DEMO: "🔴",
    }
    st.caption(
        f"{emoji_map.get(cap.level, '✅')} "
        f"Mode: {cap.level.value.upper()} │ "
        f"Methods: {', '.join(cap.sentiment_methods)} │ "
        f"Data: {', '.join(cap.data_sources)}"
    )
    if cap.warnings:
        with st.expander("⚙️ How to enable more features"):
            for w in cap.warnings:
                st.warning(w)
```

### Task G.3 — Enhance Prompts with Few-Shot Examples

**File**: `nlp_pipeline/sentiment_llm.py` — Replace SYSTEM_PROMPT with enhanced version

The new prompt MUST include:
1. Clear scoring rubric with numerical boundaries
2. **3-5 few-shot examples** (anchor points showing expected input→output pairs)
3. Domain-specific guidance (finance terminology, what to focus on vs ignore)
4. Anti-manipulation instructions (don't be influenced by hype words alone)

Write the actual prompt text into the file (not placeholder). Include examples for:
- Strong positive (+0.8 to +1.0): earnings beat, product launch, regulatory win
- Moderate positive (+0.3 to +0.7): analyst upgrade, partnership, expansion
- Neutral (-0.2 to +0.2): management change, routine announcement, minor news
- Moderate negative (-0.7 to -0.3): earnings miss, downgrade, supply issue
- Strong negative (-1.0 to -0.7): fraud, recall, major fine, bankruptcy risk

### Task G.4 — Intent-Based RAG Retrieval Strategy

**File**: `agents/research_agent.py` — Upgrade `_build_context()`

Add intent detection function `_detect_intent(question)` that classifies questions into types:
- `"causal"` → prioritize recent news articles (why did X happen?)
- `"decision"` → pull sentiment trends + agent decisions + SHAP (should I buy/sell?)
- `"compare"` → aggregate all tickers' summaries (which stock is best?)
- `"sentiment"` → pull time-series sentiment data (what's the trend?)
- `"general"` → balanced mix of everything

Each intent type maps to different ChromaDB query parameters:
- Different `top_k` values (3-10)
- Different date filters (last day / last week / last month)
- Whether to include trading logs
- Whether to include SHAP explanations

Also add a **query rewriter** that expands abbreviations:
- "tsla" → "TSLA Tesla Motors stock"
- "为什么跌" → "Why did the stock price drop decline fall"
This improves retrieval quality for bilingual queries.

---

## PHASE H: Portfolio Transformation Foundation

### Goal: Design decisions NOW that make post-assignment evolution into a personal showcase project seamless.

### Task H.1 — Abstract Base Classes for Extensibility

Create interfaces that let future contributors (including you) add new components easily.

**File**: `interfaces/__init__.py` (NEW DIRECTORY + FILE)

```python
"""Abstract base classes for pluggable components."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class SentimentMethod(ABC):
    """Interface for any sentiment analysis method."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier (e.g., 'vader', 'finbert', 'my_custom')."""
        ...
    
    @abstractmethod
    def analyze(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles.
        
        Returns DataFrame with columns:
        [date, sentiment_score, confidence, label, (optional) reasoning]
        """
        ...


class DataSource(ABC):
    """Interface for market data or news data sources."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fetch(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch data for given symbol. Return standardized DataFrame."""
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is reachable/configured."""
        ...


class RLAgent(ABC):
    """Interface for any RL-based trading agent."""
    
    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the agent on provided data."""
        ...
    
    @abstractmethod
    def predict(self, state) -> int:
        """Choose action given current state."""
        ...
    
    @abstractmethod
    def explain(self, state) -> dict:
        """Explain why the agent chose its action."""
        ...
```

Then refactor EXISTING modules to implement these interfaces (backward-compatible):

```python
# In nlp_pipeline/sentiment_vader.py:
class VADERMethod(SentimentMethod):
    @property
    def name(self) -> str:
        return "vader"
    
    def analyze(self, news_df: pd.DataFrame) -> pd.DataFrame:
        return vader_sentiment_batch(news_df)


# In nlp_pipeline/sentiment_llm.py:
class LLMMethod(SentimentMethod):
    ...
```

This means anyone can add a NEW sentiment method by:
```python
# my_method.py
class MyCoolMethod(SentimentMethod):
    def name(self): return "my_cool"
    def analyze(self, news_df): ...  # their implementation

# Then register in config.yaml:
# sentiment_methods: [vader, logistic_regression, finbert, llm, my_cool]
```

### Task H.2 — Plugin Discovery System

**File**: `utils/plugin_manager.py` (NEW FILE)

Auto-discover classes implementing ABC interfaces:

```python
"""Plugin manager: auto-discovers and registers sentiment methods & data sources."""

import importlib
import inspect
from pathlib import Path

from interfaces import SentimentMethod, DataSource

# Registry of discovered plugins
_SENTIMENT_METHODS: dict[str, type[SentimentMethod]] = {}
_DATA_SOURCES: dict[str, type[DataSource]] = {}


def discover_plugins(package_dir: Path = None):
    """Scan nlp_pipeline/ and data_ingestion/ for classes implementing interfaces."""
    # Auto-import modules, find classes implementing our ABCs
    # Register them in global dicts
    ...

def get_sentiment_method(name: str) -> SentimentMethod:
    """Instantiate a registered sentiment method by name."""

def get_data_source(name: str) -> DataSource:
    """Instantiate a registered data source by name."""

def list_available_methods() -> list[str]:
    """List all discovered sentiment method names."""

def list_available_sources() -> list[str]:
    """List all discovered data source names."""
```

### Task H.3 — Comprehensive README with Portfolio Mindset

**File**: `README.md` — When writing in Phase F, ensure it includes ALL of these sections:

```markdown
# 🧠 NLP-RL Trading Platform

[Badges]

## ✨ Features (with demo GIF/video embed)

## 🎯 The Problem We Solve (NOT "assignment requirements")

## 🏗 Architecture (Mermaid diagram)

## 🚀 Quick Start (Docker primary, 30-second setup)

## 📊 Results (ablation snapshot table)

## 🔬 Methodology Deep-Dive
### Why These 4 Sentiment Methods?
### Why DQN Not Other Algorithms?
### Why SHAP For Explainability?

## 🛠 Tech Stack (visual grid)

## 📁 Project Structure (tree)

## 🔌 Extensibility Guide
### Adding a New Sentiment Method (3 steps)
### Adding a New Data Source (3 steps)
### Adding a New RL Algorithm

## 🗺 Roadmap
### v1.0 (current) - what's done
### v1.1 (next) - planned improvements  
### v2.0 (vision) - multi-agent architecture

## 👥 Team & Contributions

## 📄 License

## 🙏 Acknowledgments
```

### Task H.4 — Demo Video Preparation Assets

**File**: `assets/demo_script.md` (NEW FILE)

Write a narrated demo script for recording a 3-minute video walkthrough:

```
DEMO SCRIPT (read while screen-recording):

[0:00-0:15] Introduction
"Hi, I'm [Name]. This is my NLP-driven reinforcement learning trading platform..."

[0:15-0:45] Architecture overview (scroll through README)
"The platform has 5 modules: data ingestion, NLP pipeline, storage, RL engine, and dashboard."

[0:45-1:30] Live demo - Market page
"Here we see real market data for Apple. Below it, four sentiment analysis methods running in parallel..."

[1:30-2:15] Live demo - Agent page + SHAP
"The DQN agent decided to BUY here. Let me click 'explain' to see why... SHAP shows sentiment was the biggest driver..."

[2:15-2:40] Live demo - AI Chat
"Now let me ask the AI assistant a question: 'Should I buy Tesla?' It retrieves relevant news and gives an informed answer..."

[2:40-3:00] Ablation results + wrap-up
"Finally, the ablation study proves NLP signals improve Sharpe by 30%. Full code is open-sourced below."
```

### Task H.5 — Commit Discipline

From this point forward, enforce this commit message format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
Scopes: `scheduler`, `llm`, `rag`, `dashboard`, `ci`, `docker`, etc.

Examples:
- `feat(llm): add Volcano Engine Doubao as 4th sentiment method with few-shot prompting`
- `feat(rag): implement ChromaDB vector store with semantic search over news articles`
- `fix(dashboard): resolve merge conflict when date column exists as both index and column`
- `test(rl): add env step/episode integration tests for FinancialTradingEnv`

---

## UPDATED EXECUTION CHECKLIST (Final)

Add Phase G and H items:

### Functionality Checklist (after ALL phases complete)
- [ ] `python main.py --skip-collect --skip-nlp` runs without errors
- [ ] `python -m data_ingestion.scheduler` starts scheduled collection
- [ ] All 4 sentiment methods execute (VADER, LR, FinBERT, LLM with few-shot prompt)
- [ ] Aggregator computes Fleiss' Kappa + weighted consensus based on agreement level
- [ ] Intent-based RAG: question type determines retrieval strategy
- [ ] DQN trains 200 episodes per ticker + SHAP explains decisions
- [ ] Ablation study with statistical significance test (t-test p-value)
- [ ] Streamlit dashboard launches with progressive-disclosure layout (not tabs)
- [ ] Capability banner shows graceful degradation when API keys missing
- [ ] AI Chat works (floating widget, not isolated tab)
- [ ] New sentiment methods can be added by implementing `SentimentMethod` ABC
- [ ] New data sources can be added by implementing `DataSource` ABC
- [ ] pytest passes, ruff lint clean, CI green on GitHub Actions
- [ ] Docker Compose builds and runs full platform
- [ ] README includes roadmap, extensibility guide, demo video embed

### Files That Should Exist After Completion (FINAL LIST)
```
interfaces/
  __init__.py                          ← NEW (ABC definitions)
utils/
  capability_manager.py                ← NEW (graceful degradation)
  plugin_manager.py                   ← NEW (auto plugin discovery)
  indicators.py                        ← EXISTS (keep)
data_ingestion/
  __init__.py                          ← EXISTS
  scheduler.py                         ← NEW (APScheduler)
  market_data.py                       ← MODIFIED (dual-source + incremental)
  news_fetcher.py                      ← MODIFIED (RSS fallback + incremental)
nlp_pipeline/
  __init__.py                          ← EXISTS
  preprocessor.py                      ← EXISTS (keep)
  sentiment_lexicon.py                 ← EXISTS (keep)
  sentiment_lr.py                      ← EXISTS (keep)
  sentiment_finbert.py                 ← EXISTS (keep)
  sentiment_llm.py                     ← NEW (few-shot prompt, Volcano Engine)
  aggregator.py                        ← MODIFIED (4-method + Kappa + weighted)
vector_store/
  __init__.py                          ← NEW
  chroma_store.py                      ← NEW (ChromaDB + sentence-transformers)
agents/
  __init__.py                          ← NEW
  research_agent.py                    ← NEW (RAG + intent detection)
data_storage/
  __init__.py                          ← EXISTS
  db_manager.py                        ← MODIFIED (LLM cache + incremental)
  schema.py                            ← MODIFIED (llm_analysis_cache table)
rl_engine/
  __init__.py                          ← EXISTS
  env.py                               ← EXISTS (keep)
  dqn.py                               ← EXISTS (keep)
  replay_buffer.py                     ← EXISTS (keep)
  train.py                             ← EXISTS (keep)
  evaluation.py                        ← EXISTS (keep)
  explainer.py                         ← NEW (SHAP for DQN decisions)
dashboard/
  __init__.py                          ← EXISTS
  app.py                              ← COMPLETE REWRITE (progressive disclosure layout)
  pages/                               ← MAY BE REMOVED (single-page design instead)
  components/
    charts.py                          ← NEW (shared chart components)
tests/
  __init__.py                          ← NEW
  conftest.py                          ← NEW
  test_market_data.py                  ← NEW
  test_nlp.py                          ← NEW
  test_rl_env.py                       ← NEW
  test_db.py                           ← NEW
  test_interfaces.py                   ← NEW (test ABC plugins)
.github/
  workflows/
    ci.yml                             ← NEW
assets/
  demo_script.md                       ← NEW (video narration script)
  architecture.png                    ← TO GENERATE (or placeholder)
  demo.gif                            ← TO RECORD (or placeholder)
  ablation_comparison.png              ← TO GENERATE after pipeline runs
Dockerfile                            ← NEW
docker-compose.yml                    ← NEW
pyproject.toml                        ← NEW
config.yaml                           ← NEW
config.py                             ← MODIFIED (YAML loading + capability checks)
requirements.txt                     ← MODIFIED (all new deps)
README.md                             ← COMPLETE REWRITE (portfolio quality)
.gitignore                            ├── MODIFIED
docs/
  architecture.md                      ← NEW
UPGRADE_PLAN.md                       ← THIS FILE (delete after completion)
.env                                  ← EXISTS (has all keys)
.example.env                         ← NEW (template without real keys)
```
