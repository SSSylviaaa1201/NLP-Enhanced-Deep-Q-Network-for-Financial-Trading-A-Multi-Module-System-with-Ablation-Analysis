"""RAG-powered research agent: semantic retrieval + LLM to answer stock questions.

Intent detection routes questions to different retrieval strategies:
- "causal" → recent news (why did X happen?)
- "decision" → sentiment + agent decisions + SHAP
- "compare" → all tickers summary
- "sentiment" → time-series trends
"""

import logging
import re

from openai import OpenAI

from config import VOLCANO_API_KEY, VOLCANO_BASE_URL, VOLCANO_MODEL_ID, LLM_ENABLED
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


def _detect_intent(question: str) -> str:
    """Classify user question into intent type for retrieval strategy."""
    q = question.lower()
    if any(w in q for w in ["why", "reason", "caused", "happened", "happen", "because", "catalyst"]):
        return "causal"
    if any(w in q for w in ["buy", "sell", "should i", "trade", "recommend", "position", "invest", "decision"]):
        return "decision"
    if any(w in q for w in ["compare", "best", "which stock", "pick", "versus", "vs"]):
        return "compare"
    if any(w in q for w in ["sentiment", "feeling", "mood", "trend", "outlook", "forecast", "prediction"]):
        return "sentiment"
    return "general"


def _rewrite_query(question: str) -> str:
    """Expand ticker abbreviations and normalize Chinese-English queries."""
    ticker_map = {
        "tsla": "TSLA Tesla Motors electric vehicle stock",
        "aapl": "AAPL Apple Inc technology stock",
        "msft": "MSFT Microsoft Corporation technology stock",
        "googl": "GOOGL Google Alphabet technology stock",
        "amzn": "AMZN Amazon e-commerce stock",
    }
    q = question
    for abbr, expansion in ticker_map.items():
        if abbr in q.lower():
            q = q.replace(abbr, expansion).replace(abbr.upper(), expansion)
    return q


def _build_context(question: str, ticker: str, db: DatabaseManager, intent: str) -> str:
    """Gather relevant context based on detected intent."""
    parts = []
    query = _rewrite_query(question)

    # Configure retrieval by intent
    intent_config = {
        "causal": {"top_k": 10, "include_agent": False},
        "decision": {"top_k": 7, "include_agent": True},
        "compare": {"top_k": 3, "include_agent": False},
        "sentiment": {"top_k": 5, "include_agent": False},
        "general": {"top_k": 5, "include_agent": True},
    }
    cfg = intent_config.get(intent, intent_config["general"])

    # 1. Semantic search
    news_results = search(query, top_k=cfg["top_k"], ticker_filter=ticker)
    if news_results:
        parts.append("=== RELEVANT NEWS ARTICLES ===")
        for i, article in enumerate(news_results, 1):
            meta = article["metadata"]
            parts.append(
                f"[{i}] {meta.get('title', '?')} ({meta.get('ticker', '?')}, {meta.get('date', '?')}) "
                f"[Relevance: {article['relevance_score']:.2f}]\n"
                f"    {article['text'][:300]}"
            )

    # 2. Sentiment data
    sent_df = db.get_sentiment(ticker)
    if not sent_df.empty:
        latest = sent_df.sort_values("date").tail(7)
        parts.append(f"\n=== RECENT SENTIMENT ({ticker}, last 7 days) ===")
        for _, row in latest.iterrows():
            d = row['date'] if hasattr(row['date'], 'strftime') else row['date']
            parts.append(f"  {d} | method={row['method']} | score={row['sentiment_score']:.3f}")

    # 3. Agent decisions (only for decision intent)
    if cfg["include_agent"]:
        logs_df = db.get_trading_logs()
        if not logs_df.empty:
            if "ticker" in logs_df.columns:
                logs_df = logs_df[logs_df["ticker"] == ticker]
            if not logs_df.empty:
                latest = logs_df.iloc[-1]
                parts.append(f"\n=== LATEST AGENT STATE ({ticker}) ===")
                parts.append(
                    f"  portfolio_value={latest.get('portfolio_value', 'N/A')}, "
                    f"position={latest.get('position', 'N/A')}, "
                    f"cash={latest.get('cash', 'N/A')}, "
                    f"last_action={latest.get('action', 'N/A')}"
                )

    return "\n\n".join(parts) if parts else "No context data available yet."


def ask(question: str, ticker: str = "AAPL") -> str:
    """Main entry point: ask a question about a ticker, get RAG-powered answer."""
    if not LLM_ENABLED:
        return "LLM service not configured. Set VOLCANO_API_KEY and VOLCANO_MODEL_ID in .env"

    db = DatabaseManager()
    intent = _detect_intent(question)
    context = _build_context(question, ticker, db, intent)

    client = OpenAI(api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL)

    try:
        resp = client.chat.completions.create(
            model=VOLCANO_MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nINTENT: {intent}\n\nQUESTION: {question}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("Q: %s | Intent: %s | A: %s", question[:80], intent, answer[:80])
        return answer
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"Sorry, I couldn't process your question: {e}"
