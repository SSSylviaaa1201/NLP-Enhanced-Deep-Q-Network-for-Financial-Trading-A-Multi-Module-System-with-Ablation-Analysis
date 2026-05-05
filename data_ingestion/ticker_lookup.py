"""Ticker lookup service — support ANY stock ticker, not just pre-configured list."""

import logging
from typing import Optional

from config import START_DATE, TICKERS
from data_ingestion.market_data import fetch_ohlcv
from data_ingestion.rss_fetcher import fetch_news_rss
from data_storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


def is_valid_us_stock(ticker: str) -> bool:
    """Basic validation: US stock ticker is 1-5 uppercase letters (or with dot)."""
    t = ticker.strip().upper().replace(".", "")
    return 1 <= len(t) <= 5 and t.isalpha()


def lookup_ticker(ticker: str, db: DatabaseManager) -> dict:
    """Query any ticker's full info. Checks local cache first, then fetches live.

    Returns:
        {ticker, has_data, market_rows, news_count, sentiment_available,
         is_realtime, price_latest, error}
    """
    ticker = ticker.strip().upper()

    result = {
        "ticker": ticker, "has_data": False, "market_rows": 0,
        "news_count": 0, "sentiment_available": False,
        "is_realtime": False, "price_latest": None, "error": None,
    }

    if not is_valid_us_stock(ticker):
        result["error"] = f"Invalid ticker format: {ticker}"
        return result

    # Step 1: Check local cache
    market_df = db.get_market_data(ticker)
    news_df = db.get_news(ticker, limit=10)
    sent_df = db.get_sentiment(ticker)

    if not market_df.empty:
        result["has_data"] = True
        result["market_rows"] = len(market_df)
        result["price_latest"] = float(market_df["close"].iloc[-1])

    if not news_df.empty:
        result["news_count"] = len(db.get_news(ticker))

    if not sent_df.empty:
        result["sentiment_available"] = True

    # Step 2: Fetch live if missing
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
        except Exception as e:
            result["error"] = f"Market data fetch failed: {e}"

    if news_df.empty:
        try:
            rss_news = fetch_news_rss(ticker, max_per_source=15)
            if rss_news:
                db.insert_news(rss_news)
                result["news_count"] = len(rss_news)
                result["is_realtime"] = True
        except Exception as e:
            logger.warning("RSS fetch failed for %s: %s", ticker, e)

    return result


def search_tickers(query: str, limit: int = 10) -> list[dict]:
    """Fuzzy ticker search for autocomplete. Returns [{symbol, name}, ...]."""
    q = query.upper().strip()
    matches = []
    for t in TICKERS:
        if t.startswith(q) or q in t:
            matches.append({"symbol": t, "name": t})
            if len(matches) >= limit:
                break
    return matches
