"""Fetch financial news: NewsAPI → RSS (rss_fetcher) → sample fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from config import TICKERS, NEWS_LOOKBACK_DAYS, NEWSAPI_KEY

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def fetch_news_newsapi(
    ticker: str,
    api_key: Optional[str] = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
) -> list[dict]:
    """Fetch news articles from NewsAPI for a given ticker."""
    key = api_key or NEWSAPI_KEY
    if not key:
        logger.info("No NewsAPI key, will fall back to RSS")
        return []

    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    params = {
        "q": f"{ticker} stock",
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey": key,
    }
    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    except requests.RequestException as e:
        logger.warning("NewsAPI request failed for %s: %s", ticker, e)
        return []

    if resp.status_code != 200:
        logger.warning("NewsAPI returned %d: %s", resp.status_code, resp.text[:200])
        return []

    articles = resp.json().get("articles", [])
    records = []
    for a in articles:
        records.append({
            "ticker": ticker,
            "source": a.get("source", {}).get("name", ""),
            "title": a.get("title", ""),
            "content": a.get("content") or a.get("description", ""),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt", ""),
        })
    return records


def _fetch_rss_fallback(ticker: str) -> list[dict]:
    """Fetch news via rss_fetcher module (4 sources) as fallback."""
    from data_ingestion.rss_fetcher import fetch_news_rss
    return fetch_news_rss(ticker, max_per_source=30)


def fetch_news_for_all_tickers(
    tickers: Optional[list[str]] = None,
) -> tuple[list[dict], list[str]]:
    """Fetch news for all tickers. Fallback chain: NewsAPI → RSS → sample."""
    tickers = tickers or TICKERS
    all_records = []
    rss_failed = []

    for t in tickers:
        records = fetch_news_newsapi(t)
        if records:
            all_records.extend(records)
            logger.info("NewsAPI: %d articles for %s", len(records), t)
            continue

        # RSS fallback (4 sources: Yahoo Finance + Google News + Seeking Alpha + MarketWatch)
        logger.info("NewsAPI exhausted, trying 4-source RSS feeds for %s...", t)
        records = _fetch_rss_fallback(t)
        if records:
            all_records.extend(records)
            logger.info("RSS: %d articles for %s", len(records), t)
        else:
            rss_failed.append(t)
            logger.warning("All sources failed for %s", t)

    return all_records, rss_failed


def fetch_sample_news(ticker: str = "AAPL") -> pd.DataFrame:
    """
    Generate synthetic sample news when NewsAPI is unavailable.
    Used for development/testing.
    """
    dates = pd.date_range(end=datetime.now(), periods=NEWS_LOOKBACK_DAYS, freq="D")
    headlines = [
        f"{ticker} reports strong quarterly earnings",
        f"{ticker} announces new product line",
        f"{ticker} faces regulatory scrutiny",
        f"Analysts upgrade {ticker} rating to Buy",
        f"{ticker} stock dips amid market uncertainty",
        f"{ticker} expands into new markets",
        f"Supply chain issues impact {ticker} production",
        f"{ticker} beats revenue expectations",
        f"Concerns grow over {ticker} valuation",
        f"{ticker} announces partnership deal",
    ]
    import random
    random.seed(42)
    records = []
    for d in dates:
        records.append({
            "ticker": ticker,
            "source": "sample",
            "title": random.choice(headlines),
            "content": random.choice(headlines),
            "url": f"sample://{ticker}/{d.date()}",
            "published_at": d.isoformat(),
        })
    return pd.DataFrame(records)
