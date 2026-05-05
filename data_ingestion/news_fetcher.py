"""Fetch financial news: NewsAPI → RSS → sample fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd
import requests

from config import TICKERS, NEWS_LOOKBACK_DAYS, NEWSAPI_KEY

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

RSS_FEEDS = {
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "google_news": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
}


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


def fetch_news_rss(ticker: str, max_articles: int = 30) -> list[dict]:
    """Fetch news via RSS feeds as free fallback when NewsAPI is unavailable."""
    records = []
    seen_urls = set()
    for source_name, url_template in RSS_FEEDS.items():
        try:
            url = url_template.format(ticker=ticker)
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                link = entry.get("link", "")
                if link in seen_urls:
                    continue
                seen_urls.add(link)
                published = entry.get("published") or entry.get("updated", "")
                records.append({
                    "ticker": ticker,
                    "source": f"rss_{source_name}",
                    "title": entry.get("title", ""),
                    "content": entry.get("description") or entry.get("summary", ""),
                    "url": link,
                    "published_at": published,
                })
        except Exception:
            logger.debug("RSS feed %s failed for %s", source_name, ticker)
            continue
    return records


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

        # RSS fallback
        logger.info("NewsAPI failed for %s, trying RSS...", t)
        records = fetch_news_rss(t)
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
