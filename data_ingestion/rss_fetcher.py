"""RSS feed aggregator — free unlimited financial news from 4 sources."""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd

from config import TICKERS, NEWS_LOOKBACK_DAYS

logger = logging.getLogger(__name__)

RSS_SOURCES = {
    "yahoo_finance": {
        "url_template": "https://finance.yahoo.com/news/rss/{ticker}",
        "parser": "_parse_yahoo_rss",
    },
    "google_news": {
        "url_template": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        "parser": "_parse_generic_rss",
    },
    "seeking_alpha": {
        "url_template": "https://seekingalpha.com/symbol/{ticker}/news/feed",
        "parser": "_parse_generic_rss",
    },
    "marketwatch": {
        "url_template": "https://www.marketwatch.com/investing/stock/{ticker}/news/rss",
        "parser": "_parse_marketwatch_rss",
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (QuantumTrade/1.0; Research Bot) (+https://github.com/SSSylviaaa1201/Fintech)"
}


def _parse_date(entry: dict) -> str:
    """Unified date parser, returns ISO format string."""
    for field in ["published_parsed", "updated_parsed"]:
        parsed = entry.get(field)
        if parsed:
            try:
                dt = datetime(*parsed[:6])
                return dt.isoformat()
            except (TypeError, ValueError):
                continue
    return entry.get("published") or entry.get("updated") or datetime.now().isoformat()


def _parse_yahoo_rss(entry: dict, ticker: str) -> Optional[dict]:
    """Parse Yahoo Finance RSS entry."""
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
    """Generic RSS parser (Google News / Seeking Alpha)."""
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
    """MarketWatch parser with HTML cleanup."""
    title = entry.get("title", "").strip()
    if not title:
        return None
    raw = entry.get("summary", "") or entry.get("description", "") or ""
    clean = re.sub(r"<[^>]+>", "", raw).strip()
    return {
        "ticker": ticker,
        "source": "MarketWatch",
        "title": title,
        "content": clean[:1000],
        "url": entry.get("link", ""),
        "published_at": _parse_date(entry),
    }


def fetch_news_rss(
    ticker: str,
    max_per_source: int = 20,
    sources: Optional[list[str]] = None,
) -> list[dict]:
    """Fetch news from all RSS sources for a ticker."""
    records = []
    active_sources = sources or list(RSS_SOURCES.keys())

    for source_name in active_sources:
        if source_name not in RSS_SOURCES:
            continue

        cfg = RSS_SOURCES[source_name]
        url = cfg["url_template"].format(ticker=ticker.lower())

        try:
            logger.info("Fetching %s RSS for %s...", source_name, ticker)
            feed = feedparser.parse(url)

            if not feed.entries:
                logger.debug("  No entries from %s for %s", source_name, ticker)
                continue

            parser_name = cfg["parser"]
            parser_func = globals().get(parser_name, _parse_generic_rss)

            count = 0
            for entry in feed.entries[:max_per_source]:
                if parser_name == "_parse_generic_rss":
                    result = parser_func(entry, ticker, source_name)
                else:
                    result = parser_func(entry, ticker)
                if result:
                    records.append(result)
                    count += 1

            logger.info("  Got %d articles from %s for %s", count, source_name, ticker)

        except Exception as e:
            logger.warning("Failed %s RSS for %s: %s", source_name, ticker, e)

    # Dedup by title[:80] + source
    seen = set()
    unique = []
    for r in records:
        key = (r["title"][:80], r["source"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    logger.info("RSS: %d unique articles for %s (%d raw)", len(unique), ticker, len(records))
    return unique


def fetch_news_rss_all_tickers(
    tickers: Optional[list[str]] = None,
    max_per_source: int = 15,
) -> list[dict]:
    """Batch fetch RSS news for multiple tickers."""
    tickers = tickers or TICKERS
    all_records = []
    for t in tickers:
        records = fetch_news_rss(t, max_per_source=max_per_source)
        all_records.extend(records)
    return all_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n=== AAPL RSS Test ===")
    results = fetch_news_rss("AAPL", max_per_source=5)
    for r in results[:5]:
        print(f"  [{r['source']}] {r['title'][:70]} | {r['published_at'][:10]}")
    print(f"Total: {len(results)} articles\n")
