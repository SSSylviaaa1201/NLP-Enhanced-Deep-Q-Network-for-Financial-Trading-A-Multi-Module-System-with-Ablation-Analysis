"""Fetch OHLCV market data: Yahoo Direct → yfinance → Alpha Vantage → synthetic fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds
TICKER_DELAY = 3  # seconds between tickers
YAHOO_API_DELAY = 0.5  # seconds between direct API calls

TICKER_PROFILES = {
    "AAPL": {"base_price": 180, "volatility": 0.015, "drift": 0.0004},
    "MSFT": {"base_price": 380, "volatility": 0.014, "drift": 0.0005},
    "GOOGL": {"base_price": 150, "volatility": 0.016, "drift": 0.0003},
    "AMZN": {"base_price": 180, "volatility": 0.018, "drift": 0.0004},
    "TSLA": {"base_price": 240, "volatility": 0.025, "drift": 0.0002},
}


def _generate_synthetic_ohlcv(
    ticker: str,
    start: str = START_DATE,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for a ticker."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq="B")

    profile = TICKER_PROFILES.get(ticker, {"base_price": 100, "volatility": 0.015, "drift": 0.0003})
    base = profile["base_price"]
    vol = profile["volatility"]
    drift = profile["drift"]

    np.random.seed(hash(ticker) % 2**31)
    returns = np.random.normal(drift, vol, len(dates))
    prices = base * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"date": dates, "close": prices})
    df["open"] = df["close"].shift(1).fillna(base) * np.random.uniform(0.995, 1.005, len(df))
    df["high"] = df[["open", "close"]].max(axis=1) * np.random.uniform(1.001, 1.02, len(df))
    df["low"] = df[["open", "close"]].min(axis=1) * np.random.uniform(0.98, 0.999, len(df))
    df["volume"] = np.random.randint(5_000_000, 80_000_000, len(df))
    df["adjusted_close"] = df["close"]
    return df


def fetch_ohlcv_alpha_vantage(ticker: str, start: str = "2020-01-01", end: Optional[str] = None, outputsize: str = "compact"):
    """Fetch OHLCV from Alpha Vantage. Returns DataFrame or None.

    Args:
        outputsize: "compact" (~100 rows, last ~5 months, free tier) or "full" (full 20-year history, premium only).
                    Free tier: 25 requests/day.
    """
    try:
        from alpha_vantage.timeseries import TimeSeries
    except ImportError:
        logger.warning("alpha-vantage not installed")
        return None

    key = os.getenv("ALPHA_VANTAGE_KEY", "")
    if not key:
        logger.info("No ALPHA_VANTAGE_KEY, skipping Alpha Vantage")
        return None

    end = end or datetime.now().strftime("%Y-%m-%d")

    for av_attempt in range(1, 4):
        try:
            ts = TimeSeries(key=key, output_format="pandas")
            df, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)

            df = df.reset_index()
            # Normalize column names: "1. open" → "1_open" → "open"
            df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

            col_map = {
                "1__open": "open", "2__high": "high",
                "3__low": "low", "4__close": "close", "5__volume": "volume",
                "1_open": "open", "2_high": "high",
                "3_low": "low", "4_close": "close", "5_volume": "volume",
                "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            if "date" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "date"})
            if "date" not in df.columns:
                date_col = df.columns[0]
                df = df.rename(columns={date_col: "date"})

            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            df["adjusted_close"] = df.get("close", df.iloc[:, -1])

            keep_cols = ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
            available = [c for c in keep_cols if c in df.columns]
            if available:
                logger.info("Alpha Vantage: %d rows for %s", len(df), ticker)
                return df[available]
        except Exception as e:
            if av_attempt < 3:
                logger.warning("Alpha Vantage attempt %d/3 for %s: %s", av_attempt, ticker, e)
                time.sleep(2)
            else:
                logger.warning("Alpha Vantage failed for %s after 3 attempts", ticker)

    return None


def _fetch_yahoo_direct(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance v8 API directly (bypasses yfinance rate limiter).

    yfinance library may get 429 even with curl_cffi; the raw API often works fine.
    """
    t1 = int(pd.Timestamp(start).timestamp())
    t2 = int(pd.Timestamp(end).timestamp())

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": t1, "period2": t2,
        "interval": "1d", "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                logger.warning("Yahoo direct API rate limited for %s after %d attempts", ticker, attempt)
                return None
            if r.status_code != 200:
                logger.warning("Yahoo direct API returned %d for %s", r.status_code, ticker)
                return None

            data = r.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            adjclose_list = None
            if "adjclose" in result["indicators"]:
                adjclose_list = result["indicators"]["adjclose"][0].get("adjclose")

            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            })
            if adjclose_list:
                df["adjusted_close"] = adjclose_list
            else:
                df["adjusted_close"] = df["close"]

            df = df.dropna(subset=["close"])
            logger.info("Yahoo direct: %d rows for %s", len(df), ticker)
            return df
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Yahoo direct attempt %d/%d for %s: %s", attempt, MAX_RETRIES, ticker, e)
                time.sleep(RETRY_BASE_DELAY * attempt)
            else:
                logger.warning("Yahoo direct failed for %s after %d attempts: %s", ticker, MAX_RETRIES, e)
    return None


def _try_yfinance(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Try fetching from yfinance with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("yfinance %s (attempt %d/%d)", ticker, attempt, MAX_RETRIES)
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if df.empty:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
                    continue
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if "adj close" in df.columns:
                df = df.rename(columns={"adj close": "adjusted_close"})
            return df
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning("Retrying yfinance (%s): %s", ticker, e)
                time.sleep(RETRY_BASE_DELAY * attempt)
            else:
                logger.warning("yfinance failed for %s: %s", ticker, e)
    return None


def fetch_ohlcv(
    ticker: str,
    start: str = START_DATE,
    end: Optional[str] = None,
    prefer: str = "yahoo_direct",
) -> pd.DataFrame:
    """Fetch OHLCV data with fallback chain.

    Args:
        prefer: "yahoo_direct" (recommended), "yfinance", or "alpha_vantage"
    """
    end = end or datetime.now().strftime("%Y-%m-%d")

    source_map = {
        "yahoo_direct": ("Yahoo Direct", lambda: _fetch_yahoo_direct(ticker, start, end)),
        "yfinance": ("yfinance", lambda: _try_yfinance(ticker, start, end)),
        "alpha_vantage": ("Alpha Vantage", lambda: fetch_ohlcv_alpha_vantage(ticker, start, end)),
    }

    # Build ordered source list: preferred first, then the rest
    ordered = [prefer] + [s for s in source_map if s != prefer]

    for source_key in ordered:
        if source_key not in source_map:
            continue
        name, fetcher = source_map[source_key]
        try:
            time.sleep(YAHOO_API_DELAY)  # be polite to APIs
            df = fetcher()
            if df is not None and not df.empty:
                logger.info("Got %d rows from %s for %s", len(df), name, ticker)
                return df
        except Exception as e:
            logger.warning("%s failed for %s: %s", name, ticker, e)

    logger.warning("All sources failed for %s, using synthetic data.", ticker)
    return _generate_synthetic_ohlcv(ticker, start, end)


def fetch_incremental(ticker: str, since_date: str) -> Optional[pd.DataFrame]:
    """Fetch only data newer than since_date. Returns None if DB has latest."""
    end = datetime.now().strftime("%Y-%m-%d")
    if since_date >= end:
        return None  # already up to date
    logger.info("Incremental fetch for %s since %s", ticker, since_date)
    return fetch_ohlcv(ticker, start=since_date, end=end)


def fetch_all_tickers(
    tickers: Optional[list[str]] = None,
    start: str = START_DATE,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all tickers with delays between requests."""
    tickers = tickers or TICKERS
    result = {}
    for i, t in enumerate(tickers):
        if i > 0:
            time.sleep(TICKER_DELAY)
        df = fetch_ohlcv(t, start=start)
        if not df.empty:
            result[t] = df
    return result
