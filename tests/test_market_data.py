"""Test market data fetching functions."""
import pandas as pd
import pytest
from data_ingestion.market_data import _generate_synthetic_ohlcv, fetch_ohlcv


def test_synthetic_data_shape():
    df = _generate_synthetic_ohlcv("AAPL", start="2024-01-01", end="2024-01-31")
    assert not df.empty
    assert "close" in df.columns
    assert "volume" in df.columns
    assert len(df) > 15


def test_synthetic_data_positive_prices():
    df = _generate_synthetic_ohlcv("TSLA")
    assert (df["close"] > 0).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()
