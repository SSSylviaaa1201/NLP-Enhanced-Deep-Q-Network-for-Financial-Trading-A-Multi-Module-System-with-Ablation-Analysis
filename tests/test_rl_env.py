"""Test RL trading environment."""
import numpy as np
import pandas as pd
import pytest
from rl_engine.env import FinancialTradingEnv, STATE_DIM


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
    assert len(state) == STATE_DIM


def test_env_step():
    df = _make_test_df()
    env = FinancialTradingEnv(df)
    env.reset()
    obs, reward, term, trunc, info = env.step(1)
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


def test_state_normalized():
    df = _make_test_df()
    env = FinancialTradingEnv(df)
    state, _ = env.reset()
    # All values should be reasonable scale (not raw $100K)
    assert -2 <= state[0] <= 5  # price_ratio
    assert 0 <= state[5] <= 1   # position_pct
    assert 0 <= state[6] <= 1   # cash_pct
