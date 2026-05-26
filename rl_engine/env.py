"""Custom Gym environment for financial trading with NLP sentiment signal.

State vector (all normalized to roughly [-1, 1] or [0, 1]):
  [price_ratio, MA50_ratio, MA200_ratio, RSI_norm, MACD_ratio, position_pct,
   cash_pct, sentiment, sentiment_ma5, sentiment_trend, sentiment_vol]

Actions: 0=Hold, 1=Buy (25% of capital), 2=Sell (25% of position)
"""

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import (
    SENTIMENT_ALIGNMENT_SCALE, MAX_DRAWDOWN_LIMIT,
    REWARD_TURNOVER_PENALTY,
    HALF_SPREAD_BPS, SLIPPAGE_BPS_PER_PCT_VOL, MAX_VOLUME_FRACTION,
)

STATE_DIM = 11  # price features(5) + portfolio(2) + sentiment(4)
N_ACTIONS = 3


class FinancialTradingEnv(gym.Env):
    """A trading environment that steps through market data day by day."""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        transaction_cost_pct: float = 0.001,
        trade_fraction: float = 0.25,
        sentiment_bonus_enabled: bool = True,
        alignment_scale: Optional[float] = None,
        turnover_penalty: Optional[float] = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_fraction = trade_fraction
        self.sentiment_bonus_enabled = sentiment_bonus_enabled
        self.alignment_scale = alignment_scale if alignment_scale is not None else SENTIMENT_ALIGNMENT_SCALE
        self.turnover_penalty = turnover_penalty if turnover_penalty is not None else REWARD_TURNOVER_PENALTY

        for col in ["close", "MA50", "MA200", "RSI", "MACD", "sentiment_score", "volume"]:
            if col not in self.df.columns:
                self.df[col] = 0.0
        for col in ["sentiment_ma5", "sentiment_trend", "sentiment_vol"]:
            if col not in self.df.columns:
                self.df[col] = 0.0

        self.df = self.df.ffill().fillna(0.0)

        if "price_baseline" in self.df.columns and self.df["price_baseline"].iloc[0] > 0:
            self._price_0 = float(self.df["price_baseline"].iloc[0])
        else:
            self._price_0 = float(self.df["close"].iloc[0]) if len(self.df) > 0 else 1.0

        self.n_steps = len(self.df)
        self.current_step = 0
        self._row = None  # cached row for current step

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )

        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = initial_capital
        self.prev_portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.trade_count = 0

    def _fetch_row(self):
        """Cache the current-step row to avoid repeated iloc calls."""
        self._row = self.df.iloc[self.current_step]

    def _norm_state(self) -> np.ndarray:
        row = self._row
        price = float(row["close"])
        if price <= 0:
            price = self._price_0

        price_ratio = price / self._price_0
        ma50_ratio = float(row["MA50"]) / price if price > 0 else 1.0
        ma200_ratio = float(row["MA200"]) / price if price > 0 else 1.0
        rsi_norm = float(row["RSI"]) / 100.0
        macd_ratio = float(row["MACD"]) / price if price > 0 else 0.0

        position_value = self.shares * price
        portfolio = self.cash + position_value
        position_pct = position_value / portfolio if portfolio > 0 else 0.0
        cash_pct = self.cash / portfolio if portfolio > 0 else 1.0

        sentiment = float(row.get("sentiment_score", 0.0))

        state = np.array([
            price_ratio, ma50_ratio, ma200_ratio, rsi_norm, macd_ratio,
            position_pct, cash_pct,
            sentiment,
            float(row.get("sentiment_ma5", sentiment)),
            float(row.get("sentiment_trend", 0.0)),
            float(row.get("sentiment_vol", 0.0)),
        ], dtype=np.float32)

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._row = None
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.trade_count = 0
        self._fetch_row()
        return self._norm_state(), {}

    def step(self, action: int):
        row = self._row
        price = float(row["close"])
        if price <= 0:
            price = self._price_0

        self.prev_portfolio_value = self.portfolio_value
        trade_cost = 0.0
        effective_price = price

        volume = float(row.get("volume", 0))
        daily_volume = max(volume, 100)

        if action == 1:  # Buy
            cash_to_use = self.cash * self.trade_fraction
            if cash_to_use >= price:
                raw_shares = int(cash_to_use / price)
                max_shares = int(daily_volume * MAX_VOLUME_FRACTION)
                gross_shares = min(raw_shares, max_shares)
                trade_pct_of_vol = gross_shares / daily_volume if daily_volume > 0 else 0.0
                slippage_bps = SLIPPAGE_BPS_PER_PCT_VOL * trade_pct_of_vol * 100
                effective_price = price * (1 + HALF_SPREAD_BPS / 10000 + slippage_bps / 10000)
                cost = effective_price * gross_shares * (1 + self.transaction_cost_pct)
                if cost <= self.cash:
                    self.shares += gross_shares
                    self.cash -= cost
                    trade_cost = effective_price * gross_shares * self.transaction_cost_pct + (effective_price - price) * gross_shares

        elif action == 2:  # Sell
            if self.shares > 0:
                raw_shares = max(1, int(self.shares * self.trade_fraction))
                max_shares = int(daily_volume * MAX_VOLUME_FRACTION)
                shares_to_sell = min(raw_shares, max_shares)
                trade_pct_of_vol = shares_to_sell / daily_volume if daily_volume > 0 else 0.0
                slippage_bps = SLIPPAGE_BPS_PER_PCT_VOL * trade_pct_of_vol * 100
                effective_price = price * (1 - HALF_SPREAD_BPS / 10000 - slippage_bps / 10000)
                revenue = effective_price * shares_to_sell * (1 - self.transaction_cost_pct)
                self.shares -= shares_to_sell
                self.cash += revenue
                trade_cost = effective_price * shares_to_sell * self.transaction_cost_pct + (price - effective_price) * shares_to_sell

        self.portfolio_value = self.cash + self.shares * price

        if self.prev_portfolio_value > 0:
            pct_return = (self.portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        else:
            pct_return = 0.0

        reward = pct_return

        position_value = self.shares * price
        total = self.cash + position_value

        sentiment = float(row.get("sentiment_score", 0.0))
        if self.sentiment_bonus_enabled and total > 0:
            reward += sentiment * (position_value / total) * self.alignment_scale

        if trade_cost > 0 and total > 0:
            self.trade_count += 1
            reward -= self.turnover_penalty * (trade_cost / total)

        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0.0

        self.current_step += 1
        time_terminated = self.current_step >= self.n_steps - 1
        drawdown_terminated = drawdown > MAX_DRAWDOWN_LIMIT
        terminated = time_terminated or drawdown_terminated
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "price": price,
            "trade_cost": trade_cost,
            "pct_return": pct_return,
            "drawdown": drawdown,
            "trade_count": self.trade_count,
            "date": row.get("date", None),
            "sentiment_score": sentiment,
        }

        if not terminated:
            self._fetch_row()
        obs = self._norm_state() if not terminated else np.zeros(STATE_DIM, dtype=np.float32)
        return obs, reward, terminated, truncated, info
