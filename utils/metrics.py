"""Financial performance metrics."""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g., -0.25 for 25% drawdown)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def cumulative_returns(equity_curve: pd.Series) -> pd.Series:
    """Cumulative return series from equity curve."""
    return equity_curve / equity_curve.iloc[0] - 1.0
