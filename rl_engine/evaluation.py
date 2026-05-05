"""Evaluation: backtesting, metrics, and ablation study runner."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL
from rl_engine.dqn import DQNAgent
from rl_engine.train import evaluate_agent, walk_forward_split, train_dqn
from utils.metrics import cumulative_returns, max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


def backtest(
    agent: DQNAgent,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Run a full backtest and return performance metrics."""
    log_df, total_reward = evaluate_agent(agent, df, initial_capital)

    equity = log_df["portfolio_value"]
    returns = log_df["returns"]

    # Buy-and-hold benchmark
    bh_shares = int(initial_capital / df["close"].iloc[0])
    bh_equity = df["close"] * bh_shares

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "final_portfolio_value": float(equity.iloc[-1]),
        "buy_and_hold_return": float(bh_equity.iloc[-1] / bh_equity.iloc[0] - 1),
        "buy_and_hold_sharpe": sharpe_ratio(bh_equity.pct_change().dropna()),
        "buy_and_hold_mdd": max_drawdown(bh_equity),
        "n_trades": int((log_df["action"] != 0).sum()),
        "log_df": log_df,
    }


def run_ablation_study(
    df_with_sentiment: pd.DataFrame,
    df_without_sentiment: pd.DataFrame,
    episodes: int = 200,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Compare RL performance with vs. without NLP sentiment signal.

    Returns dict with metrics for both conditions.
    """
    results = {}

    for label, df in [("with_nlp", df_with_sentiment), ("without_nlp", df_without_sentiment)]:
        logger.info("=== Ablation: %s ===", label)
        train_df, val_df, test_df = walk_forward_split(df)

        agent = DQNAgent()
        agent = train_dqn(train_df, val_df, episodes=episodes, initial_capital=initial_capital, agent=agent)
        metrics = backtest(agent, test_df, initial_capital=initial_capital)
        results[label] = metrics

        logger.info("%s: Sharpe=%.4f, MDD=%.4f, Return=%.4f",
                    label, metrics["sharpe_ratio"], metrics["max_drawdown"], metrics["total_return"])

    # Comparison summary
    delta_sharpe = results["with_nlp"]["sharpe_ratio"] - results["without_nlp"]["sharpe_ratio"]
    delta_return = results["with_nlp"]["total_return"] - results["without_nlp"]["total_return"]
    logger.info("Ablation delta: Sharpe=%+.4f, Return=%+.4f", delta_sharpe, delta_return)

    results["summary"] = {
        "sharpe_delta": delta_sharpe,
        "return_delta": delta_return,
        "nlp_improves_sharpe": delta_sharpe > 0,
        "nlp_improves_return": delta_return > 0,
    }

    return results
