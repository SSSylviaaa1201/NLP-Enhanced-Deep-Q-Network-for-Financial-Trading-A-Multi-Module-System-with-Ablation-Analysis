"""Evaluation: backtesting, metrics, and ablation study runner."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL
from rl_engine.dqn import DQNAgent
from rl_engine.train import backtest, compute_bh_metrics, walk_forward_split, train_dqn

logger = logging.getLogger(__name__)


def run_ablation_study(
    df_with_sentiment: pd.DataFrame,
    df_without_sentiment: pd.DataFrame,
    episodes: int = 200,
    initial_capital: float = INITIAL_CAPITAL,
    seeds: Optional[list] = None,
    ticker: str = "",
) -> dict:
    """Compare RL performance with vs. without NLP sentiment signal.

    Computes B&H benchmark once (identical for both conditions: only sentiment cols differ).
    Trains with each seed and returns seed-averaged metrics + per-seed details.
    """
    seed_list = seeds if seeds else [None]
    metric_keys = ["sharpe_ratio", "total_return", "max_drawdown",
                   "buy_and_hold_return", "buy_and_hold_sharpe", "buy_and_hold_mdd",
                   "final_portfolio_value", "n_trades"]

    # B&H computed once: test_df differs only in sentiment cols (irrelevant for price)
    _, _, test_df = walk_forward_split(df_with_sentiment)
    bh_metrics = compute_bh_metrics(test_df, initial_capital)

    all_with = []
    all_without = []

    for seed_idx, seed in enumerate(seed_list):
        for label, df in [("with_nlp", df_with_sentiment), ("without_nlp", df_without_sentiment)]:
            logger.info("=== Ablation: %s [seed=%s] ===", label, seed)
            train_df, val_df, test_df = walk_forward_split(df)

            agent = DQNAgent(seed=seed)
            curve_tag = f"{ticker}_seed{seed}_{label}"
            agent = train_dqn(train_df, val_df, episodes=episodes,
                            initial_capital=initial_capital, agent=agent, seed=seed,
                            ticker=curve_tag, sentiment_bonus_enabled=False)
            metrics = backtest(agent, test_df, bh_metrics=bh_metrics,
                             initial_capital=initial_capital,
                             episode=seed_idx + 1, ticker=ticker,
                             sentiment_bonus_enabled=False)
            metrics["seed"] = seed

            if label == "with_nlp":
                all_with.append(metrics)
            else:
                all_without.append(metrics)

            logger.info("%s [seed=%s]: Sharpe=%.4f, MDD=%.4f, Return=%.4f",
                        label, seed, metrics["sharpe_ratio"],
                        metrics["max_drawdown"], metrics["total_return"])

    def _aggregate(seed_metrics: list[dict]) -> dict:
        avg, std = {}, {}
        for k in metric_keys:
            values = [m[k] for m in seed_metrics if k in m]
            avg[k] = float(np.mean(values)) if values else 0.0
            std[k] = float(np.std(values)) if len(values) > 1 else 0.0
        avg["seed_count"] = len(seed_metrics)
        avg["seed_std"] = std
        avg["seed_details"] = seed_metrics
        return avg

    results = {
        "with_nlp": _aggregate(all_with),
        "without_nlp": _aggregate(all_without),
    }

    delta_sharpe = results["with_nlp"]["sharpe_ratio"] - results["without_nlp"]["sharpe_ratio"]
    delta_return = results["with_nlp"]["total_return"] - results["without_nlp"]["total_return"]
    delta_mdd = results["with_nlp"]["max_drawdown"] - results["without_nlp"]["max_drawdown"]
    sharpe_std = np.sqrt(
        results["with_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2 +
        results["without_nlp"]["seed_std"].get("sharpe_ratio", 0) ** 2
    )

    results["summary"] = {
        "sharpe_delta": delta_sharpe,
        "return_delta": delta_return,
        "mdd_delta": delta_mdd,
        "sharpe_delta_std": sharpe_std,
        "nlp_improves_sharpe": delta_sharpe > 0,
        "nlp_improves_return": delta_return > 0,
        "nlp_improves_mdd": delta_mdd > 0,
        "n_seeds": len(seed_list),
    }

    logger.info("Ablation delta: Sharpe=%+.4f±%.4f, Return=%+.4f, MDD=%+.4f",
                delta_sharpe, sharpe_std, delta_return, delta_mdd)
    return results
