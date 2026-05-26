"""Training loop with walk-forward validation and convergence analysis."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import EPISODES, TRAIN_SPLIT, VAL_SPLIT, INITIAL_CAPITAL
from rl_engine.dqn import DQNAgent
from rl_engine.env import FinancialTradingEnv

logger = logging.getLogger(__name__)


def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically: train / val / test. No look-ahead bias."""
    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def run_episode(
    env: FinancialTradingEnv,
    agent: DQNAgent,
    train: bool = True,
    episode: int = 0,
    ticker: str = "",
) -> tuple[float, list[dict]]:
    """Run a single episode. Returns (total_reward, log_records)."""
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    logs = []

    while not done:
        action = agent.select_action(state, evaluate=not train)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

        logs.append({
            "episode": episode, "ticker": ticker,
            "step": env.current_step, "action": action,
            "price": info["price"], "portfolio_value": info["portfolio_value"],
            "cash": info["cash"], "shares": info["shares"],
            "reward": reward, "sentiment_score": info.get("sentiment_score", 0.0),
            "date": info.get("date", None),
        })

        state = next_state
        total_reward += reward

    return total_reward, logs


def train_dqn(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    episodes: int = EPISODES,
    initial_capital: float = INITIAL_CAPITAL,
    agent: Optional[DQNAgent] = None,
    seed: Optional[int] = None,
    ticker: str = "",
    sentiment_bonus_enabled: bool = True,
    alignment_scale: Optional[float] = None,
    turnover_penalty: Optional[float] = None,
) -> DQNAgent:
    """Train DQN agent on training data with optional validation."""

    if agent is None:
        agent = DQNAgent(seed=seed)

    train_env = FinancialTradingEnv(train_df, initial_capital=initial_capital,
                                    sentiment_bonus_enabled=sentiment_bonus_enabled,
                                    alignment_scale=alignment_scale,
                                    turnover_penalty=turnover_penalty)
    val_env = (FinancialTradingEnv(val_df, initial_capital=initial_capital,
                                   sentiment_bonus_enabled=sentiment_bonus_enabled,
                                   alignment_scale=alignment_scale,
                                   turnover_penalty=turnover_penalty)
               if val_df is not None else None)

    best_val_return = -np.inf
    episode_rewards = []

    for ep in range(1, episodes + 1):
        train_reward, _ = run_episode(train_env, agent, train=True, episode=ep, ticker=ticker)
        episode_rewards.append(train_reward)
        agent.decay_epsilon()

        if (val_env is not None) and (ep % 10 == 0):
            val_reward, _ = run_episode(val_env, agent, train=False, episode=ep, ticker=ticker)
            if val_reward > best_val_return:
                best_val_return = val_reward
                agent.save_checkpoint()

        if ep % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            logger.info("Episode %d/%d | avg_reward=%.2f | epsilon=%.4f",
                        ep, episodes, avg_reward, agent.epsilon)

    rewards = np.array(episode_rewards)
    n = len(rewards)
    first = np.mean(rewards[:n // 2]) if n >= 4 else 0.0
    second = np.mean(rewards[n // 2:]) if n >= 2 else 0.0
    slope = np.polyfit(np.arange(n), rewards, 1)[0] if n >= 2 else 0.0
    tail = rewards[-max(10, n // 4):]
    tail_cv = float(np.std(tail) / max(abs(np.mean(tail)), 1e-8))

    logger.info("Train done. best_val=%.2f | reward 1st=%.3f 2nd=%.3f | slope=%.5f | tail_cv=%.2f",
                best_val_return, first, second, slope, tail_cv)

    if best_val_return > -np.inf:
        agent.load_checkpoint()
    agent.save()  # final model to disk for paper_trader/dashboard
    return agent


def backtest(
    agent: DQNAgent,
    df: pd.DataFrame,
    bh_metrics: Optional[dict] = None,
    initial_capital: float = INITIAL_CAPITAL,
    episode: int = 0,
    ticker: str = "",
    sentiment_bonus_enabled: bool = True,
) -> dict:
    """Run a full backtest and return performance metrics.

    If bh_metrics is provided, B&H is not recomputed — use pre-computed values.
    """
    from utils.metrics import sharpe_ratio, max_drawdown

    env = FinancialTradingEnv(df, initial_capital=initial_capital,
                              sentiment_bonus_enabled=sentiment_bonus_enabled)
    total_reward, logs = run_episode(env, agent, train=False, episode=episode, ticker=ticker)

    log_df = pd.DataFrame(logs)
    equity = log_df["portfolio_value"]
    returns = equity.pct_change().fillna(0)

    result = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "final_portfolio_value": float(equity.iloc[-1]),
        "n_trades": int((log_df["action"] != 0).sum()),
        "log_df": log_df,
    }

    if bh_metrics is not None:
        result.update({f"buy_and_hold_{k}": v for k, v in bh_metrics.items()})
    else:
        from config import TRANSACTION_COST_PCT
        price0 = float(df["close"].iloc[0])
        tc = TRANSACTION_COST_PCT
        bh_shares = int(initial_capital / (price0 * (1 + tc)))
        bh_equity = df["close"] * bh_shares + (initial_capital - price0 * bh_shares * (1 + tc))
        result["buy_and_hold_return"] = float(bh_equity.iloc[-1] / initial_capital - 1)
        result["buy_and_hold_sharpe"] = sharpe_ratio(bh_equity.pct_change().dropna())
        result["buy_and_hold_mdd"] = max_drawdown(bh_equity)

    return result


def compute_bh_metrics(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    """Compute Buy-and-Hold benchmark metrics once per ticker/test-set."""
    from utils.metrics import sharpe_ratio, max_drawdown
    from config import TRANSACTION_COST_PCT

    price0 = float(df["close"].iloc[0])
    tc = TRANSACTION_COST_PCT
    bh_shares = int(initial_capital / (price0 * (1 + tc)))
    bh_equity = df["close"] * bh_shares + (initial_capital - price0 * bh_shares * (1 + tc))

    return {
        "return": float(bh_equity.iloc[-1] / initial_capital - 1),
        "sharpe": sharpe_ratio(bh_equity.pct_change().dropna()),
        "mdd": max_drawdown(bh_equity),
    }
