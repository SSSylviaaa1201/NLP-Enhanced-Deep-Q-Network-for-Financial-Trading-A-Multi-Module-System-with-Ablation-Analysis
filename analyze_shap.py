"""SHAP feature importance analysis for DQN trading agent.

Quick-trains one model on a representative stock, then uses SHAP to measure
how much each feature contributes to Q-value predictions.
"""
import sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from data_storage.db_manager import DatabaseManager
from utils.indicators import compute_indicators
from rl_engine.dqn import DQNAgent, _device
from rl_engine.env import FinancialTradingEnv, STATE_DIM
from rl_engine.train import train_dqn, walk_forward_split

import torch

# ── Pick representative stocks from different sectors ──
TARGETS = ['JPM', 'AAPL', 'JNJ', 'KO', 'XOM', 'DIS']

db = DatabaseManager()

for TICKER in TARGETS:
    print(f'\n{"="*60}')
    print(f'SHAP Analysis: {TICKER}')
    print('='*60)

    market = db.get_market_data(TICKER)
    sent = db.get_sentiment(TICKER)

    # Build features (same as ablation)
    df = compute_indicators(market).reset_index(drop=True)
    sent_dates = pd.to_datetime(sent['date'].unique()) if not sent.empty else pd.DatetimeIndex([])
    if not sent.empty:
        first_news = sent_dates.min().date()
        cutoff = first_news - pd.Timedelta(days=100)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df[df['date'] >= cutoff]
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Sentiment processing
    from main import _process_sentiment_signal
    if not sent.empty:
        signal = _process_sentiment_signal(sent)
        if not signal.empty:
            sig_df = signal.reset_index()
            sig_df.columns = ['date', 'sentiment_score']
            sig_df['date'] = pd.to_datetime(sig_df['date']).dt.date
            df = df.merge(sig_df, on='date', how='left')
            df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
        else:
            df['sentiment_score'] = 0.0
    else:
        df['sentiment_score'] = 0.0

    df = df.ffill().fillna(0.0)
    df['sentiment_ma5'] = df['sentiment_score'].rolling(5, min_periods=1).mean()
    df['sentiment_ma20'] = df['sentiment_score'].rolling(20, min_periods=1).mean()
    df['sentiment_trend'] = df['sentiment_ma5'] - df['sentiment_ma20']
    df['sentiment_vol'] = df['sentiment_score'].rolling(10, min_periods=1).std()

    train_df, val_df, test_df = walk_forward_split(df)

    # ── Train model ──
    print(f'Training on {len(train_df)} rows...')
    agent = train_dqn(train_df, val_df, episodes=150, ticker=f'{TICKER}_shap',
                      sentiment_bonus_enabled=False)

    # ── Collect test states + Q-values ──
    env = FinancialTradingEnv(test_df, sentiment_bonus_enabled=False)
    state, _ = env.reset()
    states = []
    actions_taken = []
    q_values = []

    done = False
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(_device)
        with torch.no_grad():
            q = agent.q_network(state_t).cpu().numpy().flatten()

        action = agent.select_action(state, evaluate=True)
        states.append(state.copy())
        actions_taken.append(action)
        q_values.append(q)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    states = np.array(states)
    q_values = np.array(q_values)

    print(f'Test steps: {len(states)}')

    # ── SHAP using KernelExplainer ──
    import shap

    # Use 100 background samples for speed
    n_bg = min(100, len(states) // 3)
    bg_idx = np.random.choice(len(states), n_bg, replace=False)
    background = states[bg_idx]

    # Prediction function: Q-value for the best action
    def predict_q(states_array):
        results = []
        for s in states_array:
            s_t = torch.FloatTensor(s).unsqueeze(0).to(_device)
            with torch.no_grad():
                q = agent.q_network(s_t).cpu().numpy().flatten()
            results.append(q.max())  # max Q-value
        return np.array(results)

    # Also predict per-action Q
    def predict_q_buy(states_array):
        results = []
        for s in states_array:
            s_t = torch.FloatTensor(s).unsqueeze(0).to(_device)
            with torch.no_grad():
                results.append(agent.q_network(s_t).cpu().numpy().flatten()[1])  # Q(buy)
        return np.array(results)

    print('Computing SHAP values...')
    explainer = shap.KernelExplainer(predict_q, background[:50])
    shap_values = explainer.shap_values(states[:200], nsamples=100)

    # ── Feature names ──
    fnames = ['price_ratio', 'MA50_ratio', 'MA200_ratio', 'RSI', 'MACD',
              'position_pct', 'cash_pct', 'sentiment', 'sent_ma5',
              'sent_trend', 'sent_vol']

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    ranking = np.argsort(-mean_shap)

    print(f'\n{"Feature":<18} {"|SHAP|":>8} {"Rank":>5}')
    print('-'*35)
    for rank, idx in enumerate(ranking):
        marker = ' ◄ NLP' if idx >= 7 else ''
        print(f'{fnames[idx]:<18} {mean_shap[idx]:>8.4f} {rank+1:>5}{marker}')

    # Sentiment feature importance summary
    nlp_shap = mean_shap[7:].sum()
    total_shap = mean_shap.sum()
    print(f'\nNLP features SHAP: {nlp_shap:.4f} / {total_shap:.4f} = {nlp_shap/total_shap*100:.1f}%')
