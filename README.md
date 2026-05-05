# NLP-Driven Reinforcement Learning Trading Platform

An end-to-end intelligent trading platform integrating Natural Language Processing (NLP) and Reinforcement Learning (RL).

**Pipeline**: Raw Text → NLP Sentiment Signal → RL Trading Decision

## Architecture

| Module | Responsibility |
|--------|---------------|
| ① Data Ingestion | Fetch financial news + OHLCV market data |
| ② NLP Pipeline | Sentiment analysis (VADER, Logistic Regression, FinBERT) |
| ③ Data Storage | SQLite persistence for all pipeline data |
| ④ RL Trading Engine | DQN agent with custom Gym trading environment |
| ⑤ Front-End Dashboard | Streamlit visualization and monitoring |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install torch transformers     # Required for FinBERT + DQN

# 2. (Optional) Set up NewsAPI key for live news
cp .env.example .env               # Edit .env with your key from https://newsapi.org/register
                                    # Skip this step to use built-in sample data

# 3. Run the full pipeline
python main.py --ablate            # Train RL agent + ablation study

# 4. Launch dashboard
streamlit run dashboard/app.py
```

> `.env` is gitignored — your API key stays local. See `.env.example` for the template.

## State Vector

[price, MA50, MA200, RSI, MACD, position, cash, sentiment_score]

## Evaluation Metrics

- Sharpe Ratio
- Maximum Drawdown (MDD)
- Buy-and-Hold Benchmark comparison
- Walk-Forward Validation
