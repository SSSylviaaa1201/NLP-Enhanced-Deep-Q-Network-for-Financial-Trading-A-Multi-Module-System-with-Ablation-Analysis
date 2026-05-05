"""Abstract base classes for pluggable components.

Anyone can add a new sentiment method or data source by implementing these interfaces.
"""

from abc import ABC, abstractmethod

import pandas as pd


class SentimentMethod(ABC):
    """Interface for any sentiment analysis method."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier (e.g., 'vader', 'finbert', 'my_custom')."""
        ...

    @abstractmethod
    def analyze(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment. Returns DataFrame with [date, sentiment_score, confidence, label]."""
        ...


class DataSource(ABC):
    """Interface for market data or news data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fetch(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch data for given symbol. Return standardized DataFrame."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is reachable/configured."""
        ...


class RLAgent(ABC):
    """Interface for any RL-based trading agent."""

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the agent on provided data."""
        ...

    @abstractmethod
    def predict(self, state) -> int:
        """Choose action given current state."""
        ...

    @abstractmethod
    def explain(self, state) -> dict:
        """Explain why the agent chose its action."""
        ...
