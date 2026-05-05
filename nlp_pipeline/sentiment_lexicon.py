"""Lexicon-based sentiment analysis using VADER."""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()

# Financial-domain lexicon adjustments
FINANCIAL_LEXICON = {
    "profit": 1.5, "growth": 1.2, "gain": 1.0, "bullish": 1.8,
    "loss": -1.5, "decline": -1.2, "crash": -2.0, "bearish": -1.8,
    "debt": -1.0, "lawsuit": -1.5, "layoff": -1.5, "bankrupt": -2.5,
    "upgrade": 1.3, "downgrade": -1.3, "beat": 1.2, "miss": -1.2,
}
analyzer.lexicon.update(FINANCIAL_LEXICON)


def vader_sentiment(text: str) -> dict:
    """Return compound score and label for a single text."""
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"sentiment_score": compound, "confidence": abs(compound), "label": label}


def vader_sentiment_batch(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """Apply VADER sentiment to a DataFrame, return with score columns."""
    results = df[text_col].fillna("").apply(vader_sentiment)
    df = df.copy()
    df["sentiment_score"] = results.apply(lambda r: r["sentiment_score"])
    df["confidence"] = results.apply(lambda r: r["confidence"])
    df["label"] = results.apply(lambda r: r["label"])
    df["method"] = "vader"
    return df
