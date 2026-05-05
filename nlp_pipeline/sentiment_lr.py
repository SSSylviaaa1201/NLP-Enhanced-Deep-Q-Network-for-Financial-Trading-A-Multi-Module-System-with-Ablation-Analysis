"""Logistic Regression sentiment classifier with TF-IDF features."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from config import MODEL_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "lr_sentiment.pkl"
VECTORIZER_PATH = MODEL_DIR / "lr_vectorizer.pkl"


def _synthetic_labels(df: pd.DataFrame, text_col: str = "cleaned_text") -> np.ndarray:
    """Generate weak labels from keyword heuristics (for cold-start training)."""
    positive_words = {"profit", "growth", "gain", "bullish", "beat", "upgrade", "rise", "positive"}
    negative_words = {"loss", "decline", "crash", "bearish", "debt", "miss", "downgrade", "fall", "negative"}

    labels = []
    for text in df[text_col].fillna(""):
        tokens = set(str(text).lower().split())
        pos = len(tokens & positive_words)
        neg = len(tokens & negative_words)
        if pos > neg:
            labels.append(1)
        elif neg > pos:
            labels.append(0)
        else:
            labels.append(1 if np.random.random() > 0.5 else 0)  # neutral → random
    return np.array(labels)


def train_lr_sentiment(df: pd.DataFrame, text_col: str = "cleaned_text"):
    """Train a Logistic Regression sentiment model. Returns (model, vectorizer)."""
    texts = df[text_col].fillna("").tolist()
    y = _synthetic_labels(df, text_col)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred)
    logger.info("LR trained: F1=%.4f, samples=%d", f1, len(texts))

    ModelDir = MODEL_DIR
    ModelDir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer


def load_lr_model():
    """Load trained LR model and vectorizer, or None if not trained."""
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def lr_predict(texts: list[str], model=None, vectorizer=None) -> list[dict]:
    """Predict sentiment scores using LR model."""
    if model is None or vectorizer is None:
        model, vectorizer = load_lr_model()
    if model is None:
        raise RuntimeError("LR model not trained. Call train_lr_sentiment first.")

    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)

    results = []
    for i in range(len(texts)):
        prob_neg, prob_pos = probs[i]  # 0=neg, 1=pos
        score = prob_pos - prob_neg  # range [-1, 1]
        confidence = max(prob_neg, prob_pos)
        label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
        results.append({"sentiment_score": round(score, 4), "confidence": round(confidence, 4), "label": label})
    return results


def lr_sentiment_batch(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """Apply LR sentiment to a DataFrame."""
    model, vectorizer = load_lr_model()
    if model is None:
        logger.info("Training LR model on current data...")
        model, vectorizer = train_lr_sentiment(df, text_col)

    results = lr_predict(df[text_col].fillna("").tolist(), model, vectorizer)
    df = df.copy()
    df["sentiment_score"] = [r["sentiment_score"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["label"] = [r["label"] for r in results]
    df["method"] = "lr"
    return df
