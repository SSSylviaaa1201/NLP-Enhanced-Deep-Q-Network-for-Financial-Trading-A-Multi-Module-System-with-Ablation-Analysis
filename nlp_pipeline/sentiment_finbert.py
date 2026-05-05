"""FinBERT sentiment analysis via HuggingFace transformers."""

import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import FINBERT_MODEL, MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_cached_tokenizer = None
_cached_model = None


def _load_finbert():
    global _cached_tokenizer, _cached_model
    if _cached_model is None:
        logger.info("Loading FinBERT: %s on %s", FINBERT_MODEL, _device)
        _cached_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _cached_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(_device)
        _cached_model.eval()
    return _cached_tokenizer, _cached_model


def _finbert_score(text: str) -> dict:
    """Run FinBERT on a single text and return sentiment dict."""
    tokenizer, model = _load_finbert()
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {"sentiment_score": 0.0, "confidence": 0.0, "label": "neutral"}

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=MAX_SEQ_LENGTH,
    ).to(_device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # FinBERT labels: 0=negative, 1=neutral, 2=positive
    neg_prob, neu_prob, pos_prob = probs
    score = pos_prob - neg_prob  # range [-1, 1]
    confidence = float(np.max(probs))
    label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
    return {"sentiment_score": round(float(score), 4), "confidence": round(confidence, 4), "label": label}


def finbert_sentiment_batch(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Apply FinBERT sentiment to entire DataFrame with mini-batches."""
    tokenizer, model = _load_finbert()
    texts = df[text_col].fillna("").tolist()
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Filter empty strings
        valid_indices = [j for j, t in enumerate(batch_texts) if t.strip()]
        results_batch = [{"sentiment_score": 0.0, "confidence": 0.0, "label": "neutral"} for _ in batch_texts]

        if valid_indices:
            valid_texts = [batch_texts[j] for j in valid_indices]
            inputs = tokenizer(
                valid_texts, return_tensors="pt", truncation=True,
                padding=True, max_length=MAX_SEQ_LENGTH,
            ).to(_device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for idx, j in enumerate(valid_indices):
                neg_prob, neu_prob, pos_prob = probs[idx]
                score = pos_prob - neg_prob
                results_batch[j] = {
                    "sentiment_score": round(float(score), 4),
                    "confidence": round(float(np.max(probs[idx])), 4),
                    "label": "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"),
                }

        all_results.extend(results_batch)

    df = df.copy()
    df["sentiment_score"] = [r["sentiment_score"] for r in all_results]
    df["confidence"] = [r["confidence"] for r in all_results]
    df["label"] = [r["label"] for r in all_results]
    df["method"] = "finbert"
    return df
