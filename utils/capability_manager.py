"""Detect available capabilities for graceful degradation when API keys are missing."""

import os
from dataclasses import dataclass
from enum import Enum


class CapabilityLevel(Enum):
    FULL = "full"
    ENHANCED = "enhanced"
    BASIC = "basic"
    DEMO = "demo"


@dataclass
class CapabilityReport:
    level: CapabilityLevel
    sentiment_methods: list[str]
    data_sources: list[str]
    llm_enabled: bool
    rag_enabled: bool
    chat_enabled: bool
    warnings: list[str]


def detect_capabilities() -> CapabilityReport:
    """Check environment and report what's available."""
    methods = ["vader", "logistic_regression"]
    sources = ["synthetic"]
    warnings = []

    # Check FinBERT model cache
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("ProsusAI/finbert")
        methods.append("finbert")
    except Exception:
        warnings.append("FinBERT model not cached. Run pipeline once to download it.")

    # Check LLM
    llm_ok = bool(os.getenv("VOLCANO_API_KEY") and os.getenv("VOLCANO_MODEL_ID"))
    if llm_ok:
        methods.append("llm")
    else:
        warnings.append("LLM not configured. Set VOLCANO_API_KEY and VOLCANO_MODEL_ID in .env.")

    # Check data source APIs
    if os.getenv("ALPHA_VANTAGE_KEY"):
        sources.append("alpha_vantage")
    if os.getenv("NEWSAPI_KEY"):
        sources.append("newsapi")
    sources.append("rss")  # Always available (free, no key)

    # Determine level
    if llm_ok and "newsapi" in sources:
        level = CapabilityLevel.FULL
    elif "finbert" in methods and ("newsapi" in sources or "rss" in sources):
        level = CapabilityLevel.ENHANCED
    elif len(methods) >= 2:
        level = CapabilityLevel.BASIC
    else:
        level = CapabilityLevel.DEMO

    return CapabilityReport(
        level=level,
        sentiment_methods=methods,
        data_sources=sources,
        llm_enabled=llm_ok,
        rag_enabled=llm_ok,
        chat_enabled=llm_ok,
        warnings=warnings,
    )
