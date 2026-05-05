"""Database schema: DDL statements for all tables."""

CREATE_NEWS_TABLE = """
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    source TEXT,
    title TEXT,
    content TEXT,
    url TEXT UNIQUE,
    published_at TEXT,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MARKET_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    adjusted_close REAL,
    UNIQUE(ticker, date)
);
"""

CREATE_SENTIMENT_TABLE = """
CREATE TABLE IF NOT EXISTS sentiment_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    method TEXT NOT NULL,
    sentiment_score REAL,
    confidence REAL,
    label TEXT,
    reasoning TEXT,
    UNIQUE(ticker, date, method)
);
"""

CREATE_LLM_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title_hash TEXT UNIQUE,
    title TEXT,
    sentiment_score REAL,
    confidence REAL,
    label TEXT,
    reasoning TEXT,
    model_used TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TRADING_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS trading_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode INTEGER,
    step INTEGER,
    ticker TEXT,
    date TEXT,
    action TEXT,
    price REAL,
    position INTEGER,
    cash REAL,
    portfolio_value REAL,
    sentiment_score REAL,
    reward REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TRADE_ORDERS_TABLE = """
CREATE TABLE IF NOT EXISTS trade_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    action TEXT NOT NULL,
    shares INTEGER,
    price REAL,
    cost REAL,
    portfolio_value_after REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_TABLES = [
    CREATE_NEWS_TABLE,
    CREATE_MARKET_DATA_TABLE,
    CREATE_SENTIMENT_TABLE,
    CREATE_LLM_CACHE_TABLE,
    CREATE_TRADING_LOGS_TABLE,
    CREATE_TRADE_ORDERS_TABLE,
]
