"""Test database operations."""
import sqlite3
from data_storage.db_manager import DatabaseManager


def test_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    conn = sqlite3.connect(str(db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    expected = {"news", "market_data", "sentiment_signals", "trading_logs", "trade_orders"}
    assert expected.issubset(tables)


def test_insert_and_query_news(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    count = db.insert_news([{
        "ticker": "TEST", "source": "test", "title": "Hello",
        "content": "World", "url": "http://t.co", "published_at": "2024-01-01",
    }])
    assert count == 1
    news = db.get_news("TEST")
    assert len(news) == 1
    assert news.iloc[0]["title"] == "Hello"
