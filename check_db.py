"""Quick DB check script."""
import sqlite3
from pathlib import Path

db_path = Path("data/trading.db")
print(f"DB exists: {db_path.exists()}")
if db_path.exists():
    print(f"DB size: {db_path.stat().st_size / 1024:.1f} KB")

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"Tables: {tables}")

for t in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    print(f"  {t}: {count} rows")
    cols = [d[0] for d in cur.execute(f"PRAGMA table_info([{t}])").fetchall()]
    print(f"    Columns: {cols}")
    sample = cur.execute(f"SELECT * FROM [{t}] LIMIT 2").fetchall()
    for row in sample:
        print(f"    Sample: {[str(x)[:30] for x in row[:5]]}...")

conn.close()
