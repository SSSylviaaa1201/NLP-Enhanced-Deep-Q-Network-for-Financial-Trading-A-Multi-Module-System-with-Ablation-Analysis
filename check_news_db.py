import sqlite3

conn = sqlite3.connect('data/trading.db')
c = conn.cursor()

# 看所有表
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("=== Tables ===")
print(tables)

# 看 news 表结构
print()
print("=== news table schema ===")
c.execute("PRAGMA table_info(news)")
cols = c.fetchall()
for col in cols:
    print(f"  {col}")

# 实际列名
col_names = [col[1] for col in cols]
print()
print("=== news columns:", col_names)

# source 列存在吗
if 'source' in col_names:
    c.execute("SELECT source, COUNT(*) FROM news GROUP BY source ORDER BY COUNT(*) DESC")
    print()
    print("=== News by source ===")
    for r in c.fetchall():
        print(f"  {r}")

# 取前 5 条看实际内容
print()
print("=== First 5 rows (all columns) ===")
c.execute("SELECT * FROM news LIMIT 5")
for r in c.fetchall():
    print(f"  {r}")

c.execute("SELECT COUNT(*) FROM news")
print()
print(f"Total news rows: {c.fetchone()[0]}")

conn.close()
