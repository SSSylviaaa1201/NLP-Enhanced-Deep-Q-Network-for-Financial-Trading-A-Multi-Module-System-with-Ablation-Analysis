import sqlite3

conn = sqlite3.connect('data/trading.db')
c = conn.cursor()

print("=== sentiment_signals schema ===")
c.execute("PRAGMA table_info(sentiment_signals)")
cols = c.fetchall()
for col in cols: print(f"  {col}")
col_names = [col[1] for col in cols]

print()
c.execute("SELECT COUNT(*) FROM sentiment_signals")
print(f"Total sentiment rows: {c.fetchone()[0]}")

print()
print("=== by ticker ===")
c.execute("SELECT ticker, COUNT(*) FROM sentiment_signals GROUP BY ticker ORDER BY COUNT(*) DESC LIMIT 15")
for r in c.fetchall(): print(f"  {r}")

print()
print("=== sample scores (first 10) ===")
c.execute("SELECT * FROM sentiment_signals LIMIT 10")
for r in c.fetchall(): print(f"  {r}")

# 看情感分数的分布
print()
print("=== score distribution ===")
# 找分数列
score_col = None
for name in col_names:
    if 'score' in name.lower() or 'sentiment' in name.lower() or 'signal' in name.lower():
        score_col = name
        break
if score_col:
    c.execute(f"SELECT MIN({score_col}), MAX({score_col}), AVG({score_col}), COUNT(DISTINCT ROUND({score_col},3)) FROM sentiment_signals")
    r = c.fetchone()
    print(f"  column={score_col}: min={r[0]:.4f}, max={r[1]:.4f}, avg={r[2]:.4f}, distinct_vals={r[3]}")

# Check if scores look repetitive (potential template issue)
if score_col:
    c.execute(f"SELECT ROUND({score_col},3), COUNT(*) FROM sentiment_signals GROUP BY ROUND({score_col},3) ORDER BY COUNT(*) DESC LIMIT 10")
    print()
    print(f"=== Most common {score_col} values (rounded to 3dp) ===")
    for r in c.fetchall(): print(f"  score={r[0]}: {r[1]} times")

conn.close()
