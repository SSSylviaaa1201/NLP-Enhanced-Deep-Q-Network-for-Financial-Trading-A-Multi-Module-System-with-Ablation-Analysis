"""Regime-stratified analysis of NLP ablation results (no retraining needed)."""
import json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from data_storage.db_manager import DatabaseManager
from utils.indicators import compute_indicators
import config

def compute_regime_features(close: pd.Series) -> pd.DataFrame:
    """Compute continuous regime indicators for each day."""
    r = close.pct_change()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    vol20 = r.rolling(20).std() * np.sqrt(252)
    vol252 = r.rolling(252).std() * np.sqrt(252)
    peak252 = close.rolling(252).max()

    return pd.DataFrame({
        'trend_pct': (close - ma200) / ma200 * 100,
        'vol_ratio': vol20 / vol252.replace(0, np.nan),
        'drawdown': (close - peak252) / peak252 * 100,
        'ma50_200': (ma50 - ma200) / ma200 * 100,
    })

def classify_regime(row: pd.Series) -> str:
    """Classify a single day into a regime category."""
    t = row['trend_pct']
    v = row['vol_ratio']
    d = row['drawdown']

    if d < -20:
        return 'Crisis'
    if v > 1.5:
        return 'HighVol'
    if t > 5:
        return 'Bull'
    if t < -5:
        return 'Bear'
    return 'Sideways'

# ── Load ablation results ──
with open('data/ablation_results.json') as f:
    results = json.load(f)

tickers_data = results['layer1_ablation']['tickers']

# ── Per-ticker regime distribution ──
db = DatabaseManager()
rows = []

for ticker, summary in tickers_data.items():
    df = db.get_market_data(ticker)
    if df.empty:
        continue
    df = compute_indicators(df)
    n = len(df)
    # Test period = last 20%
    test = df.iloc[int(n * 0.8):].copy()

    regime_df = compute_regime_features(test['close'])
    regime_df['regime'] = regime_df.apply(classify_regime, axis=1)
    regime_df = regime_df.dropna()

    counts = regime_df['regime'].value_counts()
    total = len(regime_df)
    pct_bull = counts.get('Bull', 0) / total * 100
    pct_bear = counts.get('Bear', 0) / total * 100
    pct_crisis = counts.get('Crisis', 0) / total * 100
    pct_highvol = counts.get('HighVol', 0) / total * 100
    pct_sideways = counts.get('Sideways', 0) / total * 100

    avg_vol = regime_df['vol_ratio'].mean()
    avg_trend = regime_df['trend_pct'].mean()
    avg_dd = regime_df['drawdown'].mean()

    rows.append({
        'ticker': ticker,
        'nlp_delta': summary['sharpe_delta'],
        'nlp_helps': summary['nlp_improves_sharpe'],
        'pct_bull': pct_bull, 'pct_bear': pct_bear,
        'pct_crisis': pct_crisis, 'pct_highvol': pct_highvol,
        'pct_sideways': pct_sideways,
        'avg_vol': avg_vol, 'avg_trend': avg_trend, 'avg_dd': avg_dd,
    })

regime_df = pd.DataFrame(rows)

# ── Analysis ──
print('=' * 70)
print('REGIME-STRATIFIED NLP ANALYSIS')
print('=' * 70)

# 1. Overall regime distribution
print(f"\n--- Test Period Regime Distribution (avg across 60 stocks) ---")
print(f"  Bull:     {regime_df['pct_bull'].mean():.1f}%")
print(f"  Bear:     {regime_df['pct_bear'].mean():.1f}%")
print(f"  Sideways: {regime_df['pct_sideways'].mean():.1f}%")
print(f"  HighVol:  {regime_df['pct_highvol'].mean():.1f}%")
print(f"  Crisis:   {regime_df['pct_crisis'].mean():.1f}%")

# 2. NLP delta by dominant regime
print(f"\n--- NLP Effectiveness by Dominant Regime ---")
for reg in ['Bull', 'Bear', 'Sideways', 'HighVol', 'Crisis']:
    col = f'pct_{reg.lower()}'
    # Stocks where this regime > 30% of test days
    subset = regime_df[regime_df[col] > 30]
    if len(subset) >= 3:
        pos = (subset['nlp_delta'] > 0.01).sum()
        avg = subset['nlp_delta'].mean()
        print(f"  {reg:10s}: {len(subset):2d} stocks, NLP pos={pos}/{len(subset)}, avg Δ={avg:+.4f}")
    else:
        print(f"  {reg:10s}: {len(subset):2d} stocks (too few for analysis)")

# 3. Correlation: regime features vs NLP delta
print(f"\n--- Correlation: Regime Features vs NLP Δ ---")
for feat in ['pct_bull', 'pct_bear', 'pct_crisis', 'avg_vol', 'avg_trend', 'avg_dd']:
    corr = regime_df[feat].corr(regime_df['nlp_delta'])
    print(f"  {feat:15s}: r={corr:+.3f}")

# 4. NLP effectiveness by market condition
print(f"\n--- NLP Win Rate by Market Condition ---")
# Split stocks by test period characteristics
high_vol = regime_df[regime_df['avg_vol'] > regime_df['avg_vol'].median()]
low_vol = regime_df[regime_df['avg_vol'] <= regime_df['avg_vol'].median()]
bull_mkt = regime_df[regime_df['avg_trend'] > 0]
bear_mkt = regime_df[regime_df['avg_trend'] <= 0]

for label, subset in [('HighVol test', high_vol), ('LowVol test', low_vol),
                       ('Bull trend', bull_mkt), ('Bear trend', bear_mkt)]:
    if len(subset) >= 5:
        pos = (subset['nlp_helps'] == True).sum()
        avg_d = subset['nlp_delta'].mean()
        print(f"  {label:15s}: {pos}/{len(subset)} positive, avg Δ={avg_d:+.4f}")

# 5. Top stocks by regime diversity (stocks that saw multiple regimes)
print(f"\n--- Stocks with Most Diverse Regimes ---")
regime_df['regime_diversity'] = (
    (regime_df['pct_bull'] > 10).astype(int) +
    (regime_df['pct_bear'] > 10).astype(int) +
    (regime_df['pct_crisis'] > 10).astype(int) +
    (regime_df['pct_sideways'] > 10).astype(int)
)
diverse = regime_df[regime_df['regime_diversity'] >= 3]
print(f"  {len(diverse)} stocks saw >=3 regime types in test period")
print(f"  NLP positive: {(diverse['nlp_helps']==True).sum()}/{len(diverse)}")
print(f"  Avg NLP Δ: {diverse['nlp_delta'].mean():+.4f}")

# ── Save for report ──
regime_df.to_csv('data/regime_analysis.csv', index=False)
print("\nSaved to data/regime_analysis.csv")
