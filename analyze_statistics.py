"""Per-sector statistical tests + FDR correction on NLP ablation results."""
import json, sys
import numpy as np
from scipy import stats

sys.path.insert(0, '.')

with open('data/ablation_results.json') as f:
    d = json.load(f)

tickers = d['layer1_ablation']['tickers']

# Sector mapping
sector_map = dict(zip(
    ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','ADBE','INTC','CRM',
     'JPM','BAC','V','MA','GS','BLK','AXP','MS','WFC','C',
     'JNJ','UNH','PFE','ABBV','MRK','TMO','ABT','BMY','GILD','LLY',
     'KO','PEP','WMT','COST','NKE','HD','MCD','PG','SBUX','LOW',
     'XOM','CVX','CAT','BA','GE','COP','DE','UPS','LMT','RTX',
     'DIS','NFLX','NEE','T','VZ','CMCSA','TMUS','SO','DUK','CHTR'],
    ['Tech']*10 + ['Finance']*10 + ['Healthcare']*10 + ['Consumer']*10 +
    ['Energy/Industrial']*10 + ['Comm/Utility']*10))

# Group NLP deltas by sector
sector_deltas = {}
for t, s in tickers.items():
    sec = sector_map.get(t, 'Other')
    sector_deltas.setdefault(sec, []).append(s['sharpe_delta'])

print('=' * 70)
print('PER-SECTOR NLP EFFECTIVENESS — STATISTICAL TESTS')
print('=' * 70)

all_pvalues = []
sector_results = []

for sec in ['Tech', 'Finance', 'Healthcare', 'Consumer', 'Energy/Industrial', 'Comm/Utility']:
    deltas = sector_deltas.get(sec, [])
    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0.01)
    neg = sum(1 for d in deltas if d < -0.01)
    mean_d = np.mean(deltas)
    std_d = np.std(deltas, ddof=1)
    se_d = std_d / np.sqrt(n) if n > 1 else 0

    # One-sample t-test: H0 = mean delta is 0
    if n >= 3:
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
        cohens_d = mean_d / std_d if std_d > 0 else 0
    else:
        t_stat, p_val, cohens_d = 0, 1.0, 0

    all_pvalues.append(p_val)
    sector_results.append({
        'sector': sec, 'n': n, 'positive': pos, 'negative': neg,
        'mean_delta': mean_d, 'std': std_d, 'se': se_d,
        't_stat': t_stat, 'p_value': p_val, 'cohens_d': cohens_d,
    })

    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
    direction = 'POSITIVE' if mean_d > 0 else 'NEGATIVE'
    print(f'{sec:<20s} n={n:2d}  pos={pos}/{n}  mean={mean_d:+.4f}  '
          f't={t_stat:+.3f}  p={p_val:.4f}{sig:3s}  d={cohens_d:+.3f}  {direction}')

# ── FDR correction (Benjamini-Hochberg) ──
print(f'\n{"─"*50}')
print('FDR CORRECTION (Benjamini-Hochberg)')
print(f'{"─"*50}')

pvals = np.array(all_pvalues)
n_tests = len(pvals)
sorted_idx = np.argsort(pvals)
sorted_p = pvals[sorted_idx]
bh_thresholds = 0.05 * (np.arange(n_tests) + 1) / n_tests

for i, (idx, p, thresh) in enumerate(zip(sorted_idx, sorted_p, bh_thresholds)):
    sec = sector_results[idx]['sector']
    reject = 'REJECT H0' if p <= thresh else 'accept'
    print(f'  {sec:<20s} p={p:.4f}  threshold={thresh:.4f}  {reject}')

# ── Overall NLP significance ──
print(f'\n{"─"*50}')
print('OVERALL NLP SIGNIFICANCE')
print(f'{"─"*50}')
all_deltas = [s['sharpe_delta'] for s in tickers.values()]
t_all, p_all = stats.ttest_1samp(all_deltas, 0)
d_all = np.mean(all_deltas) / np.std(all_deltas, ddof=1)
pos_all = sum(1 for d in all_deltas if d > 0.01)
print(f'  N={len(all_deltas)}, NLP positive={pos_all} ({pos_all/len(all_deltas)*100:.1f}%)')
print(f'  Mean delta: {np.mean(all_deltas):+.4f} +- {np.std(all_deltas, ddof=1):.4f}')
print(f'  One-sample t-test: t={t_all:.3f}, p={p_all:.4f}')
print(f'  Cohens d: {d_all:+.3f}')
print(f'  95% CI: [{np.mean(all_deltas)-1.96*np.std(all_deltas,ddof=1)/np.sqrt(len(all_deltas)):+.4f}, '
      f'{np.mean(all_deltas)+1.96*np.std(all_deltas,ddof=1)/np.sqrt(len(all_deltas)):+.4f}]')

# ── Correlation: NLP effectiveness vs data quality ──
print(f'\n{"─"*50}')
print('NLP EFFECTIVENESS vs DATA QUALITY')
print(f'{"─"*50}')
from data_storage.db_manager import DatabaseManager
db = DatabaseManager()
import pandas as pd
rows = []
for t in tickers:
    s = db.get_sentiment(t)
    if not s.empty:
        n_news = s['date'].nunique()
        # Method agreement (if multiple methods)
        pivot = s.pivot_table(index='date', columns='method', values='sentiment_score', aggfunc='mean')
        if pivot.shape[1] >= 2:
            corr = pivot.corr().values
            triu_idx = np.triu_indices_from(corr, k=1)
            agreement = float(corr[triu_idx].mean()) if len(triu_idx[0]) > 0 else 0
        else:
            agreement = 0
        rows.append({'ticker': t, 'nlp_delta': tickers[t]['sharpe_delta'],
                     'n_news_days': n_news, 'method_agreement': agreement})

qd = pd.DataFrame(rows)
c1 = qd['nlp_delta'].corr(qd['n_news_days'])
c2 = qd['nlp_delta'].corr(qd['method_agreement'])
print(f'  Corr(NLP delta, news_days):       {c1:+.3f}')
print(f'  Corr(NLP delta, method_agreement): {c2:+.3f}')
# Split by data quality
hi_news = qd[qd['n_news_days'] > qd['n_news_days'].median()]
lo_news = qd[qd['n_news_days'] <= qd['n_news_days'].median()]
h_pos = (hi_news['nlp_delta'] > 0.01).sum()
l_pos = (lo_news['nlp_delta'] > 0.01).sum()
h_mean = hi_news['nlp_delta'].mean()
l_mean = lo_news['nlp_delta'].mean()
print('  High news days: NLP pos=%d/%d, mean=%+.4f' % (h_pos, len(hi_news), h_mean))
print('  Low news days:  NLP pos=%d/%d, mean=%+.4f' % (l_pos, len(lo_news), l_mean))
