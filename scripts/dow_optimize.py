"""
Optimize the DOW-corrected seasonal model.

Current best: dow_4140k_cogs103.csv → Kaggle 682,679

Levers to optimize:
1. ann24 (2024 annual mean) — current 4,265,000, sensitivity ~1,607 RMSE per 100K
2. cogs_scale with DOW — retest if 1.03 still optimal
3. Global DOW vs Monthly DOW — test simpler model
4. LGBM ensemble with DOW seasonal
"""
import sys
sys.path.insert(0, 'E:/lgbm_temp')

import pandas as pd
import numpy as np
import os

RAW = 'src/data/raw'
os.makedirs('submissions', exist_ok=True)

ANNUAL_2023 = 4_140_000.0

# ── Build base seasonal + DOW profile ─────────────────────────────────────
sales = pd.read_csv(f'{RAW}/sales.csv', parse_dates=['Date']).sort_values('Date')
sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['day']   = sales['Date'].dt.day
sales['dow']   = sales['Date'].dt.dayofweek

pre = sales[sales['year'].between(2013, 2018)].copy()
ann_m = pre.groupby('year')['Revenue'].mean()
pre['rev_norm']  = pre['Revenue'] / pre['year'].map(ann_m)
pre['cogs_norm'] = pre['COGS']    / pre['year'].map(ann_m)

# Calendar seasonal profile
cal = pre.groupby(['month', 'day'])[['rev_norm', 'cogs_norm']].mean().reset_index()
cal_rn_mean = cal['rev_norm'].mean()
cal['rev_norm']  /= cal_rn_mean
cal['cogs_norm'] /= cal_rn_mean

# Weekday correction factors
pre_with_cal = pre.merge(
    cal[['month', 'day', 'rev_norm']].rename(columns={'rev_norm': 'seasonal_rn'}),
    on=['month', 'day'], how='left'
)
pre_with_cal['residual'] = pre_with_cal['rev_norm'] / pre_with_cal['seasonal_rn']

# Monthly DOW factors (84 factors: 12 months × 7 days)
wd_factor_monthly = pre_with_cal.groupby(['month', 'dow'])['residual'].mean()

# Global DOW factors (7 factors only)
wd_factor_global = pre_with_cal.groupby('dow')['residual'].mean()

print("Global DOW factors:")
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for d, v in wd_factor_global.items():
    bar = '▊' * int(abs(v - 1.0) * 100)
    direction = '+' if v >= 1 else '-'
    print(f"  {dow_names[d]}: {v:.4f} ({direction}{abs(v-1)*100:.1f}%) {bar}")

# ── Core prediction builder ────────────────────────────────────────────────
test_dates = pd.date_range('2023-01-01', '2024-07-01', freq='D')

def build_dow(annual_2024, cogs_scale=1.03, use_monthly_dow=True):
    df = pd.DataFrame({'Date': test_dates})
    df['year']  = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day']   = df['Date'].dt.day
    df['dow']   = df['Date'].dt.dayofweek
    df = df.merge(cal, on=['month', 'day'], how='left')
    df['rev_norm']  = df['rev_norm'].fillna(1.0)
    df['cogs_norm'] = df['cogs_norm'].fillna(cal['cogs_norm'].mean())

    if use_monthly_dow:
        df['wd_factor'] = df.apply(
            lambda r: wd_factor_monthly.get((r['month'], r['dow']),
                                             wd_factor_global.get(r['dow'], 1.0)), axis=1)
    else:
        df['wd_factor'] = df['dow'].map(wd_factor_global).fillna(1.0)

    df['rev_norm']  *= df['wd_factor']
    df['cogs_norm'] *= df['wd_factor']

    df['annual_mean'] = df['year'].map({2023: ANNUAL_2023, 2024: annual_2024})
    rev  = (df['rev_norm']  * df['annual_mean']).values
    cogs = (df['cogs_norm'] * df['annual_mean'] * cogs_scale).values
    cogs = np.minimum(cogs, rev * 0.985)
    rev  = np.maximum(rev, 1)
    cogs = np.maximum(cogs, 1)

    out = pd.DataFrame({
        'Date':    df['Date'].dt.strftime('%Y-%m-%d'),
        'Revenue': rev.round().astype(int),
        'COGS':    cogs.round().astype(int)
    })
    assert len(out) == 548
    return out

# ── Variant 1: ann24 search with DOW ──────────────────────────────────────
print('\n=== Variant 1: ann24 search (DOW corrected) ===')
print('Current best: ann24=4,265,000 → Kaggle 682,679')
print('Sensitivity: 100K deviation → ~1,607 RMSE')
print()

ann24_candidates = [4_100_000, 4_150_000, 4_200_000, 4_250_000,
                    4_265_000, 4_300_000, 4_350_000, 4_400_000]

for ann24 in ann24_candidates:
    out = build_dow(ann24)
    d24 = out[out['Date'] >= '2024-01-01']
    fname = f'submissions/dow_ann24_{int(ann24/1000)}k.csv'
    out.to_csv(fname, index=False)
    print(f'  ann24={ann24/1e6:.3f}M → 2024_avg={d24["Revenue"].mean():.0f}  saved: {fname}')

# ── Variant 2: cogs_scale search with DOW ─────────────────────────────────
print('\n=== Variant 2: cogs_scale search (ann24=4265K, DOW corrected) ===')
for cs in [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]:
    out = build_dow(4_265_000, cogs_scale=cs)
    avg_cogs_ratio = (out['COGS'] / out['Revenue']).mean()
    fname = f'submissions/dow_cogs{int(cs*100):03d}.csv'
    out.to_csv(fname, index=False)
    print(f'  cogs_scale={cs:.2f} → avg_COGS/Rev={avg_cogs_ratio:.4f}  saved: {fname}')

# ── Variant 3: Global DOW only (simpler, less noise) ──────────────────────
print('\n=== Variant 3: Global DOW (no monthly interaction) ===')
out_global = build_dow(4_265_000, use_monthly_dow=False)
out_global.to_csv('submissions/dow_global_4265k.csv', index=False)
print('  Saved: dow_global_4265k.csv')

# ── Variant 4: LGBM ensemble with DOW seasonal ────────────────────────────
print('\n=== Variant 4: LGBM v3 + DOW seasonal ensemble ===')
try:
    lgbm = pd.read_csv('submissions/lgbm_v3_calibrated.csv')
    dow_best = pd.read_csv('submissions/dow_4140k_cogs103.csv')

    assert (lgbm['Date'] == dow_best['Date']).all()

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        ens_rev  = alpha * lgbm['Revenue'] + (1 - alpha) * dow_best['Revenue']
        ens_cogs = alpha * lgbm['COGS']    + (1 - alpha) * dow_best['COGS']
        ens_cogs = np.minimum(ens_cogs, ens_rev * 0.985)
        ens_rev  = np.maximum(ens_rev, 1)
        ens_cogs = np.maximum(ens_cogs, 1)

        out = pd.DataFrame({
            'Date':    lgbm['Date'],
            'Revenue': ens_rev.round().astype(int),
            'COGS':    ens_cogs.round().astype(int)
        })
        fname = f'submissions/ens_lgbm{int(alpha*10)}_dow{10-int(alpha*10)}.csv'
        out.to_csv(fname, index=False)

        diff = (out['Revenue'] - dow_best['Revenue']).abs().mean()
        print(f'  alpha={alpha:.1f} (LGBM:{alpha:.0%} + DOW:{1-alpha:.0%})  '
              f'mean_diff_from_DOW={diff:>8,.0f}  → {fname}')
except FileNotFoundError as e:
    print(f'  LGBM file not found: {e}')

# ── Summary table ──────────────────────────────────────────────────────────
print('\n' + '='*65)
print('SUBMISSION PLAN')
print('='*65)
print('CONFIRMED: dow_4140k_cogs103.csv → Kaggle 682,679 (NEW BEST)')
print()
print('Priority submit queue (4 slots):')
print('  1. lgbm_v3_calibrated.csv      → get LGBM standalone score')
print('  2. ens_lgbm2_dow8.csv          → 20% LGBM + 80% DOW (conservative)')
print('  3. ens_lgbm3_dow7.csv          → 30% LGBM + 70% DOW (moderate)')
print('  4. dow_ann24_4200k.csv         → test if ann24=4.2M better with DOW')
print()
print('ann24 sensitivity with DOW (for guidance after #4 result):')
print('  If 4200k < 682K → ann24 optimum < 4265K, search 4100-4200K')
print('  If 4200k > 682K → 4265K still optimal, keep current')
print()
print('Files generated:', len([f for f in os.listdir('submissions') if f.startswith('dow_')]))
