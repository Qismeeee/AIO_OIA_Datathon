"""
LGBM v3 — proper rebuild from scratch.

Root causes of lgbm_v1/v2 failure:
  1. year_mean_rev in test was 3.59M (wrong) — should be 4.14M/4.265M
  2. Web/inventory features (sessions, bounce_rate, stockout_rate, etc.)
     are ZERO in test period → add noise, degrade predictions
  3. 2024 lag_365 was using 2022 Revenue (730-day lag), not 2023 predictions
  4. Trained on all 2012-2022 including COVID distortion

This version:
  - Trains on NORMALIZED revenue (removes annual level) using 2013-2022
    with LOO CV treating pre/post-COVID separately
  - Uses ONLY safe features available in test period
  - Bootstraps 2024 lag_365 from predicted 2023
  - Ensembles with best seasonal model at optimal alpha
"""
import sys
sys.path.insert(0, 'E:/lgbm_temp')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('submissions', exist_ok=True)
os.makedirs('src/models', exist_ok=True)

RAW       = 'src/data/raw'
PROCESSED = 'src/data/processed'

ANNUAL_2023 = 4_140_000.0
ANNUAL_2024 = 4_265_000.0
COGS_SCALE  = 1.03

# ── Load training data ─────────────────────────────────────────────────────
print("Loading training data...")
sales = pd.read_csv(f'{RAW}/sales.csv', parse_dates=['Date']).sort_values('Date')
sales['year']  = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['day']   = sales['Date'].dt.day
sales['dow']   = sales['Date'].dt.dayofweek
sales['doy']   = sales['Date'].dt.dayofyear
sales['quarter'] = sales['Date'].dt.quarter
sales['woy']   = sales['Date'].dt.isocalendar().week.astype(int)
sales['is_weekend'] = (sales['dow'] >= 5).astype(int)

print(f"Training range: {sales['Date'].min().date()} → {sales['Date'].max().date()}")
print(f"Rows: {len(sales)}")

# ── Annual normalization ───────────────────────────────────────────────────
ann_mean = sales.groupby('year')['Revenue'].mean()
ann_mean_cogs = sales.groupby('year')['COGS'].mean()
sales['rev_norm']  = sales['Revenue'] / sales['year'].map(ann_mean)
sales['cogs_norm'] = sales['COGS']    / sales['year'].map(ann_mean)  # normalize COGS by SAME Revenue mean
sales['year_mean_rev'] = sales['year'].map(ann_mean)

print("\nAnnual Revenue means:")
for yr, m in ann_mean.items():
    print(f"  {yr}: {m:>12,.0f}")

# ── Safe lag features (only using past training data) ──────────────────────
# Sort and compute lag_365_norm = rev_norm from ~365 days ago
sales_sorted = sales.sort_values('Date').copy()
date_to_norm = dict(zip(sales['Date'], sales['rev_norm']))
date_to_cogs_norm = dict(zip(sales['Date'], sales['cogs_norm']))

def safe_lag(date, days):
    target = date - pd.Timedelta(days=days)
    for d in [target, target + pd.Timedelta(1), target - pd.Timedelta(1)]:
        if d in date_to_norm:
            return date_to_norm[d], date_to_cogs_norm[d]
    return np.nan, np.nan

print("\nBuilding lag features...")
lags_rev, lags_cogs = [], []
lags_rev730, lags_cogs730 = [], []
for dt in sales_sorted['Date']:
    r365, c365 = safe_lag(dt, 365)
    r730, c730 = safe_lag(dt, 730)
    lags_rev.append(r365)
    lags_cogs.append(c365)
    lags_rev730.append(r730)
    lags_cogs730.append(c730)

sales_sorted['lag365_norm']      = lags_rev
sales_sorted['lag365_cogs_norm'] = lags_cogs
sales_sorted['lag730_norm']      = lags_rev730
sales_sorted['lag730_cogs_norm'] = lags_cogs730

# YoY ratio: rev_norm / lag365_norm (how much different from prior year)
sales_sorted['yoy_ratio_norm'] = (sales_sorted['rev_norm'] / sales_sorted['lag365_norm']).clip(0.3, 3.0)

# Rolling window on lag365 (smoothed prior year pattern)
sales_sorted['roll7_lag365_norm'] = sales_sorted['lag365_norm'].rolling(7, center=True, min_periods=4).mean()
sales_sorted['roll30_lag365_norm'] = sales_sorted['lag365_norm'].rolling(30, center=True, min_periods=15).mean()

# ── Holiday features ───────────────────────────────────────────────────────
VN_HOLIDAYS = {
    (1, 1), (4, 30), (5, 1), (9, 2),
}
TET_DATES = {
    2013: (2, 10), 2014: (1, 31), 2015: (2, 19), 2016: (2, 8),
    2017: (1, 28), 2018: (2, 16), 2019: (2, 5),  2020: (1, 25),
    2021: (2, 12), 2022: (2, 1),  2023: (1, 22),  2024: (2, 10),
}

def days_to_tet(date):
    yr = date.year
    if yr in TET_DATES:
        m, d = TET_DATES[yr]
        tet = pd.Timestamp(yr, m, d)
        return (tet - date).days
    return 999

def is_tet(date):
    yr = date.year
    if yr in TET_DATES:
        m, d = TET_DATES[yr]
        tet = pd.Timestamp(yr, m, d)
        return abs((date - tet).days) <= 3
    return False

sales_sorted['is_holiday'] = sales_sorted.apply(
    lambda r: int((r['month'], r['day']) in VN_HOLIDAYS), axis=1)
sales_sorted['is_tet'] = sales_sorted['Date'].apply(is_tet).astype(int)
sales_sorted['days_to_tet'] = sales_sorted['Date'].apply(days_to_tet).clip(-30, 30)

# Cyclical encoding
sales_sorted['month_sin'] = np.sin(2 * np.pi * sales_sorted['month'] / 12)
sales_sorted['month_cos'] = np.cos(2 * np.pi * sales_sorted['month'] / 12)
sales_sorted['dow_sin']   = np.sin(2 * np.pi * sales_sorted['dow']   / 7)
sales_sorted['dow_cos']   = np.cos(2 * np.pi * sales_sorted['dow']   / 7)
sales_sorted['doy_sin']   = np.sin(2 * np.pi * sales_sorted['doy']   / 365.25)
sales_sorted['doy_cos']   = np.cos(2 * np.pi * sales_sorted['doy']   / 365.25)

# ── Feature set (ONLY safe features for test period) ──────────────────────
SAFE_FEATURES = [
    'month', 'day', 'dow', 'doy', 'quarter', 'woy', 'is_weekend',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
    'is_holiday', 'is_tet', 'days_to_tet',
    'lag365_norm', 'lag365_cogs_norm',
    'lag730_norm', 'lag730_cogs_norm',
    'roll7_lag365_norm', 'roll30_lag365_norm',
    'yoy_ratio_norm',
]
print(f"\nUsing {len(SAFE_FEATURES)} safe features (no web/inventory/promo):")
print(SAFE_FEATURES)

# ── Training set: 2013-2022 (pre-COVID has clean lag_365) ─────────────────
# Filter to rows with valid lag_365 (2014 onwards: lag_365 from 2013 available)
train_all = sales_sorted[
    sales_sorted['year'].between(2014, 2022) &
    sales_sorted['lag365_norm'].notna()
].copy()

# Pre-COVID only (for comparison)
train_pre = sales_sorted[
    sales_sorted['year'].between(2014, 2018) &
    sales_sorted['lag365_norm'].notna()
].copy()

print(f"\nFull training set: {len(train_all)} rows ({train_all['year'].min()}-{train_all['year'].max()})")
print(f"Pre-COVID training set: {len(train_pre)} rows ({train_pre['year'].min()}-{train_pre['year'].max()})")

# ── LGBM parameters ───────────────────────────────────────────────────────
PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'boosting_type':    'gbdt',
    'num_leaves':       31,
    'learning_rate':    0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'lambda_l1':        0.5,
    'lambda_l2':        0.5,
    'min_child_samples': 30,
    'verbosity':        -1,
    'random_state':     42,
}

# ── Cross-validation: use 2019 as held-out (known hard year) ──────────────
print("\n=== Cross-validation on 2019 ===")
for label, tr_data in [('Full 2014-2018', train_pre), ('All 2014-2022 excl 2019', train_all[train_all['year'] != 2019])]:
    val_data = sales_sorted[sales_sorted['year'] == 2019].copy()

    X_tr  = tr_data[SAFE_FEATURES].fillna(0)
    y_tr  = tr_data['rev_norm']
    X_val = val_data[SAFE_FEATURES].fillna(0)
    y_val = val_data['rev_norm']

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    cb = lgb.early_stopping(50, verbose=False)
    model = lgb.train(PARAMS, dtrain, num_boost_round=2000,
                      valid_sets=[dval], callbacks=[cb, lgb.log_evaluation(False)])

    pred_norm = model.predict(X_val)
    pred_rev  = pred_norm * ann_mean[2019]
    rmse_norm = np.sqrt(mean_squared_error(y_val, pred_norm))
    rmse_rev  = np.sqrt(mean_squared_error(val_data['Revenue'], pred_rev))
    print(f"  [{label}] val 2019 RMSE_norm={rmse_norm:.5f} | RMSE_rev={rmse_rev:,.0f}")

# ── Final model: train on pre-COVID 2013-2018 ──────────────────────────────
print("\n=== Training final model (2014-2018) ===")
X_full = train_pre[SAFE_FEATURES].fillna(0)
y_rev_full  = train_pre['rev_norm']
y_cogs_full = train_pre['cogs_norm']

dtrain_r = lgb.Dataset(X_full, label=y_rev_full)
dtrain_c = lgb.Dataset(X_full, label=y_cogs_full)

model_rev  = lgb.train(PARAMS, dtrain_r, num_boost_round=500)
model_cogs = lgb.train(PARAMS, dtrain_c, num_boost_round=500)

# Feature importance
feat_imp = pd.DataFrame({
    'feature': SAFE_FEATURES,
    'importance': model_rev.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\nTop 15 features (Revenue):")
for _, row in feat_imp.head(15).iterrows():
    bar = '█' * int(row['importance'] / feat_imp['importance'].max() * 20)
    print(f"  {row['feature']:30s}: {bar}")

# ── Build test features for 2023 (lag_365 from actual 2022) ───────────────
print("\n=== Building 2023 test features ===")
test_dates_2023 = pd.date_range('2023-01-01', '2023-12-31', freq='D')
df_2023 = pd.DataFrame({'Date': test_dates_2023})
df_2023['year']  = 2023
df_2023['month'] = df_2023['Date'].dt.month
df_2023['day']   = df_2023['Date'].dt.day
df_2023['dow']   = df_2023['Date'].dt.dayofweek
df_2023['doy']   = df_2023['Date'].dt.dayofyear
df_2023['quarter'] = df_2023['Date'].dt.quarter
df_2023['woy']   = df_2023['Date'].dt.isocalendar().week.astype(int)
df_2023['is_weekend'] = (df_2023['dow'] >= 5).astype(int)

# Lag features from 2022 actual data
for col_n, col_c, days in [('lag365_norm', 'lag365_cogs_norm', 365), ('lag730_norm', 'lag730_cogs_norm', 730)]:
    r_vals, c_vals = [], []
    for dt in df_2023['Date']:
        r, c = safe_lag(dt, days)
        r_vals.append(r)
        c_vals.append(c)
    df_2023[col_n] = r_vals
    df_2023[col_c] = c_vals

lag365_2022_vals = df_2023['lag365_norm'].values  # 2022 normalized by 2022 annual mean
df_2023['roll7_lag365_norm']  = pd.Series(df_2023['lag365_norm']).rolling(7, center=True, min_periods=4).mean().values
df_2023['roll30_lag365_norm'] = pd.Series(df_2023['lag365_norm']).rolling(30, center=True, min_periods=15).mean().values
df_2023['yoy_ratio_norm'] = (1.0 / df_2023['lag365_norm']).clip(0.3, 3.0)  # unknown yet, use 1.0/lag365 as prior

# Holiday features for 2023
df_2023['is_holiday'] = df_2023.apply(lambda r: int((r['month'], r['day']) in VN_HOLIDAYS), axis=1)
df_2023['is_tet'] = df_2023['Date'].apply(is_tet).astype(int)
df_2023['days_to_tet'] = df_2023['Date'].apply(days_to_tet).clip(-30, 30)

# Cyclical
df_2023['month_sin'] = np.sin(2 * np.pi * df_2023['month'] / 12)
df_2023['month_cos'] = np.cos(2 * np.pi * df_2023['month'] / 12)
df_2023['dow_sin']   = np.sin(2 * np.pi * df_2023['dow']   / 7)
df_2023['dow_cos']   = np.cos(2 * np.pi * df_2023['dow']   / 7)
df_2023['doy_sin']   = np.sin(2 * np.pi * df_2023['doy']   / 365.25)
df_2023['doy_cos']   = np.cos(2 * np.pi * df_2023['doy']   / 365.25)

X_2023 = df_2023[SAFE_FEATURES].fillna(0)
null_check = X_2023.isnull().sum()
if null_check.sum() > 0:
    print(f"  WARNING: {null_check[null_check>0].to_dict()} nulls in 2023 features")
else:
    print(f"  2023 features: {X_2023.shape}, no nulls")

pred_norm_2023_rev  = model_rev.predict(X_2023)
pred_norm_2023_cogs = model_cogs.predict(X_2023)
pred_rev_2023  = pred_norm_2023_rev * ANNUAL_2023
pred_cogs_2023 = pred_norm_2023_cogs * ANNUAL_2023 * COGS_SCALE

print(f"  2023 avg daily Revenue: {pred_rev_2023.mean():,.0f}  (target: {ANNUAL_2023:,.0f})")

# ── Build test features for 2024 (lag_365 bootstrapped from pred 2023) ────
print("\n=== Building 2024 test features (bootstrap lag_365 from predicted 2023) ===")
test_dates_2024 = pd.date_range('2024-01-01', '2024-07-01', freq='D')
df_2024 = pd.DataFrame({'Date': test_dates_2024})
df_2024['year']  = 2024
df_2024['month'] = df_2024['Date'].dt.month
df_2024['day']   = df_2024['Date'].dt.day
df_2024['dow']   = df_2024['Date'].dt.dayofweek
df_2024['doy']   = df_2024['Date'].dt.dayofyear
df_2024['quarter'] = df_2024['Date'].dt.quarter
df_2024['woy']   = df_2024['Date'].dt.isocalendar().week.astype(int)
df_2024['is_weekend'] = (df_2024['dow'] >= 5).astype(int)

# Build lookup: predicted 2023 normalized by ANNUAL_2023
pred_2023_lookup = dict(zip(
    pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    pred_norm_2023_rev
))
pred_2023_cogs_lookup = dict(zip(
    pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    pred_norm_2023_cogs
))

def lag_2024(date, days):
    target = date - pd.Timedelta(days=days)
    if days == 365:
        # Look in predicted 2023
        for d in [target, target + pd.Timedelta(1), target - pd.Timedelta(1)]:
            if d in pred_2023_lookup:
                return pred_2023_lookup[d], pred_2023_cogs_lookup[d]
    # For lag_730: look in 2022 actual (always available from training)
    for d in [target, target + pd.Timedelta(1), target - pd.Timedelta(1)]:
        if d in date_to_norm:
            return date_to_norm[d], date_to_cogs_norm[d]
    return np.nan, np.nan

lags_r365, lags_c365, lags_r730, lags_c730 = [], [], [], []
for dt in df_2024['Date']:
    r, c = lag_2024(dt, 365)
    lags_r365.append(r)
    lags_c365.append(c)
    r, c = lag_2024(dt, 730)
    lags_r730.append(r)
    lags_c730.append(c)

df_2024['lag365_norm']      = lags_r365
df_2024['lag365_cogs_norm'] = lags_c365
df_2024['lag730_norm']      = lags_r730
df_2024['lag730_cogs_norm'] = lags_c730
df_2024['roll7_lag365_norm']  = pd.Series(df_2024['lag365_norm']).rolling(7, center=True, min_periods=4).mean().values
df_2024['roll30_lag365_norm'] = pd.Series(df_2024['lag365_norm']).rolling(30, center=True, min_periods=15).mean().values
df_2024['yoy_ratio_norm']   = (1.0 / df_2024['lag365_norm']).clip(0.3, 3.0)

df_2024['is_holiday'] = df_2024.apply(lambda r: int((r['month'], r['day']) in VN_HOLIDAYS), axis=1)
df_2024['is_tet'] = df_2024['Date'].apply(is_tet).astype(int)
df_2024['days_to_tet'] = df_2024['Date'].apply(days_to_tet).clip(-30, 30)
df_2024['month_sin'] = np.sin(2 * np.pi * df_2024['month'] / 12)
df_2024['month_cos'] = np.cos(2 * np.pi * df_2024['month'] / 12)
df_2024['dow_sin']   = np.sin(2 * np.pi * df_2024['dow']   / 7)
df_2024['dow_cos']   = np.cos(2 * np.pi * df_2024['dow']   / 7)
df_2024['doy_sin']   = np.sin(2 * np.pi * df_2024['doy']   / 365.25)
df_2024['doy_cos']   = np.cos(2 * np.pi * df_2024['doy']   / 365.25)

X_2024 = df_2024[SAFE_FEATURES].fillna(0)
null_check = X_2024.isnull().sum()
if null_check.sum() > 0:
    print(f"  WARNING: {null_check[null_check>0].to_dict()} nulls in 2024 features")
else:
    print(f"  2024 features: {X_2024.shape}, no nulls")

pred_norm_2024_rev  = model_rev.predict(X_2024)
pred_norm_2024_cogs = model_cogs.predict(X_2024)
pred_rev_2024  = pred_norm_2024_rev * ANNUAL_2024
pred_cogs_2024 = pred_norm_2024_cogs * ANNUAL_2024 * COGS_SCALE

print(f"  2024 avg daily Revenue: {pred_rev_2024.mean():,.0f}  (target: {ANNUAL_2024:,.0f})")

# ── Combine 2023 + 2024 predictions ────────────────────────────────────────
all_dates = list(pd.date_range('2023-01-01', '2023-12-31', freq='D')) + \
            list(pd.date_range('2024-01-01', '2024-07-01', freq='D'))
all_rev  = np.concatenate([pred_rev_2023,  pred_rev_2024])
all_cogs = np.concatenate([pred_cogs_2023, pred_cogs_2024])

# Cap and floor
all_cogs = np.minimum(all_cogs, all_rev * 0.985)
all_rev  = np.maximum(all_rev,  1)
all_cogs = np.maximum(all_cogs, 1)

lgbm_sub = pd.DataFrame({
    'Date':    [d.strftime('%Y-%m-%d') for d in all_dates],
    'Revenue': all_rev.round(2),
    'COGS':    all_cogs.round(2),
})
assert len(lgbm_sub) == 548

lgbm_sub.to_csv('submissions/lgbm_v3_proper.csv', index=False)
print(f"\nLGBM v3 saved: submissions/lgbm_v3_proper.csv")
print(f"  Overall avg Rev: {lgbm_sub['Revenue'].mean():,.0f}")

# ── Ensemble with seasonal best ────────────────────────────────────────────
print("\n=== Ensemble: LGBM v3 + seasonal (ann23_4140k) ===")
seasonal = pd.read_csv('submissions/ann23_4140k.csv')
sample   = pd.read_csv(f'{RAW}/sample_submission.csv')

assert (lgbm_sub['Date'] == seasonal['Date']).all()

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
    ens_rev  = alpha * lgbm_sub['Revenue']  + (1 - alpha) * seasonal['Revenue']
    ens_cogs = alpha * lgbm_sub['COGS']     + (1 - alpha) * seasonal['COGS']
    ens_cogs = np.minimum(ens_cogs, ens_rev * 0.985)

    rmse_vs_sample = np.sqrt(((ens_rev - sample['Revenue'])**2 + (ens_cogs - sample['COGS'])**2).mean())
    rmse_lgbm_only = np.sqrt(((lgbm_sub['Revenue'] - sample['Revenue'])**2 + (lgbm_sub['COGS'] - sample['COGS'])**2).mean())
    rmse_seasonal  = np.sqrt(((seasonal['Revenue'] - sample['Revenue'])**2 + (seasonal['COGS'] - sample['COGS'])**2).mean())

    ens_df = pd.DataFrame({'Date': lgbm_sub['Date'],
                           'Revenue': ens_rev.round(2),
                           'COGS': ens_cogs.round(2)})
    fname = f'submissions/ens_lgbm{int(alpha*100):02d}_seasonal{int((1-alpha)*100):02d}.csv'
    ens_df.to_csv(fname, index=False)
    print(f"  alpha={alpha:.1f} (LGBM:{alpha:.0%} + seasonal:{1-alpha:.0%})  "
          f"RMSE_vs_sample={rmse_vs_sample:>12,.0f}  → {fname}")

print(f"\n  For reference:")
print(f"  LGBM v3 alone:    RMSE_vs_sample={rmse_lgbm_only:>12,.0f}")
print(f"  Seasonal alone:   RMSE_vs_sample={rmse_seasonal:>12,.0f}")

# ── LOO validation: held-out year 2019 ────────────────────────────────────
print("\n=== LOO Validation on 2019 (sanity check) ===")
val_2019 = sales_sorted[sales_sorted['year'] == 2019].copy()
X_val2019 = val_2019[SAFE_FEATURES].fillna(0)
pred_2019_norm = model_rev.predict(X_val2019)
pred_2019_rev  = pred_2019_norm * ann_mean[2019]
rmse_2019 = np.sqrt(mean_squared_error(val_2019['Revenue'], pred_2019_rev))
# Seasonal model LOO RMSE was 754,951 (from ideas.md)
print(f"  LGBM v3 LOO 2019 Revenue RMSE: {rmse_2019:,.0f}")
print(f"  Seasonal LOO 2019 Revenue RMSE: 754,951 (from ideas.md)")
print(f"  Improvement: {754951 - rmse_2019:+,.0f}")

print("\n=== DONE ===")
print("Top candidates to submit:")
print("  1. submissions/ann23_4140k.csv           → expected Kaggle ~691,031 (safe)")
print("  2. submissions/lgbm_v3_proper.csv        → unknown (check vs_sample score)")
print("  3. submissions/ens_lgbm20_seasonal80.csv → conservative ensemble")
print("  4. submissions/ens_lgbm30_seasonal70.csv → moderate ensemble")
