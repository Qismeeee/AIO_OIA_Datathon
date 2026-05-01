"""Baseline model script: Seasonal Naive (lag_365)"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, json
warnings.filterwarnings('ignore')

os.makedirs('../submissions', exist_ok=True)
os.makedirs('../report/figures', exist_ok=True)

PROCESSED = "../src/data/processed"
RAW = "../src/data/raw"

# Load data
train = pd.read_parquet(f"{PROCESSED}/train_features.parquet")
test  = pd.read_parquet(f"{PROCESSED}/test_features.parquet")

with open(f"{PROCESSED}/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train Revenue: mean={train['Revenue'].mean():,.0f}, std={train['Revenue'].std():,.0f}")

# Baseline 1 — Seasonal Naive (lag_365)
baseline_rev  = test["rev_lag_365"].values
baseline_cogs = test["cogs_lag_365"].values

# Sanity check and cap
ratio = baseline_cogs / np.maximum(baseline_rev, 1)
print(f"\nCOGS/Rev ratio in baseline: mean={ratio.mean():.4f}, max={ratio.max():.4f}")
baseline_cogs = np.minimum(baseline_cogs, baseline_rev * 0.99)

# Validate on training tail 2022
val = train[train["year"] == 2022].copy()
val_pred_rev  = val["rev_lag_365"].values
val_pred_cogs = val["cogs_lag_365"].values

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

mask = (val["rev_lag_365"] > 0) & (~np.isnan(val_pred_rev))
val_clean = val[mask]
pred_clean = val_pred_rev[mask]

rmse_rev  = rmse(val_clean["Revenue"].values, pred_clean)
mape_rev  = mape(val_clean["Revenue"].values, pred_clean)
rmse_cogs = rmse(val_clean["COGS"].values, val_pred_cogs[mask])

print(f"\nBaseline Validation (2022):")
print(f"  Revenue RMSE: {rmse_rev:>12,.0f}")
print(f"  Revenue MAPE: {mape_rev:>12.2f}%")
print(f"  COGS RMSE:    {rmse_cogs:>12,.0f}")

# Save baseline submission
baseline_sub = pd.DataFrame({
    "Date": test["Date"].dt.strftime("%Y-%m-%d"),
    "Revenue": np.maximum(baseline_rev, 0).round(2),
    "COGS":    np.maximum(baseline_cogs, 0).round(2),
})

sample = pd.read_csv(f"{RAW}/sample_submission.csv")
assert list(baseline_sub.columns) == list(sample.columns), "Column mismatch"
assert len(baseline_sub) == len(sample), f"Row mismatch: {len(baseline_sub)} vs {len(sample)}"

baseline_sub.to_csv("../submissions/baseline_lag365.csv", index=False)
print(f"\nBaseline submission saved: {len(baseline_sub)} rows")
print(baseline_sub.head(10))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

sample_dates = pd.to_datetime(sample["Date"])
axes[0].plot(sample_dates, sample["Revenue"], lw=1.0, label="Sample submission", color='steelblue')
axes[0].plot(pd.to_datetime(baseline_sub["Date"]), baseline_sub["Revenue"], lw=1.0,
             label="Baseline (lag_365)", color='orange', linestyle='--')
axes[0].set_title("Revenue: Baseline vs Sample Submission")
axes[0].legend()

axes[1].plot(sample_dates, sample["COGS"], lw=1.0, label="Sample submission", color='steelblue')
axes[1].plot(pd.to_datetime(baseline_sub["Date"]), baseline_sub["COGS"], lw=1.0,
             label="Baseline (lag_365)", color='orange', linestyle='--')
axes[1].set_title("COGS: Baseline vs Sample Submission")
axes[1].legend()

plt.tight_layout()
plt.savefig("../report/figures/07_baseline_vs_sample.png", dpi=120, bbox_inches="tight")
plt.close()
print("Figure saved.")

print("\n=== BASELINE DONE ===")
