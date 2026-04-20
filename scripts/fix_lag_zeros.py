import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("src/data/raw")
PROCESSED = Path("src/data/processed")

# Load raw sales for lookup
sales = pd.read_csv(RAW / "sales.csv", parse_dates=["Date"])
sales = sales.sort_values("Date").reset_index(drop=True)

rev_lookup = sales.set_index("Date")["Revenue"]
cogs_lookup = sales.set_index("Date")["COGS"]


def safe_lag(dates, lookup, lag_days):
    lag_dates = pd.to_datetime(dates) - pd.Timedelta(days=lag_days)
    return lag_dates.map(lookup).values


for fname, df_path in [
    ("train", PROCESSED / "train_features.parquet"),
    ("test",  PROCESSED / "test_features.parquet"),
]:
    df = pd.read_parquet(df_path)
    dates = pd.to_datetime(df["Date"])

    # Before stats
    zero_before = (df["rev_lag_365"] == 0).sum()
    mean_before = df["rev_lag_365"].mean()
    print(f"{fname} BEFORE: rev_lag_365 zeros={zero_before}, mean={mean_before:,.0f}")

    # Recompute lag_730 from raw (authoritative)
    lag730_rev = safe_lag(dates, rev_lookup,  730)
    lag730_cogs = safe_lag(dates, cogs_lookup, 730)
    lag365_rev = safe_lag(dates, rev_lookup,  365)
    lag365_cogs = safe_lag(dates, cogs_lookup, 365)

    # median fallback (for dates before training start, e.g. 2012 needs lag_730 from 2010)
    median_rev = sales["Revenue"].median()
    median_cogs = sales["COGS"].median()

    # Apply: lag_365, fallback to lag_730, fallback to median
    df["rev_lag_365"] = np.where(~np.isnan(lag365_rev),  lag365_rev,
                                 np.where(~np.isnan(lag730_rev),  lag730_rev, median_rev))
    df["cogs_lag_365"] = np.where(~np.isnan(lag365_cogs), lag365_cogs,
                                  np.where(~np.isnan(lag730_cogs), lag730_cogs, median_cogs))
    df["rev_lag_730"] = np.where(
        ~np.isnan(lag730_rev),  lag730_rev,  median_rev)
    df["cogs_lag_730"] = np.where(
        ~np.isnan(lag730_cogs), lag730_cogs, median_cogs)

    # Also fix lag_364, lag_366
    if "rev_lag_364" in df.columns:
        lag364 = safe_lag(dates, rev_lookup, 364)
        df["rev_lag_364"] = np.where(
            ~np.isnan(lag364), lag364, df["rev_lag_365"].values)
    if "rev_lag_366" in df.columns:
        lag366 = safe_lag(dates, rev_lookup, 366)
        df["rev_lag_366"] = np.where(
            ~np.isnan(lag366), lag366, df["rev_lag_365"].values)

    # Fix rolling lag features if they have zeros
    if "rev_rolling7_lag365" in df.columns:
        zeros_mask = (df["rev_rolling7_lag365"] == 0)
        if zeros_mask.any():
            df.loc[zeros_mask,
                   "rev_rolling7_lag365"] = df.loc[zeros_mask, "rev_lag_365"]

    if "rev_rolling30_lag365" in df.columns:
        zeros_mask = (df["rev_rolling30_lag365"] == 0)
        if zeros_mask.any():
            df.loc[zeros_mask,
                   "rev_rolling30_lag365"] = df.loc[zeros_mask, "rev_lag_365"]

    # Verify
    zero_check = (df["rev_lag_365"] == 0).sum()
    min_val = df["rev_lag_365"].min()
    mean_val = df["rev_lag_365"].mean()
    print(f"{fname} AFTER:  rev_lag_365 zeros={zero_check}, min={min_val:,.0f}, mean={mean_val:,.0f}")

    # Full check all lag cols
    lag_cols = [c for c in df.columns if 'lag' in c]
    for col in lag_cols:
        z = (df[col] == 0).sum()
        if z > 0:
            print(f"  WARNING: {col} still has {z} zeros")

    df.to_parquet(df_path, index=False)
    print(f"  Saved {fname}: {df.shape}\n")

print("Done!")
