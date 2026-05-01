"""
Temporal Feature Engineering
Extract time-based features (day_of_week, seasonality, etc.)
"""
import pandas as pd
import numpy as np
from datetime import datetime

def add_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Add temporal features to DataFrame
    
    Args:
        df: Input DataFrame with date column
        date_col: Name of date column
    
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Basic temporal features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["day_of_month"] = df[date_col].dt.day
    df["day_of_week"] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week
    
    # Categorical temporal features
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # Saturday=5, Sunday=6
    df["is_month_start"] = df["day_of_month"].isin([1, 2, 3]).astype(int)
    df["is_month_end"] = df["day_of_month"].isin([28, 29, 30, 31]).astype(int)
    df["is_quarter_start"] = df["day_of_year"].isin([1, 91, 182, 273]).astype(int)
    df["is_quarter_end"] = df["day_of_year"].isin([90, 181, 272, 365, 366]).astype(int)
    
    # Seasonality encoding (circular)
    # Month seasonality (12-month cycle)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Day of week seasonality (7-day cycle)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Day of year seasonality (365-day cycle)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    return df


def get_day_name(day_of_week: int) -> str:
    """Get day name from day_of_week (0=Monday)"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[day_of_week]


def get_month_name(month: int) -> str:
    """Get month name from month number"""
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return months[month]
