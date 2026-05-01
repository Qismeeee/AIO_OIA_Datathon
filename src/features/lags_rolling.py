"""
Lag and Rolling Window Features
Historical patterns for time series forecasting
"""
import pandas as pd
import numpy as np
from src.config import LAG_DAYS, ROLLING_WINDOWS


def add_lag_features(df: pd.DataFrame, 
                    target_col: str = "Revenue",
                    lag_days: list = None) -> pd.DataFrame:
    """
    Add lag features for time series
    
    Args:
        df: DataFrame with time series data (must be sorted by date)
        target_col: Column to create lags from
        lag_days: List of lag periods (default from config)
    
    Returns:
        DataFrame with lag features
    """
    if lag_days is None:
        lag_days = LAG_DAYS
    
    df = df.copy()
    
    # Create lag features
    for lag in lag_days:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame,
                        target_col: str = "Revenue",
                        rolling_windows: list = None) -> pd.DataFrame:
    """
    Add rolling window features
    
    Args:
        df: DataFrame with time series data
        target_col: Column to compute rolling stats from
        rolling_windows: List of window sizes (default from config)
    
    Returns:
        DataFrame with rolling features
    """
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS
    
    df = df.copy()
    
    for window in rolling_windows:
        # Mean
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Std
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(
            window=window, min_periods=1
        ).std()
        
        # Min/Max
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(
            window=window, min_periods=1
        ).min()
        
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(
            window=window, min_periods=1
        ).max()
    
    return df


def add_expanding_features(df: pd.DataFrame, target_col: str = "Revenue") -> pd.DataFrame:
    """
    Add expanding window features (cumulative statistics)
    
    Args:
        df: DataFrame with time series data
        target_col: Column to compute expanding stats from
    
    Returns:
        DataFrame with expanding features
    """
    df = df.copy()
    
    df[f"{target_col}_expanding_mean"] = df[target_col].expanding().mean()
    df[f"{target_col}_expanding_std"] = df[target_col].expanding().std()
    
    return df


def add_seasonal_lags(df: pd.DataFrame,
                     target_col: str = "Revenue",
                     seasonal_period: int = 365) -> pd.DataFrame:
    """
    Add seasonal lag features (same day last year, last week, last month)
    
    Args:
        df: DataFrame with time series data
        target_col: Column to create seasonal lags from
        seasonal_period: Period of seasonality (365 for daily, 52 for weekly)
    
    Returns:
        DataFrame with seasonal lag features
    """
    df = df.copy()
    
    # Yearly seasonality (365 days)
    df[f"{target_col}_seasonal_lag_365"] = df[target_col].shift(365)
    
    # Monthly seasonality (30 days)
    df[f"{target_col}_seasonal_lag_30"] = df[target_col].shift(30)
    
    # Weekly seasonality (7 days)
    df[f"{target_col}_seasonal_lag_7"] = df[target_col].shift(7)
    
    return df


def add_diff_features(df: pd.DataFrame,
                     target_col: str = "Revenue",
                     periods: list = [1, 7, 365]) -> pd.DataFrame:
    """
    Add difference features (momentum/velocity)
    
    Args:
        df: DataFrame with time series data
        target_col: Column to compute differences from
        periods: List of difference periods
    
    Returns:
        DataFrame with diff features
    """
    df = df.copy()
    
    for period in periods:
        # First difference
        df[f"{target_col}_diff_{period}"] = df[target_col].diff(period)
        
        # Percentage change
        df[f"{target_col}_pct_change_{period}"] = df[target_col].pct_change(period)
    
    return df


def add_interaction_features(df: pd.DataFrame,
                            col1: str = "Revenue",
                            col2: str = "web_traffic_sessions") -> pd.DataFrame:
    """
    Add interaction features between variables
    
    Args:
        df: DataFrame with both columns
        col1: First column name
        col2: Second column name
    
    Returns:
        DataFrame with interaction features
    """
    df = df.copy()
    
    if col1 in df.columns and col2 in df.columns:
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)  # Avoid division by zero
    
    return df
