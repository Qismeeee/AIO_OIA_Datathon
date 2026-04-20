import pandas as pd
from datetime import datetime

# Vietnam holidays
VIETNAM_HOLIDAYS = {
    "2012-01-01": "New Year",
    "2012-01-23": "Lunar New Year",
    "2012-01-24": "Lunar New Year",
    "2012-01-25": "Lunar New Year",

    "2013-02-10": "Lunar New Year",
    "2013-02-11": "Lunar New Year",
    "2013-02-12": "Lunar New Year",

    "2014-01-31": "Lunar New Year",
    "2014-02-01": "Lunar New Year",
    "2014-02-02": "Lunar New Year",

    "2015-02-19": "Lunar New Year",
    "2015-02-20": "Lunar New Year",
    "2015-02-21": "Lunar New Year",

    "2016-02-08": "Lunar New Year",
    "2016-02-09": "Lunar New Year",
    "2016-02-10": "Lunar New Year",

    "2017-01-28": "Lunar New Year",
    "2017-01-29": "Lunar New Year",
    "2017-01-30": "Lunar New Year",

    "2018-02-16": "Lunar New Year",
    "2018-02-17": "Lunar New Year",
    "2018-02-18": "Lunar New Year",

    "2019-02-05": "Lunar New Year",
    "2019-02-06": "Lunar New Year",
    "2019-02-07": "Lunar New Year",

    "2020-01-25": "Lunar New Year",
    "2020-01-26": "Lunar New Year",
    "2020-01-27": "Lunar New Year",

    "2021-02-12": "Lunar New Year",
    "2021-02-13": "Lunar New Year",
    "2021-02-14": "Lunar New Year",

    "2022-02-01": "Lunar New Year",
    "2022-02-02": "Lunar New Year",
    "2022-02-03": "Lunar New Year",

    # Fixed holidays
    **{f"{year}-04-30": "Reunification Day" for year in range(2012, 2023)},
    **{f"{year}-05-01": "International Labor Day" for year in range(2012, 2023)},
    **{f"{year}-09-02": "National Day" for year in range(2012, 2023)},
    **{f"{year}-03-08": "Women's Day" for year in range(2012, 2023)},
}


def add_holiday_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Add holiday-related features

    Args:
        df: Input DataFrame with date column
        date_col: Name of date column

    Returns:
        DataFrame with added holiday features
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Check if date is holiday
    df["is_holiday"] = df[date_col].dt.strftime("%Y-%m-%d").map(
        lambda x: 1 if x in VIETNAM_HOLIDAYS else 0
    )

    # Holiday name (nullable)
    df["holiday_name"] = df[date_col].dt.strftime("%Y-%m-%d").map(
        VIETNAM_HOLIDAYS
    )

    # Days until/since holiday
    holiday_dates = pd.to_datetime(list(VIETNAM_HOLIDAYS.keys()))

    def days_to_next_holiday(date):
        future_holidays = holiday_dates[holiday_dates > date]
        if len(future_holidays) == 0:
            return 365  # Large number if no holiday in future
        return (future_holidays.min() - pd.Timestamp(date)).days

    df["days_to_holiday"] = df[date_col].apply(days_to_next_holiday)
    df["is_near_holiday"] = (df["days_to_holiday"] <= 7).astype(int)

    return df


def get_holiday_impact_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Advanced holiday features: holiday season, cluster patterns

    Args:
        df: Input DataFrame
        date_col: Name of date column

    Returns:
        DataFrame with advanced holiday features
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Pre-compute holiday dates before nested function uses it
    holiday_dates = pd.to_datetime(list(VIETNAM_HOLIDAYS.keys()))

    # Rolling window: is there a holiday within X days?
    df["holiday_in_7days"] = df["is_holiday"].rolling(
        window=7, min_periods=1).max().astype(int)
    df["holiday_in_14days"] = df["is_holiday"].rolling(
        window=14, min_periods=1).max().astype(int)

    # Post-holiday recovery (days since last holiday)
    def days_since_last_holiday(date):
        past_holidays = holiday_dates[holiday_dates <= date]
        if len(past_holidays) == 0:
            return 1000  # Large number if no past holiday
        return (pd.Timestamp(date) - past_holidays.max()).days

    df["days_since_holiday"] = df[date_col].apply(days_since_last_holiday)

    return df
