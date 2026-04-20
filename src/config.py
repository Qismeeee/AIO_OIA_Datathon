import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent          
DATA_DIR = SRC_DIR / "data"            
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = SRC_DIR / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

# Create directories if not exist
for directory in [PROCESSED_DATA_DIR, SUBMISSIONS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============ DATA CONFIG ============
DATA_START_DATE = "2012-07-04"
DATA_END_DATE = "2022-12-31"
TRAIN_TEST_SPLIT_DATE = "2022-06-30"  

# Files
CSV_FILES = {
    "sales": RAW_DATA_DIR / "sales.csv",
    "sales_test": RAW_DATA_DIR / "sales_test.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "customers": RAW_DATA_DIR / "customers.csv",
    "orders": RAW_DATA_DIR / "orders.csv",
    "order_items": RAW_DATA_DIR / "order_items.csv",
    "promotions": RAW_DATA_DIR / "promotions.csv",
    "payments": RAW_DATA_DIR / "payments.csv",
    "shipments": RAW_DATA_DIR / "shipments.csv",
    "returns": RAW_DATA_DIR / "returns.csv",
    "reviews": RAW_DATA_DIR / "reviews.csv",
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "web_traffic": RAW_DATA_DIR / "web_traffic.csv",
    "geography": RAW_DATA_DIR / "geography.csv",
}

# Processed data
PROCESSED_FILES = {
    "daily_features": PROCESSED_DATA_DIR / "daily_features.parquet",
    "cv_splits": PROCESSED_DATA_DIR / "cv_splits.pickle",
    "train_data": PROCESSED_DATA_DIR / "train_data.parquet",
    "test_data": PROCESSED_DATA_DIR / "test_data.parquet",
}

# ============ MODEL CONFIG ============
RANDOM_SEED = 42
N_SPLITS = 5  # Time series cross-validation splits
TEST_SIZE = 0.2

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
}

LGBM_TRAIN_PARAMS = {
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
}

# Prophet hyperparameters
PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10,
    "interval_width": 0.95,
}

# SARIMA parameters (p,d,q)x(P,D,Q,s)
SARIMA_PARAMS = (1, 1, 1, 365)  # (p,d,q) x seasonal_period

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "prophet": 0.2,
    "lightgbm": 0.6,
    "sarima": 0.2,
}

# ============ FEATURE CONFIG ============
# Lag features (days)
LAG_DAYS = [1, 7, 30, 365]

# Rolling windows (days)
ROLLING_WINDOWS = [7, 30]

# Holiday dates (Vietnam)
VIETNAM_HOLIDAYS = {
    "2022-01-01": "New Year",
    "2022-02-01": "Lunar New Year",
    "2022-02-02": "Lunar New Year",
    "2022-02-03": "Lunar New Year",
    "2022-04-30": "Reunification Day",
    "2022-05-01": "International Labor Day",
    "2022-09-02": "National Day",
    "2022-03-08": "Women's Day",
    # Add more as needed
}

# ============ LOGGING & DEBUG ============
LOG_LEVEL = "INFO"
VERBOSE = True

# ============ SUBMISSION CONFIG ============
SUBMISSION_TEMPLATE = {
    "date": None,  # Will be filled with test dates
    "revenue": None,  # Will be filled with predictions
}

# ============ PERFORMANCE TARGETS ============
BASELINE_RMSE_TARGET = 20000  # Seasonal naive baseline
LGBM_RMSE_TARGET = 15000
ENSEMBLE_RMSE_TARGET = 12000

# ============ ENSEMBLE METHOD ============
ENSEMBLE_METHOD = "weighted_average"  # or "stacking", "voting"
