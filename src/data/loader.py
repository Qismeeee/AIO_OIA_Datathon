"""
Data Loading Module
Load & validate CSV files from Kaggle dataset
"""
import pandas as pd
import logging
from pathlib import Path
from src.config import CSV_FILES, RAW_DATA_DIR

logger = logging.getLogger(__name__)

def load_csv(file_key: str) -> pd.DataFrame:
    """
    Load CSV file by key
    
    Args:
        file_key: Key from CSV_FILES dict (e.g., 'sales', 'products')
    
    Returns:
        pd.DataFrame with parsed dates
    """
    filepath = CSV_FILES.get(file_key)
    if filepath is None:
        raise ValueError(f"Unknown file key: {file_key}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}\nMake sure to download from Kaggle!")
    
    logger.info(f"Loading {file_key} from {filepath}")
    
    # Read CSV with appropriate date parsing
    date_cols = []
    if file_key in ["sales", "sales_test"]:
        date_cols = ["Date"]
    elif file_key == "customers":
        date_cols = ["signup_date"]
    elif file_key == "orders":
        date_cols = ["order_date"]
    elif file_key == "promotions":
        date_cols = ["start_date", "end_date"]
    elif file_key == "shipments":
        date_cols = ["ship_date", "delivery_date"]
    elif file_key == "returns":
        date_cols = ["return_date"]
    elif file_key == "reviews":
        date_cols = ["review_date"]
    elif file_key == "inventory":
        date_cols = ["snapshot_date"]
    elif file_key == "web_traffic":
        date_cols = ["date"]
    
    df = pd.read_csv(filepath, parse_dates=date_cols)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def load_all_data() -> dict:
    """
    Load all required CSV files
    
    Returns:
        dict with all DataFrames
    """
    data = {}
    for file_key in CSV_FILES.keys():
        try:
            data[file_key] = load_csv(file_key)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {file_key}: {e}")
    
    return data


def validate_data_exists():
    """Check if all required CSV files exist"""
    missing = []
    for file_key, filepath in CSV_FILES.items():
        if not filepath.exists():
            missing.append(file_key)
    
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}\n"
            f"Please download from Kaggle: "
            f"https://www.kaggle.com/competitions/datathon-2026-round-1/data"
        )
    
    logger.info("✅ All required CSV files found!")
