"""
Data Validation Module
Check data integrity, FK relationships, constraints
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def check_null_values(df: pd.DataFrame, name: str, threshold: float = 0.5) -> dict:
    """
    Check for null values in DataFrame
    
    Args:
        df: DataFrame to check
        name: Name of dataset (for logging)
        threshold: Alert if null% > threshold
    
    Returns:
        dict with null statistics
    """
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    
    result = {
        "columns_with_nulls": [],
        "high_null_columns": [],
    }
    
    for col in df.columns:
        if null_counts[col] > 0:
            result["columns_with_nulls"].append({
                "column": col,
                "count": int(null_counts[col]),
                "percentage": float(null_pcts[col])
            })
            
            if null_pcts[col] > threshold * 100:
                result["high_null_columns"].append(col)
    
    if result["high_null_columns"]:
        logger.warning(
            f"[{name}] High null columns: {result['high_null_columns']}"
        )
    
    return result


def check_foreign_keys(df_child: pd.DataFrame, 
                       df_parent: pd.DataFrame,
                       child_col: str,
                       parent_col: str,
                       child_name: str = "child",
                       parent_name: str = "parent") -> bool:
    """
    Check referential integrity (FK constraint)
    
    Args:
        df_child: Child DataFrame (has FK)
        df_parent: Parent DataFrame (has PK)
        child_col: Column name in child table
        parent_col: Column name in parent table
        child_name: Name for logging
        parent_name: Name for logging
    
    Returns:
        True if all FK values exist in parent, False otherwise
    """
    missing_refs = set(df_child[child_col].unique()) - set(df_parent[parent_col].unique())
    
    if missing_refs:
        logger.warning(
            f"[FK Check] {len(missing_refs)} missing references in "
            f"{parent_name}.{parent_col} from {child_name}.{child_col}"
        )
        return False
    
    logger.info(f"✅ FK Check passed: {child_name}.{child_col} → {parent_name}.{parent_col}")
    return True


def check_date_ranges(df: pd.DataFrame, date_col: str, name: str):
    """
    Check date column for consistency
    
    Args:
        df: DataFrame to check
        date_col: Date column name
        name: Name for logging
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    logger.info(f"[{name}] {date_col} range: {min_date} to {max_date}")
    
    # Check for duplicates
    duplicates = df[date_col].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"[{name}] {duplicates} duplicate dates in {date_col}")
    
    return {"min": min_date, "max": max_date, "duplicates": duplicates}


def check_numeric_ranges(df: pd.DataFrame, col: str, name: str) -> dict:
    """
    Check numeric column for outliers
    
    Args:
        df: DataFrame to check
        col: Numeric column name
        name: Name for logging
    
    Returns:
        dict with statistics
    """
    stats = {
        "min": df[col].min(),
        "max": df[col].max(),
        "mean": df[col].mean(),
        "median": df[col].median(),
        "std": df[col].std(),
        "negative_count": (df[col] < 0).sum(),
        "zero_count": (df[col] == 0).sum(),
    }
    
    if stats["negative_count"] > 0:
        logger.warning(f"[{name}] {col} has {stats['negative_count']} negative values")
    
    return stats


def run_full_validation(data: dict) -> dict:
    """
    Run comprehensive data validation
    
    Args:
        data: dict of DataFrames from load_all_data()
    
    Returns:
        dict with validation results
    """
    results = {}
    
    logger.info("=" * 60)
    logger.info("STARTING FULL DATA VALIDATION")
    logger.info("=" * 60)
    
    # 1. Check main sales data
    if "sales" in data:
        logger.info("\n[SALES] Validating...")
        results["sales_nulls"] = check_null_values(data["sales"], "sales")
        results["sales_date_range"] = check_date_ranges(data["sales"], "Date", "sales")
        results["sales_revenue"] = check_numeric_ranges(data["sales"], "Revenue", "sales")
    
    # 2. Check FK relationships
    logger.info("\n[FOREIGN KEYS] Validating...")
    if "orders" in data and "customers" in data:
        check_foreign_keys(
            data["orders"], data["customers"],
            "customer_id", "customer_id",
            "orders", "customers"
        )
    
    if "order_items" in data and "orders" in data:
        check_foreign_keys(
            data["order_items"], data["orders"],
            "order_id", "order_id",
            "order_items", "orders"
        )
    
    if "order_items" in data and "products" in data:
        check_foreign_keys(
            data["order_items"], data["products"],
            "product_id", "product_id",
            "order_items", "products"
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DATA VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    return results
