"""
cleaning_tools.py

Utilities for automatically cleaning tabular datasets:
- drop columns with too many missing values
- impute missing values
- handle duplicate rows
- detect (and optionally remove) outliers

Designed to be used by the Automatic Data Cleaning & Analysis Agent.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Default cleaning configuration
# =============================================================================

DEFAULT_CLEANING_CONFIG: Dict[str, Any] = {
    # If more than this fraction of a column is missing, drop the column
    "missing_threshold_drop_column": 0.6,   # e.g. 0.6 = 60%

    # Strategy for numeric imputation: "mean" or "median"
    "impute_numeric": "median",

    # Strategy for categorical imputation: "mode" or "constant"
    "impute_categorical": "mode",
    "impute_categorical_constant": "Unknown",

    # Remove duplicate rows?
    "remove_duplicates": True,

    # Outlier detection method & parameters
    "outlier_method": "iqr",               # currently only "iqr" implemented
    "outlier_iqr_multiplier": 1.5,

    # If True, drop rows that contain outliers
    # If False, only report them
    "remove_outliers": False,
}


# =============================================================================
# Config utilities
# =============================================================================

def merge_config(
    user_config: Optional[Dict[str, Any]],
    default: Dict[str, Any] = DEFAULT_CLEANING_CONFIG,
) -> Dict[str, Any]:
    """
    Merge user-provided config with default config.

    :param user_config: Optional partial config.
    :param default: Default config to start from.
    :return: Merged config dictionary.
    """
    if user_config is None:
        return default.copy()
    merged = default.copy()
    merged.update(user_config)
    return merged


# =============================================================================
# Missing values handling
# =============================================================================

def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with a fraction of missing values > threshold.

    :param df: Input DataFrame.
    :param threshold: Fraction of missing values above which to drop.
    :return: (reduced_df, list_of_dropped_columns)
    """
    missing_fraction = df.isna().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    if cols_to_drop:
        df_reduced = df.drop(columns=cols_to_drop)
    else:
        df_reduced = df.copy()
    return df_reduced, cols_to_drop


def impute_missing_values(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Impute missing values for numeric and categorical columns.

    For numeric columns:
        - strategy: "mean" or "median"
    For categorical columns:
        - strategy: "mode" or "constant"

    :param df: DataFrame to impute.
    :param config: Cleaning configuration.
    :return: (imputed_df, imputation_report)
    """
    df_imputed = df.copy()
    imputation_report: Dict[str, Dict[str, Any]] = {}

    numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_imputed.select_dtypes(exclude=np.number).columns.tolist()

    # --- Numeric imputation ---
    for col in numeric_cols:
        n_missing = int(df_imputed[col].isna().sum())
        if n_missing == 0:
            continue

        strategy = config["impute_numeric"]
        if strategy == "median":
            value = df_imputed[col].median()
        elif strategy == "mean":
            value = df_imputed[col].mean()
        else:
            raise ValueError(f"Unknown numeric imputation strategy: {strategy}")

        df_imputed[col].fillna(value, inplace=True)
        imputation_report[col] = {
            "type": "numeric",
            "strategy": strategy,
            "value_used": float(value),
            "n_imputed": n_missing,
        }

    # --- Categorical imputation ---
    for col in categorical_cols:
        n_missing = int(df_imputed[col].isna().sum())
        if n_missing == 0:
            continue

        strategy = config["impute_categorical"]
        if strategy == "mode":
            # If column is entirely NaN, fall back to constant
            if df_imputed[col].dropna().empty:
                value = config["impute_categorical_constant"]
            else:
                value = df_imputed[col].mode(dropna=True)[0]
        elif strategy == "constant":
            value = config["impute_categorical_constant"]
        else:
            raise ValueError(f"Unknown categorical imputation strategy: {strategy}")

        df_imputed[col].fillna(value, inplace=True)
        imputation_report[col] = {
            "type": "categorical",
            "strategy": strategy,
            "value_used": str(value),
            "n_imputed": n_missing,
        }

    return df_imputed, imputation_report


# =============================================================================
# Duplicate rows handling
# =============================================================================

def handle_duplicates(
    df: pd.DataFrame,
    remove: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Detect (and optionally remove) duplicate rows.

    :param df: Input DataFrame.
    :param remove: If True, remove duplicate rows.
    :return: (df_without_duplicates, number_of_duplicates_removed)
    """
    duplicated_mask = df.duplicated()
    n_duplicates = int(duplicated_mask.sum())

    if remove and n_duplicates > 0:
        df_dedup = df[~duplicated_mask].copy()
        n_removed = n_duplicates
    else:
        df_dedup = df.copy()
        n_removed = 0

    return df_dedup, n_removed


# =============================================================================
# Outlier detection (IQR)
# =============================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Detect outliers per numeric column using the IQR rule.

    For each numeric column:
        - compute Q1, Q3, IQR
        - define lower = Q1 - multiplier * IQR
        - define upper = Q3 + multiplier * IQR
        - any value outside [lower, upper] is considered an outlier

    :param df: DataFrame to inspect.
    :param multiplier: IQR multiplier (default 1.5).
    :return:
        dict where keys are column names and values contain:
          - lower_bound
          - upper_bound
          - indices (list of row indices)
          - n_outliers
    """
    outlier_indices: Dict[str, Dict[str, Any]] = {}
    numeric_df = df.select_dtypes(include=np.number)

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
        idx = numeric_df[mask].index.tolist()

        if idx:
            outlier_indices[col] = {
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "indices": idx,
                "n_outliers": len(idx),
            }

    return outlier_indices


# =============================================================================
# Main cleaning pipeline
# =============================================================================

def clean_dataset(
    df: pd.DataFrame,
    user_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full cleaning pipeline:
      1) Drop columns with too many missing values
      2) Impute remaining missing values
      3) Handle duplicate rows
      4) Detect (and optionally remove) outliers

    :param df: Raw input DataFrame.
    :param user_config: Optional dict to override default behavior.
    :return: (cleaned_df, cleaning_report)
    """
    config = merge_config(user_config, DEFAULT_CLEANING_CONFIG)
    report: Dict[str, Any] = {
        "dropped_columns": [],
        "imputations": {},
        "duplicates": {
            "n_duplicates_before": 0,
            "n_removed": 0,
        },
        "outliers": {
            "method": config["outlier_method"],
            "details": {},
            "n_rows_removed": 0,
        },
        "shape_before": df.shape,
        "shape_after": None,
    }

    # 1) Drop high-missing columns
    df_step, cols_dropped = drop_high_missing_columns(
        df,
        threshold=config["missing_threshold_drop_column"],
    )
    report["dropped_columns"] = cols_dropped

    # 2) Impute missing values
    df_step, imputation_report = impute_missing_values(df_step, config)
    report["imputations"] = imputation_report

    # 3) Handle duplicates
    n_duplicates_before = int(df_step.duplicated().sum())
    df_step, n_removed_dup = handle_duplicates(
        df_step,
        remove=config["remove_duplicates"],
    )
    report["duplicates"]["n_duplicates_before"] = n_duplicates_before
    report["duplicates"]["n_removed"] = n_removed_dup

    # 4) Detect outliers
    if config["outlier_method"] == "iqr":
        outlier_details = detect_outliers_iqr(
            df_step,
            multiplier=config["outlier_iqr_multiplier"],
        )
    else:
        raise ValueError(f"Unknown outlier method: {config['outlier_method']}")

    report["outliers"]["details"] = outlier_details

    # Optionally remove outliers
    if config["remove_outliers"] and outlier_details:
        all_indices = set()
        for info in outlier_details.values():
            all_indices.update(info["indices"])
        n_rows_outliers = len(all_indices)
        report["outliers"]["n_rows_removed"] = n_rows_outliers

        df_step = df_step.drop(index=list(all_indices))

    # Final shape
    report["shape_after"] = df_step.shape

    return df_step, report


# =============================================================================
# Optional: pretty-print helper for the report
# =============================================================================

def display_cleaning_report(report: Dict[str, Any]) -> None:
    """
    Nicely print a cleaning report dictionary to the console.
    """
    print("=== Cleaning Report ===")
    print(f"Shape before: {report['shape_before']}")
    print(f"Shape after : {report['shape_after']}")
    print("\nDropped columns:", report["dropped_columns"])

    print("\nImputations:")
    if report["imputations"]:
        for col, info in report["imputations"].items():
            print(f"  - {col}: {info}")
    else:
        print("  None")

    print("\nDuplicates:")
    print(f"  Duplicates before: {report['duplicates']['n_duplicates_before']}")
    print(f"  Duplicates removed: {report['duplicates']['n_removed']}")

    print("\nOutliers:")
    print(f"  Method: {report['outliers']['method']}")
    print(f"  Rows removed (if configured): {report['outliers']['n_rows_removed']}")
    print("  Details per column:")
    if report["outliers"]["details"]:
        for col, info in report["outliers"]["details"].items():
            print(f"    - {col}: {info['n_outliers']} outliers")
    else:
        print("    None")


__all__ = [
    "DEFAULT_CLEANING_CONFIG",
    "merge_config",
    "drop_high_missing_columns",
    "impute_missing_values",
    "handle_duplicates",
    "detect_outliers_iqr",
    "clean_dataset",
    "display_cleaning_report",
]
###------------------------------------------------------------------------------
def detect_file_type(path: str) -> str:
    """
    Detect the dataset file type (.csv or .xlsx).
    Returns: "csv" or "excel"
    Raises: ValueError if unsupported.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".csv":
        return "csv"
    if ext in [".xlsx", ".xls"]:
        return "excel"
    
    raise ValueError(f"Unsupported file type: {ext}")
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.
    Returns a pandas DataFrame.
    """
    file_type = detect_file_type(path)

    try:
        if file_type == "csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    return df

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze dataset structure and return a profiling dictionary.
    """
    
    profile = {}

    # Shape
    profile["shape"] = df.shape

    # Columns & data types
    profile["columns"] = list(df.columns)
    profile["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missing values
    profile["missing_values"] = df.isna().sum().to_dict()

    # Duplicate rows
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Basic stats (for numeric columns only)
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] > 0:
        profile["numeric_stats"] = numeric_df.describe().to_dict()
    else:
        profile["numeric_stats"] = {}

    return profile

def display_profile(profile: dict):
    """Nicely print profiling information."""
    
    print("=== Dataset Profile ===")
    print(f"Shape: {profile['shape']}")
    print("\nColumns:")
    for col in profile["columns"]:
        print(f"  - {col}  ({profile['dtypes'][col]})")

    print("\nMissing Values:")
    for col, mv in profile["missing_values"].items():
        print(f"  {col}: {mv}")

    print("\nDuplicate Rows:", profile["duplicate_rows"])

    if profile["numeric_stats"]:
        print("\nNumeric Stats:")
        display(pd.DataFrame(profile["numeric_stats"]))
    else:
        print("\n(No numeric columns detected)")



import os

os.makedirs("../data", exist_ok=True)
print("Folder 'data' ready.")