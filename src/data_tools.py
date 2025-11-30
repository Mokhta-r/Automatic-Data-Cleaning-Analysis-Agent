# src/data_tools.py

import pandas as pd
import numpy as np
def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Profile the dataset: shape, columns, missing values, dtypes.
    """
    profile = {}
    profile["shape"] = df.shape
    profile["columns"] = list(df.columns)
    profile["dtypes"] = df.dtypes.astype(str).to_dict()
    profile["missing_values"] = df.isna().sum().to_dict()
    profile["duplicate_rows"] = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=np.number)
    profile["numeric_stats"] = numeric_df.describe().to_dict() if not numeric_df.empty else {}

    return profile
