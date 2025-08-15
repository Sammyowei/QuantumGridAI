"""
utils/time_features.py
----------------------
Adds time-based features (hour, dayofweek, month, etc.) and lag features.
"""

import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df["hour"] = df[ts_col].dt.hour
    df["dayofweek"] = df[ts_col].dt.dayofweek
    df["month"] = df[ts_col].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def make_lag_features(df: pd.DataFrame, y_col: str = "Load", lags=(1, 24, 168)) -> pd.DataFrame:
    df = df.copy()
    for L in lags:
        df[f"{y_col}_lag{L}"] = df[y_col].shift(L)
    df.dropna(inplace=True)
    return df
