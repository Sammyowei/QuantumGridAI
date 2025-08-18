import pandas as pd
import numpy as np

def make_lag_features(df: pd.DataFrame, target_col: str = None, y_col: str = None, lags=(1,)) -> pd.DataFrame:
    """
    Create lag features for a time series DataFrame while preserving the timestamp column.

    Args:
        df (pd.DataFrame): Must include 'timestamp' and target_col / y_col.
        target_col (str): Name of the target column (e.g., 'Load').
        y_col (str): Alias for target_col for backward compatibility.
        lags (tuple or list): Lag steps to create.

    Returns:
        pd.DataFrame: DataFrame with lag features added.
    """
    # Allow y_col as alias for target_col
    if target_col is None and y_col is not None:
        target_col = y_col
    elif target_col is None and y_col is None:
        raise ValueError("Either target_col or y_col must be provided.")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    # Drop rows with NaN values caused by lagging
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_datetime_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Add common datetime features for machine learning models.

    Args:
        df (pd.DataFrame): DataFrame with timestamp column.
        timestamp_col (str): Column name containing timestamps.

    Returns:
        pd.DataFrame: DataFrame with extra datetime feature columns.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df["hour"] = df[timestamp_col].dt.hour
    df["day"] = df[timestamp_col].dt.day
    df["weekday"] = df[timestamp_col].dt.weekday
    df["month"] = df[timestamp_col].dt.month
    df["year"] = df[timestamp_col].dt.year
    df["dayofyear"] = df[timestamp_col].dt.dayofyear
    df["weekofyear"] = df[timestamp_col].dt.isocalendar().week.astype(int)

    return df


def normalize_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Normalize numeric features to range [0, 1].

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list): List of column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized features.
    """
    df = df.copy()
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    return df


def difference_series(df: pd.DataFrame, target_col: str, periods: int = 1) -> pd.DataFrame:
    """
    Difference a series to make it stationary.

    Args:
        df (pd.DataFrame): DataFrame with target column.
        target_col (str): Column name of target series.
        periods (int): Number of periods to difference.

    Returns:
        pd.DataFrame: DataFrame with differenced column.
    """
    df = df.copy()
    df[f"{target_col}_diff"] = df[target_col].diff(periods=periods)
    df.dropna(inplace=True)
    return df
