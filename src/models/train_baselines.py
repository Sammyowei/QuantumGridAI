"""
models/train_baselines.py
-------------------------
Two baselines:
1) NaiveLast: y_hat(t) = y(t-1)
2) Linear Regression on lag features [1, 24, 168]
Saves forecasts to data/forecasts/*.csv
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from common_paths import TRAIN_CSV, TEST_CSV, SCALER_PATH, FORECASTS_DIR
from utils.time_features import make_lag_features
from utils.io import write_csv

def naive_last_forecast(test_series: np.ndarray) -> np.ndarray:
    # Predict each point as previous point; first prediction equals first value
    y = test_series
    yhat = np.roll(y, 1)
    yhat[0] = y[0]
    return yhat

def main():
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    test_df  = pd.read_csv(TEST_CSV,  parse_dates=["timestamp"])
    scaler = joblib.load(SCALER_PATH)

    # We'll train baseline models **on the unscaled space** for interpretability.
    # (LSTM uses scaled, baselines here use raw—fine for comparisons.)
    # --- NaiveLast
    test_y = test_df["Load"].values
    naive_pred = naive_last_forecast(test_y)
    write_csv(pd.DataFrame({"Predicted_Load": naive_pred}), FORECASTS_DIR / "baseline_naive.csv")

    # --- Linear Regression with lag features
    full = pd.concat([train_df, test_df], ignore_index=True)
    full_lag = make_lag_features(full, y_col="Load", lags=(1, 24, 168))
    # Split back: rows that belong to test period
    full_lag["is_test"] = full_lag["timestamp"] >= test_df["timestamp"].min()
    train_lag = full_lag[~full_lag["is_test"]]
    test_lag  = full_lag[full_lag["is_test"]]

    X_cols = ["Load_lag1", "Load_lag24", "Load_lag168"]
    lr = LinearRegression()
    lr.fit(train_lag[X_cols].values, train_lag["Load"].values)
    lr_pred = lr.predict(test_lag[X_cols].values)

    write_csv(pd.DataFrame({"Predicted_Load": lr_pred}), FORECASTS_DIR / "baseline_linear.csv")
    print("✅ Baseline forecasts saved: baseline_naive.csv, baseline_linear.csv")

if __name__ == "__main__":
    main()
