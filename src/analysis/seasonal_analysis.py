"""
analysis/seasonal_analysis.py
-----------------------------
- Yearly & Monthly error tables (uses test set actual vs model forecast)
- Seasonal boxplot of load
- Error trend per year
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.common_paths import TEST_CSV, FORECASTS_DIR, TABLES_DIR, FIGURES_DIR
from src.utils.metrics import mae, rmse, mape

def main():
    test_df = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])
    y_true = test_df["Load"].values

    # Use LSTM comparison output for alignment
    lstm_pred = pd.read_csv(FORECASTS_DIR / "test_forecast.csv")["Predicted_Load"].values
    n = min(len(y_true), len(lstm_pred))
    y_true = y_true[:n]; y_pred = lstm_pred[:n]

    df = test_df.iloc[:n].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    # Yearly metrics
    yearly = df.groupby("year").apply(
        lambda g: pd.Series({
            "MAE": mae(g["y_true"].values, g["y_pred"].values),
            "RMSE": rmse(g["y_true"].values, g["y_pred"].values),
            "MAPE": mape(g["y_true"].values, g["y_pred"].values)
        })
    ).reset_index()
    yearly.to_csv(TABLES_DIR / "yearly_metrics.csv", index=False)

    # Monthly metrics (across test period)
    monthly = df.groupby("month").apply(
        lambda g: pd.Series({
            "MAE": mae(g["y_true"].values, g["y_pred"].values),
            "RMSE": rmse(g["y_true"].values, g["y_pred"].values),
            "MAPE": mape(g["y_true"].values, g["y_pred"].values)
        })
    ).reset_index()
    monthly.to_csv(TABLES_DIR / "monthly_metrics.csv", index=False)

    # Seasonal boxplot of actual load by month
    plt.figure(figsize=(10,5))
    df.boxplot(column="y_true", by="month")
    plt.title("Seasonal Load Distribution (Test)"); plt.suptitle("")
    plt.xlabel("Month"); plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "seasonal_boxplot.png"); plt.close()

    # Error trend per year
    yearly.plot(x="year", y=["MAE","RMSE","MAPE"], kind="bar", figsize=(10,5), title="Yearly Error Trend")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "yearly_error_trend.png"); plt.close()

    print("✅ Seasonal analysis saved (tables + figures).")

if __name__ == "__main__":
    main()
