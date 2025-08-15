"""
models/evaluate_models.py
-------------------------
- Evaluates LSTM + baselines on test set
- Saves metrics table + comparison plots
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from src.common_paths import TEST_CSV, SCALER_PATH, LSTM_MODEL, FORECASTS_DIR, TABLES_DIR, FIGURES_DIR
from src.utils.metrics import mae, rmse, mape
from src.utils.plotting import plot_prediction_vs_actual, plot_metrics_bar

def load_lstm_forecast():
    # Rebuild test sequences to infer predictions aligned to test range
    test_df = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])
    scaler = joblib.load(SCALER_PATH)
    model = keras.models.load_model(LSTM_MODEL, compile=False)

    from models.train_lstm import WINDOW, make_xy
    scaled = scaler.transform(test_df[["Load"]].values)
    X_test, y_test = make_xy(scaled.flatten(), WINDOW)
    yhat_scaled = model.predict(X_test, verbose=0).flatten()
    yhat = scaler.inverse_transform(yhat_scaled.reshape(-1,1)).flatten()

    # Align y_true to same length
    y_true = test_df["Load"].values[WINDOW:]
    return y_true, yhat

def main():
    # LSTM
    y_true, lstm_pred = load_lstm_forecast()

    # Baselines
    naive = pd.read_csv(FORECASTS_DIR / "baseline_naive.csv")["Predicted_Load"].values[-len(y_true):]
    linear = pd.read_csv(FORECASTS_DIR / "baseline_linear.csv")["Predicted_Load"].values[-len(y_true):]

    metrics = {}
    for name, pred in [("LSTM", lstm_pred), ("Naive", naive), ("Linear", linear)]:
        metrics[name] = {
            "MAE":  mae(y_true, pred),
            "RMSE": rmse(y_true, pred),
            "MAPE": mape(y_true, pred)
        }

    # Save metrics table
    mdf = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    mdf.to_csv(TABLES_DIR / "forecast_model_comparison.csv", index=False)

    # Plots
    plot_prediction_vs_actual(y_true, {"LSTM": lstm_pred, "Naive": naive, "Linear": linear},
                              FIGURES_DIR / "forecast_compare.png",
                              title="Forecast Comparison (Test Set)")
    plot_metrics_bar(metrics, FIGURES_DIR / "forecast_metrics_bar.png")

    print("✅ Model comparison complete. Tables and figures saved.")

if __name__ == "__main__":
    main()
