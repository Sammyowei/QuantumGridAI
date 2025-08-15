"""
models/predict_future.py
------------------------
- Uses trained LSTM + scaler
- Takes last WINDOW hours from full series to predict next H steps
- Saves to data/forecasts/future_forecast_H<steps>.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from tensorflow import keras
from common_paths import TRAIN_CSV, TEST_CSV, SCALER_PATH, LSTM_MODEL, FORECASTS_DIR

from models.train_lstm import WINDOW

def rolling_predict(last_series_scaled: np.ndarray, model, steps: int) -> np.ndarray:
    """
    Greedy roll-out: each new prediction is appended to the window and used
    to predict the next step.
    """
    history = last_series_scaled.copy().reshape(-1).tolist()  # 1D list
    preds = []
    for _ in range(steps):
        window = np.array(history[-WINDOW:]).reshape(1, WINDOW, 1)
        pred_scaled = model.predict(window, verbose=0)[0,0]
        preds.append(pred_scaled)
        history.append(pred_scaled)
    return np.array(preds)

def main(horizon: int):
    # Load full (train+test) to grab the latest timestamp and tail window
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    test_df  = pd.read_csv(TEST_CSV,  parse_dates=["timestamp"])
    full = pd.concat([train_df, test_df], ignore_index=True)

    scaler = joblib.load(SCALER_PATH)
    model  = keras.models.load_model(LSTM_MODEL, compile=False)

    # Scale full load
    full_scaled = scaler.transform(full[["Load"]].values).flatten()

    # Predict future
    preds_scaled = rolling_predict(full_scaled[-WINDOW:], model, steps=horizon)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

    # Build future timestamps
    last_ts = full["timestamp"].iloc[-1]
    future_idx = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="H")
    out = pd.DataFrame({"timestamp": future_idx, "Predicted_Load": preds})
    out_path = FORECASTS_DIR / f"future_forecast_H{horizon}.csv"
    out.to_csv(out_path, index=False)

    print(f"✅ Future forecast saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=168, help="Number of hours to predict into the future")
    args = ap.parse_args()
    main(args.horizon)
