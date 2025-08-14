"""
predict_future.py
-----------------
- Predicts future electrical load using trained LSTM model
- Saves future forecast to CSV
- Can auto-run QUBO optimizer for future dispatch

Usage:
    python predict_future.py
    # Enter horizon (hours) when prompted (e.g., 168 for 1 week)
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# 1. File Paths
# -----------------------------
MODEL_PATH = "../models/lstm_load_forecast.keras"      # trained LSTM model
SCALER_PATH = "../models/scaler.pkl"                # fitted MinMax scaler
DATA_PATH = "../data/raw/load_data.csv"      # historical data used for training
FUTURE_FORECAST_PATH = "../data/forecasts/future_forecast.csv"

os.makedirs("../data/forecasts", exist_ok=True)

# -----------------------------
# 2. Load Model & Scaler
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Trained LSTM model & scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit(1)

# -----------------------------
# 3. Load Historical Data
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
    load_series = df["Load"].values  # Column name depends on your processed dataset
    print(f"✅ Loaded historical load data: {len(load_series)} hours.")
except Exception as e:
    print(f"❌ Error loading historical data: {e}")
    exit(1)

# -----------------------------
# 4. Configure Forecast Horizon
# -----------------------------
try:
    horizon = int(input("Enter future forecast horizon (hours, e.g., 168 for 1 week): ") or 168)
except ValueError:
    print("⚠ Invalid input. Defaulting to 168 hours (1 week).")
    horizon = 168

LOOK_BACK = 24  # same as training window

# -----------------------------
# 5. Prepare Initial Window 
# -----------------------------
scaled_series = scaler.transform(load_series.reshape(-1, 1))
last_window = scaled_series[-LOOK_BACK:]  # last 24 hours
uuu
future_predictions_scaled = []

for step in range(horizon):
    # Reshape for LSTM: (batch_size, timesteps, features)
    x_input = last_window.reshape(1, LOOK_BACK, 1)
    next_scaled = model.predict(x_input, verbose=0)[0][0]

    # Save prediction
    future_predictions_scaled.append(next_scaled)

    # Slide window
    last_window = np.append(last_window[1:], [[next_scaled]], axis=0)

# Inverse transform to MW
future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()

# -----------------------------
# 6. Save Future Forecast
# -----------------------------
future_df = pd.DataFrame({
    "Hour": np.arange(1, horizon + 1),
    "Predicted_Load": future_predictions
})
future_df.to_csv(FUTURE_FORECAST_PATH, index=False)

print(f"📄 Future forecast saved to {FUTURE_FORECAST_PATH}")
print(f"🔮 Forecasted {horizon} hours. Example:")
print(future_df.head())

# -----------------------------
# 7. Optional: Trigger QUBO Optimization
# -----------------------------
run_qubo = input("Do you want to run QUBO optimization for this forecast? (y/n): ").lower()
if run_qubo == "y":
    try:
        print("⚡ Running QUBO optimization for future forecast...")
        os.system("python quantum_optimizer.py")
    except Exception as e:
        print(f"❌ Could not run QUBO optimizer: {e}")

print("🎉 Future load prediction pipeline complete!")
