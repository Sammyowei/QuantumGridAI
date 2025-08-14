"""
evaluate.py (Final Version with Metrics)
----------------------------------------
This script checks how good our trained AI model is at predicting electricity load.

What it does:
1. Loads the trained LSTM model and scaler
2. Makes predictions on unseen test data (future hours)
3. Saves predictions and actual values to a CSV file
4. Calculates error metrics: MAE, RMSE, MAPE
5. Plots a graph showing Actual vs Predicted load for 500 test hours

Think of this as giving the AI model its final "exam."
"""

# -----------------------------
# 0. Import Libraries
# -----------------------------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing import load_and_prepare_data

# -----------------------------
# 1. File Paths
# -----------------------------
RAW_DATA_PATH = "../data/raw/load_data.csv"             # Input dataset
MODEL_PATH = "../models/lstm_load_forecast.h5"          # Trained model (HDF5 legacy)
SCALER_PATH = "../models/scaler.pkl"                    # Scaler for inverse-transform
FORECAST_PATH = "../data/forecasts/test_forecast.csv"   # Output CSV

TIME_STEPS = 24  # AI looks at 24 past hours to predict the next hour

# -----------------------------
# 2. Load Data and Model
# -----------------------------
print("📂 Loading data, model, and scaler...")

# Prepare the dataset (scaled & windowed)
X, y, scaler = load_and_prepare_data(RAW_DATA_PATH, time_steps=TIME_STEPS)

# Split data: 70% training, 30% testing (same as in training)
split = int(0.7 * len(X))
X_test, y_test = X[split:], y[split:]

# Load the model safely (compile=False for legacy HDF5 format)
print("🤖 Loading trained AI model...")
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='mse')  # Compile for prediction

# Load the saved scaler
scaler = joblib.load(SCALER_PATH)

print(f"✅ Model and scaler loaded. Test samples: {X_test.shape[0]}")

# -----------------------------
# 3. Make Predictions
# -----------------------------
print("🔮 Generating AI forecasts...")
predictions = model.predict(X_test)

# Convert predictions and actual values back to MW
predicted_load = scaler.inverse_transform(predictions)
actual_load = scaler.inverse_transform(y_test)

# -----------------------------
# 4. Save Predictions to CSV
# -----------------------------
os.makedirs("../data/forecasts", exist_ok=True)

forecast_df = pd.DataFrame({
    "Actual_Load": actual_load.flatten(),
    "Predicted_Load": predicted_load.flatten()
})
forecast_df.to_csv(FORECAST_PATH, index=False)

print(f"✅ Forecast saved to {FORECAST_PATH}")

# -----------------------------
# 5. Calculate Error Metrics
# -----------------------------
# MAE: Average error in MW
mae = mean_absolute_error(actual_load, predicted_load)

# RMSE: Square root of mean squared error (penalizes big mistakes)
rmse = np.sqrt(mean_squared_error(actual_load, predicted_load))

# MAPE: Average % error
mape = np.mean(np.abs((actual_load - predicted_load) / actual_load)) * 100

print("\n📊 Model Performance Metrics:")
print(f"MAE  (Mean Absolute Error)      : {mae:.2f} MW")
print(f"RMSE (Root Mean Squared Error)  : {rmse:.2f} MW")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%\n")

# -----------------------------
# 6. Visualize First 500 Hours
# -----------------------------
print("📈 Plotting first 500 hours of test set...")

plt.figure(figsize=(12, 5))
plt.plot(actual_load[:500], label="Actual Load", color="blue")
plt.plot(predicted_load[:500], label="Predicted Load", color="orange")
plt.title("AI Electricity Load Forecast vs Actual")
plt.xlabel("Hour (Test Period)")
plt.ylabel("Load (MW)")
plt.legend()
plt.tight_layout()
plt.show()

print("🎉 Evaluation complete. Your AI is ready for simulation & quantum optimization!")
