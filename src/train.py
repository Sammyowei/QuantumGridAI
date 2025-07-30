"""
train.py
--------
This file is like the 'coach' for our AI model.
It loads the electricity data, prepares it, 
trains the AI to predict the next hour's load, 
and saves the trained model for future use.
"""

import joblib
from data_preprocessing import load_and_prepare_data
from model import create_lstm_model
from tensorflow.keras.models import load_model
import os

# -----------------------------
# 1. Paths to important files
# -----------------------------
RAW_DATA_PATH = "../data/raw/load_data.csv"
MODEL_PATH = "../models/lstm_load_forecast.h5"
SCALER_PATH = "../models/scaler.pkl"

# How many past hours should the AI use to predict the next one?
TIME_STEPS = 24

# -----------------------------
# 2. Load and prepare the data
# -----------------------------
print("📂 Loading and preparing the dataset...")
X, y, scaler = load_and_prepare_data(RAW_DATA_PATH, time_steps=TIME_STEPS)

# Split the data: 70% for training, 30% for testing
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ Data ready. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -----------------------------
# 3. Train the LSTM model
# -----------------------------
print("🚀 Training the AI model to learn electricity patterns...")
model = create_lstm_model(time_steps=TIME_STEPS)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 4. Save the model and scaler
# -----------------------------
os.makedirs("../models", exist_ok=True)
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"✅ Training complete. Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")
