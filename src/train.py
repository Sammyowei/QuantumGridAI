"""
train.py
--------
This file teaches our AI model to predict electricity load.

Imagine the AI as a student:
- We show it the last 24 hours of electricity usage (the story)
- It tries to guess what happens in the next hour (the test)
- We repeat this thousands of times until it gets smart

Features of this training script:
1. It saves the model after every 'lesson' (epoch)
2. It stops automatically if the AI stops improving
3. It can continue training from where it left off
"""

import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_preprocessing import load_and_prepare_data
from model import create_lstm_model

# -----------------------------
# 1. File Paths
# -----------------------------
RAW_DATA_PATH = "../data/raw/load_data.csv"        # Our 20-year dataset
MODEL_PATH = "../models/lstm_load_forecast.h5"     # Where the AI brain is saved
SCALER_PATH = "../models/scaler.pkl"               # Save the scaler to handle new data

TIME_STEPS = 24      # AI looks at 24 past hours
BATCH_SIZE = 32      # Larger batch size = faster training
MAX_EPOCHS = 200     # Maximum training rounds

# -----------------------------
# 2. Load and Prepare Data
# -----------------------------
print("📂 Loading and preparing the 20-year dataset...")

X, y, scaler = load_and_prepare_data(RAW_DATA_PATH, time_steps=TIME_STEPS)

# Split the data: 70% for training, 30% for testing
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ Data ready. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# -----------------------------
# 3. Load or Create the Model
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("ℹ️ Found an existing model. Loading it to continue training...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 No saved model found. Creating a new LSTM model...")
    model = create_lstm_model(time_steps=TIME_STEPS)

# -----------------------------
# 4. Create Checkpoints and Early Stopping
# -----------------------------
# Checkpoint: Save the best model after each epoch
checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    save_best_only=True,        # Only keep the best model
    monitor="val_loss",         # Compare using validation loss
    mode="min",
    verbose=1
)

# Early stopping: Stop if no improvement for 10 epochs
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,                # Wait 10 epochs for improvement
    restore_best_weights=True,  # Keep the best version
    verbose=1
)

# -----------------------------
# 5. Train the Model
# -----------------------------
print(f"🎯 Starting training for up to {MAX_EPOCHS} epochs...")
print("💾 Model will save automatically and stop early if it stops improving.")

history = model.fit(
    X_train, y_train,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# -----------------------------
# 6. Save the Scaler
# -----------------------------
# Only save once (scaler never changes)
if not os.path.exists(SCALER_PATH):
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")

print("🎉 Training complete. Best model saved in models/")
