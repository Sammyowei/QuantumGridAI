"""
models/train_lstm.py
--------------------
- Builds sequences from processed train/test
- Scales with saved scaler
- Trains LSTM with checkpoint + early stopping
- Saves model as .keras
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from src.common_paths import TRAIN_CSV, TEST_CSV, SCALER_PATH, LSTM_MODEL, MODELS_DIR, FIGURES_DIR
from src.models.build_lstm import build_lstm_model
import matplotlib.pyplot as plt

WINDOW = 168  # 1 week history -> predict next hour

def make_xy(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def main():
    # Load data
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    test_df  = pd.read_csv(TEST_CSV,  parse_dates=["timestamp"])
    scaler   = joblib.load(SCALER_PATH)

    # Scale only the Load column
    train_scaled = scaler.transform(train_df[["Load"]].values)
    test_scaled  = scaler.transform(test_df[["Load"]].values)

    X_train, y_train = make_xy(train_scaled.flatten(), WINDOW)
    X_test,  y_test  = make_xy(test_scaled.flatten(),  WINDOW)

    # Build model
    model = build_lstm_model(WINDOW, 1)

    # Callbacks: save best + early stop
    ckpt_path = LSTM_MODEL
    ckpt = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    es   = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print("💾 Model will save automatically and stop early if it stops improving.")
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200, batch_size=256, verbose=1,
        callbacks=[ckpt, es]
    )

    # Plot training curve
    plt.figure(figsize=(8,4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.tight_layout()
    (FIGURES_DIR / "lstm_training_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "lstm_training_curve.png")
    plt.close()

    print(f"✅ LSTM saved to {ckpt_path}")

if __name__ == "__main__":
    main()
