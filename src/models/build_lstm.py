"""
models/build_lstm.py
--------------------
Builds a small LSTM model for 1-step-ahead forecasting on sequences.
"""

from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(input_len: int, n_features: int = 1) -> keras.Model:
    """
    input_len: window size (timesteps)
    n_features: number of features per timestep (we use 1: just Load)
    """
    inp = keras.Input(shape=(input_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)  # predict next Load
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.MeanSquaredError(name="mse")])
    return model
