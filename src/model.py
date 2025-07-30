"""
model.py
--------
This file builds the AI brain (LSTM model) that will predict electricity demand.

LSTM stands for "Long Short-Term Memory" — it is a smart type of AI 
that is very good at learning patterns over time, like:

- Morning electricity usage goes up
- Afternoon is moderate
- Evening peaks when everyone comes home

We give it 24 hours, and it tries to predict the 25th hour.
"""

import tensorflow as tf

def create_lstm_model(time_steps=24, features=1):
    """
    Create and compile a simple LSTM model.
    
    Parameters:
    - time_steps: how many past hours the model will look at (like 24 hours)
    - features: how many columns of input data we have (usually 1 for single load)
    
    Returns:
    - model: a ready-to-train LSTM model
    """
    
    # The AI brain: a simple stack of layers
    model = tf.keras.Sequential([
        # First layer: LSTM with 64 memory cells
        tf.keras.layers.LSTM(64, input_shape=(time_steps, features)),
        # Output layer: 1 value (the predicted next load)
        tf.keras.layers.Dense(1)
    ])
    
    # We use 'adam' to learn, and 'mse' (mean squared error) to check mistakes
    model.compile(optimizer='adam', loss='mse')
    
    return model
