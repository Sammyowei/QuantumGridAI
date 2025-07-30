"""
data_preprocessing.py
---------------------
This file handles preparing the electricity load data for AI training.

Steps:
1. Load the dataset (CSV) that contains electricity load values.
2. Scale the data so all numbers are between 0 and 1 (LSTMs like this).
3. Cut the data into "windows" of 24 hours to predict the next hour.
   - Example: If we know the last 24 hours, can we guess the 25th hour?

Imagine this like learning a daily rhythm: 
if the last 24 hours were morning → afternoon → evening, 
the model can guess what happens next (probably night).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(file_path, time_steps=24):
    """
    Load CSV, scale data, and make it ready for LSTM training.
    
    Parameters:
    - file_path: where the CSV is stored (like "data/raw/load_data.csv")
    - time_steps: how many hours to look back to predict the next hour
    
    Returns:
    - X: sequences of past hours (like 24-hour windows)
    - y: the next hour after each window
    - scaler: the object used to scale the data (we need it for predictions)
    """
    
    # 1. Load the dataset from CSV
    df = pd.read_csv(file_path)
    
    # Make sure the CSV has a "Load" column
    if 'Load' not in df.columns:
        raise ValueError("CSV must contain a 'Load' column with electricity demand values.")

    # Keep only the Load column values as a 2D array for scaling
    values = df[['Load']].values

    # 2. Scale the data between 0 and 1 so the LSTM learns easier
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # 3. Create sequences of 24 hours → predict the 25th hour
    X, y = [], []
    for i in range(len(scaled) - time_steps):
        X.append(scaled[i:i+time_steps])  # Take 24 hours
        y.append(scaled[i+time_steps])    # Next hour value
    
    # Convert lists to numpy arrays for AI training
    return np.array(X), np.array(y), scaler
