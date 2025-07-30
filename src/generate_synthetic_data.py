"""
generate_synthetic_data.py
--------------------------
Generates a 10-year synthetic electricity load dataset for the QuantumGridAI project.
The dataset simulates realistic hourly demand patterns with daily, seasonal, 
and random variations to train the LSTM forecasting model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Configuration
# -----------------------------

YEARS = 20                 # Number of years to simulate
HOURS_PER_YEAR = 365 * 24  # Hours in a non-leap year
TOTAL_HOURS = YEARS * HOURS_PER_YEAR

BASE_LOAD = 50             # MW, base load of the grid
DAILY_VARIATION = 20       # MW, daily peak-to-offpeak difference
SEASONAL_VARIATION = 10    # MW, seasonal effect (e.g., summer/winter)
RANDOM_NOISE_STD = 3       # MW, standard deviation of random noise

SAVE_PATH = "../data/raw/load_data.csv"

# -----------------------------
# 2. Generate Time Index
# -----------------------------

# Create hourly timestamps for 10 years
date_range = pd.date_range(start="2015-01-01", periods=TOTAL_HOURS, freq="H")

# -----------------------------
# 3. Generate Synthetic Load
# -----------------------------

# 3.1 Daily cycle: sinusoidal pattern peaking in the evening (~18:00)
hours = np.arange(TOTAL_HOURS)
daily_cycle = DAILY_VARIATION * np.sin((2*np.pi/24) * hours - np.pi/2)

# 3.2 Seasonal cycle: yearly sinusoidal pattern
seasonal_cycle = SEASONAL_VARIATION * np.sin((2*np.pi/(24*365)) * hours - np.pi/2)

# 3.3 Combine base load + daily + seasonal + random noise
np.random.seed(42)  # For reproducibility
load = BASE_LOAD + daily_cycle + seasonal_cycle + np.random.normal(0, RANDOM_NOISE_STD, TOTAL_HOURS)

# Ensure no negative load
load = np.clip(load, 0, None)

# -----------------------------
# 4. Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "Datetime": date_range,
    "Load": load
})

# -----------------------------
# 5. Save Dataset
# -----------------------------

os.makedirs("../data/raw", exist_ok=True)
df.to_csv(SAVE_PATH, index=False)
print(f"✅ Synthetic dataset generated and saved to {SAVE_PATH}")
print(f"Dataset shape: {df.shape}")

# -----------------------------
# 6. Visualization
# -----------------------------

plt.figure(figsize=(12,5))
plt.plot(df["Datetime"][:24*7], df["Load"][:24*7])  # Plot first 7 days
plt.title("Synthetic Load Demand - First Week (MW)")
plt.xlabel("Datetime")
plt.ylabel("Load (MW)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df["Datetime"][:24*365], df["Load"][:24*365])  # Plot first year
plt.title("Synthetic Load Demand - First Year (MW)")
plt.xlabel("Datetime")
plt.ylabel("Load (MW)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
