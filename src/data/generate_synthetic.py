"""
data/generate_synthetic.py
--------------------------
Creates a 10–20 year hourly dataset with:
- daily/weekly/seasonal patterns
- growth trend
- random weather-like noise
- random holidays effect
Saves to data/raw/load_data.csv with columns: timestamp, Load
"""

import numpy as np
import pandas as pd
from datetime import datetime
from src.common_paths import RAW_LOAD
from src.utils.io import write_csv

YEARS = 10  # set to 20 if you want larger dataset quickly

def make_series(years=YEARS, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2005-01-01 00:00:00")
    periods = years * 365 * 24  # approximate (ignores leap-day detail)
    idx = pd.date_range(start=start, periods=periods, freq="H")

    # Base seasonal patterns
    hour = idx.hour.values
    dayofweek = idx.dayofweek.values
    month = idx.month.values

    # Daily curve (peak evening)
    daily = 100 * (np.sin(2*np.pi*(hour-17)/24) + 1.2)

    # Weekly: weekends lower
    weekly = 50 * (1 - (dayofweek >= 5)*0.3)

    # Yearly: hotter months -> higher demand
    monthly = 80 * (np.sin(2*np.pi*(month-7)/12) + 1.1)

    # Growth trend (slowly increasing)
    trend = np.linspace(500, 900, periods)

    # Noise
    noise = rng.normal(0, 20, size=periods)

    # Holiday dip spikes (random)
    holiday_mask = rng.random(periods) < 0.002
    holiday_effect = -80 * holiday_mask.astype(float)

    load = trend + daily + weekly + monthly + noise + holiday_effect
    load = np.clip(load, a_min=50, a_max=None)

    df = pd.DataFrame({"timestamp": idx, "Load": load.round(2)})
    return df

if __name__ == "__main__":
    df = make_series()
    write_csv(df, RAW_LOAD)
    print(f"✅ Synthetic dataset created: {RAW_LOAD}  (rows={len(df)})")
