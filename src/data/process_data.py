"""
data/process_data.py
--------------------
- Loads raw CSV (flexible column names)
- Parses/cleans timestamp
- Resamples to hourly (if needed)
- Splits Train/Test (last N days for test)
- Saves processed CSVs + scaler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.common_paths import RAW_LOAD, TRAIN_CSV, TEST_CSV, SCALER_PATH
from src.utils.io import read_csv_flexible, write_csv

TEST_DAYS = 30  # last 30 days = test

def main():
    # 1) Load raw with flexible renaming
    df = read_csv_flexible(RAW_LOAD, required=["timestamp","Load"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # If your raw is not strictly hourly, enforce hourly with mean
    df = df.set_index("timestamp").resample("H").mean().interpolate()
    df.reset_index(inplace=True)

    # 2) Split train/test
    split_time = df["timestamp"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df = df[df["timestamp"] <= split_time].copy()
    test_df  = df[df["timestamp"]  > split_time].copy()

    # 3) Fit scaler on Train Load only (to avoid leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[["Load"]].values)

    # 4) Save files
    write_csv(train_df, TRAIN_CSV)
    write_csv(test_df, TEST_CSV)

    # 5) Save scaler
    import joblib
    joblib.dump(scaler, SCALER_PATH)

    print(f"✅ Processed: Train={len(train_df)} rows | Test={len(test_df)} rows")
    print(f"💾 Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
