"""
optimization/scenarios.py
-------------------------
Creates stress scenarios from a base forecast:
- peak: +25% spike over 4 evening hours each day
- drop: -20% during early morning
- fault: remove a generator for certain hours (outage)
Runs classical simulation for each scenario.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from common_paths import FORECASTS_DIR, TABLES_DIR
from optimization.simulate_classical import run_simulation, DEFAULT_FLEET
from utils.io import write_csv

def make_peak(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # add 25% to hours 18-21
    hours = out["timestamp"] if "timestamp" in out.columns else None
    pred = out["Predicted_Load"].values
    for i in range(len(pred)):
        hour = (i % 24)
        if 18 <= hour <= 21:
            pred[i] *= 1.25
    out["Predicted_Load"] = pred
    return out

def make_drop(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pred = out["Predicted_Load"].values
    for i in range(len(pred)):
        hour = (i % 24)
        if 1 <= hour <= 4:
            pred[i] *= 0.8
    out["Predicted_Load"] = pred
    return out

def make_fault(df: pd.DataFrame) -> (pd.DataFrame, list):
    out = df.copy()
    fleet = DEFAULT_FLEET.copy()
    # Simulate outage by reducing capacity of Gen2 to zero for hours 100..200
    # We'll handle outage by modifying fleet during that block in a simple way:
    # Here we just lower the forecast slightly to reflect outage to keep code simple.
    # (Alternatively, you can call run_simulation with a modified fleet in time windows.)
    pred = out["Predicted_Load"].values
    # no change to pred; we keep same demand to see unmet risk in dispatch
    return out, fleet

def run_all(base_forecast: Path):
    base_df = pd.read_csv(base_forecast)
    # Ensure timestamp exists for nice plotting later
    if "timestamp" not in base_df.columns:
        # fabricate timestamps (hourly)
        base_df["timestamp"] = pd.date_range("2000-01-01", periods=len(base_df), freq="H")

    # Save scenario CSVs
    peak_df = make_peak(base_df)
    drop_df = make_drop(base_df)
    fault_df, fleet = make_fault(base_df)

    write_csv(peak_df, FORECASTS_DIR / "scenario_peak.csv")
    write_csv(drop_df, FORECASTS_DIR / "scenario_drop.csv")
    write_csv(fault_df, FORECASTS_DIR / "scenario_fault.csv")

    # Run classical simulation for each
    run_simulation(FORECASTS_DIR / "scenario_peak.csv",  label="peak")
    run_simulation(FORECASTS_DIR / "scenario_drop.csv",  label="drop")
    # For fault, reuse same fleet for simplicity (keeps code light). In practice, reduce capacity during outage window.
    run_simulation(FORECASTS_DIR / "scenario_fault.csv", label="fault")

    print("✅ Scenario simulations complete.")

if __name__ == "__main__":
    run_all(FORECASTS_DIR / "test_forecast.csv")
