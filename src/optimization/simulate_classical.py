"""
optimization/simulate_classical.py
----------------------------------
- Takes a forecast file (Predicted_Load) and a generator fleet
- Produces least-cost dispatch meeting demand (greedy cost/MW)
- Saves schedule + metrics + figure
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from common_paths import TABLES_DIR, FIGURES_DIR
from utils.metrics import mae, rmse, mape
from utils.plotting import plot_dispatch

# Default simple fleet (you can swap with a config JSON if you want)
DEFAULT_FLEET = [
    {"name": "Gen1", "capacity": 20, "cost": 5},
    {"name": "Gen2", "capacity": 30, "cost": 8},
    {"name": "Gen3", "capacity": 50, "cost": 12}
]

def greedy_dispatch(demand: float, fleet) -> (dict, float):
    """Greedy: sort by cost per MW and switch on until demand met."""
    order = sorted(fleet, key=lambda g: g["cost"]/g["capacity"])
    out = {}
    remaining = demand
    cost = 0.0
    for g in order:
        gen = min(g["capacity"], max(0.0, remaining))
        out[g["name"]] = gen
        cost += gen * g["cost"]
        remaining -= gen
        if remaining <= 1e-9:
            break
    # If still remaining, unmet
    unmet = max(0.0, remaining)
    out["Unmet_Load"] = unmet
    out["Total_Generation"] = sum(out[k] for k in out if k not in ["Unmet_Load", "Total_Generation"])
    return out, cost

def run_simulation(forecast_csv: Path, fleet=DEFAULT_FLEET, label="normal"):
    df = pd.read_csv(forecast_csv)
    pred = df["Predicted_Load"].values

    schedule_rows = []
    total_cost = 0.0
    total_unmet = 0.0

    for h, d in enumerate(pred):
        row, cost = greedy_dispatch(float(d), fleet)
        row["Hour"] = h
        total_cost += cost
        total_unmet += row["Unmet_Load"]
        schedule_rows.append(row)

    sched_df = pd.DataFrame(schedule_rows)
    sched_df.to_csv(TABLES_DIR / f"classical_schedule_{label}.csv", index=False)

    # If you have real test target in same df, compute metrics. Otherwise compare to pred itself.
    y_true = pred
    y_gen  = sched_df["Total_Generation"].values
    METR = {
        "MAE":  mae(y_true, y_gen),
        "RMSE": rmse(y_true, y_gen),
        "MAPE": mape(y_true, y_gen)
    }
    pd.DataFrame([{
        "Label": label,
        "Total Cost": total_cost,
        "Total Unmet (MW)": total_unmet,
        **METR
    }]).to_csv(TABLES_DIR / f"classical_metrics_{label}.csv", index=False)

    plot_dispatch(pred, y_gen, FIGURES_DIR / f"classical_dispatch_{label}.png",
                  title=f"Classical Dispatch vs Predicted Load ({label})")

    print(f"✅ Classical '{label}' schedule saved. Cost={total_cost:.2f} | Unmet={total_unmet:.2f} MW")
    print(f"📊 MAE={METR['MAE']:.2f} | RMSE={METR['RMSE']:.2f} | MAPE={METR['MAPE']:.2f}%")
    return sched_df, METR

if __name__ == "__main__":
    # Example run point (edit path if needed)
    run_simulation(Path("../data/forecasts/test_forecast.csv"))
