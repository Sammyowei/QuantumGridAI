"""
optimization/renewables.py
--------------------------
Adds simple renewable profiles:
- Solar: bell-shaped daytime curve, zero at night
- Wind: random availability 30–70%
Then runs classical dispatch with renewables included.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.common_paths import FORECASTS_DIR, TABLES_DIR, FIGURES_DIR
from src.optimization.simulate_classical import greedy_dispatch
from src.utils.metrics import mae, rmse, mape
from src.utils.plotting import plot_dispatch

def solar_profile(hours: int, peak_mw: float = 40.0) -> np.ndarray:
    # Make a simple daily solar curve with peak at noon
    base = np.array([max(0.0, np.sin((h-6)/24*np.pi)) for h in range(24)])
    base = base / base.max()
    prof = np.tile(base, hours//24 + 1)[:hours] * peak_mw
    return prof

def wind_profile(hours: int, nameplate: float = 50.0, seed=123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    avail = rng.uniform(0.3, 0.7, size=hours)
    return nameplate * avail

def run_with_renewables(forecast_csv: Path, thermal_fleet, label="renewables"):
    df = pd.read_csv(forecast_csv)
    pred = df["Predicted_Load"].values.astype(float)
    H = len(pred)

    solar = solar_profile(H, peak_mw=40.0)
    wind  = wind_profile(H, nameplate=50.0)

    # Treat renewables as zero-cost, variable-capacity units available first
    gen_total = np.zeros(H)
    total_cost = 0.0
    total_unmet = 0.0

    for h in range(H):
        # First, take all renewables available up to demand
        ren = min(pred[h], solar[h] + wind[h])
        remaining = pred[h] - ren
        # Then dispatch thermal fleet greedily
        row, cost = greedy_dispatch(remaining, thermal_fleet)
        gen = ren + row["Total_Generation"]
        total_cost += cost
        total_unmet += max(0.0, row["Unmet_Load"])
        gen_total[h] = gen

    # Metrics
    MAE, RMSE, MAPE = mae(pred, gen_total), rmse(pred, gen_total), mape(pred, gen_total)
    pd.DataFrame([{
        "Label": label, "Total Cost": total_cost, "Total Unmet (MW)": total_unmet,
        "MAE": MAE, "RMSE": RMSE, "MAPE": MAPE
    }]).to_csv(TABLES_DIR / f"classical_metrics_{label}.csv", index=False)

    plot_dispatch(pred, gen_total, FIGURES_DIR / f"classical_dispatch_{label}.png",
                  title="Dispatch with Renewables vs Predicted Load")

    print(f"✅ Renewables run saved. Cost={total_cost:.2f} | Unmet={total_unmet:.2f} MW")
    print(f"📊 MAE={MAE:.2f} | RMSE={RMSE:.2f} | MAPE={MAPE:.2f}%")

if __name__ == "__main__":
    THERMAL = [
        {"name": "Gen1", "capacity": 20, "cost": 5},
        {"name": "Gen2", "capacity": 30, "cost": 8},
        {"name": "Gen3", "capacity": 50, "cost": 12}
    ]
    run_with_renewables(Path("../data/forecasts/test_forecast.csv"), THERMAL)
