"""
simulate.py (Improved)
----------------------
Simulates classical power grid operation using AI-predicted load
and calculates cost and accuracy metrics for thesis purposes.

Steps:
1. Load AI forecast (Predicted Load from LSTM)
2. Simulate 3-generator grid dispatch (economic dispatch: cheapest first)
3. Calculate hourly schedule, unmet load, and total operating cost
4. Compute error metrics: MAE, RMSE, MAPE between predicted load and generated load
5. Save results as CSV and plot dispatch for thesis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. File Paths
# -----------------------------
FORECAST_PATH = "../data/forecasts/test_forecast.csv"
SCHEDULE_PATH = "../results/tables/classical_schedule.csv"
FIGURE_PATH = "../results/figures/classical_dispatch.png"

# Ensure results folders exist
os.makedirs("../results/tables", exist_ok=True)
os.makedirs("../results/figures", exist_ok=True)

# -----------------------------
# 2. Define Grid Generators
# -----------------------------
# Format: {"name": str, "capacity": MW, "cost": currency/MW}
generators = [
    {"name": "Gen1", "capacity": 20, "cost": 5},   # Cheap but small
    {"name": "Gen2", "capacity": 30, "cost": 8},   # Medium
    {"name": "Gen3", "capacity": 50, "cost": 12}   # Expensive but large
]

# -----------------------------
# 3. Load AI Forecast
# -----------------------------
forecast_df = pd.read_csv(FORECAST_PATH)
predicted_load = forecast_df["Predicted_Load"].values
actual_load = forecast_df["Actual_Load"].values  # For additional comparison if needed

print(f"✅ Loaded forecast with {len(predicted_load)} hours.")

# -----------------------------
# 4. Dispatch Generators Hourly
# -----------------------------
schedule = []
total_cost = 0
total_unmet = 0

for hour, demand in enumerate(predicted_load):
    remaining_load = demand
    hour_schedule = {"Hour": hour}

    # Sort generators by cost (cheapest first)
    sorted_gens = sorted(generators, key=lambda x: x["cost"])

    for gen in sorted_gens:
        output = min(gen["capacity"], remaining_load)
        hour_schedule[gen["name"]] = output

        # Add cost = MW * cost per MW
        total_cost += output * gen["cost"]

        # Reduce remaining demand
        remaining_load -= output

        if remaining_load <= 0:
            break

    # Calculate unmet load if demand exceeds total capacity
    unmet_load = max(0, remaining_load)
    hour_schedule["Unmet_Load"] = unmet_load
    total_unmet += unmet_load

    schedule.append(hour_schedule)

# -----------------------------
# 5. Save Schedule to CSV
# -----------------------------
schedule_df = pd.DataFrame(schedule)
schedule_df["Total_Generation"] = schedule_df[["Gen1","Gen2","Gen3"]].sum(axis=1)
schedule_df.to_csv(SCHEDULE_PATH, index=False)

print(f"✅ Classical dispatch schedule saved to {SCHEDULE_PATH}")
print(f"💰 Total operating cost: {total_cost:.2f} currency units")
print(f"⚡ Total unmet load: {total_unmet:.2f} MW\n")

# -----------------------------
# 6. Compute Error Metrics
# -----------------------------
generated_load = schedule_df["Total_Generation"].values

mae = mean_absolute_error(predicted_load, generated_load)
rmse = np.sqrt(mean_squared_error(predicted_load, generated_load))
mape = np.mean(np.abs((predicted_load - generated_load) / predicted_load)) * 100

print("📊 Dispatch Performance Metrics (Predicted vs Generated Load):")
print(f"MAE  (Mean Absolute Error)      : {mae:.2f} MW")
print(f"RMSE (Root Mean Squared Error)  : {rmse:.2f} MW")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%\n")

# -----------------------------
# 7. Plot Dispatch vs Predicted Load
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(predicted_load[:200], label="Predicted Load (MW)", color="blue")
plt.plot(generated_load[:200], label="Generated Load (MW)", color="orange")
plt.title("Classical Grid Dispatch vs AI Predicted Load (First 200 Hours)")
plt.xlabel("Hour")
plt.ylabel("MW")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_PATH)
plt.show()

print(f"📊 Dispatch plot saved to {FIGURE_PATH}")
print("🎉 Simulation complete! Classical results with metrics are thesis-ready.")
