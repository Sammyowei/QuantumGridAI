"""
run_all.py
----------
Runs the whole pipeline in the 6 phases you defined.

PHASE 1: Forecasting completeness (baselines + LSTM + evaluation)
PHASE 2: Scenario simulations (peak/drop/fault)
PHASE 3: Renewables integration (solar + wind)
PHASE 4: Statistical & seasonal analysis
PHASE 5: Visualization & reporting bundling (already saved along the way)
PHASE 6: Thesis-ready export (collect artifacts)

If you don't have a raw dataset yet, run data/generate_synthetic.py first.
"""

import sys
from pathlib import Path

from common_paths import FORECASTS_DIR, TABLES_DIR, FIGURES_DIR
from data.process_data import main as process_data_main
from models.train_baselines import main as train_baselines_main
from models.train_lstm import main as train_lstm_main
from models.evaluate_models import main as eval_models_main
from optimization.simulate_classical import run_simulation
from optimization.scenarios import run_all as scenarios_main
from optimization.renewables import run_with_renewables
from optimization.quantum_qubo_simulator import run_qubo_sim
from analysis.seasonal_analysis import main as seasonal_main
from analysis.report_export import main as report_main

def banner(msg): print(f"\n>>\n{msg}\n")

def safe_call(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return True
    except Exception as e:
        print(f"❌ {fn.__name__} failed: {e}")
        return False

def main():
    # PHASE 1
    banner("🚀 PHASE 1: Processing data, training baselines + LSTM, evaluating models")
    if not safe_call(process_data_main): sys.exit(1)
    if not safe_call(train_baselines_main): sys.exit(1)
    if not safe_call(train_lstm_main): sys.exit(1)
    if not safe_call(eval_models_main): sys.exit(1)

    # Save LSTM test forecast to standard name (for downstream)
    # evaluate_models computed LSTM predictions internally; we also need a file:
    # Let's create test_forecast.csv by reusing evaluate_models' logic quickly:
    try:
        from tensorflow import keras
        import joblib, pandas as pd
        from common_paths import TEST_CSV, SCALER_PATH, LSTM_MODEL
        from models.train_lstm import WINDOW, make_xy
        test_df = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])
        scaler = joblib.load(SCALER_PATH)
        model = keras.models.load_model(LSTM_MODEL, compile=False)
        scaled = scaler.transform(test_df[["Load"]].values)
        X_test, y_test = make_xy(scaled.flatten(), WINDOW)
        yhat = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
        # align timestamps for yhat
        out_ts = test_df["timestamp"].iloc[WINDOW:].reset_index(drop=True)
        (FORECASTS_DIR / "test_forecast.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": out_ts, "Predicted_Load": yhat}).to_csv(FORECASTS_DIR / "test_forecast.csv", index=False)
        print("✅ test_forecast.csv saved for downstream steps.")
    except Exception as e:
        print(f"⚠️ Could not save test_forecast.csv: {e}")

    # PHASE 2: Scenarios
    banner("🌪️ PHASE 2: Scenario simulations (peak / drop / fault)")
    safe_call(scenarios_main, FORECASTS_DIR / "test_forecast.csv")

    # PHASE 3: Renewables
    banner("☀️🌬️ PHASE 3: Renewables integration")
    THERMAL = [
        {"name": "Gen1", "capacity": 20, "cost": 5},
        {"name": "Gen2", "capacity": 30, "cost": 8},
        {"name": "Gen3", "capacity": 50, "cost": 12}
    ]
    safe_call(run_with_renewables, FORECASTS_DIR / "test_forecast.csv", THERMAL, "renewables")

    # PHASE 4: Seasonal analysis
    banner("📈 PHASE 4: Seasonal & statistical analysis")
    safe_call(seasonal_main)

    # PHASE 5: QUBO simulator (scalable)
    banner("🧠 PHASE 5: QUBO simulator (scalable streaming)")
    safe_call(run_qubo_sim, FORECASTS_DIR / "test_forecast.csv", THERMAL, "scalable")

    # PHASE 6: Export report artifacts
    banner("📦 PHASE 6: Export report artifacts")
    safe_call(report_main)

    print("\n🎓 All phases completed. Check results/tables and results/figures.")

if __name__ == "__main__":
    main()
