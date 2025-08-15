"""
common_paths.py
---------------
Single source of truth for all important folders/files.
Uses pathlib so Windows/Mac/Linux paths work the same.
"""

from pathlib import Path

# Find the project root (folder containing "QuantumGridAI")
# If your folder has a different name, change the string below.
PROJECT_NAME = "QuantumGridAI"

def get_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if parent.name == PROJECT_NAME:
            return parent
    # Fallback: two levels up (src -> QuantumGridAI)
    return Path(__file__).resolve().parents[1]

ROOT = get_project_root()

# Data directories
DATA_DIR       = ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
FORECASTS_DIR  = DATA_DIR / "forecasts"

# Model + results directories
MODELS_DIR     = ROOT / "models"
RESULTS_DIR    = ROOT / "results"
TABLES_DIR     = RESULTS_DIR / "tables"
FIGURES_DIR    = RESULTS_DIR / "figures"
REPORTS_DIR    = RESULTS_DIR / "reports"

# Canonical file names
RAW_LOAD       = RAW_DIR / "load_data.csv"             # expects 'timestamp','Load'
TRAIN_CSV      = PROCESSED_DIR / "train_data.csv"
TEST_CSV       = PROCESSED_DIR / "test_data.csv"
SCALER_PATH    = MODELS_DIR / "scaler.pkl"
LSTM_MODEL     = MODELS_DIR / "lstm_load_forecast.keras"

# Convenience: make sure folders exist
for d in [RAW_DIR, PROCESSED_DIR, FORECASTS_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
