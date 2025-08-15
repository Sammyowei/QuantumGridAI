"""
analysis/report_export.py
-------------------------
Collects key tables/figures into a 'reports' folder for quick thesis insertion.
(Here we just ensure copies exist; for PDFs use your text editor to insert PNG/CSVs.)
"""

import shutil
from common_paths import TABLES_DIR, FIGURES_DIR, REPORTS_DIR

def main():
    # Copy a few headline artifacts (customize as you like)
    for p in [
        TABLES_DIR / "forecast_model_comparison.csv",
        TABLES_DIR / "classical_metrics_normal.csv",
        TABLES_DIR / "quantum_qubo_scalable_metrics.csv",
        TABLES_DIR / "yearly_metrics.csv",
        TABLES_DIR / "monthly_metrics.csv",
        FIGURES_DIR / "forecast_compare.png",
        FIGURES_DIR / "forecast_metrics_bar.png",
        FIGURES_DIR / "classical_dispatch_normal.png",
        FIGURES_DIR / "quantum_qubo_scalable_dispatch.png",
        FIGURES_DIR / "seasonal_boxplot.png",
        FIGURES_DIR / "yearly_error_trend.png",
    ]:
        if p.exists():
            shutil.copy(p, REPORTS_DIR / p.name)
    print(f"📦 Report artifacts updated in: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
