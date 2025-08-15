"""
utils/plotting.py
-----------------
Simple plotting helpers for comparisons and dispatch.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_prediction_vs_actual(y_true, preds_dict, save_path: Path, title="Predictions vs Actual"):
    plt.figure(figsize=(12,5))
    plt.plot(y_true, label="Actual")
    for name, yhat in preds_dict.items():
        plt.plot(yhat, label=name, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time Index (test set)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_bar(metrics_dict, save_path: Path, title="Error Metrics (lower is better)"):
    # metrics_dict: {model: {"MAE":..., "RMSE":..., "MAPE":...}}
    labels = list(metrics_dict.keys())
    mae_vals = [metrics_dict[m]["MAE"] for m in labels]
    rmse_vals = [metrics_dict[m]["RMSE"] for m in labels]
    mape_vals = [metrics_dict[m]["MAPE"] for m in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12,5))
    plt.bar(x - width, mae_vals, width, label="MAE")
    plt.bar(x, rmse_vals, width, label="RMSE")
    plt.bar(x + width, mape_vals, width, label="MAPE")
    plt.xticks(x, labels, rotation=15)
    plt.title(title)
    plt.ylabel("Error / %")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_dispatch(pred_load, gen_load, save_path: Path, title="Dispatch vs Predicted Load"):
    plt.figure(figsize=(12,5))
    plt.plot(pred_load, label="Predicted Load (MW)")
    plt.plot(gen_load, label="Dispatched Generation (MW)")
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
