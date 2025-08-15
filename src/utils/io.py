"""
utils/io.py
-----------
Helper functions to load/save CSVs, and to normalize column names.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def read_csv_flexible(path: Path,
                      rename_map: Optional[Dict[str, str]] = None,
                      required: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reads CSV and tries to standardize common column name variations.
    - rename_map: user-provided mapping to rename columns
    - required: columns that must exist after rename
    """
    df = pd.read_csv(path)
    # Common normalizations
    auto_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "load_mw": "Load",
        "loadmw": "Load",
        "load": "Load",
        "demand": "Load"
    }
    df.rename(columns={**auto_map, **(rename_map or {})}, inplace=True)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after rename: {missing}")
    return df

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
