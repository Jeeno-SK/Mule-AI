"""Simple drift-monitoring foundation (Population Stability Index).

PSI describes how much a distribution moved between two samples. It does NOT show that future fraud
patterns can or cannot be detected. Common PSI rules of thumb (<0.1 small, 0.1-0.25 moderate, >0.25 large)
are conventions, not validated thresholds for this domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float | None:
    e, a = expected.dropna().astype(float), actual.dropna().astype(float)
    if len(e) < 20 or len(a) < 20 or e.nunique() < 2:
        return None
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def compare_feature_tables(base: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(base[c], current[c]), "base_median": float(base[c].median()) if len(base) else None,
             "current_median": float(current[c].median()) if len(current) else None}
            for c in columns if c in base and c in current]
    return pd.DataFrame(rows, columns=["feature", "psi", "base_median", "current_median"])


def temporal_split_drift(tx: pd.DataFrame) -> pd.DataFrame:
    """Compare transaction amount and hour-of-day between the first and second half of the dataset's time span."""
    if len(tx) < 40:
        return pd.DataFrame(columns=["feature", "psi", "base_median", "current_median"])
    mid = tx["timestamp"].min() + (tx["timestamp"].max() - tx["timestamp"].min()) / 2
    a, b = tx[tx["timestamp"] <= mid], tx[tx["timestamp"] > mid]
    return compare_feature_tables(a.assign(hour=a["timestamp"].dt.hour), b.assign(hour=b["timestamp"].dt.hour), ["amount", "hour"])
