"""Shared test fixtures (plain functions; no pytest dependency)."""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from src.database.repository import Repository


def make_tx(rows: list[tuple]) -> pd.DataFrame:
    """rows: (id, 'YYYY-MM-DD HH:MM', src, dst, amount[, type[, is_fraud]])"""
    recs = []
    for r in rows:
        recs.append({"transaction_id": r[0], "timestamp": pd.Timestamp(r[1]), "source_account": r[2], "destination_account": r[3],
                     "amount": float(r[4]), "transaction_type": r[5] if len(r) > 5 else "UPI",
                     "is_fraud": r[6] if len(r) > 6 else float("nan"), "extra_json": None})
    df = pd.DataFrame(recs)
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    return df


def temp_repo() -> Repository:
    return Repository(os.path.join(tempfile.mkdtemp(), "test.db"))


CHAIN = [("T1", "2025-01-01 10:00", "V", "M1", 10000), ("T2", "2025-01-01 10:30", "M1", "M2", 9600),
         ("T3", "2025-01-01 11:00", "M2", "M3", 9200), ("T4", "2025-01-01 11:20", "M3", "M4", 8900)]
