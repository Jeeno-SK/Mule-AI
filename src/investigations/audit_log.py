"""Audit trail for important analyst and system actions (stored in SQLite, append-only from the app)."""
from __future__ import annotations

import pandas as pd

from src.database.repository import Repository


def log_action(repo: Repository, actor: str, action: str, dataset_id: int | None = None, target: str | None = None,
               details: dict | None = None) -> None:
    repo.insert_audit(actor or "unknown", action, dataset_id, target, details or {})


def read_log(repo: Repository, dataset_id: int | None = None, limit: int = 500) -> pd.DataFrame:
    return repo.get_audit(dataset_id, limit)
