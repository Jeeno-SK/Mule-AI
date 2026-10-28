"""Safe CSV reading and dataset ingestion (never executes uploaded content)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.data.validator import ValidationResult, validate_transactions
from src.database.repository import Repository
from src.utils.exceptions import DataValidationError


def read_csv_bytes(data: bytes, max_bytes: int | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Parse CSV bytes as plain text (all columns as strings) with size and row limits."""
    s = get_settings()
    max_bytes = max_bytes or s.max_upload_mb * 1024 * 1024
    max_rows = max_rows or s.max_rows
    if not data:
        raise DataValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise DataValidationError(f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if b"\x00" in data[:4096]:
        raise DataValidationError("The file looks binary, not a CSV text file.")
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig",
                         nrows=max_rows + 1, on_bad_lines="error", engine="python")
    except UnicodeDecodeError as exc:
        raise DataValidationError("The file is not valid UTF-8 text.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DataValidationError(f"The CSV could not be parsed: {str(exc)[:200]}") from exc
    if len(df) > max_rows:
        raise DataValidationError(f"File exceeds the {max_rows:,}-row limit.")
    return df


def read_csv_path(path: str | Path) -> tuple[pd.DataFrame, bytes]:
    data = Path(path).read_bytes()
    return read_csv_bytes(data), data


def ingest_dataframe(repo: Repository, raw: pd.DataFrame, name: str, source: str, origin: str,
                     raw_bytes: bytes | None = None, extra_metadata: dict | None = None,
                     activate: bool = True) -> tuple[int | None, ValidationResult]:
    """Validate `raw` and persist the clean rows as a new dataset. Returns (dataset_id or None, result)."""
    result = validate_transactions(raw)
    if not result.schema_ok or result.clean.empty:
        return None, result
    digest = hashlib.sha256(raw_bytes if raw_bytes is not None else raw.to_csv(index=False).encode("utf-8")).hexdigest()
    metadata = {"validation_summary": result.summary, "messages": result.messages, **(extra_metadata or {})}
    dataset_id = repo.create_dataset(name=name, source=source, origin=origin,
                                     label_status=result.summary["label_status"], sha256=digest,
                                     row_count=len(result.clean), metadata=metadata)
    repo.insert_transactions(dataset_id, result.clean)
    if activate:
        repo.set_active_dataset(dataset_id)
    return dataset_id, result
