"""Transaction schema validation, cleaning and reporting.

Nothing is silently discarded: every dropped, normalised or suspicious row is
recorded in the returned `issues` table.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["transaction_id", "timestamp", "source_account", "destination_account", "amount"]
OPTIONAL_COLUMNS = ["transaction_type", "is_fraud"]
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ["extra_json"]

# Column aliases so public datasets (e.g. PaySim-style exports) can be mapped.
COLUMN_ALIASES = {
    "txn_id": "transaction_id", "tx_id": "transaction_id", "id": "transaction_id",
    "time": "timestamp", "datetime": "timestamp", "date": "timestamp",
    "nameorig": "source_account", "sender": "source_account", "from_account": "source_account",
    "namedest": "destination_account", "receiver": "destination_account", "to_account": "destination_account",
    "type": "transaction_type", "isfraud": "is_fraud", "label": "is_fraud",
}
ACCOUNT_RE = re.compile(r"^[A-Z0-9_.:@/\-]{1,64}$")
TRUE_VALUES = {"1", "true", "t", "yes", "y", "fraud"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
MAX_EXTRA_COLUMNS = 20
MAX_AMOUNT = 1e12
PAYSIM_EPOCH = "2024-01-01 00:00:00"


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    issues: pd.DataFrame
    summary: dict
    schema_ok: bool = True
    messages: list[str] = field(default_factory=list)


def normalise_account(value: object) -> str | None:
    """Trim, collapse whitespace and upper-case an account identifier."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", "", str(value)).upper()
    return text or None


def _normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    cols = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    renamed = []
    for original, c in zip(df.columns, cols):
        target = COLUMN_ALIASES.get(c, c)
        if target != c:
            notes.append(f"Column '{original}' was mapped to '{target}'.")
        renamed.append(target)
    out = df.copy()
    out.columns = renamed
    out = out.loc[:, ~pd.Index(renamed).duplicated(keep="first")]
    return out, notes


def _issue(rows: list[dict], row_number: int, tx_id: object, field_: str, issue: str, severity: str, action: str) -> None:
    rows.append({"row_number": row_number, "transaction_id": "" if tx_id is None else str(tx_id),
                 "field": field_, "issue": issue, "severity": severity, "action": action})


def _empty_summary(rows_in: int, label_status: str = "none") -> dict:
    return {"rows_in": rows_in, "rows_valid": 0, "rows_dropped": rows_in, "label_status": label_status,
            "unique_accounts": 0, "total_amount": 0.0}


def validate_transactions(raw: pd.DataFrame) -> ValidationResult:
    """Validate a raw (string-typed) transaction table and return cleaned data plus a full report."""
    rows_in = len(raw)
    messages: list[str] = []
    if raw is None or raw.empty:
        return ValidationResult(pd.DataFrame(columns=CANONICAL_COLUMNS),
                                pd.DataFrame(columns=["row_number", "transaction_id", "field", "issue", "severity", "action"]),
                                _empty_summary(0), schema_ok=False,
                                messages=["The file contains no data rows."])
    df, notes = _normalise_columns(raw)
    messages.extend(notes)
    issues: list[dict] = []

    # PaySim-style 'step' (hours) fallback for missing timestamps; flagged as derived.
    if "timestamp" not in df.columns and "step" in df.columns:
        steps = pd.to_numeric(df["step"], errors="coerce")
        df["timestamp"] = (pd.Timestamp(PAYSIM_EPOCH) + pd.to_timedelta(steps, unit="h")).astype(str)
        df.loc[steps.isna(), "timestamp"] = None
        messages.append(f"No 'timestamp' column: derived from 'step' (hours since {PAYSIM_EPOCH}). Timestamps are synthetic.")
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"ROW{i + 1:09d}" for i in range(len(df))]
        messages.append("No 'transaction_id' column: row-based identifiers were generated.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return ValidationResult(pd.DataFrame(columns=CANONICAL_COLUMNS),
                                pd.DataFrame(columns=["row_number", "transaction_id", "field", "issue", "severity", "action"]),
                                _empty_summary(rows_in), schema_ok=False,
                                messages=messages + [f"Missing required column(s): {', '.join(missing_cols)}."])

    df = df.reset_index(drop=True)
    df["_row"] = np.arange(2, len(df) + 2)  # spreadsheet-style row numbers (header = row 1)
    for col in REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col] == "", col] = pd.NA

    keep = pd.Series(True, index=df.index)

    def drop(mask: pd.Series, field_: str, issue: str) -> None:
        mask = mask & keep
        for idx in df.index[mask]:
            _issue(issues, int(df.at[idx, "_row"]), df.at[idx, "transaction_id"], field_, issue, "error", "dropped")
        keep.loc[mask] = False

    for col in REQUIRED_COLUMNS:
        drop(df[col].isna(), col, "missing value")

    # Amount
    amount = pd.to_numeric(df["amount"].str.replace(",", "", regex=False), errors="coerce")
    drop(amount.isna() & df["amount"].notna(), "amount", "non-numeric amount")
    drop(np.isfinite(amount) & (amount <= 0), "amount", "amount must be greater than zero")
    drop(~np.isfinite(amount) & amount.notna(), "amount", "non-finite amount")
    drop(amount > MAX_AMOUNT, "amount", f"amount exceeds sanity limit ({MAX_AMOUNT:g})")
    df["amount"] = amount.astype(float)

    # Timestamp
    ts = pd.to_datetime(df["timestamp"].astype("object"), errors="coerce", utc=True, format="mixed")
    drop(ts.isna() & df["timestamp"].notna(), "timestamp", "unparseable timestamp")
    df["timestamp"] = ts.dt.tz_convert(None).astype("datetime64[ns]")

    # Accounts
    for col in ("source_account", "destination_account"):
        norm = df[col].map(lambda v: normalise_account(None if pd.isna(v) else v)).astype("string")
        changed = keep & df[col].notna() & (norm != df[col])
        for idx in df.index[changed]:
            _issue(issues, int(df.at[idx, "_row"]), df.at[idx, "transaction_id"], col, "identifier normalised (trim/upper-case)", "info", "normalised")
        df[col] = norm
        bad = df[col].notna() & ~df[col].fillna("").str.match(ACCOUNT_RE)
        drop(bad, col, "invalid account identifier (allowed: A-Z 0-9 _ . : @ / -, max 64 chars)")

    # Duplicate transaction ids (first occurrence kept)
    dup = df["transaction_id"].notna() & df["transaction_id"].duplicated(keep="first")
    drop(dup, "transaction_id", "duplicate transaction_id (first occurrence kept)")

    # Warnings on retained rows
    selfmask = keep & (df["source_account"] == df["destination_account"])
    for idx in df.index[selfmask]:
        _issue(issues, int(df.at[idx, "_row"]), df.at[idx, "transaction_id"], "destination_account", "self-transfer (source equals destination)", "warning", "kept")
    content_dup = keep & df.duplicated(subset=["source_account", "destination_account", "amount", "timestamp"], keep="first")
    for idx in df.index[content_dup]:
        _issue(issues, int(df.at[idx, "_row"]), df.at[idx, "transaction_id"], "transaction_id", "possible duplicate: identical source, destination, amount and time under a different id", "warning", "kept")

    # Labels
    label_status = "none"
    if "is_fraud" in df.columns:
        low = df["is_fraud"].str.lower()
        parsed = pd.Series(np.nan, index=df.index, dtype="float")
        parsed[low.isin(TRUE_VALUES)] = 1.0
        parsed[low.isin(FALSE_VALUES)] = 0.0
        unknown = low.notna() & parsed.isna() & keep
        for idx in df.index[unknown]:
            _issue(issues, int(df.at[idx, "_row"]), df.at[idx, "transaction_id"], "is_fraud", "unrecognised label value treated as unlabelled", "warning", "kept")
        df["is_fraud"] = parsed
        kept_labels = parsed[keep]
        if kept_labels.notna().all() and len(kept_labels):
            label_status = "full"
        elif kept_labels.notna().any():
            label_status = "partial"
    else:
        df["is_fraud"] = np.nan
    if "transaction_type" in df.columns:
        df["transaction_type"] = df["transaction_type"].str.upper().fillna("UNSPECIFIED").astype(str)
    else:
        df["transaction_type"] = "UNSPECIFIED"

    # Extra columns preserved as JSON (bounded)
    known = set(CANONICAL_COLUMNS) | {"_row", "step"}
    extras = [c for c in df.columns if c not in known][:MAX_EXTRA_COLUMNS]
    if extras:
        df["extra_json"] = [json.dumps({c: (None if pd.isna(r[c]) else str(r[c])[:200]) for c in extras})
                            for _, r in df[extras].iterrows()]
        messages.append(f"{len(extras)} additional column(s) preserved in extra_json: {', '.join(extras)}.")
    else:
        df["extra_json"] = None

    clean = df.loc[keep, CANONICAL_COLUMNS].sort_values("timestamp", kind="stable").reset_index(drop=True)
    issues_df = pd.DataFrame(issues, columns=["row_number", "transaction_id", "field", "issue", "severity", "action"])
    summary = _summarise(rows_in, clean, issues_df, label_status)
    return ValidationResult(clean, issues_df, summary, True, messages)


def _summarise(rows_in: int, clean: pd.DataFrame, issues: pd.DataFrame, label_status: str) -> dict:
    n = len(clean)
    counts = issues.groupby("issue").size().to_dict() if len(issues) else {}
    labelled = clean["is_fraud"].dropna() if n else pd.Series(dtype=float)
    return {
        "rows_in": int(rows_in), "rows_valid": int(n), "rows_dropped": int(rows_in - n),
        "issue_counts": {str(k): int(v) for k, v in counts.items()},
        "missing_required_values": int((issues["issue"] == "missing value").sum()) if len(issues) else 0,
        "invalid_amounts": int(issues["issue"].str.contains("amount", na=False).sum()) if len(issues) else 0,
        "invalid_timestamps": int((issues["issue"] == "unparseable timestamp").sum()) if len(issues) else 0,
        "duplicate_ids": int(issues["issue"].str.startswith("duplicate transaction_id", na=False).sum()) if len(issues) else 0,
        "self_transfers": int(issues["issue"].str.startswith("self-transfer", na=False).sum()) if len(issues) else 0,
        "label_status": label_status,
        "labelled_rows": int(len(labelled)),
        "fraud_labelled_rows": int((labelled == 1).sum()),
        "unique_accounts": int(pd.concat([clean["source_account"], clean["destination_account"]]).nunique()) if n else 0,
        "total_amount": float(clean["amount"].sum()) if n else 0.0,
        "amount_min": float(clean["amount"].min()) if n else None,
        "amount_median": float(clean["amount"].median()) if n else None,
        "amount_max": float(clean["amount"].max()) if n else None,
        "date_min": str(clean["timestamp"].min()) if n else None,
        "date_max": str(clean["timestamp"].max()) if n else None,
    }
