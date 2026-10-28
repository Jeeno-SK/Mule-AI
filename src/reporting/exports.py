"""CSV / JSON exports with provenance. Cells are sanitised against spreadsheet formula injection."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

DISCLAIMER = ("Risk indicators are unverified and for authorised human review only. They are not findings of criminal "
              "conduct. Scores are heuristic, not calibrated probabilities.")


def provenance(dataset: dict, detection_basis: str) -> dict:
    synthetic = dataset["origin"] == "synthetic"
    label = {"none": "no labels in dataset (unlabelled rows are not assumed legitimate)",
             "partial": "partially labelled by dataset provider", "full": "labelled by dataset provider"}[dataset["label_status"]]
    return {"dataset_id": dataset["dataset_id"], "dataset_name": dataset["name"], "dataset_sha256": dataset.get("sha256"),
            "data_origin": "SYNTHETIC / SIMULATED - not real banking evidence" if synthetic else "imported by user (authenticity not verified by this tool)",
            "labels": label, "detection_basis": detection_basis,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "disclaimer": DISCLAIMER}


def sanitise_cell(value):
    """Neutralise values a spreadsheet could interpret as formulas."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame, prov: dict) -> bytes:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].map(sanitise_cell)
    for key in ("dataset_name", "data_origin", "labels", "detection_basis", "exported_at_utc"):
        out[key] = prov[key]
    return out.to_csv(index=False).encode("utf-8")


def to_json_bytes(records: pd.DataFrame | dict | list, prov: dict) -> bytes:
    data = records.to_dict(orient="records") if isinstance(records, pd.DataFrame) else records
    return json.dumps({"provenance": prov, "data": data}, indent=2, default=str).encode("utf-8")
