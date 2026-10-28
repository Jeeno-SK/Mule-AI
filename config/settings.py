"""Central, configurable application settings (no secrets, no network endpoints)."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("MULESHIELD_APP_NAME", "MuleShield AI")
    subtitle: str = "Explainable Mule Account and Suspicious Financial Network Detection Platform"
    db_path: Path = Path(os.getenv("MULESHIELD_DB", str(ROOT / "data" / "muleshield.db")))
    models_dir: Path = ROOT / "models"
    rules_path: Path = ROOT / "config" / "rules.yaml"
    demo_csv: Path = ROOT / "data" / "demo_transactions.csv"
    max_upload_mb: int = 25
    max_rows: int = 500_000
    max_graph_nodes: int = 250
    default_analyst: str = os.getenv("MULESHIELD_ANALYST", "analyst")


def get_settings() -> Settings:
    return Settings()


def load_rules_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML detection configuration (safe_load only)."""
    path = path or get_settings().rules_path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return a new dict where nested `override` values replace those in `base`."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
