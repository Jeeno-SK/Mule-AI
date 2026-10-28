"""All SQL lives here. Every statement is parameterised; no string-built values."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.connection import get_connection, init_db

TX_COLUMNS = ["transaction_id", "timestamp", "source_account", "destination_account", "amount",
              "transaction_type", "is_fraud", "extra_json"]


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class Repository:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        init_db(self.db_path)

    def connect(self):
        return get_connection(self.db_path)

    # ---- datasets ----
    def create_dataset(self, name: str, source: str, origin: str, label_status: str, sha256: str | None,
                       row_count: int, metadata: dict) -> int:
        with self.connect() as c:
            cur = c.execute(
                "INSERT INTO datasets (name, source, origin, label_status, sha256, row_count, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (name, source, origin, label_status, sha256, row_count, utcnow(), json.dumps(metadata, default=str)))
            return int(cur.lastrowid)

    def insert_transactions(self, dataset_id: int, df: pd.DataFrame) -> None:
        rows = []
        for r in df[TX_COLUMNS].itertuples(index=False):
            fraud = None if pd.isna(r.is_fraud) else int(r.is_fraud)
            rows.append((dataset_id, str(r.transaction_id), pd.Timestamp(r.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                         str(r.source_account), str(r.destination_account), float(r.amount),
                         None if pd.isna(r.transaction_type) else str(r.transaction_type), fraud,
                         None if r.extra_json is None or pd.isna(r.extra_json) else str(r.extra_json)))
        with self.connect() as c:
            c.executemany("INSERT INTO transactions (dataset_id, transaction_id, timestamp, source_account, "
                          "destination_account, amount, transaction_type, is_fraud, extra_json) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def set_active_dataset(self, dataset_id: int) -> None:
        with self.connect() as c:
            c.execute("UPDATE datasets SET is_active = 0")
            c.execute("UPDATE datasets SET is_active = 1 WHERE dataset_id = ?", (dataset_id,))

    def get_active_dataset_id(self) -> int | None:
        with self.connect() as c:
            row = c.execute("SELECT dataset_id FROM datasets WHERE is_active = 1 LIMIT 1").fetchone()
        return int(row["dataset_id"]) if row else None

    def get_dataset(self, dataset_id: int) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
        return d

    def list_datasets(self) -> pd.DataFrame:
        with self.connect() as c:
            rows = c.execute("SELECT dataset_id, name, origin, label_status, row_count, created_at, analyzed_at, is_active "
                             "FROM datasets ORDER BY dataset_id DESC").fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["dataset_id", "name", "origin", "label_status", "row_count",
                                                              "created_at", "analyzed_at", "is_active"])

    def delete_dataset(self, dataset_id: int) -> None:
        with self.connect() as c:
            c.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))

    def get_transactions(self, dataset_id: int) -> pd.DataFrame:
        with self.connect() as c:
            rows = c.execute("SELECT transaction_id, timestamp, source_account, destination_account, amount, "
                             "transaction_type, is_fraud, extra_json FROM transactions WHERE dataset_id = ? "
                             "ORDER BY timestamp, transaction_id", (dataset_id,)).fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=TX_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
        df["amount"] = df["amount"].astype(float)
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
        return df

    # ---- analysis results ----
    def save_analysis(self, dataset_id: int, alerts: pd.DataFrame, alert_txs: pd.DataFrame,
                      account_risk: pd.DataFrame, params: dict) -> None:
        a_rows = [(dataset_id, r.alert_id, r.rule_id, r.rule_name, r.severity, r.account,
                   r.transaction_id if isinstance(r.transaction_id, str) and r.transaction_id else None,
                   r.timestamp if isinstance(r.timestamp, str) and r.timestamp else None,
                   r.description, r.observed_json, r.threshold_json, r.evidence)
                  for r in alerts.itertuples(index=False)]
        t_rows = [(dataset_id, r.alert_id, r.transaction_id) for r in alert_txs.itertuples(index=False)]
        k_rows = [(dataset_id, r.account, float(r.risk_score), r.risk_band, float(r.rule_points), float(r.graph_points),
                   float(r.ml_points), r.rules_triggered, int(r.n_alerts), r.basis, r.explanation, r.features_json)
                  for r in account_risk.itertuples(index=False)]
        with self.connect() as c:
            c.execute("DELETE FROM alerts WHERE dataset_id = ?", (dataset_id,))
            c.execute("DELETE FROM alert_transactions WHERE dataset_id = ?", (dataset_id,))
            c.execute("DELETE FROM account_risk WHERE dataset_id = ?", (dataset_id,))
            c.executemany("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", a_rows)
            c.executemany("INSERT OR IGNORE INTO alert_transactions VALUES (?,?,?)", t_rows)
            c.executemany("INSERT INTO account_risk VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", k_rows)
            meta_row = c.execute("SELECT metadata_json FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            meta = json.loads(meta_row["metadata_json"]) if meta_row else {}
            meta["analysis_params"] = params
            c.execute("UPDATE datasets SET analyzed_at = ?, metadata_json = ? WHERE dataset_id = ?",
                      (utcnow(), json.dumps(meta, default=str), dataset_id))

    def get_alerts(self, dataset_id: int) -> pd.DataFrame:
        """Alerts joined with their persisted analyst status (default 'open')."""
        sql = ("SELECT a.alert_id, a.rule_id, a.rule_name, a.severity, a.account, a.transaction_id, a.timestamp, "
               "a.description, a.observed_json, a.threshold_json, a.evidence, "
               "COALESCE(s.status, 'open') AS status, s.resolution_reason, s.updated_at, s.updated_by "
               "FROM alerts a LEFT JOIN alert_status s ON s.dataset_id = a.dataset_id AND s.alert_id = a.alert_id "
               "WHERE a.dataset_id = ? ORDER BY CASE a.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
               "a.timestamp DESC, a.alert_id")
        with self.connect() as c:
            rows = c.execute(sql, (dataset_id,)).fetchall()
        cols = ["alert_id", "rule_id", "rule_name", "severity", "account", "transaction_id", "timestamp", "description",
                "observed_json", "threshold_json", "evidence", "status", "resolution_reason", "updated_at", "updated_by"]
        return pd.DataFrame([dict(r) for r in rows], columns=cols)

    def get_alert_transactions(self, dataset_id: int) -> pd.DataFrame:
        with self.connect() as c:
            rows = c.execute("SELECT alert_id, transaction_id FROM alert_transactions WHERE dataset_id = ?",
                             (dataset_id,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["alert_id", "transaction_id"])

    def get_account_risk(self, dataset_id: int) -> pd.DataFrame:
        with self.connect() as c:
            rows = c.execute("SELECT account, risk_score, risk_band, rule_points, graph_points, ml_points, rules_triggered, "
                             "n_alerts, basis, explanation, features_json FROM account_risk WHERE dataset_id = ? "
                             "ORDER BY risk_score DESC, account", (dataset_id,)).fetchall()
        cols = ["account", "risk_score", "risk_band", "rule_points", "graph_points", "ml_points", "rules_triggered",
                "n_alerts", "basis", "explanation", "features_json"]
        return pd.DataFrame([dict(r) for r in rows], columns=cols)

    # ---- search ----
    def search(self, dataset_id: int, query: str, limit: int = 50) -> dict[str, pd.DataFrame]:
        q = _like(query.strip().upper())
        with self.connect() as c:
            acc = c.execute(
                "SELECT account FROM (SELECT source_account AS account FROM transactions WHERE dataset_id = ? "
                "UNION SELECT destination_account FROM transactions WHERE dataset_id = ?) "
                "WHERE account LIKE ? ESCAPE '\\' ORDER BY account LIMIT ?", (dataset_id, dataset_id, q, limit)).fetchall()
            txs = c.execute(
                "SELECT transaction_id, timestamp, source_account, destination_account, amount FROM transactions "
                "WHERE dataset_id = ? AND UPPER(transaction_id) LIKE ? ESCAPE '\\' ORDER BY timestamp LIMIT ?",
                (dataset_id, q, limit)).fetchall()
        return {"accounts": pd.DataFrame([dict(r) for r in acc], columns=["account"]),
                "transactions": pd.DataFrame([dict(r) for r in txs], columns=["transaction_id", "timestamp", "source_account",
                                                                              "destination_account", "amount"])}

    # ---- model runs & settings ----
    def save_model_run(self, dataset_id: int, kind: str, metadata: dict) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO model_runs (dataset_id, kind, created_at, metadata_json) VALUES (?, ?, ?, ?)",
                      (dataset_id, kind, utcnow(), json.dumps(metadata, default=str)))

    def get_model_runs(self, dataset_id: int) -> list[dict]:
        with self.connect() as c:
            rows = c.execute("SELECT run_id, kind, created_at, metadata_json FROM model_runs WHERE dataset_id = ? "
                             "ORDER BY run_id DESC", (dataset_id,)).fetchall()
        return [{"run_id": r["run_id"], "kind": r["kind"], "created_at": r["created_at"],
                 "metadata": json.loads(r["metadata_json"])} for r in rows]

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as c:
            row = c.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO app_settings (key, value_json) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json", (key, json.dumps(value)))

    def delete_setting(self, key: str) -> None:
        with self.connect() as c:
            c.execute("DELETE FROM app_settings WHERE key = ?", (key,))

    # ---- investigations & audit ----
    def alert_ids_for_account(self, dataset_id: int, account: str) -> list[str]:
        with self.connect() as c:
            rows = c.execute("SELECT alert_id FROM alerts WHERE dataset_id = ? AND account = ?", (dataset_id, account)).fetchall()
        return [r["alert_id"] for r in rows]

    def upsert_alert_status(self, dataset_id: int, alert_ids: list[str], status: str, reason: str | None, actor: str) -> None:
        now = utcnow()
        with self.connect() as c:
            c.executemany(
                "INSERT INTO alert_status (dataset_id, alert_id, status, resolution_reason, updated_at, updated_by) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(dataset_id, alert_id) DO UPDATE SET status = excluded.status, "
                "resolution_reason = excluded.resolution_reason, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                [(dataset_id, a, status, reason, now, actor) for a in alert_ids])

    def insert_note(self, dataset_id: int, account: str, alert_id: str | None, note: str, author: str) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO notes (dataset_id, account, alert_id, note, author, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                      (dataset_id, account, alert_id, note, author, utcnow()))

    def get_notes(self, dataset_id: int, account: str | None = None) -> pd.DataFrame:
        sql = "SELECT note_id, account, alert_id, note, author, created_at FROM notes WHERE dataset_id = ?"
        args: list = [dataset_id]
        if account:
            sql += " AND account = ?"
            args.append(account)
        with self.connect() as c:
            rows = c.execute(sql + " ORDER BY note_id DESC", args).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["note_id", "account", "alert_id", "note", "author", "created_at"])

    def insert_audit(self, actor: str, action: str, dataset_id: int | None, target: str | None, details: dict) -> None:
        with self.connect() as c:
            c.execute("INSERT INTO audit_log (ts, actor, action, dataset_id, target, details_json) VALUES (?, ?, ?, ?, ?, ?)",
                      (utcnow(), actor, action, dataset_id, target, json.dumps(details, default=str)))

    def get_audit(self, dataset_id: int | None = None, limit: int = 500) -> pd.DataFrame:
        sql, args = "SELECT audit_id, ts, actor, action, dataset_id, target, details_json FROM audit_log", []
        if dataset_id is not None:
            sql += " WHERE dataset_id = ?"
            args.append(dataset_id)
        with self.connect() as c:
            rows = c.execute(sql + " ORDER BY audit_id DESC LIMIT ?", args + [limit]).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["audit_id", "ts", "actor", "action", "dataset_id", "target", "details_json"])
