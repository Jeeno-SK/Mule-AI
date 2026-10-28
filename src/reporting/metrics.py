"""Dashboard metrics computed only from the active dataset and persisted analysis results."""
from __future__ import annotations

import pandas as pd


def compute_dashboard_metrics(tx: pd.DataFrame, risk: pd.DataFrame, alert_txs: pd.DataFrame, review_min_score: float) -> dict:
    if tx.empty:
        return {"n_transactions": 0, "n_accounts": 0, "flagged_accounts": 0, "suspicious_transactions": 0,
                "total_value": 0.0, "flagged_value": 0.0}
    suspicious_ids = set(alert_txs["transaction_id"]) if len(alert_txs) else set()
    flagged_tx = tx[tx["transaction_id"].isin(suspicious_ids)]
    return {
        "n_transactions": int(len(tx)),
        "n_accounts": int(pd.concat([tx["source_account"], tx["destination_account"]]).nunique()),
        "flagged_accounts": int((risk["risk_score"] >= review_min_score).sum()) if len(risk) else 0,
        "suspicious_transactions": int(len(flagged_tx)),
        "total_value": float(tx["amount"].sum()),
        "flagged_value": float(flagged_tx["amount"].sum()),
    }


def daily_activity(tx: pd.DataFrame, suspicious_ids: set[str]) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame(columns=["date", "transactions", "value", "suspicious_transactions"])
    d = tx.assign(date=tx["timestamp"].dt.floor("D"), susp=tx["transaction_id"].isin(suspicious_ids))
    return d.groupby("date").agg(transactions=("amount", "size"), value=("amount", "sum"),
                                 suspicious_transactions=("susp", "sum")).reset_index()


def in_out_for_accounts(tx: pd.DataFrame, accounts: list[str]) -> pd.DataFrame:
    inc = tx[tx["destination_account"].isin(accounts)].groupby("destination_account")["amount"].sum()
    out = tx[tx["source_account"].isin(accounts)].groupby("source_account")["amount"].sum()
    return pd.DataFrame({"account": accounts, "incoming": inc.reindex(accounts).fillna(0).values,
                         "outgoing": out.reindex(accounts).fillna(0).values})
