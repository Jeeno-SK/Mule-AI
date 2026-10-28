"""Case report assembly for a single account (evidence + analyst context)."""
from __future__ import annotations

import json

import pandas as pd

from src.reporting.exports import DISCLAIMER, provenance


def build_case_report(account: str, dataset: dict, detection_basis: str, risk_row: pd.Series | None, alerts: pd.DataFrame,
                      account_tx: pd.DataFrame, notes: pd.DataFrame, evidence_tx_ids: set[str]) -> dict:
    """Assemble a JSON-serialisable case report. Only stored/observed values are included."""
    tx = account_tx.copy()
    tx["timestamp"] = tx["timestamp"].astype(str)
    evidence = tx[tx["transaction_id"].isin(evidence_tx_ids)] if evidence_tx_ids else tx.head(0)
    cps = pd.concat([tx.loc[tx.source_account == account, "destination_account"], tx.loc[tx.destination_account == account, "source_account"]])
    risk = None if risk_row is None else {"risk_score": float(risk_row["risk_score"]), "risk_band": risk_row["risk_band"],
                                          "basis": risk_row["basis"], "rules_triggered": risk_row["rules_triggered"],
                                          "explanation": risk_row["explanation"]}
    return {
        "provenance": provenance(dataset, detection_basis),
        "account": account,
        "risk_assessment": risk,
        "alerts": [{"alert_id": a.alert_id, "rule_id": a.rule_id, "rule_name": a.rule_name, "severity": a.severity,
                    "status": a.status, "resolution_reason": a.resolution_reason, "timestamp": a.timestamp, "evidence": a.evidence,
                    "observed": json.loads(a.observed_json), "threshold": json.loads(a.threshold_json)}
                   for a in alerts.itertuples(index=False)],
        "connected_accounts": cps.value_counts().head(25).rename_axis("account").reset_index(name="transactions").to_dict("records"),
        "supporting_transactions": evidence.head(100).drop(columns=["extra_json"], errors="ignore").to_dict("records"),
        "total_transactions_for_account": int(len(tx)),
        "analyst_notes": notes.to_dict("records"),
        "disclaimer": DISCLAIMER,
    }


def report_to_markdown(rep: dict) -> str:
    p = rep["provenance"]
    lines = [f"# Case report: {rep['account']}", "",
             f"**Dataset:** {p['dataset_name']} (id {p['dataset_id']})  ", f"**Data origin:** {p['data_origin']}  ",
             f"**Labels:** {p['labels']}  ", f"**Detection basis:** {p['detection_basis']}  ", f"**Generated (UTC):** {p['exported_at_utc']}", "",
             f"> {rep['disclaimer']}", ""]
    r = rep["risk_assessment"]
    if r:
        lines += ["## Risk assessment", f"Score {r['risk_score']:.1f}/100, band **{r['risk_band']}** (basis: {r['basis']}).", "", "```", r["explanation"], "```", ""]
    lines += ["## Alerts (unverified indicators)"]
    lines += [f"- `{a['alert_id']}` {a['rule_id']} {a['rule_name']} [{a['severity']}] status **{a['status']}**"
              + (f" (reason: {a['resolution_reason']})" if a["resolution_reason"] else "") + f"\n  - {a['evidence']}" for a in rep["alerts"]] or ["- none"]
    lines += ["", "## Connected accounts (by transaction count)"]
    lines += [f"- {c['account']}: {c['transactions']}" for c in rep["connected_accounts"]] or ["- none"]
    lines += ["", f"## Supporting transactions ({len(rep['supporting_transactions'])} shown of {rep['total_transactions_for_account']} for this account)"]
    lines += [f"- {t['transaction_id']} | {t['timestamp']} | {t['source_account']} -> {t['destination_account']} | {t['amount']:,.2f}"
              for t in rep["supporting_transactions"]] or ["- none linked to alerts"]
    lines += ["", "## Analyst notes"]
    lines += [f"- {n['created_at']} ({n['author']}): {n['note']}" for n in rep["analyst_notes"]] or ["- none"]
    return "\n".join(lines)
