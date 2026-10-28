"""Deterministic, configurable rule engine. Every alert carries evidence, threshold, severity and supporting transactions.

Thresholds are tunable heuristics (see config/rules.yaml); they are not validated industry standards.
Alerts are *unverified indicators for human review*, not findings of wrongdoing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALERT_COLUMNS = ["alert_id", "rule_id", "rule_name", "severity", "account", "transaction_id", "timestamp",
                 "description", "observed_json", "threshold_json", "evidence"]
MAX_EVIDENCE_TX = 25


@dataclass
class RuleResult:
    alerts: pd.DataFrame
    alert_txs: pd.DataFrame


def make_alert_id(rule_id: str, account: str, transaction_id: str = "") -> str:
    return hashlib.sha1(f"{rule_id}|{account}|{transaction_id}".encode("utf-8")).hexdigest()[:16]


class _Collector:
    def __init__(self, cfg: dict):
        self.cfg = cfg["rules"]
        self.rows: list[dict] = []
        self.txs: list[tuple[str, str]] = []

    def enabled(self, rid: str) -> bool:
        return bool(self.cfg.get(rid, {}).get("enabled", False))

    def params(self, rid: str) -> dict:
        return self.cfg[rid].get("params", {})

    def add(self, rid: str, account: str, observed: dict, threshold: dict, evidence: str, tx_ids: list[str],
            timestamp=None, transaction_id: str = "") -> None:
        r = self.cfg[rid]
        aid = make_alert_id(rid, account, transaction_id)
        self.rows.append({
            "alert_id": aid, "rule_id": rid, "rule_name": r["name"], "severity": r["severity"], "account": account,
            "transaction_id": transaction_id, "timestamp": None if timestamp is None or pd.isna(timestamp) else str(pd.Timestamp(timestamp)),
            "description": r["description"], "observed_json": json.dumps(observed, default=_json_default),
            "threshold_json": json.dumps(threshold, default=_json_default), "evidence": evidence})
        self.txs.extend((aid, t) for t in list(dict.fromkeys(tx_ids))[:MAX_EVIDENCE_TX])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _relative_threshold(series: pd.Series, minimum: float, percentile: float) -> float:
    """max(absolute minimum, dataset percentile among accounts with activity)."""
    active = series[series > 0].dropna()
    return float(max(minimum, np.percentile(active, percentile))) if len(active) else float(minimum)


def _account_tx_ids(tx: pd.DataFrame, account: str, direction: str | None = None, limit: int = MAX_EVIDENCE_TX) -> list[str]:
    if direction == "in":
        m = tx["destination_account"] == account
    elif direction == "out":
        m = tx["source_account"] == account
    else:
        m = (tx["source_account"] == account) | (tx["destination_account"] == account)
    return tx.loc[m, "transaction_id"].head(limit).tolist()


def run_rules(tx: pd.DataFrame, tx_feat: pd.DataFrame, acct_feat: pd.DataFrame, rapid_pairs: pd.DataFrame,
              chains: pd.DataFrame, cycles: list[list[str]], G, config: dict) -> RuleResult:
    """Evaluate all enabled rules and return alerts plus alert->transaction evidence links."""
    c = _Collector(config)
    window_h = config["features"]["window_hours"]
    if acct_feat.empty:
        return RuleResult(pd.DataFrame(columns=ALERT_COLUMNS), pd.DataFrame(columns=["alert_id", "transaction_id"]))
    af = acct_feat.set_index("account")

    if c.enabled("R001"):
        p = c.params("R001")
        th = _relative_threshold(af["peak_tx_window"], p["min_tx_in_window"], p["dataset_percentile"])
        for a, r in af[af["peak_tx_window"] >= th].iterrows():
            c.add("R001", a, {"peak_tx_in_window": int(r.peak_tx_window), "window_hours": window_h},
                  {"threshold": th, **p}, f"{int(r.peak_tx_window)} transactions within {window_h:g}h (threshold {th:g}).",
                  _account_tx_ids(tx, a), r.last_seen)
    if c.enabled("R002"):
        p = c.params("R002")
        for a, g in rapid_pairs.groupby("account"):
            val = float(g["out_amount"].sum())
            if g["out_tx_id"].nunique() >= p["min_rapid_forwards"] and val >= p["min_forwarded_value"]:
                c.add("R002", a, {"rapid_forwards": int(g["out_tx_id"].nunique()), "forwarded_value": round(val, 2),
                                  "median_dwell_hours": round(float(g["dwell_hours"].median()), 2)},
                      {k: v for k, v in p.items()},
                      f"{g['out_tx_id'].nunique()} outgoing transfers left within {config['features']['rapid_forward_hours']:g}h of an incoming "
                      f"transfer, forwarding {val:,.2f} in comparable amounts.",
                      g["in_tx_id"].tolist() + g["out_tx_id"].tolist(), g["out_time"].max())
    if c.enabled("R003"):
        p = c.params("R003")
        th = _relative_threshold(af["unique_senders"], p["min_unique_senders"], p["dataset_percentile"])
        for a, r in af[af["unique_senders"] >= th].iterrows():
            c.add("R003", a, {"unique_senders": int(r.unique_senders), "in_value": round(float(r.in_value), 2)}, {"threshold": th, **p},
                  f"Received from {int(r.unique_senders)} distinct accounts (threshold {th:g}).", _account_tx_ids(tx, a, "in"), r.last_seen)
    if c.enabled("R004"):
        p = c.params("R004")
        th = _relative_threshold(af["unique_receivers"], p["min_unique_receivers"], p["dataset_percentile"])
        for a, r in af[af["unique_receivers"] >= th].iterrows():
            c.add("R004", a, {"unique_receivers": int(r.unique_receivers), "out_value": round(float(r.out_value), 2)}, {"threshold": th, **p},
                  f"Sent to {int(r.unique_receivers)} distinct accounts (threshold {th:g}).", _account_tx_ids(tx, a, "out"), r.last_seen)
    if c.enabled("R005"):
        p = c.params("R005")
        both = np.minimum(af["peak_in_value_window"], af["peak_out_value_window"])
        th = _relative_threshold(both, p["min_window_value"], p["dataset_percentile"])
        for a in both[both >= th].index:
            r = af.loc[a]
            c.add("R005", a, {"peak_in_value": round(float(r.peak_in_value_window), 2), "peak_out_value": round(float(r.peak_out_value_window), 2),
                              "window_hours": window_h}, {"threshold": th, **p},
                  f"Within {window_h:g}h: up to {r.peak_in_value_window:,.2f} in and {r.peak_out_value_window:,.2f} out (threshold {th:,.2f} each).",
                  _account_tx_ids(tx, a), r.last_seen)
    if c.enabled("R006") and not tx_feat.empty:
        p = c.params("R006")
        m = tx_feat[(tx_feat["source_history_count"] >= p["min_history"]) & (tx_feat["amount_robust_z"] >= p["robust_z_threshold"])
                    & (tx_feat["amount"] >= p["min_amount"])]
        for r in m.itertuples(index=False):
            c.add("R006", r.source_account, {"amount": round(float(r.amount), 2), "robust_z": round(float(r.amount_robust_z), 2),
                                             "history_count": int(r.source_history_count)}, p,
                  f"Amount {r.amount:,.2f} has robust z-score {r.amount_robust_z:.1f} versus this account's {int(r.source_history_count)} sent transactions.",
                  [r.transaction_id], r.timestamp, r.transaction_id)
    if c.enabled("R007") and not chains.empty:
        p = c.params("R007")
        seen_accounts: set[str] = set()   # one alert per account: its longest chain (chains are sorted by hops)
        for ch in chains[chains["hops"] >= p["min_hops"]].itertuples(index=False):
            for a in dict.fromkeys(ch.accounts):
                if a in seen_accounts:
                    continue
                seen_accounts.add(a)
                c.add("R007", a, {"chain_id": ch.chain_id, "hops": int(ch.hops), "duration_hours": round(float(ch.duration_hours), 2),
                                  "initial_amount": round(ch.initial_amount, 2), "final_amount": round(ch.final_amount, 2)}, p,
                      f"Part of chain {ch.chain_id}: {ch.hops} time-ordered transfers across {len(set(ch.accounts))} accounts in {ch.duration_hours:.1f}h "
                      f"(amount {ch.initial_amount:,.2f} -> {ch.final_amount:,.2f}).", list(ch.transaction_ids), ch.start_time, ch.chain_id)
    if c.enabled("R008"):
        p = c.params("R008")
        for i, cyc in enumerate(cycles):
            ids = []
            for k in range(len(cyc)):
                s, d = cyc[k], cyc[(k + 1) % len(cyc)]
                ids += tx.loc[(tx.source_account == s) & (tx.destination_account == d), "transaction_id"].head(5).tolist()
            val = float(G.edges[cyc[0], cyc[1]]["weight"]) if G.has_edge(cyc[0], cyc[1]) else 0.0
            for a in cyc:
                c.add("R008", a, {"cycle": cyc, "cycle_length": len(cyc), "first_edge_value": round(val, 2)}, p,
                      f"Account is on a directed transfer cycle of {len(cyc)} accounts: {' -> '.join(cyc + [cyc[0]])}.", ids, None, f"CYC{i}")
    if c.enabled("R010"):
        p = c.params("R010")
        m = af[(af["active_span_hours"] <= p["max_span_hours"]) & (af["tx_count"] >= p["min_tx_count"])
               & ((af["in_value"] + af["out_value"]) >= p["min_total_value"])]
        for a, r in m.iterrows():
            c.add("R010", a, {"observed_span_hours": round(float(r.active_span_hours), 2), "tx_count": int(r.tx_count),
                              "total_value": round(float(r.in_value + r.out_value), 2)}, p,
                  f"{int(r.tx_count)} transactions inside an observed window of {r.active_span_hours:.1f}h. "
                  "This is the span seen in this dataset, not the account's age.", _account_tx_ids(tx, a), r.last_seen)

    alerts = pd.DataFrame(c.rows, columns=ALERT_COLUMNS)
    if c.enabled("R009") and not alerts.empty:
        p = c.params("R009")
        distinct = alerts[alerts["rule_id"] != "R009"].groupby("account")["rule_id"].agg(lambda s: sorted(set(s)))
        acct_alert_ids = alerts.groupby("account")["alert_id"].agg(set).to_dict()
        tx_by_alert: dict[str, list[str]] = {}
        for aid, t in c.txs:
            tx_by_alert.setdefault(aid, []).append(t)
        for a, rules in distinct.items():
            if len(rules) >= p["min_distinct_rules"]:
                ids = [t for aid in acct_alert_ids[a] for t in tx_by_alert.get(aid, [])]
                c.add("R009", a, {"rules": rules, "distinct_rule_count": len(rules)}, p,
                      f"{len(rules)} distinct rules triggered together: {', '.join(rules)}.", ids)
        alerts = pd.DataFrame(c.rows, columns=ALERT_COLUMNS)
    alerts = alerts.drop_duplicates("alert_id").reset_index(drop=True)
    alert_txs = pd.DataFrame(c.txs, columns=["alert_id", "transaction_id"]).drop_duplicates()
    alert_txs = alert_txs[alert_txs["alert_id"].isin(alerts["alert_id"])]
    return RuleResult(alerts, alert_txs.reset_index(drop=True))
