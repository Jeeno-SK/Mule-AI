"""Transparent, configurable heuristic risk scoring.

score = 100 * (rule_points + graph_points + ml_points) / available_max, clipped to [0, 100]
  rule_points  = min(max_rule_points, sum over DISTINCT scoring rules of severity points)
  graph_points = graph_weight * min(1, flagged_counterparty_share / share_for_full)   [only if the account has
                 its own rule evidence: association alone never raises an account]
  ml_points    = ml_weight * ml_value  (supervised score if available, else rescaled top-decile anomaly percentile)
available_max drops the ML weight when no ML output exists, so rules-only scores still span 0-100.
This is a prioritisation heuristic, NOT a calibrated probability of fraud and NOT a finding of wrongdoing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.detection.ml_detector import MLResult

PERCENTILE_FEATURES = ["in_value", "out_value", "unique_senders", "unique_receivers", "peak_tx_window",
                       "peak_in_value_window", "rapid_forward_count", "flow_balance_ratio"]
DISCLAIMER = "Heuristic risk indicator for prioritising human review. Not a probability of fraud and not a finding of wrongdoing."


def risk_band(score: float, bands: dict) -> str:
    if score <= 0:
        return "none"
    for name in ("critical", "high", "medium", "low"):
        if score >= bands[name]:
            return name
    return "low"


def compute_risk_scores(acct_feat: pd.DataFrame, graph_feat: pd.DataFrame, alerts: pd.DataFrame, ml: MLResult | None,
                        tx: pd.DataFrame, scoring_cfg: dict) -> pd.DataFrame:
    cols = ["account", "risk_score", "risk_band", "rule_points", "graph_points", "ml_points", "rules_triggered",
            "n_alerts", "basis", "explanation", "features_json"]
    if acct_feat.empty:
        return pd.DataFrame(columns=cols)
    sev = scoring_cfg["severity_points"]
    w = scoring_cfg["weights"]
    non_scoring = set(scoring_cfg.get("non_scoring_rules", []))
    accounts = acct_feat["account"].tolist()
    a = alerts if alerts is not None and len(alerts) else pd.DataFrame(columns=["rule_id", "rule_name", "severity", "account", "evidence"])
    scoring = a[~a["rule_id"].isin(non_scoring) & ~a["rule_id"].str.startswith("ML")]

    # distinct-rule points per account
    per_rule = scoring.assign(pts=scoring["severity"].map(sev)).groupby(["account", "rule_id"])["pts"].max().reset_index()
    rule_pts = per_rule.groupby("account")["pts"].sum().clip(upper=scoring_cfg["max_rule_points"]).reindex(accounts).fillna(0.0)

    # flagged counterparty share
    flagged = set(rule_pts[rule_pts > 0].index)
    if len(tx):
        pr = pd.concat([tx[["source_account", "destination_account"]].set_axis(["account", "cp"], axis=1),
                        tx[["destination_account", "source_account"]].set_axis(["account", "cp"], axis=1)]).drop_duplicates()
        pr = pr[pr["account"] != pr["cp"]]
        pr["cp_flagged"] = pr["cp"].isin(flagged)
        g = pr.groupby("account")["cp_flagged"].agg(["sum", "size"])
    else:
        g = pd.DataFrame(columns=["sum", "size"])
    n_flag = g["sum"].reindex(accounts).fillna(0)
    n_cp = g["size"].reindex(accounts).fillna(0)
    share = (n_flag / n_cp.where(n_cp > 0, 1)).where(n_cp > 0, 0.0)
    graph_pts = (w["graph"] * (share / scoring_cfg["graph_neighbour_share_for_full_points"]).clip(upper=1.0)).where((rule_pts > 0) & (n_flag >= 1), 0.0)

    # ML
    ml_ok = bool(ml is not None and ml.available)
    ml_sc = ml.scores.set_index("account") if ml_ok else pd.DataFrame()
    sup = ml_sc["ml_supervised_score"].reindex(accounts) if ml_ok else pd.Series(np.nan, index=accounts)
    ano = ml_sc["ml_anomaly_percentile"].reindex(accounts) if ml_ok else pd.Series(np.nan, index=accounts)
    ml_val = sup.where(sup.notna(), ((ano - 0.9) / 0.1).clip(lower=0, upper=1)).fillna(0.0)
    ml_pts = (w["ml"] * ml_val) if ml_ok else pd.Series(0.0, index=accounts)
    avail = w["rules"] + w["graph"] + (w["ml"] if ml_ok else 0)
    raw = rule_pts.values + graph_pts.values + ml_pts.values
    # rule_points are on a 0..max_rule_points scale; normalise against configured rule weight
    rule_scaled = rule_pts.values * (w["rules"] / max(scoring_cfg["max_rule_points"], 1))
    score = np.clip(100.0 * (rule_scaled + graph_pts.values + ml_pts.values) / avail, 0, 100).round(1)

    pct = acct_feat.set_index("account")[PERCENTILE_FEATURES].rank(pct=True)
    by_acct = {k: grp for k, grp in a.groupby("account")}
    basis = "rules only" if not ml_ok else ("rules + supervised model" if sup.notna().any() else "rules + anomaly model")
    feat_idx = acct_feat.set_index("account")
    rows = []
    for i, acct in enumerate(accounts):
        grp = by_acct.get(acct)
        rules_list = sorted(set(grp["rule_id"])) if grp is not None else []
        lines = [f"Priority {risk_band(score[i], scoring_cfg['bands']).upper()} (score {score[i]:.1f}/100, basis: {basis}). {DISCLAIMER}"]
        if grp is not None:
            for r in grp.sort_values("severity").itertuples(index=False):
                lines.append(f"- {r.rule_id} {r.rule_name} [{r.severity}]: {r.evidence}")
        if graph_pts.iloc[i] > 0:
            lines.append(f"- Graph context: {int(n_flag.iloc[i])} of {int(n_cp.iloc[i])} direct counterparties also have rule alerts "
                         "(connectivity is context, not proof of coordination).")
        if ml_ok and ml_val.iloc[i] > 0:
            lines.append(f"- Model signal: {'supervised score %.2f' % sup.iloc[i] if sup.notna().iloc[i] else 'anomaly percentile %.3f' % ano.iloc[i]} (uncalibrated; not a probability).")
        top = pct.loc[acct][pct.loc[acct] >= 0.95].sort_values(ascending=False).head(3)
        if len(top) and score[i] > 0:
            obs = ", ".join(f"{k}={feat_idx.at[acct, k]:,.2f} (dataset percentile {v:.0%})" for k, v in top.items())
            lines.append(f"- Notable observed values: {obs}.")
        if score[i] == 0:
            lines = [f"No risk indicators were triggered in this dataset (basis: {basis}). This does not establish that the account is legitimate."]
        fj = {k: (None if pd.isna(feat_idx.at[acct, k]) else round(float(feat_idx.at[acct, k]), 4)) for k in
              ["in_value", "out_value", "in_count", "out_count", "unique_senders", "unique_receivers", "flow_balance_ratio",
               "peak_tx_window", "rapid_forward_count", "median_dwell_hours", "active_span_hours"]}
        rows.append({"account": acct, "risk_score": float(score[i]), "risk_band": risk_band(score[i], scoring_cfg["bands"]),
                     "rule_points": float(rule_scaled[i]), "graph_points": float(graph_pts.iloc[i]), "ml_points": float(ml_pts.iloc[i]),
                     "rules_triggered": ",".join(rules_list), "n_alerts": 0 if grp is None else len(grp), "basis": basis,
                     "explanation": "\n".join(lines), "features_json": json.dumps(fj)})
    return pd.DataFrame(rows, columns=cols).sort_values(["risk_score", "account"], ascending=[False, True]).reset_index(drop=True)
