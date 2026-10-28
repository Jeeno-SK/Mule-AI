"""Module I: alert queue, account workspace, notes, status workflow, case reports, audit log."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.investigations import case_manager as cm
from src.investigations.audit_log import read_log
from src.reporting.exports import provenance, to_csv_bytes, to_json_bytes
from src.reporting.report_builder import build_case_report, report_to_markdown
from src.ui.common import AppContext, banner, empty_state, require_analysis, safe_run

NOTE = "Alerts are unverified risk indicators for human review. They are not evidence that an account holder committed a crime."


def _go_graph(account: str) -> None:
    st.session_state["graph_focus"] = account
    st.session_state["nav"] = "Network Graph"


def _alert_queue(ctx: AppContext) -> None:
    a = ctx.alerts
    if a.empty:
        empty_state("No alerts were produced for this dataset.")
        return
    risk_band = ctx.risk.set_index("account")["risk_band"]
    a = a.assign(account_risk_band=a["account"].map(risk_band))
    c1, c2, c3, c4 = st.columns(4)
    status = c1.multiselect("Status", cm.STATUSES, default=["open", "under_review", "escalated"])
    sev = c2.multiselect("Severity", ["high", "medium", "low"])
    rule = c3.multiselect("Rule", sorted(a["rule_id"].unique()))
    band = c4.multiselect("Account risk band", ["low", "medium", "high", "critical"])
    text = st.text_input("Search account or alert id", key="alert_search")
    f = a
    if status:
        f = f[f["status"].isin(status)]
    if sev:
        f = f[f["severity"].isin(sev)]
    if rule:
        f = f[f["rule_id"].isin(rule)]
    if band:
        f = f[f["account_risk_band"].isin(band)]
    if text.strip():
        t = text.strip().upper()
        f = f[f["account"].str.upper().str.contains(t, regex=False) | f["alert_id"].str.upper().str.contains(t, regex=False)]
    st.caption(f"{len(f):,} alerts match ({len(a):,} total). {NOTE}")
    show = ["alert_id", "timestamp", "rule_id", "rule_name", "severity", "account", "account_risk_band", "status", "evidence"]
    st.dataframe(f[show].head(1500), use_container_width=True, hide_index=True)
    prov = provenance(ctx.dataset, ctx.detection_basis)
    e1, e2 = st.columns(2)
    e1.download_button("Export filtered alerts (CSV)", to_csv_bytes(f, prov), "alerts_filtered.csv", "text/csv")
    e2.download_button("Export filtered alerts (JSON)", to_json_bytes(f, prov), "alerts_filtered.json", "application/json")
    if f.empty:
        return
    st.markdown("**Update one alert**")
    options = f["alert_id"].head(1500).tolist()
    labels = f.set_index("alert_id")
    pick = st.selectbox("Alert", options, format_func=lambda i: f"{i} | {labels.at[i, 'rule_id']} | {labels.at[i, 'account']} | {labels.at[i, 'status']}")
    row = labels.loc[pick]
    st.write(row["evidence"])
    with st.expander("Observed values and thresholds"):
        st.json({"observed": json.loads(row["observed_json"]), "threshold": json.loads(row["threshold_json"])})
    _status_form(ctx, [pick], row["account"], "single")
    st.button("Open account workspace", on_click=lambda: st.session_state.update(selected_account=row["account"]), key="open_ws")


def _status_form(ctx: AppContext, alert_ids: list[str], account: str, key: str) -> None:
    with st.form(f"status_{key}"):
        c1, c2 = st.columns([1, 3])
        new = c1.selectbox("New status", cm.STATUSES, key=f"st_{key}")
        reason = c2.text_input("Resolution reason (required when resolving)", key=f"rs_{key}")
        apply_all = st.checkbox("Apply to ALL alerts for this account", key=f"all_{key}")
        if st.form_submit_button("Save status"):
            if apply_all:
                ok, n = safe_run(cm.set_status_for_account, ctx.repo, ctx.dataset_id, account, new, reason, ctx.actor)
            else:
                ok, n = safe_run(cm.set_alert_status, ctx.repo, ctx.dataset_id, alert_ids, new, reason, ctx.actor)
            if ok:
                st.session_state["flash"] = f"Updated {n} alert(s) to '{new}'."
                st.rerun()


def _workspace(ctx: AppContext) -> None:
    q = st.text_input("Search accounts or transaction ids", key="ws_search")
    options: list[str] = ctx.risk[ctx.risk["risk_score"] > 0]["account"].head(100).tolist()
    if q.strip():
        hits = ctx.repo.search(ctx.dataset_id, q)
        tx_accts = pd.concat([hits["transactions"]["source_account"], hits["transactions"]["destination_account"]]).tolist() if len(hits["transactions"]) else []
        options = list(dict.fromkeys(hits["accounts"]["account"].tolist() + tx_accts))
        if len(hits["transactions"]):
            st.caption("Matching transactions")
            st.dataframe(hits["transactions"], use_container_width=True, hide_index=True)
        if not options:
            empty_state("No accounts or transactions match that search.")
            return
    sel = st.session_state.get("selected_account")
    if sel and sel not in options:
        options = [sel] + options
    if not options:
        empty_state("No accounts with risk indicators. Use search to open any account.")
        return
    idx = options.index(sel) if sel in options else 0
    account = st.selectbox("Account", options, index=idx)
    st.session_state["selected_account"] = account

    rr = ctx.risk[ctx.risk["account"] == account]
    risk_row = rr.iloc[0] if len(rr) else None
    atx = ctx.tx[(ctx.tx["source_account"] == account) | (ctx.tx["destination_account"] == account)]
    c = st.columns(4)
    c[0].metric("Risk score (heuristic)", "n/a" if risk_row is None else f"{risk_row['risk_score']:.1f}")
    c[1].metric("Risk band", "n/a" if risk_row is None else risk_row["risk_band"])
    c[2].metric("Transactions", f"{len(atx):,}")
    c[3].metric("Alerts", 0 if risk_row is None else int(risk_row["n_alerts"]))
    st.caption(NOTE)
    if risk_row is not None:
        st.markdown("**Why this account was prioritised**")
        st.text(risk_row["explanation"])
        feats = json.loads(risk_row["features_json"])
        with st.expander("Observed account features"):
            st.dataframe(pd.DataFrame({"feature": list(feats), "value": list(feats.values())}), hide_index=True, use_container_width=True)
    st.button("Show in network graph", on_click=_go_graph, args=(account,), key="to_graph")

    aal = ctx.alerts[ctx.alerts["account"] == account]
    st.markdown("**Alerts and evidence**")
    if aal.empty:
        st.write("No alerts for this account.")
    else:
        st.dataframe(aal[["alert_id", "timestamp", "rule_id", "rule_name", "severity", "status", "resolution_reason", "evidence"]],
                     use_container_width=True, hide_index=True)
        _status_form(ctx, aal["alert_id"].tolist(), account, "acct")
    ev_ids = set(ctx.alert_txs.loc[ctx.alert_txs["alert_id"].isin(aal["alert_id"]), "transaction_id"]) if len(aal) else set()
    t1, t2, t3 = st.tabs(["Incoming", "Outgoing", "Connected accounts"])
    cols = ["transaction_id", "timestamp", "source_account", "destination_account", "amount", "transaction_type"]
    t1.dataframe(atx[atx["destination_account"] == account][cols], use_container_width=True, hide_index=True)
    t2.dataframe(atx[atx["source_account"] == account][cols], use_container_width=True, hide_index=True)
    with t3:
        inc = atx[atx["destination_account"] == account].groupby("source_account").agg(transactions=("amount", "size"), value=("amount", "sum"))
        out = atx[atx["source_account"] == account].groupby("destination_account").agg(transactions=("amount", "size"), value=("amount", "sum"))
        conn = pd.concat([inc.assign(direction="received from"), out.assign(direction="sent to")]).rename_axis("counterparty").reset_index()
        bands = ctx.risk.set_index("account")["risk_band"]
        conn["counterparty_risk_band"] = conn["counterparty"].map(bands)
        st.dataframe(conn.sort_values("value", ascending=False), use_container_width=True, hide_index=True)
    if ev_ids:
        st.markdown(f"**Supporting transactions referenced by alerts** ({len(ev_ids)})")
        st.dataframe(atx[atx["transaction_id"].isin(ev_ids)][cols], use_container_width=True, hide_index=True)

    st.markdown("**Analyst notes**")
    with st.form("note_form", clear_on_submit=True):
        note = st.text_area("Add an investigation note", max_chars=cm.MAX_NOTE)
        if st.form_submit_button("Save note"):
            ok, _ = safe_run(cm.add_note, ctx.repo, ctx.dataset_id, account, note, ctx.actor)
            if ok:
                st.session_state["flash"] = "Note saved."
                st.rerun()
    notes = ctx.repo.get_notes(ctx.dataset_id, account)
    if notes.empty:
        st.caption("No notes yet.")
    else:
        st.dataframe(notes[["created_at", "author", "note"]], use_container_width=True, hide_index=True)

    rep = build_case_report(account, ctx.dataset, ctx.detection_basis, risk_row, aal, atx, notes, ev_ids)
    d1, d2 = st.columns(2)
    d1.download_button("Export case report (Markdown)", report_to_markdown(rep), f"case_{account}.md", "text/markdown")
    d2.download_button("Export case report (JSON)", json.dumps(rep, indent=2, default=str), f"case_{account}.json", "application/json")


def render(ctx: AppContext) -> None:
    st.title("Alerts & investigations")
    banner(ctx)
    if not require_analysis(ctx):
        return
    t1, t2, t3 = st.tabs(["Alert queue", "Account workspace", "Audit log"])
    with t1:
        _alert_queue(ctx)
    with t2:
        _workspace(ctx)
    with t3:
        log = read_log(ctx.repo, ctx.dataset_id)
        if log.empty:
            empty_state("No audited actions yet for this dataset.")
        else:
            st.dataframe(log, use_container_width=True, hide_index=True)
