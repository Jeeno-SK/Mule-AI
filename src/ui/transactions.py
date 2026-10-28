"""Modules B, C, J: data management, synthetic generation, validation reporting, transaction explorer and exports."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.data.loader import ingest_dataframe, read_csv_bytes, read_csv_path
from src.data.synthetic_generator import SyntheticConfig, generate_synthetic
from src.data.validator import validate_transactions
from src.investigations.audit_log import log_action
from src.pipeline import run_analysis
from src.reporting.exports import provenance, to_csv_bytes, to_json_bytes
from src.ui.common import AppContext, banner, empty_state, safe_run


def _show_validation(res) -> None:
    s = res.summary
    for msg in res.messages:
        st.warning(msg) if "Missing required" in msg else st.caption(msg)
    if not res.schema_ok:
        st.error("The file does not match the required schema. Nothing was imported.")
        return
    c = st.columns(4)
    c[0].metric("Rows read", f"{s['rows_in']:,}")
    c[1].metric("Rows valid", f"{s['rows_valid']:,}")
    c[2].metric("Rows dropped", f"{s['rows_dropped']:,}")
    c[3].metric("Label status", s["label_status"])
    if s["label_status"] == "none":
        st.info("No fraud labels found. Supervised ML will be skipped; unlabelled rows are NOT treated as legitimate.")
    if len(res.issues):
        st.write("**Validation report** (every dropped, normalised or flagged row)")
        st.dataframe(res.issues.head(1000), use_container_width=True, hide_index=True)
        st.download_button("Download full validation report (CSV)", res.issues.to_csv(index=False).encode("utf-8"),
                           "validation_report.csv", "text/csv")
    else:
        st.success("No validation issues found.")


def _finish_import(ctx: AppContext, dataset_id: int | None, res, run: bool, what: str) -> None:
    if dataset_id is None:
        _show_validation(res)
        return
    log_action(ctx.repo, ctx.actor, "dataset_imported", dataset_id, what, {"rows": res.summary["rows_valid"], "dropped": res.summary["rows_dropped"]})
    if run:
        with st.spinner("Running analysis (features, graph, rules, ML, scoring)..."):
            ok, out = safe_run(run_analysis, ctx.repo, dataset_id)
        if ok:
            st.session_state["flash"] = f"Imported {res.summary['rows_valid']:,} rows from {what} and analysed: {out['alerts']} alerts, {out['flagged_accounts']} accounts flagged for review."
    else:
        st.session_state["flash"] = f"Imported {res.summary['rows_valid']:,} rows from {what}. Run analysis to see results."
    st.session_state["last_validation"] = res
    st.rerun()


def _load_tab(ctx: AppContext) -> None:
    d = ctx.dataset
    if d:
        st.markdown(f"**Active dataset:** {d['name']} | origin **{d['origin']}** | {d['row_count']:,} rows | labels: {d['label_status']} | sha256 `{(d['sha256'] or '')[:16]}...` | source: {d['source']}")
        if st.button("Run analysis on active dataset", type="primary"):
            log_action(ctx.repo, ctx.actor, "analysis_run", ctx.dataset_id, d["name"], {})
            with st.spinner("Running analysis..."):
                ok, out = safe_run(run_analysis, ctx.repo, ctx.dataset_id)
            if ok:
                st.session_state["flash"] = f"Analysis complete: {out['alerts']} alerts, {out['flagged_accounts']} accounts flagged for review."
                st.rerun()
    if "last_validation" in st.session_state:
        with st.expander("Last validation report", expanded=False):
            _show_validation(st.session_state["last_validation"])

    st.subheader("Synthetic demonstration data")
    st.caption("Generated locally with a random seed. Account identifiers are random 'SYN-######' strings; no real people or bank details. "
               "Ground-truth roles are stored separately and never used as model features.")
    with st.form("synthetic_form"):
        a, b, c = st.columns(3)
        seed = a.number_input("Random seed", 0, 10_000_000, 42)
        n_acc = b.number_input("Legitimate accounts", 100, 20000, 1200, step=100)
        n_tx = c.number_input("Background transactions", 500, 200000, 8000, step=500)
        a, b, c, d2 = st.columns(4)
        days = a.number_input("Days", 7, 120, 30)
        rings = b.number_input("Collection rings", 0, 20, 4)
        chains = c.number_input("Pass-through chains", 0, 20, 3)
        fan = d2.number_input("Fan-out networks", 0, 10, 2)
        cyc = st.number_input("Circular-flow rings", 0, 10, 1)
        name = st.text_input("Dataset name", "Synthetic demo", key="syn_name")
        run = st.checkbox("Run analysis after generating", True)
        go = st.form_submit_button("Generate and load synthetic dataset")
    if go:
        cfg = SyntheticConfig(seed=int(seed), n_accounts=int(n_acc), n_legit_tx=int(n_tx), days=int(days), n_collection_rings=int(rings),
                              n_passthrough_chains=int(chains), n_fanout_networks=int(fan), n_cycle_rings=int(cyc))
        with st.spinner("Generating..."):
            syn = generate_synthetic(cfg)
            gt = {r.account: [r.role, bool(r.is_mule_by_design)] for r in syn.ground_truth_accounts.itertuples(index=False)}
            did, res = ingest_dataframe(ctx.repo, syn.transactions.astype(str), f"{name} (seed {cfg.seed})", "synthetic generator", "synthetic",
                                        extra_metadata={"synthetic_config": asdict(cfg), "ground_truth_accounts": gt})
        _finish_import(ctx, did, res, run, "synthetic generator")

    demo = ctx.settings.demo_csv
    if demo.exists() and st.button("Load bundled demo CSV (data/demo_transactions.csv)"):
        ok, out = safe_run(read_csv_path, demo)
        if ok:
            raw, data = out
            did, res = ingest_dataframe(ctx.repo, raw, "Bundled demo CSV (synthetic)", demo.name, "synthetic", raw_bytes=data)
            _finish_import(ctx, did, res, True, demo.name)

    st.subheader("Upload a CSV")
    st.caption("Required columns: transaction_id, timestamp, source_account, destination_account, amount. Optional: transaction_type, is_fraud (0/1). "
               f"Limit {ctx.settings.max_upload_mb} MB / {ctx.settings.max_rows:,} rows. Files are parsed as plain text only; nothing is executed.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        ds_name = st.text_input("Dataset name", up.name.rsplit(".", 1)[0], key="up_name")
        run = st.checkbox("Run analysis after import", True, key="run_after_upload")
        c1, c2 = st.columns(2)
        data = up.getvalue()
        ok, raw = safe_run(read_csv_bytes, data)
        if ok:
            if c1.button("Validate only (preview)"):
                _show_validation(validate_transactions(raw))
            if c2.button("Validate and import", type="primary"):
                did, res = ingest_dataframe(ctx.repo, raw, ds_name or up.name, up.name, "imported", raw_bytes=data)
                _finish_import(ctx, did, res, run, up.name)

    st.subheader("Dataset library")
    lib = ctx.repo.list_datasets()
    if lib.empty:
        empty_state("No datasets yet.")
        return
    st.dataframe(lib, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    target = c1.selectbox("Dataset", lib["dataset_id"].tolist(), format_func=lambda i: f"{i}: {lib.set_index('dataset_id').at[i, 'name']}")
    if c1.button("Make active"):
        ctx.repo.set_active_dataset(int(target))
        st.rerun()
    confirm = c2.checkbox("I understand deleting removes the dataset, its alerts, notes and statuses")
    if c2.button("Delete dataset") and confirm:
        ctx.repo.delete_dataset(int(target))
        log_action(ctx.repo, ctx.actor, "dataset_deleted", int(target), None, {})
        st.rerun()


def _explorer_tab(ctx: AppContext) -> None:
    if ctx.dataset is None or ctx.tx.empty:
        empty_state("Load a dataset to explore transactions.")
        return
    tx = ctx.tx.drop(columns=["extra_json"]).copy()
    risk_map = ctx.risk.set_index("account")["risk_band"].to_dict() if len(ctx.risk) else {}
    if ctx.analysed and len(ctx.alert_txs):
        link = ctx.alert_txs.merge(ctx.alerts[["alert_id", "rule_id", "status"]], on="alert_id")
        rules = link.groupby("transaction_id")["rule_id"].agg(lambda s: ",".join(sorted(set(s))))
        stat = link.groupby("transaction_id")["status"].agg(lambda s: ",".join(sorted(set(s))))
        tx["triggered_rules"] = tx["transaction_id"].map(rules).fillna("")
        tx["alert_status"] = tx["transaction_id"].map(stat).fillna("")
    else:
        tx["triggered_rules"], tx["alert_status"] = "", ""
    tx["source_risk_band"] = tx["source_account"].map(risk_map).fillna("unscored")
    tx["destination_risk_band"] = tx["destination_account"].map(risk_map).fillna("unscored")

    q = st.text_input("Search account or transaction id")
    c1, c2, c3 = st.columns(3)
    dmin, dmax = tx["timestamp"].min().date(), tx["timestamp"].max().date()
    dr = c1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    amin, amax = float(tx["amount"].min()), float(tx["amount"].max())
    ar = c2.slider("Amount range", amin, amax, (amin, amax)) if amax > amin else (amin, amax)
    direction = c3.selectbox("Direction (relative to searched account)", ["all", "incoming", "outgoing"])
    c4, c5, c6 = st.columns(3)
    bands = c4.multiselect("Risk band (either account)", ["none", "low", "medium", "high", "critical", "unscored"])
    rule_opts = sorted({r for v in tx["triggered_rules"] for r in v.split(",") if r})
    rules_sel = c5.multiselect("Triggered rule", rule_opts)
    status_sel = c6.multiselect("Investigation status", ["open", "under_review", "resolved", "escalated"])

    v = tx
    if q.strip():
        t = q.strip().upper()
        acct_hit = (v["source_account"].str.contains(t, regex=False)) | (v["destination_account"].str.contains(t, regex=False))
        if direction == "incoming":
            acct_hit = v["destination_account"].str.contains(t, regex=False)
        elif direction == "outgoing":
            acct_hit = v["source_account"].str.contains(t, regex=False)
        v = v[acct_hit | v["transaction_id"].str.upper().str.contains(t, regex=False)]
    if isinstance(dr, tuple) and len(dr) == 2:
        v = v[(v["timestamp"] >= pd.Timestamp(dr[0])) & (v["timestamp"] < pd.Timestamp(dr[1]) + pd.Timedelta(days=1))]
    v = v[v["amount"].between(ar[0], ar[1])]
    if bands:
        v = v[v["source_risk_band"].isin(bands) | v["destination_risk_band"].isin(bands)]
    if rules_sel:
        v = v[v["triggered_rules"].map(lambda s: any(r in s.split(",") for r in rules_sel))]
    if status_sel:
        v = v[v["alert_status"].map(lambda s: any(r in s.split(",") for r in status_sel))]
    st.caption(f"{len(v):,} of {len(tx):,} transactions match. Showing up to 2,000 rows.")
    st.dataframe(v.head(2000), use_container_width=True, hide_index=True)
    prov = provenance(ctx.dataset, ctx.detection_basis)
    d1, d2 = st.columns(2)
    d1.download_button("Export filtered transactions (CSV)", to_csv_bytes(v, prov), "filtered_transactions.csv", "text/csv")
    d2.download_button("Export filtered transactions (JSON)", to_json_bytes(v, prov), "filtered_transactions.json", "application/json")
    if ctx.analysed:
        e1, e2 = st.columns(2)
        e1.download_button("Export account risk summary (CSV)", to_csv_bytes(ctx.risk.drop(columns=["features_json"]), prov),
                           "account_risk_summary.csv", "text/csv")
        e2.download_button("Export all alerts (CSV)", to_csv_bytes(ctx.alerts, prov), "alerts.csv", "text/csv")


def render(ctx: AppContext) -> None:
    st.title("Data & transactions")
    banner(ctx)
    t1, t2 = st.tabs(["Load and validate data", "Transaction explorer"])
    with t1:
        _load_tab(ctx)
    with t2:
        _explorer_tab(ctx)
