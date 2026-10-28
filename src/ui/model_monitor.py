"""Module K: data quality, label/detection distributions, model evaluation, feature comparison and drift foundation."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.transaction_features import compute_account_features
from src.monitoring.drift import compare_feature_tables, temporal_split_drift
from src.ui.common import BAND_COLORS, BAND_ORDER, AppContext, banner, require_analysis, style_fig


@st.cache_data(show_spinner=False)
def _features(_tx: pd.DataFrame, dataset_id: int, analyzed_at: str | None) -> pd.DataFrame:
    return compute_account_features(_tx)


def _quality(ctx: AppContext) -> None:
    v = ctx.dataset["metadata"].get("validation_summary", {})
    c = st.columns(4)
    c[0].metric("Rows read", f"{v.get('rows_in', 0):,}")
    c[1].metric("Rows valid", f"{v.get('rows_valid', 0):,}")
    c[2].metric("Rows dropped", f"{v.get('rows_dropped', 0):,}")
    c[3].metric("Self-transfers kept", f"{v.get('self_transfers', 0):,}")
    c = st.columns(4)
    c[0].metric("Missing required values", v.get("missing_required_values", 0))
    c[1].metric("Invalid amounts", v.get("invalid_amounts", 0))
    c[2].metric("Invalid timestamps", v.get("invalid_timestamps", 0))
    c[3].metric("Duplicate ids", v.get("duplicate_ids", 0))
    if v.get("issue_counts"):
        st.dataframe(pd.DataFrame({"issue": list(v["issue_counts"]), "rows": list(v["issue_counts"].values())}), hide_index=True, use_container_width=True)
    st.write(f"Date range in data: {v.get('date_min')} to {v.get('date_max')}")
    st.subheader("Label distribution")
    tx = ctx.tx
    if ctx.dataset["label_status"] == "none":
        st.info("This dataset has no labels. Class distribution and supervised evaluation are unavailable; unlabelled rows are not assumed legitimate.")
    else:
        lab = tx["is_fraud"].map({1.0: "labelled fraud", 0.0: "labelled non-fraud"}).fillna("unlabelled").value_counts()
        st.dataframe(lab.rename_axis("class").reset_index(name="transactions"), hide_index=True, use_container_width=True)
        st.caption("Labels are provided by the dataset (for synthetic data: by the generator). They are not verified real-world outcomes.")


def _detections(ctx: AppContext) -> None:
    c1, c2 = st.columns(2)
    with c1:
        r = ctx.alerts["rule_id"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=r.index, y=r.values, marker_color="#22B8CF"))
        fig.update_layout(title="Alerts by rule")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        b = ctx.risk["risk_band"].value_counts().reindex(BAND_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=b.index, y=b.values, marker_color=[BAND_COLORS[x] for x in b.index]))
        fig.update_layout(title="Accounts by risk band")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.dataframe(ctx.alerts.groupby(["rule_id", "rule_name", "severity"]).size().reset_index(name="alerts"), hide_index=True, use_container_width=True)


def _models(ctx: AppContext) -> None:
    runs = ctx.repo.get_model_runs(ctx.dataset_id)
    if not runs:
        st.info("No model run recorded.")
        return
    meta = runs[0]["metadata"]
    st.caption(f"Latest run #{runs[0]['run_id']} at {runs[0]['created_at']} UTC | scikit-learn {meta.get('sklearn_version')} | {meta.get('note', '')}")
    for key, title in (("unsupervised", "Unsupervised: IsolationForest"), ("supervised", "Supervised: RandomForest")):
        m = meta.get(key, {})
        st.markdown(f"**{title}** - status: `{m.get('status')}`")
        if m.get("status") != "trained":
            st.write(m.get("reason", "Not trained."))
            continue
        if key == "unsupervised":
            st.write(f"Trained on {m['n_accounts']} accounts. Output is a within-dataset anomaly percentile, not a fraud probability.")
            continue
        st.write(f"Label definition: {m['label_definition']}. Labelled accounts: {m['n_labelled_accounts']} "
                 f"({m['n_positive_accounts']} positive / {m['n_negative_accounts']} negative).")
        for name, ev in m["evaluation"].items():
            st.write(f"*{name.replace('_', ' ')}*")
            if ev.get("status") == "skipped":
                st.write(f"Skipped: {ev['reason']}")
                continue
            cc = st.columns(5)
            cc[0].metric("Precision", f"{ev['precision']:.3f}")
            cc[1].metric("Recall", f"{ev['recall']:.3f}")
            cc[2].metric("F1", f"{ev['f1']:.3f}")
            cc[3].metric("PR-AUC", "n/a" if ev["pr_auc"] is None else f"{ev['pr_auc']:.3f}")
            cc[4].metric("Test accounts", ev["n_test"])
            cm = ev["confusion_matrix"]
            st.dataframe(pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["predicted 0", "predicted 1"]), use_container_width=True)
            if ev.get("caveat"):
                st.caption(ev["caveat"])
        st.caption("Evaluation is on held-out accounts of this dataset only. It says nothing about performance on real banking data.")
    with st.expander("Model metadata (JSON)"):
        st.json(meta)


def _features_and_drift(ctx: AppContext) -> None:
    feats = _features(ctx.tx, ctx.dataset_id, ctx.dataset["analyzed_at"])
    flagged = set(ctx.risk.loc[ctx.risk["risk_score"] >= ctx.review_min_score, "account"])
    f = feats.assign(group=feats["account"].map(lambda a: "flagged for review" if a in flagged else "not flagged"))
    cols = ["in_value", "out_value", "unique_senders", "unique_receivers", "flow_balance_ratio", "peak_tx_window",
            "rapid_forward_count", "median_dwell_hours", "active_span_hours"]
    st.write("**Median feature values: flagged vs not flagged** (descriptive; flagged accounts were selected using these signals)")
    st.dataframe(f.groupby("group")[cols].median().T.rename_axis("feature").reset_index(), hide_index=True, use_container_width=True)
    pick = st.selectbox("Distribution of feature", cols)
    fig = px.histogram(f, x=pick, color="group", barmode="overlay", nbins=40, log_y=True, opacity=0.7, color_discrete_sequence=["#64748B", "#FB923C"])
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    st.subheader("Drift foundation (Population Stability Index)")
    st.caption("PSI only describes how a distribution moved. It does not show whether future fraud patterns can be detected. "
               "Rule of thumb: <0.1 small, 0.1-0.25 moderate, >0.25 large (a convention, not validated here).")
    st.write("First half vs second half of this dataset's time span")
    st.dataframe(temporal_split_drift(ctx.tx), hide_index=True, use_container_width=True)
    lib = ctx.repo.list_datasets()
    others = lib[lib["dataset_id"] != ctx.dataset_id]
    if others.empty:
        st.caption("Load a second dataset to compare account-feature distributions across datasets.")
        return
    other = st.selectbox("Compare with dataset", others["dataset_id"].tolist(), format_func=lambda i: f"{i}: {others.set_index('dataset_id').at[i, 'name']}")
    otx = ctx.repo.get_transactions(int(other))
    st.dataframe(compare_feature_tables(_features(otx, int(other), None), feats, cols), hide_index=True, use_container_width=True)


def _synthetic_check(ctx: AppContext) -> None:
    gt = ctx.dataset["metadata"].get("ground_truth_accounts")
    if ctx.dataset["origin"] != "synthetic" or not gt:
        st.info("Ground-truth comparison is only available for generated synthetic datasets.")
        return
    g = pd.DataFrame([{"account": a, "role": r[0], "designed_mule": r[1]} for a, r in gt.items()])
    m = ctx.risk.merge(g, on="account", how="inner")
    m["flagged"] = m["risk_score"] >= ctx.review_min_score
    tp = int((m["flagged"] & m["designed_mule"]).sum())
    fp = int((m["flagged"] & ~m["designed_mule"]).sum())
    fn = int((~m["flagged"] & m["designed_mule"]).sum())
    c = st.columns(3)
    c[0].metric("Designed mule accounts", int(m["designed_mule"].sum()))
    c[1].metric("Recall on designed mules", f"{tp / max(tp + fn, 1):.2f}")
    c[2].metric("Precision of flagged set", f"{tp / max(tp + fp, 1):.2f}")
    st.dataframe(m.groupby("role").agg(accounts=("account", "size"), flagged=("flagged", "sum"), mean_score=("risk_score", "mean")).reset_index()
                 .sort_values("accounts", ascending=False), hide_index=True, use_container_width=True)
    st.caption("This compares output with the generator's own design on synthetic data. It is a sanity check, not a benchmark, and says nothing about real-world accuracy. "
               "Victim, merchant and payroll accounts are intentionally included as hard negatives.")


def render(ctx: AppContext) -> None:
    st.title("Data quality & model monitoring")
    banner(ctx)
    if not require_analysis(ctx):
        return
    d = ctx.dataset
    st.caption(f"Dataset id {d['dataset_id']} | sha256 `{(d['sha256'] or '')[:24]}...` | origin {d['origin']} | created {d['created_at']} UTC | analysis config version {ctx.config.get('version')}")
    tabs = st.tabs(["Data quality", "Detections", "Model evaluation", "Features & drift", "Synthetic ground-truth check"])
    with tabs[0]:
        _quality(ctx)
    with tabs[1]:
        _detections(ctx)
    with tabs[2]:
        _models(ctx)
    with tabs[3]:
        _features_and_drift(ctx)
    with tabs[4]:
        _synthetic_check(ctx)
