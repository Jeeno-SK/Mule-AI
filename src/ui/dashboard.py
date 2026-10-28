"""Module A: Executive dashboard. Every number is computed from the active dataset and stored analysis."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.reporting.metrics import compute_dashboard_metrics, daily_activity, in_out_for_accounts
from src.ui.common import ACCENT, BAND_COLORS, BAND_ORDER, AppContext, banner, money, require_analysis, style_fig


def render(ctx: AppContext) -> None:
    st.title("Executive dashboard")
    banner(ctx)
    if not require_analysis(ctx):
        return
    m = compute_dashboard_metrics(ctx.tx, ctx.risk, ctx.alert_txs, ctx.review_min_score)
    st.caption(f"Active dataset: **{ctx.dataset['name']}** (id {ctx.dataset_id}) | {ctx.dataset['origin']} | "
               f"labels: {ctx.dataset['label_status']} | analysed {ctx.dataset['analyzed_at']} UTC | detection basis: {ctx.detection_basis}")
    row1 = st.columns(3)
    row1[0].metric("Transactions analysed", f"{m['n_transactions']:,}")
    row1[1].metric("Unique accounts", f"{m['n_accounts']:,}")
    row1[2].metric("Accounts flagged for review", f"{m['flagged_accounts']:,}",
                   help=f"Accounts with heuristic risk score >= {ctx.review_min_score:g}. Flagged means 'review', not 'guilty'.")
    row2 = st.columns(3)
    row2[0].metric("Transactions linked to alerts", f"{m['suspicious_transactions']:,}")
    row2[1].metric("Total transaction value", money(m["total_value"]))
    row2[2].metric("Value linked to alerts", money(m["flagged_value"]),
                   delta=f"{100 * m['flagged_value'] / m['total_value']:.1f}% of total" if m["total_value"] else None, delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        dist = ctx.risk["risk_band"].value_counts().reindex(BAND_ORDER).fillna(0).astype(int)
        fig = go.Figure(go.Bar(x=dist.index, y=dist.values, marker_color=[BAND_COLORS[b] for b in dist.index], text=dist.values, textposition="outside"))
        fig.update_layout(title="Account risk distribution (heuristic bands)", yaxis_title="Accounts")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        daily = daily_activity(ctx.tx, set(ctx.alert_txs["transaction_id"]))
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["transactions"], name="Transactions", marker_color="#1F6F8B")
        fig.add_scatter(x=daily["date"], y=daily["suspicious_transactions"], name="Linked to alerts", mode="lines+markers",
                        line=dict(color="#FB923C"), yaxis="y2")
        fig.update_layout(title="Transaction activity over time", yaxis_title="Transactions",
                          yaxis2=dict(overlaying="y", side="right", title="Linked to alerts", showgrid=False))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    top = ctx.risk[ctx.risk["risk_score"] > 0].head(12)
    c3, c4 = st.columns(2)
    with c3:
        io = in_out_for_accounts(ctx.tx, top["account"].tolist())
        fig = go.Figure()
        fig.add_bar(x=io["account"], y=io["incoming"], name="Incoming value", marker_color=ACCENT)
        fig.add_bar(x=io["account"], y=io["outgoing"], name="Outgoing value", marker_color="#A78BFA")
        fig.update_layout(title="Incoming vs outgoing value: top review accounts", barmode="group")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c4:
        p = ctx.dataset["metadata"].get("analysis_params", {})
        st.subheader("Network activity summary")
        n1, n2 = st.columns(2)
        n1.metric("Connected components", p.get("n_components", 0))
        n2.metric("Communities (Louvain)", p.get("n_communities", 0))
        n3, n4 = st.columns(2)
        n3.metric("Rapid transfer chains", p.get("n_chains", 0))
        n4.metric("Short transfer cycles", p.get("n_cycles", 0))
        st.caption("Connectivity and communities are structural context. They do not prove coordination.")

    st.subheader("Top accounts requiring review")
    cols = ["account", "risk_score", "risk_band", "rules_triggered", "n_alerts", "basis"]
    st.dataframe(ctx.risk[ctx.risk["risk_score"] >= ctx.review_min_score][cols].head(15), use_container_width=True, hide_index=True)
    st.subheader("Recent alerts")
    recent = ctx.alerts.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False).head(15)
    st.dataframe(recent[["timestamp", "rule_id", "rule_name", "severity", "account", "status", "evidence"]], use_container_width=True, hide_index=True)
