"""Module G: interactive directed transaction graph (NetworkX layout, Plotly rendering)."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.graph import graph_analytics as ga
from src.graph.network_builder import build_graph, filter_transactions
from src.ui.common import BAND_COLORS, AppContext, banner, empty_state, require_analysis

MAX_ARROWS = 250


def _figure(G: nx.DiGraph, focus: str | None) -> go.Figure:
    pos = ga.compute_layout(G)
    fig = go.Figure()
    xs, ys = [], []
    for u, v in G.edges:
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(width=0.8, color="#3B4A68"), hoverinfo="skip", showlegend=False)
    ex, ey, et = [], [], []
    for u, v, d in G.edges(data=True):
        ex.append((pos[u][0] + pos[v][0]) / 2)
        ey.append((pos[u][1] + pos[v][1]) / 2)
        et.append(f"{u} -> {v}<br>transfers: {d['count']}<br>total: {d['weight']:,.2f}<br>{d['first_ts']} to {d['last_ts']}<br>types: {d['types']}")
    fig.add_scatter(x=ex, y=ey, mode="markers", marker=dict(size=7, color="rgba(0,0,0,0)"), hovertext=et, hoverinfo="text", showlegend=False)
    for i, (u, v) in enumerate(G.edges):
        if i >= MAX_ARROWS:
            break
        fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=0.8, arrowcolor="#5B6B8C", standoff=8, opacity=0.8)
    for band in ["unscored", "none", "low", "medium", "high", "critical"]:
        nodes = [n for n in G.nodes if G.nodes[n]["risk_band"] == band]
        if not nodes:
            continue
        fig.add_scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=band,
            marker=dict(size=[min(34, 9 + 3 * (G.degree(n)) ** 0.5 + (8 if n == focus else 0)) for n in nodes], color=BAND_COLORS[band],
                        line=dict(width=[3 if n == focus else 0.5 for n in nodes], color="#FFFFFF")),
            hovertext=[f"<b>{n}</b><br>risk score: {G.nodes[n]['risk_score']:.1f} ({band})<br>in-degree {G.in_degree(n)}, out-degree {G.out_degree(n)}"
                       f"<br>received {G.in_degree(n, weight='weight'):,.0f}, sent {G.out_degree(n, weight='weight'):,.0f}" for n in nodes], hoverinfo="text")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=640, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h", y=1.02), dragmode="pan")
    return fig


@st.cache_data(show_spinner=False)
def _chains_cycles(_tx: pd.DataFrame, dataset_id: int, analyzed_at: str, g: dict, min_hops: int, min_value: float):
    chains, warn = ga.find_rapid_chains(_tx, g["max_chain_gap_hours"], g["chain_ratio_min"], g["chain_ratio_max"], min_hops, g["max_chains"])
    G = build_graph(_tx)
    cycles = ga.find_cycles(G, g["max_cycle_length"], g["max_cycles"], 3, min_value, g.get("max_cycle_span_hours", 72.0))
    return chains, warn, cycles


def render(ctx: AppContext) -> None:
    st.title("Network graph")
    banner(ctx)
    if not require_analysis(ctx):
        return
    tx = ctx.tx
    st.caption("Nodes are accounts, arrows are aggregated transfers. Colour = heuristic risk band. **Graph connectivity is context for review, not proof of criminal coordination.**")
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("View", ["Account neighbourhood", "Highest-risk accounts"], horizontal=True)
        hops = c2.slider("Neighbourhood hops", 1, 3, 1)
        max_nodes = c3.slider("Maximum nodes drawn", 20, ctx.settings.max_graph_nodes, 120)
        d1, d2, d3 = st.columns(3)
        dmin, dmax = tx["timestamp"].min().date(), tx["timestamp"].max().date()
        dr = d1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax, key="g_dates")
        amin, amax = float(tx["amount"].min()), float(tx["amount"].max())
        ar = d2.slider("Edge amount range (per transaction)", amin, amax, (amin, amax), key="g_amt") if amax > amin else (amin, amax)
        types = d3.multiselect("Transaction types", sorted(tx["transaction_type"].dropna().unique()))
        min_risk = st.slider("Minimum node risk score (Highest-risk view)", 0, 100, int(ctx.review_min_score))
    sub = filter_transactions(tx, tuple(dr) if isinstance(dr, tuple) and len(dr) == 2 else None, ar, types or None)
    if sub.empty:
        empty_state("No transactions match these filters.")
        return
    G = build_graph(sub, ctx.risk)

    focus = None
    if mode == "Account neighbourhood":
        top = ctx.risk[ctx.risk["risk_score"] > 0]["account"].head(200).tolist()
        pre = st.session_state.get("graph_focus") or st.session_state.get("selected_account")
        typed = st.text_input("Focus account (type an exact id, or pick below)", value=pre or "")
        options = ([typed.strip().upper()] if typed.strip().upper() in G else []) + [a for a in top if a in G]
        options = list(dict.fromkeys(options))
        if not options:
            empty_state("The focus account is not present under the current filters.", "Clear the filters or pick another account.")
            return
        focus = st.selectbox("Account", options)
        st.session_state["graph_focus"] = focus
        H = ga.neighbourhood(G, focus, hops, max_nodes)
    else:
        seeds = [n for n in G.nodes if G.nodes[n]["risk_score"] >= min_risk]
        if not seeds:
            empty_state("No accounts reach that risk score under the current filters.")
            return
        seeds = sorted(seeds, key=lambda n: G.nodes[n]["risk_score"], reverse=True)[: max(5, max_nodes // 3)]
        nodes = set(seeds)
        for s in seeds:
            nodes |= set(G.predecessors(s)) | set(G.successors(s))
        nodes = set(sorted(nodes, key=lambda n: G.nodes[n]["risk_score"], reverse=True)[:max_nodes])
        H = G.subgraph(nodes).copy()
    st.caption(f"Showing {H.number_of_nodes()} accounts and {H.number_of_edges()} edges (of {G.number_of_nodes()} / {G.number_of_edges()} under the filters). "
               + ("Arrowheads are limited to the first %d edges." % MAX_ARROWS if H.number_of_edges() > MAX_ARROWS else ""))
    st.plotly_chart(_figure(H, focus), use_container_width=True)

    st.subheader("Inspect an account")
    inspect = st.selectbox("Account in view", sorted(H.nodes, key=lambda n: -H.nodes[n]["risk_score"]), key="g_inspect")
    c1, c2 = st.columns(2)
    inc = pd.DataFrame([{"from": u, "transfers": d["count"], "total": d["weight"], "last": d["last_ts"], "from_risk": G.nodes[u]["risk_band"]}
                        for u, _, d in G.in_edges(inspect, data=True)])
    out = pd.DataFrame([{"to": v, "transfers": d["count"], "total": d["weight"], "last": d["last_ts"], "to_risk": G.nodes[v]["risk_band"]}
                        for _, v, d in G.out_edges(inspect, data=True)])
    c1.write("Incoming connections")
    c1.dataframe(inc.sort_values("total", ascending=False) if len(inc) else inc, use_container_width=True, hide_index=True)
    c2.write("Outgoing connections")
    c2.dataframe(out.sort_values("total", ascending=False) if len(out) else out, use_container_width=True, hide_index=True)
    st.button("Open in investigation workspace", on_click=lambda: st.session_state.update(selected_account=inspect, nav="Alerts & Investigations"))

    with st.expander("Network analytics (whole dataset, unfiltered)"):
        cfg = ctx.config
        r7, r8 = cfg["rules"]["R007"]["params"], cfg["rules"]["R008"]["params"]
        chains, warn, cycles = _chains_cycles(ctx.tx, ctx.dataset_id, ctx.dataset["analyzed_at"], cfg["graph"], r7["min_hops"], r8["min_cycle_value"])
        full = build_graph(ctx.tx, ctx.risk)
        if warn:
            st.warning(warn)
        st.write("**Connected components (weak)**")
        st.dataframe(ga.component_table(full, 10), use_container_width=True, hide_index=True)
        st.write("**Time-ordered rapid transfer chains**")
        st.dataframe(chains.assign(accounts=chains["accounts"].map(" -> ".join), transaction_ids=chains["transaction_ids"].map(len)).head(50)
                     if len(chains) else chains, use_container_width=True, hide_index=True)
        st.write("**Short transfer cycles (time- and value-filtered)**")
        st.dataframe(pd.DataFrame({"cycle": [" -> ".join(c + [c[0]]) for c in cycles]}), use_container_width=True, hide_index=True)
