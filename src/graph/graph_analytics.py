"""Graph analytics: degrees, components, communities, cycles, time-ordered transfer chains, neighbourhoods.

Graph connectivity is NOT proof of coordination; these outputs are leads for human review.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def degree_table(G: nx.DiGraph) -> pd.DataFrame:
    rows = [{"account": n, "in_degree": G.in_degree(n), "out_degree": G.out_degree(n),
             "weighted_in": G.in_degree(n, weight="weight"), "weighted_out": G.out_degree(n, weight="weight")} for n in G.nodes]
    return pd.DataFrame(rows, columns=["account", "in_degree", "out_degree", "weighted_in", "weighted_out"])


def component_table(G: nx.DiGraph, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for i, comp in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):
        sub = G.subgraph(comp)
        rows.append({"component_id": i, "accounts": len(comp), "edges": sub.number_of_edges(),
                     "total_value": sum(d["weight"] for _, _, d in sub.edges(data=True)),
                     "max_risk_score": max((G.nodes[n].get("risk_score", 0.0) for n in comp), default=0.0)})
    return pd.DataFrame(rows, columns=["component_id", "accounts", "edges", "total_value", "max_risk_score"]).head(top_n)


def component_membership(G: nx.DiGraph) -> dict[str, tuple[int, int]]:
    """account -> (component_id, component_size) using weakly connected components."""
    out: dict[str, tuple[int, int]] = {}
    for i, comp in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):
        for n in comp:
            out[n] = (i, len(comp))
    return out


def detect_communities(G: nx.DiGraph, seed: int = 42) -> dict[str, int]:
    """Louvain communities on the undirected weighted graph (deterministic for a given seed)."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes)}
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d["weight"] + (U[u][v]["weight"] if U.has_edge(u, v) else 0.0)
        U.add_edge(u, v, weight=w)
    comms = nx.community.louvain_communities(U, weight="weight", seed=seed)
    return {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}


def find_cycles(G: nx.DiGraph, max_len: int = 4, max_cycles: int = 200, min_len: int = 3,
                min_edge_value: float = 0.0, max_span_hours: float | None = None) -> list[list[str]]:
    """Short directed cycles of `min_len`..`max_len` accounts, found with a bounded depth-first search.

    Edges below `min_edge_value`, or whose own activity spans more than `max_span_hours`, are ignored, and the whole
    cycle must complete inside `max_span_hours`. This matters because reciprocal payments between regular
    counterparties are common, so unfiltered cycles are weak evidence. Approximation: an edge's span is
    (last - first) of all its transfers. Output is capped by `max_cycles` and canonicalised (smallest id first).
    """
    limit = None if max_span_hours is None else pd.Timedelta(hours=max_span_hours)
    adj: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for u, v, d in G.edges(data=True):
        if d["weight"] < min_edge_value:
            continue
        first, last = pd.Timestamp(d["first_ts"]), pd.Timestamp(d["last_ts"])
        if limit is not None and last - first > limit:
            continue
        adj.setdefault(u, []).append((v, first, last))
    seen: set[tuple[str, ...]] = set()
    cycles: list[list[str]] = []
    for start in sorted(adj):
        stack = [([start], None, None)]
        while stack:
            path, tmin, tmax = stack.pop()
            for nv, f, l in adj.get(path[-1], ()):
                lo, hi = (f if tmin is None else min(tmin, f)), (l if tmax is None else max(tmax, l))
                if limit is not None and hi - lo > limit:
                    continue
                if nv == start:
                    if len(path) >= min_len:
                        k = path.index(min(path))
                        canon = tuple(path[k:] + path[:k])
                        if canon not in seen:
                            seen.add(canon)
                            cycles.append(list(canon))
                            if len(cycles) >= max_cycles:
                                return cycles
                elif nv not in path and len(path) < max_len:
                    stack.append((path + [nv], lo, hi))
    return cycles


def find_rapid_chains(tx: pd.DataFrame, max_gap_hours: float = 12.0, ratio_min: float = 0.6, ratio_max: float = 1.02,
                      min_hops: int = 3, max_chains: int = 500, max_pairs: int = 2_000_000) -> tuple[pd.DataFrame, str]:
    """Time-ordered chains tx1 -> tx2 -> ... where each next transfer leaves the previous recipient soon after,
    with an amount in [ratio_min, ratio_max] x the previous one. Returns (chains, warning)."""
    cols = ["chain_id", "hops", "accounts", "transaction_ids", "start_time", "duration_hours", "initial_amount", "final_amount"]
    if tx is None or tx.empty:
        return pd.DataFrame(columns=cols), ""
    t = tx[tx["source_account"] != tx["destination_account"]]
    in_c = t.groupby("destination_account").size()
    out_c = t.groupby("source_account").size()
    if int((in_c * out_c.reindex(in_c.index).fillna(0)).sum()) > max_pairs:
        return pd.DataFrame(columns=cols), "Chain search skipped: dataset too dense for the pairwise join limit."
    a = t[["transaction_id", "destination_account", "timestamp", "amount", "source_account"]].rename(
        columns={"transaction_id": "a_id", "destination_account": "mid", "timestamp": "a_ts", "amount": "a_amt", "source_account": "a_src"})
    b = t[["transaction_id", "source_account", "timestamp", "amount", "destination_account"]].rename(
        columns={"transaction_id": "b_id", "source_account": "mid", "timestamp": "b_ts", "amount": "b_amt", "destination_account": "b_dst"})
    m = a.merge(b, on="mid")
    gap = (m["b_ts"] - m["a_ts"]).dt.total_seconds() / 3600.0
    ratio = m["b_amt"] / m["a_amt"]
    m = m[(gap >= 0) & (gap <= max_gap_hours) & (ratio >= ratio_min) & (ratio <= ratio_max) & (m["a_id"] != m["b_id"])]
    if m.empty:
        return pd.DataFrame(columns=cols), ""
    nxt: dict[str, list[str]] = {}
    for r in m.itertuples(index=False):
        nxt.setdefault(r.a_id, []).append(r.b_id)
    has_prev = set(m["b_id"])
    info = t.set_index("transaction_id")[["source_account", "destination_account", "timestamp", "amount"]]
    paths: list[list[str]] = []
    for start in [s for s in nxt if s not in has_prev]:
        stack = [[start]]
        while stack and len(paths) < max_chains * 4:
            path = stack.pop()
            ext = [n for n in nxt.get(path[-1], [])[:5] if n not in path] if len(path) < 12 else []
            if not ext:
                if len(path) >= min_hops:
                    paths.append(path)
            else:
                stack.extend(path + [n] for n in ext)
    rows = []
    for p in sorted(paths, key=len, reverse=True)[:max_chains]:
        first, last = info.loc[p[0]], info.loc[p[-1]]
        rows.append({"hops": len(p), "accounts": [first["source_account"]] + [info.at[x, "destination_account"] for x in p],
                     "transaction_ids": p, "start_time": first["timestamp"],
                     "duration_hours": (last["timestamp"] - first["timestamp"]).total_seconds() / 3600.0,
                     "initial_amount": float(first["amount"]), "final_amount": float(last["amount"])})
    out = pd.DataFrame(rows, columns=cols[1:])
    out.insert(0, "chain_id", [f"CH{i + 1:04d}" for i in range(len(out))])
    return out, ""


def neighbourhood(G: nx.DiGraph, account: str, hops: int = 1, max_nodes: int = 150) -> nx.DiGraph:
    """Ego network (treating edges as undirected for reachability). Trimmed to the highest-weight neighbours."""
    if account not in G:
        return nx.DiGraph()
    sub = nx.ego_graph(G.to_undirected(as_view=True), account, radius=hops)
    nodes = list(sub.nodes)
    if len(nodes) > max_nodes:
        strength = {n: G.degree(n, weight="weight") for n in nodes if n != account}
        keep = [account] + sorted(strength, key=strength.get, reverse=True)[: max_nodes - 1]
        nodes = keep
    return G.subgraph(nodes).copy()


def compute_layout(G: nx.DiGraph, seed: int = 42) -> dict[str, tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() == 1:
        return {next(iter(G.nodes)): (0.0, 0.0)}
    k = 1.2 / np.sqrt(G.number_of_nodes())
    return {n: (float(x), float(y)) for n, (x, y) in nx.spring_layout(G, k=k, seed=seed, iterations=60).items()}
