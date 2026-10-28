"""Build a directed account graph from transactions and filter transactions for exploration."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def filter_transactions(tx: pd.DataFrame, date_range: tuple | None = None, amount_range: tuple[float, float] | None = None,
                        types: list[str] | None = None, accounts: set[str] | None = None) -> pd.DataFrame:
    """Filter by date, amount, transaction type and (optionally) accounts on either side."""
    m = pd.Series(True, index=tx.index)
    if date_range and date_range[0] is not None:
        m &= tx["timestamp"] >= pd.Timestamp(date_range[0])
    if date_range and len(date_range) > 1 and date_range[1] is not None:
        m &= tx["timestamp"] < pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    if amount_range:
        m &= tx["amount"].between(amount_range[0], amount_range[1])
    if types:
        m &= tx["transaction_type"].isin(types)
    if accounts is not None:
        m &= tx["source_account"].isin(accounts) | tx["destination_account"].isin(accounts)
    return tx[m]


def build_graph(tx: pd.DataFrame, account_risk: pd.DataFrame | None = None) -> nx.DiGraph:
    """Directed graph: node = account, edge = aggregated transfers src->dst.

    Edge attributes: weight (total amount), count, first_ts, last_ts, types. Self-transfers are excluded.
    Node attributes: risk_score, risk_band (defaults 0 / 'unscored' when no scoring is supplied).
    """
    G = nx.DiGraph()
    if tx is not None and not tx.empty:
        t = tx[tx["source_account"] != tx["destination_account"]]
        agg = t.groupby(["source_account", "destination_account"]).agg(
            weight=("amount", "sum"), count=("amount", "size"), first_ts=("timestamp", "min"), last_ts=("timestamp", "max"),
            types=("transaction_type", lambda s: ",".join(sorted(set(map(str, s)))))).reset_index()
        for r in agg.itertuples(index=False):
            G.add_edge(r.source_account, r.destination_account, weight=float(r.weight), count=int(r.count),
                       first_ts=str(r.first_ts), last_ts=str(r.last_ts), types=r.types)
    risk = {}
    if account_risk is not None and not account_risk.empty:
        risk = account_risk.set_index("account")[["risk_score", "risk_band"]].to_dict("index")
    for n in G.nodes:
        info = risk.get(n)
        G.nodes[n]["risk_score"] = float(info["risk_score"]) if info else 0.0
        G.nodes[n]["risk_band"] = info["risk_band"] if info else "unscored"
    return G
