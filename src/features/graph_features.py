"""Graph-derived account features (degree, component, community, cycle membership)."""
from __future__ import annotations

import networkx as nx
import pandas as pd

from src.graph import graph_analytics as ga

GRAPH_FEATURE_COLUMNS = ["account", "in_degree", "out_degree", "weighted_in", "weighted_out", "component_id",
                         "component_size", "community_id", "community_size", "in_cycle"]


def compute_graph_features(G: nx.DiGraph, cycles: list[list[str]] | None = None, seed: int = 42) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=GRAPH_FEATURE_COLUMNS)
    deg = ga.degree_table(G).set_index("account")
    comp = ga.component_membership(G)
    comm = ga.detect_communities(G, seed=seed)
    comm_sizes = pd.Series(comm).value_counts()
    in_cycle = {a for c in (cycles if cycles is not None else ga.find_cycles(G)) for a in c}
    deg["component_id"] = [comp[a][0] for a in deg.index]
    deg["component_size"] = [comp[a][1] for a in deg.index]
    deg["community_id"] = [comm.get(a, -1) for a in deg.index]
    deg["community_size"] = [int(comm_sizes.get(comm.get(a, -1), 1)) for a in deg.index]
    deg["in_cycle"] = [a in in_cycle for a in deg.index]
    return deg.reset_index()[GRAPH_FEATURE_COLUMNS]
