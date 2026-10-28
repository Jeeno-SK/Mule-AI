import unittest

from src.features.graph_features import compute_graph_features
from src.graph import graph_analytics as ga
from src.graph.network_builder import build_graph, filter_transactions
from tests.helpers import CHAIN, make_tx


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tx = make_tx(CHAIN + [("T5", "2025-01-02 10:00", "X", "Y", 50), ("T6", "2025-01-02 11:00", "X", "Y", 70)])
        self.G = build_graph(self.tx)

    def test_edges_are_directed_and_aggregated(self):
        self.assertTrue(self.G.has_edge("V", "M1"))
        self.assertFalse(self.G.has_edge("M1", "V"))
        e = self.G.edges["X", "Y"]
        self.assertEqual((e["count"], e["weight"]), (2, 120.0))

    def test_degrees_and_components(self):
        d = ga.degree_table(self.G).set_index("account")
        self.assertEqual((d.loc["M1", "in_degree"], d.loc["M1", "out_degree"]), (1, 1))
        comps = ga.component_table(self.G)
        self.assertEqual(sorted(comps["accounts"]), [2, 5])

    def test_rapid_chain_found(self):
        chains, warn = ga.find_rapid_chains(self.tx, min_hops=3)
        self.assertEqual(warn, "")
        self.assertEqual(int(chains["hops"].max()), 4)
        self.assertEqual(chains.iloc[0]["accounts"], ["V", "M1", "M2", "M3", "M4"])

    def test_chain_requires_comparable_amounts(self):
        tx = make_tx([("A", "2025-01-01 10:00", "S", "M", 1000), ("B", "2025-01-01 10:30", "M", "N", 100), ("C", "2025-01-01 11:00", "N", "O", 90)])
        chains, _ = ga.find_rapid_chains(tx, min_hops=2)
        self.assertEqual(len(chains[chains["hops"] >= 3]), 0)

    def test_cycle_detection(self):
        tx = make_tx([("A", "2025-01-01 10:00", "P", "Q", 5000), ("B", "2025-01-01 11:00", "Q", "R", 5000), ("C", "2025-01-01 12:00", "R", "P", 5000)])
        cyc = ga.find_cycles(build_graph(tx), 4, 10, 3, 1000, 72)
        self.assertEqual(cyc, [["P", "Q", "R"]])

    def test_neighbourhood(self):
        n1 = ga.neighbourhood(self.G, "M2", hops=1)
        self.assertEqual(set(n1.nodes), {"M1", "M2", "M3"})
        self.assertEqual(set(ga.neighbourhood(self.G, "M2", hops=2).nodes), {"V", "M1", "M2", "M3", "M4"})
        self.assertEqual(ga.neighbourhood(self.G, "NOPE").number_of_nodes(), 0)

    def test_filtering(self):
        f = filter_transactions(self.tx, amount_range=(1000, 20000))
        self.assertEqual(len(f), 4)
        self.assertEqual(len(filter_transactions(self.tx, types=["NEFT"])), 0)

    def test_graph_features_and_empty_graph(self):
        gf = compute_graph_features(self.G).set_index("account")
        self.assertEqual(gf.loc["M1", "component_size"], 5)
        self.assertTrue(compute_graph_features(build_graph(self.tx.iloc[0:0])).empty)

    def test_louvain_deterministic(self):
        self.assertEqual(ga.detect_communities(self.G, 1), ga.detect_communities(self.G, 1))


if __name__ == "__main__":
    unittest.main()
