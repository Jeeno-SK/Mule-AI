import copy
import json
import unittest

from config.settings import load_rules_config
from src.features import transaction_features as tf
from src.graph import graph_analytics as ga
from src.graph.network_builder import build_graph
from src.detection.rule_engine import make_alert_id, run_rules
from tests.helpers import CHAIN, make_tx


def run(tx, cfg=None):
    cfg = cfg or load_rules_config()
    f = cfg["features"]
    acct = tf.compute_account_features(tx, f["window_hours"], f["rapid_forward_hours"], f["forward_ratio_min"], f["forward_ratio_max"])
    pairs = tf.compute_rapid_forward_pairs(tx, f["rapid_forward_hours"], f["forward_ratio_min"], f["forward_ratio_max"])
    G = build_graph(tx)
    chains, _ = ga.find_rapid_chains(tx, min_hops=3)
    cycles = ga.find_cycles(G, 4, 100, 3, 1000, 72)
    return run_rules(tx, tf.add_transaction_features(tx), acct, pairs, chains, cycles, G, cfg)


class RuleEngineTests(unittest.TestCase):
    def test_pass_through_and_chain_trigger_with_evidence(self):
        cfg = copy.deepcopy(load_rules_config())
        cfg["rules"]["R002"]["params"]["min_rapid_forwards"] = 1   # each mule in this tiny chain forwards once
        res = run(make_tx(CHAIN), cfg)
        a = res.alerts
        self.assertIn("R002", set(a["rule_id"]))
        self.assertIn("R007", set(a["rule_id"]))
        row = a[(a["rule_id"] == "R007")].iloc[0]
        self.assertEqual(json.loads(row["observed_json"])["hops"], 4)
        for col in ("rule_id", "description", "severity", "account", "observed_json", "threshold_json", "evidence"):
            self.assertTrue(str(row[col]))
        linked = res.alert_txs[res.alert_txs["alert_id"] == row["alert_id"]]["transaction_id"]
        self.assertEqual(set(linked), {"T1", "T2", "T3", "T4"})

    def test_only_one_chain_alert_per_account(self):
        a = run(make_tx(CHAIN)).alerts
        r7 = a[a["rule_id"] == "R007"]
        self.assertEqual(len(r7), r7["account"].nunique())

    def test_unrelated_activity_triggers_nothing(self):
        tx = make_tx([("A", "2025-01-01 10:00", "X", "Y", 100), ("B", "2025-01-05 10:00", "Z", "W", 120)])
        self.assertTrue(run(tx).alerts.empty)

    def test_disabled_rule_does_not_fire(self):
        cfg = copy.deepcopy(load_rules_config())
        cfg["rules"]["R002"]["enabled"] = False
        cfg["rules"]["R007"]["enabled"] = False
        self.assertFalse({"R002", "R007"} & set(run(make_tx(CHAIN), cfg).alerts["rule_id"]))

    def test_threshold_is_configurable(self):
        cfg = copy.deepcopy(load_rules_config())
        cfg["rules"]["R002"]["params"]["min_rapid_forwards"] = 1
        self.assertIn("R002", set(run(make_tx(CHAIN), cfg).alerts["rule_id"]))
        cfg["rules"]["R002"]["params"]["min_rapid_forwards"] = 5
        self.assertNotIn("R002", set(run(make_tx(CHAIN), cfg).alerts["rule_id"]))

    def test_fan_in_rule(self):
        rows = [(f"T{i}", f"2025-01-01 1{i % 10}:00", f"S{i}", "HUB", 500 + i) for i in range(12)]
        a = run(make_tx(rows)).alerts
        self.assertIn(("R003", "HUB"), set(zip(a["rule_id"], a["account"])))

    def test_alert_ids_are_deterministic(self):
        self.assertEqual(make_alert_id("R002", "A"), make_alert_id("R002", "A"))
        self.assertNotEqual(make_alert_id("R002", "A"), make_alert_id("R003", "A"))

    def test_circular_flow_requires_time_and_value(self):
        ring = [("C1", "2025-01-01 10:00", "A", "B", 5000), ("C2", "2025-01-01 12:00", "B", "C", 4900), ("C3", "2025-01-01 14:00", "C", "A", 4800)]
        self.assertIn("R008", set(run(make_tx(ring)).alerts["rule_id"]))
        slow = [("C1", "2025-01-01 10:00", "A", "B", 5000), ("C2", "2025-02-01 12:00", "B", "C", 4900), ("C3", "2025-03-01 14:00", "C", "A", 4800)]
        self.assertNotIn("R008", set(run(make_tx(slow)).alerts["rule_id"]))


if __name__ == "__main__":
    unittest.main()
