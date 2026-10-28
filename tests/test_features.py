import unittest

import numpy as np

from src.features.transaction_features import add_transaction_features, compute_account_features, compute_rapid_forward_pairs
from tests.helpers import CHAIN, make_tx


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tx = make_tx(CHAIN)
        self.f = compute_account_features(self.tx).set_index("account")

    def test_basic_aggregates(self):
        m1 = self.f.loc["M1"]
        self.assertEqual(m1["in_value"], 10000)
        self.assertEqual(m1["out_value"], 9600)
        self.assertEqual((m1["in_count"], m1["out_count"], m1["unique_senders"], m1["unique_receivers"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m1["flow_balance_ratio"], 0.96)
        self.assertAlmostEqual(m1["retained_value"], 400)

    def test_zero_denominator_is_safe(self):
        self.assertEqual(self.f.loc["V", "flow_balance_ratio"], 0.0)   # only sends
        self.assertEqual(self.f.loc["M4", "flow_balance_ratio"], 0.0)  # only receives

    def test_rapid_forward_and_dwell(self):
        self.assertEqual(self.f.loc["M1", "rapid_forward_count"], 1)
        self.assertAlmostEqual(self.f.loc["M1", "median_dwell_hours"], 0.5)
        self.assertEqual(self.f.loc["V", "rapid_forward_count"], 0)
        self.assertTrue(np.isnan(self.f.loc["V", "median_dwell_hours"]))

    def test_forward_ratio_bounds_exclude_unrelated_amounts(self):
        tx = make_tx([("A", "2025-01-01 10:00", "X", "M", 1000), ("B", "2025-01-01 10:10", "M", "Y", 100)])
        self.assertTrue(compute_rapid_forward_pairs(tx).empty)

    def test_slow_forwarding_is_not_rapid(self):
        tx = make_tx([("A", "2025-01-01 10:00", "X", "M", 1000), ("B", "2025-01-03 10:00", "M", "Y", 950)])
        self.assertTrue(compute_rapid_forward_pairs(tx, rapid_hours=6).empty)

    def test_observed_span_not_account_age(self):
        self.assertNotIn("account_age", self.f.columns)
        self.assertAlmostEqual(self.f.loc["M1", "active_span_hours"], 0.5)

    def test_window_peak(self):
        self.assertEqual(self.f.loc["M2", "peak_tx_window"], 2)

    def test_empty_dataset(self):
        self.assertTrue(compute_account_features(self.tx.iloc[0:0]).empty)
        self.assertTrue(add_transaction_features(self.tx.iloc[0:0]).empty)

    def test_labels_never_used_as_features(self):
        tx = make_tx([(r[0], r[1], r[2], r[3], r[4], "UPI", 1) for r in CHAIN])
        self.assertNotIn("is_fraud", compute_account_features(tx).columns)

    def test_transaction_features_robust_z(self):
        rows = [(f"T{i}", f"2025-01-0{i + 1} 10:00", "A", "B", 100 + i) for i in range(7)] + [("TX", "2025-01-09 10:00", "A", "B", 100000)]
        tf = add_transaction_features(make_tx(rows)).set_index("transaction_id")
        self.assertGreater(tf.loc["TX", "amount_robust_z"], 10)
        self.assertEqual(tf.loc["TX", "source_history_count"], 8)


if __name__ == "__main__":
    unittest.main()
