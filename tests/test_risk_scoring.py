import unittest

import numpy as np
import pandas as pd

from config.settings import load_rules_config
from src.detection.ml_detector import MLResult
from src.detection.risk_scoring import compute_risk_scores, risk_band
from src.features.transaction_features import compute_account_features
from tests.helpers import CHAIN, make_tx

CFG = load_rules_config()["scoring"]


def alert(rule, acct, sev="high"):
    return {"alert_id": f"{rule}{acct}", "rule_id": rule, "rule_name": rule, "severity": sev, "account": acct, "evidence": f"evidence {rule}"}


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.tx = make_tx(CHAIN)
        self.acct = compute_risk_input = compute_account_features(self.tx)

    def score(self, alerts, ml=None):
        r = compute_risk_scores(self.acct, pd.DataFrame(), pd.DataFrame(alerts), ml, self.tx, CFG)
        return r.set_index("account")

    def test_no_alerts_scores_zero_with_honest_text(self):
        r = self.score([])
        self.assertTrue((r["risk_score"] == 0).all())
        self.assertIn("does not establish", r.loc["M1", "explanation"])

    def test_rules_only_scale_and_explanation(self):
        r = self.score([alert("R002", "M1"), alert("R007", "M1")])
        self.assertEqual(r.loc["M1", "rules_triggered"], "R002,R007")
        self.assertGreater(r.loc["M1", "risk_score"], 50)
        self.assertEqual(r.loc["M1", "basis"], "rules only")
        self.assertIn("R002", r.loc["M1", "explanation"])
        self.assertIn("Not a probability", r.loc["M1", "explanation"])

    def test_same_rule_twice_counts_once(self):
        a = self.score([alert("R007", "M1")])["risk_score"]["M1"]
        b = self.score([alert("R007", "M1"), {**alert("R007", "M1"), "alert_id": "other"}])["risk_score"]["M1"]
        self.assertEqual(a, b)

    def test_derived_rule_does_not_double_count(self):
        a = self.score([alert("R002", "M1")])["risk_score"]["M1"]
        b = self.score([alert("R002", "M1"), alert("R009", "M1")])["risk_score"]["M1"]
        self.assertEqual(a, b)

    def test_graph_points_need_own_evidence(self):
        r = self.score([alert("R002", "M2")])
        self.assertEqual(r.loc["M1", "graph_points"], 0)   # neighbour of flagged M2 but no own evidence
        self.assertEqual(r.loc["M1", "risk_score"], 0)
        self.assertGreater(r.loc["M3", "graph_points"] + 1, 0)

    def test_ml_adds_points_and_basis(self):
        scores = pd.DataFrame({"account": self.acct["account"], "ml_supervised_score": 0.9, "ml_anomaly_percentile": np.nan, "ml_in_training": False})
        ml = MLResult(scores, {"supervised": {"status": "trained"}, "unsupervised": {"status": "skipped"}})
        r = self.score([alert("R002", "M1")], ml)
        self.assertEqual(r.loc["M1", "basis"], "rules + supervised model")
        self.assertGreater(r.loc["M1", "ml_points"], 0)
        self.assertLessEqual(r["risk_score"].max(), 100)

    def test_bands(self):
        b = CFG["bands"]
        self.assertEqual(risk_band(0, b), "none")
        self.assertEqual(risk_band(10, b), "low")
        self.assertEqual(risk_band(b["medium"], b), "medium")
        self.assertEqual(risk_band(b["high"], b), "high")
        self.assertEqual(risk_band(100, b), "critical")

    def test_empty_features(self):
        self.assertTrue(compute_risk_scores(self.acct.iloc[0:0], pd.DataFrame(), pd.DataFrame(), None, self.tx.iloc[0:0], CFG).empty)


if __name__ == "__main__":
    unittest.main()
