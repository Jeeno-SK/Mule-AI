import unittest

import pandas as pd

from src.data.loader import ingest_dataframe
from src.data.synthetic_generator import SyntheticConfig, generate_synthetic
from src.detection.ml_detector import build_account_labels
from src.pipeline import run_analysis
from src.reporting.exports import provenance, sanitise_cell, to_csv_bytes, to_json_bytes
from src.reporting.metrics import compute_dashboard_metrics
from src.reporting.report_builder import build_case_report, report_to_markdown
from src.utils.exceptions import AnalysisError
from tests.helpers import make_tx, temp_repo

CFG = SyntheticConfig(seed=7, n_accounts=400, n_legit_tx=2500, days=20, n_collection_rings=2, n_passthrough_chains=2,
                      n_fanout_networks=1, n_cycle_rings=1, n_merchants=2, n_payroll=1)


class SyntheticTests(unittest.TestCase):
    def test_reproducible_and_seed_sensitive(self):
        a, b = generate_synthetic(CFG), generate_synthetic(CFG)
        self.assertTrue(a.transactions.equals(b.transactions))
        other = generate_synthetic(SyntheticConfig(seed=8))
        self.assertFalse(a.transactions["source_account"].equals(other.transactions["source_account"]))

    def test_ground_truth_is_separate_and_not_in_ids(self):
        d = generate_synthetic(CFG)
        self.assertNotIn("role", d.transactions.columns)
        self.assertTrue(d.transactions["source_account"].str.startswith("SYN-").all())
        self.assertGreater(d.ground_truth_accounts["is_mule_by_design"].sum(), 0)
        self.assertGreater(d.transactions["is_fraud"].sum(), 0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.syn = generate_synthetic(CFG)
        cls.repo = temp_repo()
        cls.did, _ = ingest_dataframe(cls.repo, cls.syn.transactions.astype(str), "labelled", "t", "synthetic")
        cls.summary = run_analysis(cls.repo, cls.did)

    def test_labelled_run_uses_ml_and_reports_metrics(self):
        self.assertTrue(self.summary["ml_available"])
        meta = self.repo.get_model_runs(self.did)[0]["metadata"]
        sup = meta["supervised"]
        self.assertEqual(sup["status"], "trained")
        ev = sup["evaluation"]["stratified_split"]
        for k in ("precision", "recall", "f1", "confusion_matrix", "pr_auc"):
            self.assertIn(k, ev)
        self.assertEqual(meta["unsupervised"]["status"], "trained")
        self.assertIn("not a probability", meta["note"])

    def test_rules_find_designed_patterns(self):
        risk = self.repo.get_account_risk(self.did).merge(self.syn.ground_truth_accounts, on="account")
        mules = risk[risk["is_mule_by_design"]]
        self.assertGreater((mules["risk_score"] >= 25).mean(), 0.5)
        self.assertTrue(((risk["risk_score"] >= 0) & (risk["risk_score"] <= 100)).all())

    def test_alerts_have_evidence_and_links(self):
        a = self.repo.get_alerts(self.did)
        self.assertFalse(a.empty)
        self.assertTrue((a["evidence"].str.len() > 0).all())
        self.assertFalse(self.repo.get_alert_transactions(self.did).empty)

    def test_dashboard_metrics_come_from_data(self):
        tx, risk = self.repo.get_transactions(self.did), self.repo.get_account_risk(self.did)
        m = compute_dashboard_metrics(tx, risk, self.repo.get_alert_transactions(self.did), 25)
        self.assertEqual(m["n_transactions"], len(tx))
        self.assertAlmostEqual(m["total_value"], tx["amount"].sum())
        self.assertLessEqual(m["flagged_value"], m["total_value"])
        # a different dataset yields different metrics
        other = ingest_dataframe(self.repo, generate_synthetic(SyntheticConfig(seed=9, n_accounts=120, n_legit_tx=400, days=10)).transactions.astype(str),
                                 "other", "t", "synthetic")[0]
        run_analysis(self.repo, other)
        m2 = compute_dashboard_metrics(self.repo.get_transactions(other), self.repo.get_account_risk(other), self.repo.get_alert_transactions(other), 25)
        self.assertNotEqual(m["n_transactions"], m2["n_transactions"])
        self.repo.set_active_dataset(self.did)

    def test_unlabelled_dataset_skips_supervised_but_still_works(self):
        repo = temp_repo()
        raw = self.syn.transactions.drop(columns=["is_fraud"]).astype(str)
        did, _ = ingest_dataframe(repo, raw, "unlabelled", "t", "imported")
        run_analysis(repo, did)
        meta = repo.get_model_runs(did)[0]["metadata"]
        self.assertEqual(meta["supervised"]["status"], "skipped")
        self.assertIn("no fraud labels", meta["supervised"]["reason"])
        self.assertEqual(meta["unsupervised"]["status"], "trained")
        self.assertFalse(repo.get_alerts(did).empty)

    def test_tiny_dataset_falls_back_to_rules(self):
        repo = temp_repo()
        raw = pd.DataFrame({"transaction_id": ["A", "B", "C"], "timestamp": ["2025-01-01 10:00", "2025-01-01 10:20", "2025-01-01 10:40"],
                            "source_account": ["P", "Q", "R"], "destination_account": ["Q", "R", "S"], "amount": ["9000", "8800", "8500"]})
        did, _ = ingest_dataframe(repo, raw, "tiny", "t", "imported")
        s = run_analysis(repo, did)
        self.assertFalse(s["ml_available"])
        self.assertTrue(any("skipped" in m or "No ML" in m for m in s["ml_messages"]))

    def test_missing_dataset_and_empty_dataset(self):
        with self.assertRaises(AnalysisError):
            run_analysis(temp_repo(), 999)
        repo = temp_repo()
        did = repo.create_dataset("empty", "t", "imported", "none", None, 0, {})
        with self.assertRaises(AnalysisError):
            run_analysis(repo, did)

    def test_account_labels_exclude_unknowns(self):
        tx = make_tx([("1", "2025-01-01 10:00", "A", "B", 10, "UPI", 1), ("2", "2025-01-01 11:00", "C", "D", 10, "UPI", 0),
                      ("3", "2025-01-01 12:00", "E", "F", 10)])
        lab = build_account_labels(tx)
        self.assertEqual(lab["B"], 1.0)
        self.assertEqual(lab["A"], 0.0)
        self.assertNotIn("E", lab.index)  # unlabelled is unknown, not legitimate

    def test_exports_and_case_report(self):
        ds = self.repo.get_dataset(self.did)
        prov = provenance(ds, "rule-based")
        self.assertIn("SYNTHETIC", prov["data_origin"])
        self.assertEqual(sanitise_cell("=SUM(A1)"), "'=SUM(A1)")
        self.assertEqual(sanitise_cell("normal"), "normal")
        df = pd.DataFrame({"x": ["=cmd|' /C calc'!A0", "ok"]})
        csv = to_csv_bytes(df, prov).decode()
        self.assertIn("'=cmd", csv)
        self.assertIn("data_origin", csv)
        self.assertIn(b"provenance", to_json_bytes(df, prov))
        risk = self.repo.get_account_risk(self.did)
        acct = risk.iloc[0]["account"]
        tx = self.repo.get_transactions(self.did)
        atx = tx[(tx.source_account == acct) | (tx.destination_account == acct)]
        alerts = self.repo.get_alerts(self.did)
        aal = alerts[alerts.account == acct]
        rep = build_case_report(acct, ds, "rule-based", risk.iloc[0], aal, atx, self.repo.get_notes(self.did, acct), set())
        md = report_to_markdown(rep)
        self.assertIn("Case report", md)
        self.assertIn("not findings of criminal", md)


if __name__ == "__main__":
    unittest.main()
