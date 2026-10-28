import os
import tempfile
import unittest

import pandas as pd

from src.data.loader import ingest_dataframe
from src.database.repository import Repository
from src.investigations import case_manager as cm
from src.investigations.audit_log import read_log
from src.pipeline import run_analysis
from src.data.synthetic_generator import SyntheticConfig, generate_synthetic
from src.utils.exceptions import CaseError
from tests.helpers import temp_repo


def small_dataset(repo, seed=3):
    syn = generate_synthetic(SyntheticConfig(seed=seed, n_accounts=150, n_legit_tx=900, days=14, n_collection_rings=1,
                                             n_passthrough_chains=1, n_fanout_networks=1, n_cycle_rings=1, n_merchants=1, n_payroll=1))
    did, res = ingest_dataframe(repo, syn.transactions.astype(str), "t", "test", "synthetic")
    return did, syn


class DatabaseTests(unittest.TestCase):
    def test_auto_init_and_dataset_roundtrip(self):
        repo = temp_repo()
        self.assertTrue(repo.list_datasets().empty)
        did, syn = small_dataset(repo)
        self.assertEqual(repo.get_active_dataset_id(), did)
        tx = repo.get_transactions(did)
        self.assertEqual(len(tx), len(syn.transactions))
        self.assertEqual(repo.get_dataset(did)["origin"], "synthetic")

    def test_unlabelled_dataset_keeps_null_labels(self):
        repo = temp_repo()
        syn = generate_synthetic(SyntheticConfig(seed=1, n_accounts=80, n_legit_tx=300, days=10))
        raw = syn.transactions.drop(columns=["is_fraud"]).astype(str)
        did, res = ingest_dataframe(repo, raw, "u", "test", "imported")
        self.assertEqual(repo.get_dataset(did)["label_status"], "none")
        self.assertTrue(repo.get_transactions(did)["is_fraud"].isna().all())

    def test_search_is_injection_safe(self):
        repo = temp_repo()
        did, _ = small_dataset(repo)
        for q in ["'; DROP TABLE transactions; --", "%", "_", "SYN-"]:
            repo.search(did, q)
        self.assertGreater(len(repo.get_transactions(did)), 0)
        self.assertEqual(len(repo.search(did, "%")["accounts"]), 0)  # wildcard is escaped, not interpreted

    def test_status_notes_and_audit_persist_across_restart(self):
        repo = temp_repo()
        did, _ = small_dataset(repo)
        run_analysis(repo, did)
        alerts = repo.get_alerts(did)
        self.assertFalse(alerts.empty)
        aid, acct = alerts.iloc[0]["alert_id"], alerts.iloc[0]["account"]
        with self.assertRaises(CaseError):
            cm.set_alert_status(repo, did, [aid], "resolved", "", "tester")
        with self.assertRaises(CaseError):
            cm.set_alert_status(repo, did, [aid], "convicted", "x", "tester")
        cm.set_alert_status(repo, did, [aid], "resolved", "legitimate payroll", "tester")
        cm.add_note(repo, did, acct, "checked counterparties", "tester", aid)
        with self.assertRaises(CaseError):
            cm.add_note(repo, did, acct, "   ", "tester")
        repo2 = Repository(repo.db_path)    # simulates an application restart
        row = repo2.get_alerts(did).set_index("alert_id").loc[aid]
        self.assertEqual((row["status"], row["resolution_reason"]), ("resolved", "legitimate payroll"))
        self.assertEqual(len(repo2.get_notes(did, acct)), 1)
        actions = set(read_log(repo2, did)["action"])
        self.assertTrue({"alert_status_changed", "note_added"} <= actions)

    def test_status_survives_reanalysis(self):
        repo = temp_repo()
        did, _ = small_dataset(repo)
        run_analysis(repo, did)
        aid = repo.get_alerts(did).iloc[0]["alert_id"]
        cm.set_alert_status(repo, did, [aid], "escalated", None, "tester")
        run_analysis(repo, did)
        self.assertEqual(repo.get_alerts(did).set_index("alert_id").loc[aid, "status"], "escalated")

    def test_delete_cascades(self):
        repo = temp_repo()
        did, _ = small_dataset(repo)
        run_analysis(repo, did)
        repo.delete_dataset(did)
        self.assertTrue(repo.get_transactions(did).empty)
        self.assertTrue(repo.get_alerts(did).empty)

    def test_settings_roundtrip(self):
        repo = temp_repo()
        repo.set_setting("k", {"a": 1})
        self.assertEqual(repo.get_setting("k"), {"a": 1})
        repo.delete_setting("k")
        self.assertIsNone(repo.get_setting("k"))

    def test_schema_constraints(self):
        import sqlite3
        repo = temp_repo()
        did, _ = small_dataset(repo)
        with self.assertRaises(sqlite3.IntegrityError):
            with repo.connect() as c:
                c.execute("INSERT INTO alert_status VALUES (?, 'x', 'bogus', NULL, 'now', 'me')", (did,))


if __name__ == "__main__":
    unittest.main()
