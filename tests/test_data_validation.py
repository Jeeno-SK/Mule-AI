import unittest

import pandas as pd

from src.data.loader import read_csv_bytes
from src.data.validator import normalise_account, validate_transactions
from src.utils.exceptions import DataValidationError


def raw(rows, cols=("transaction_id", "timestamp", "source_account", "destination_account", "amount")):
    return pd.DataFrame(rows, columns=list(cols)).astype(str)


GOOD = [("T1", "2025-01-01 10:00:00", " acc 1 ", "ACC2", "100.50"), ("T2", "2025-01-02 10:00:00", "ACC2", "ACC3", "50")]


class ValidationTests(unittest.TestCase):
    def test_valid_rows_and_normalisation(self):
        r = validate_transactions(raw(GOOD))
        self.assertTrue(r.schema_ok)
        self.assertEqual(len(r.clean), 2)
        self.assertIn("ACC1", set(r.clean["source_account"]))
        self.assertTrue((r.issues["action"] == "normalised").any())

    def test_missing_required_column(self):
        r = validate_transactions(raw([("T1", "2025-01-01", "A", "B")], cols=("transaction_id", "timestamp", "source_account", "destination_account")))
        self.assertFalse(r.schema_ok)
        self.assertTrue(r.clean.empty)

    def test_invalid_rows_are_reported_not_silent(self):
        rows = GOOD + [("T3", "not a date", "A", "B", "10"), ("T4", "2025-01-01", "A", "B", "-5"), ("T5", "2025-01-01", "A", "B", "abc"),
                       ("T2", "2025-01-03", "A", "B", "10"), ("T6", "2025-01-01", "", "B", "10")]
        r = validate_transactions(raw(rows))
        self.assertEqual(len(r.clean), 2)
        self.assertEqual(r.summary["rows_dropped"], 5)
        self.assertEqual(len(r.issues[r.issues["action"] == "dropped"]), 5)
        self.assertEqual(r.summary["invalid_timestamps"], 1)
        self.assertEqual(r.summary["duplicate_ids"], 1)

    def test_missing_labels_stay_unknown_not_legitimate(self):
        r = validate_transactions(raw(GOOD))
        self.assertEqual(r.summary["label_status"], "none")
        self.assertTrue(r.clean["is_fraud"].isna().all())

    def test_partial_and_full_labels(self):
        cols = ("transaction_id", "timestamp", "source_account", "destination_account", "amount", "is_fraud")
        partial = validate_transactions(raw([GOOD[0] + ("1",), GOOD[1] + ("",)], cols))
        full = validate_transactions(raw([GOOD[0] + ("1",), GOOD[1] + ("0",)], cols))
        self.assertEqual(partial.summary["label_status"], "partial")
        self.assertEqual(full.summary["label_status"], "full")

    def test_self_transfer_and_possible_duplicate_warned_but_kept(self):
        rows = [("T1", "2025-01-01", "A", "A", "10"), ("T2", "2025-01-02", "A", "B", "10"), ("T3", "2025-01-02", "A", "B", "10")]
        r = validate_transactions(raw(rows))
        self.assertEqual(len(r.clean), 3)
        self.assertEqual(r.summary["self_transfers"], 1)
        self.assertTrue(r.issues["issue"].str.contains("possible duplicate").any())

    def test_empty_input(self):
        r = validate_transactions(pd.DataFrame())
        self.assertFalse(r.schema_ok)

    def test_paysim_style_columns(self):
        df = pd.DataFrame({"step": ["1", "2"], "type": ["TRANSFER", "CASH_OUT"], "amount": ["10", "20"], "nameOrig": ["C1", "C2"],
                           "nameDest": ["C3", "C4"], "isFraud": ["0", "1"]})
        r = validate_transactions(df)
        self.assertTrue(r.schema_ok)
        self.assertEqual(len(r.clean), 2)
        self.assertTrue(any("synthetic" in m for m in r.messages))

    def test_account_normalisation_helper(self):
        self.assertEqual(normalise_account("  ab 12 "), "AB12")
        self.assertIsNone(normalise_account(None))

    def test_csv_reader_limits(self):
        with self.assertRaises(DataValidationError):
            read_csv_bytes(b"")
        with self.assertRaises(DataValidationError):
            read_csv_bytes(b"a,b\n1,2\n", max_bytes=3)
        with self.assertRaises(DataValidationError):
            read_csv_bytes(b"a,b\x00\n1,2\n")
        with self.assertRaises(DataValidationError):
            read_csv_bytes(b"a,b\n1,2\n3,4\n5,6\n", max_rows=2)
        self.assertEqual(len(read_csv_bytes(b"a,b\n1,2\n")), 1)


if __name__ == "__main__":
    unittest.main()
