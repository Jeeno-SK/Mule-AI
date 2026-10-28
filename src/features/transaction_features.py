"""Account-level and transaction-level behavioural features.

Feature definitions (all computed only from the supplied transactions; `is_fraud` is never used):
- in_value / out_value        : sum of amounts received / sent.
- in_count / out_count        : number of incoming / outgoing transactions.
- unique_senders (fan-in)     : distinct source accounts that sent to the account.
- unique_receivers (fan-out)  : distinct destination accounts the account sent to.
- unique_counterparties       : distinct accounts on either side.
- first_seen / last_seen      : first/last timestamp the account appears in the dataset (NOT account age).
- active_span_hours           : (last_seen - first_seen) in hours; 0 if only one timestamp.
- flow_balance_ratio          : min(in_value, out_value) / max(in_value, out_value); 0 if either is zero.
- retained_value              : in_value - out_value (can be negative).
- median_gap_hours/min_gap_hours : gaps between consecutive events of the account (NaN if < 2 events).
- peak_tx_window              : max number of events in any sliding window of `window_hours`.
- peak_in_value_window / peak_out_value_window : max incoming / outgoing value in any such window.
- rapid_forward_count/value   : outgoing transfers made within `rapid_hours` of the most recent incoming one,
                                 with out/in amount ratio inside [ratio_min, ratio_max]. Approximation: only the
                                 single most recent prior incoming transfer is matched.
- median_dwell_hours          : median delay between an outgoing transfer and the preceding incoming transfer.
- recent_activity_share       : share of events in the second half of the account's observed span (NaN if span is 0).
Zero denominators and missing history produce 0 or NaN, never exceptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACCOUNT_FEATURE_COLUMNS = [
    "account", "in_value", "out_value", "in_count", "out_count", "tx_count", "unique_senders", "unique_receivers",
    "unique_counterparties", "first_seen", "last_seen", "active_span_hours", "avg_in_amount", "avg_out_amount",
    "max_amount", "flow_balance_ratio", "retained_value", "median_gap_hours", "min_gap_hours", "peak_tx_window",
    "peak_in_value_window", "peak_out_value_window", "rapid_forward_count", "rapid_forward_value",
    "median_dwell_hours", "recent_activity_share",
]
ML_FEATURES = [
    "in_value", "out_value", "in_count", "out_count", "unique_senders", "unique_receivers", "unique_counterparties",
    "active_span_hours", "avg_in_amount", "avg_out_amount", "max_amount", "flow_balance_ratio", "median_gap_hours",
    "min_gap_hours", "peak_tx_window", "peak_in_value_window", "peak_out_value_window", "rapid_forward_count",
    "median_dwell_hours", "recent_activity_share",
]


def _prepare(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df["amount"] = df["amount"].astype(float)
    return df


def add_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features: hour, gap since the source's previous transaction, robust amount z-score.

    amount_robust_z = 0.6745 * (amount - median) / MAD over the source account's observed amounts
    (NaN when MAD is 0). Uses the full dataset (non-causal); see README limitations.
    """
    if tx.empty:
        out = tx.copy()
        for c in ("hour", "hours_since_prev_source_tx", "source_history_count", "amount_robust_z"):
            out[c] = pd.Series(dtype=float)
        return out
    df = _prepare(tx).sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["hours_since_prev_source_tx"] = df.groupby("source_account")["timestamp"].diff().dt.total_seconds() / 3600.0
    grp = df.groupby("source_account")["amount"]
    med = grp.transform("median")
    mad = (df["amount"] - med).abs().groupby(df["source_account"]).transform("median")
    df["source_history_count"] = grp.transform("size")
    df["amount_robust_z"] = np.where(mad > 0, 0.6745 * (df["amount"] - med) / mad.where(mad > 0, 1.0), np.nan)
    return df


def _events(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"account": df["source_account"], "ts": df["timestamp"], "amount": df["amount"], "direction": "out"})
    inc = pd.DataFrame({"account": df["destination_account"], "ts": df["timestamp"], "amount": df["amount"], "direction": "in"})
    return pd.concat([out, inc], ignore_index=True).sort_values(["account", "ts"], kind="stable").reset_index(drop=True)


def compute_rapid_forward_pairs(tx: pd.DataFrame, rapid_hours: float = 6.0, ratio_min: float = 0.6,
                                ratio_max: float = 1.0) -> pd.DataFrame:
    """Pairs (incoming tx -> outgoing tx) on the same account that look like pass-through forwarding."""
    cols = ["account", "in_tx_id", "out_tx_id", "in_time", "out_time", "dwell_hours", "in_amount", "out_amount", "ratio"]
    if tx.empty:
        return pd.DataFrame(columns=cols)
    df = _prepare(tx)
    out = df[["transaction_id", "timestamp", "source_account", "amount"]].rename(
        columns={"transaction_id": "out_tx_id", "timestamp": "out_time", "source_account": "account", "amount": "out_amount"})
    inc = df[["transaction_id", "timestamp", "destination_account", "amount"]].rename(
        columns={"transaction_id": "in_tx_id", "timestamp": "in_time", "destination_account": "account", "amount": "in_amount"})
    out = out.sort_values("out_time", kind="stable")
    inc = inc.sort_values("in_time", kind="stable")
    m = pd.merge_asof(out, inc, left_on="out_time", right_on="in_time", by="account", direction="backward")
    m = m.dropna(subset=["in_time"])
    m = m[m["in_tx_id"] != m["out_tx_id"]]
    m["dwell_hours"] = (m["out_time"] - m["in_time"]).dt.total_seconds() / 3600.0
    m["ratio"] = m["out_amount"] / m["in_amount"]
    m = m[(m["dwell_hours"] <= rapid_hours) & (m["ratio"] >= ratio_min) & (m["ratio"] <= ratio_max + 1e-9)]
    return m[cols].reset_index(drop=True)


def _dwell_table(tx: pd.DataFrame) -> pd.Series:
    """Median dwell (hours) between each outgoing tx and the preceding incoming tx, by account."""
    df = _prepare(tx)
    out = df[["timestamp", "source_account"]].rename(columns={"timestamp": "out_time", "source_account": "account"}).sort_values("out_time", kind="stable")
    inc = df[["timestamp", "destination_account"]].rename(columns={"timestamp": "in_time", "destination_account": "account"}).sort_values("in_time", kind="stable")
    m = pd.merge_asof(out, inc, left_on="out_time", right_on="in_time", by="account", direction="backward").dropna(subset=["in_time"])
    m["dwell"] = (m["out_time"] - m["in_time"]).dt.total_seconds() / 3600.0
    return m.groupby("account")["dwell"].median()


def _window_peaks(ev: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    """Peak event count and peak in/out value in any sliding window, per account."""
    w = np.int64(window_hours * 3600 * 1e9)
    rows = []
    for acct, g in ev.groupby("account", sort=False):
        t = g["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        v = g["amount"].to_numpy()
        is_in = (g["direction"] == "in").to_numpy()
        left = np.searchsorted(t, t - w, side="left")
        idx = np.arange(len(t))
        peak_cnt = int((idx - left + 1).max())
        res = [acct, peak_cnt]
        for mask in (is_in, ~is_in):
            vals = np.where(mask, v, 0.0)
            cs = np.concatenate([[0.0], np.cumsum(vals)])
            res.append(float((cs[idx + 1] - cs[left]).max()))
        rows.append(res)
    return pd.DataFrame(rows, columns=["account", "peak_tx_window", "peak_in_value_window", "peak_out_value_window"])


def compute_account_features(tx: pd.DataFrame, window_hours: float = 24.0, rapid_hours: float = 6.0,
                             ratio_min: float = 0.6, ratio_max: float = 1.0) -> pd.DataFrame:
    """Compute one feature row per account (see module docstring for definitions)."""
    if tx is None or tx.empty:
        return pd.DataFrame(columns=ACCOUNT_FEATURE_COLUMNS)
    df = _prepare(tx)
    out_g = df.groupby("source_account").agg(out_value=("amount", "sum"), out_count=("amount", "size"),
                                              unique_receivers=("destination_account", "nunique"),
                                              avg_out_amount=("amount", "mean"), max_out=("amount", "max"))
    in_g = df.groupby("destination_account").agg(in_value=("amount", "sum"), in_count=("amount", "size"),
                                                  unique_senders=("source_account", "nunique"),
                                                  avg_in_amount=("amount", "mean"), max_in=("amount", "max"))
    f = out_g.join(in_g, how="outer").fillna(0.0)
    f.index.name = "account"
    f["max_amount"] = f[["max_out", "max_in"]].max(axis=1)
    f = f.drop(columns=["max_out", "max_in"])
    for c in ("out_count", "in_count", "unique_receivers", "unique_senders"):
        f[c] = f[c].astype(int)
    f["tx_count"] = f["in_count"] + f["out_count"]

    pairs = pd.concat([df[["source_account", "destination_account"]].rename(columns={"source_account": "account", "destination_account": "cp"}),
                       df[["destination_account", "source_account"]].rename(columns={"destination_account": "account", "source_account": "cp"})])
    f["unique_counterparties"] = pairs[pairs["account"] != pairs["cp"]].groupby("account")["cp"].nunique().reindex(f.index).fillna(0).astype(int)

    ev = _events(df)
    g = ev.groupby("account")["ts"]
    f["first_seen"], f["last_seen"] = g.min(), g.max()
    f["active_span_hours"] = (f["last_seen"] - f["first_seen"]).dt.total_seconds() / 3600.0
    ev["gap_h"] = g.diff().dt.total_seconds() / 3600.0
    f["median_gap_hours"] = ev.groupby("account")["gap_h"].median()
    f["min_gap_hours"] = ev.groupby("account")["gap_h"].min()
    mid = f["first_seen"] + (f["last_seen"] - f["first_seen"]) / 2
    ev = ev.join(mid.rename("mid"), on="account")
    late = (ev["ts"] > ev["mid"]).groupby(ev["account"]).mean()
    f["recent_activity_share"] = late.where(f["active_span_hours"] > 0)

    hi, lo = f[["in_value", "out_value"]].max(axis=1), f[["in_value", "out_value"]].min(axis=1)
    f["flow_balance_ratio"] = np.where(hi > 0, lo / hi.where(hi > 0, 1.0), 0.0)
    f["retained_value"] = f["in_value"] - f["out_value"]

    f = f.join(_window_peaks(ev, window_hours).set_index("account"))
    pairs_rf = compute_rapid_forward_pairs(df, rapid_hours, ratio_min, ratio_max)
    rf = pairs_rf.groupby("account").agg(rapid_forward_count=("out_tx_id", "nunique"), rapid_forward_value=("out_amount", "sum"))
    f = f.join(rf)
    f[["rapid_forward_count", "rapid_forward_value"]] = f[["rapid_forward_count", "rapid_forward_value"]].fillna(0)
    f["rapid_forward_count"] = f["rapid_forward_count"].astype(int)
    f["median_dwell_hours"] = _dwell_table(df).reindex(f.index)
    f = f.reset_index()
    return f[ACCOUNT_FEATURE_COLUMNS]
