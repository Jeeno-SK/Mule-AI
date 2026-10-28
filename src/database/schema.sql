PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('synthetic', 'imported')),
    label_status  TEXT NOT NULL CHECK (label_status IN ('none', 'partial', 'full')),
    sha256        TEXT,
    row_count     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    analyzed_at   TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    is_active     INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS transactions (
    dataset_id          INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    transaction_id      TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    source_account      TEXT NOT NULL,
    destination_account TEXT NOT NULL,
    amount              REAL NOT NULL CHECK (amount > 0),
    transaction_type    TEXT,
    is_fraud            INTEGER CHECK (is_fraud IN (0, 1)),
    extra_json          TEXT,
    PRIMARY KEY (dataset_id, transaction_id)
);
CREATE INDEX IF NOT EXISTS idx_tx_src ON transactions(dataset_id, source_account);
CREATE INDEX IF NOT EXISTS idx_tx_dst ON transactions(dataset_id, destination_account);
CREATE INDEX IF NOT EXISTS idx_tx_ts  ON transactions(dataset_id, timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    dataset_id     INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    alert_id       TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    rule_name      TEXT NOT NULL,
    severity       TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    account        TEXT NOT NULL,
    transaction_id TEXT,
    timestamp      TEXT,
    description    TEXT NOT NULL,
    observed_json  TEXT NOT NULL,
    threshold_json TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    PRIMARY KEY (dataset_id, alert_id)
);
CREATE INDEX IF NOT EXISTS idx_alert_acct ON alerts(dataset_id, account);

CREATE TABLE IF NOT EXISTS alert_transactions (
    dataset_id     INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    alert_id       TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    PRIMARY KEY (dataset_id, alert_id, transaction_id)
);

CREATE TABLE IF NOT EXISTS account_risk (
    dataset_id      INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    account         TEXT NOT NULL,
    risk_score      REAL NOT NULL,
    risk_band       TEXT NOT NULL CHECK (risk_band IN ('none', 'low', 'medium', 'high', 'critical')),
    rule_points     REAL NOT NULL,
    graph_points    REAL NOT NULL,
    ml_points       REAL NOT NULL,
    rules_triggered TEXT NOT NULL,
    n_alerts        INTEGER NOT NULL,
    basis           TEXT NOT NULL,
    explanation     TEXT NOT NULL,
    features_json   TEXT NOT NULL,
    PRIMARY KEY (dataset_id, account)
);

CREATE TABLE IF NOT EXISTS alert_status (
    dataset_id        INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    alert_id          TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('open', 'under_review', 'resolved', 'escalated')),
    resolution_reason TEXT,
    updated_at        TEXT NOT NULL,
    updated_by        TEXT NOT NULL,
    PRIMARY KEY (dataset_id, alert_id)
);

CREATE TABLE IF NOT EXISTS notes (
    note_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    account    TEXT NOT NULL,
    alert_id   TEXT,
    note       TEXT NOT NULL CHECK (length(note) > 0),
    author     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_acct ON notes(dataset_id, account);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    dataset_id   INTEGER,
    target       TEXT,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
