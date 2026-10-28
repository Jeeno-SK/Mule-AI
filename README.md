# MuleShield AI
**Explainable Mule Account and Suspicious Financial Network Detection Platform**

A local, modular prototype that turns transaction records into account features, a directed transfer graph, deterministic rule alerts,
baseline ML signals and an explainable heuristic risk score, then supports human investigation (notes, statuses, audit log, case reports).

> **Honesty first.** The bundled data is **synthetic**. A high score is a *prioritisation indicator for human review*, never a finding that
> anyone is a criminal. Scores are heuristics, **not calibrated probabilities**. Nothing here has been tested on real bank data.

## 1. Setup on Windows (PowerShell)
```powershell
cd muleshield-ai
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python scripts/generate_demo_data.py   # optional: writes data/demo_transactions.csv (synthetic, seed 42)
streamlit run app.py
```
Open http://localhost:8501. The SQLite file `data/muleshield.db` is created automatically on first run (override with `MULESHIELD_DB`).
Python 3.11+ recommended (developed against 3.12). No API keys, Docker, cloud or external network calls are used.

**First use:** *Data & Transactions* -> **Generate and load synthetic dataset** (or *Load bundled demo CSV*) -> analysis runs automatically -> explore the other pages.

## 2. Tests
```powershell
python -m pytest -q                       # or: python -m unittest discover -s tests -t .
```

## 3. Modules (all connected to the same SQLite state)
| Module | Where |
|---|---|
| A Executive dashboard (all numbers computed from the active dataset) | `src/ui/dashboard.py`, `src/reporting/metrics.py` |
| B CSV upload, schema validation, validation report (downloadable), provenance | `src/data/validator.py`, `loader.py`, `src/ui/transactions.py` |
| C Synthetic generator (seeded; collection fan-in, rapid chains, fan-out, cycles, hard negatives) | `src/data/synthetic_generator.py` |
| D Features (formulas documented in module docstring) | `src/features/transaction_features.py`, `graph_features.py` |
| E Rule engine R001-R010 (config in `config/rules.yaml`, editable in UI) | `src/detection/rule_engine.py` |
| F ML: IsolationForest (unsupervised), RandomForest (only with enough labels), checksummed joblib artefacts | `src/detection/ml_detector.py` |
| G Graph: NetworkX analytics + Plotly view, neighbourhood, filters, chains, cycles, Louvain communities | `src/graph/`, `src/ui/graph_view.py` |
| H Risk scoring + explanations | `src/detection/risk_scoring.py` |
| I Investigation workspace, statuses, notes, audit log | `src/investigations/`, `src/ui/investigations.py` |
| J Search/filters/exports (CSV/JSON with provenance columns) | `src/ui/transactions.py`, `src/reporting/exports.py` |
| K Data quality, model evaluation, PSI drift foundation | `src/ui/model_monitor.py`, `src/monitoring/drift.py` |

## 4. CSV schema
Required: `transaction_id, timestamp, source_account, destination_account, amount`. Optional: `transaction_type`, `is_fraud` (0/1).
Extra columns are kept in `extra_json`. Common aliases (e.g. PaySim `nameOrig`, `nameDest`, `type`, `isFraud`, `step`) are mapped; a PaySim `step`
becomes a **synthetic** timestamp and the UI says so. Dropped rows (missing values, bad amounts/timestamps, bad identifiers, duplicate ids) are always
listed in the validation report; self-transfers and "same content, different id" are kept with a warning. Missing labels stay *unknown* and are never
treated as legitimate. Limits: 25 MB, 500,000 rows, UTF-8 text only; content is never executed.

## 5. How scoring works
`score = 100 x (rule_points + graph_points + ml_points) / available_max`
- `rule_points`: distinct triggered rules, severity points (low 8 / medium 16 / high 28), capped; derived rule R009 not double counted.
- `graph_points`: share of direct counterparties that also have rule alerts, **only for accounts with their own rule evidence** (guilt-by-association alone never raises an account).
- `ml_points`: supervised score if trained, else top-decile anomaly percentile rescaled. If no model ran, its weight is removed so rules-only scores still span 0-100.
- Bands and weights are in `config/rules.yaml`. Every account gets a text explanation listing the rules, observed values, graph context and model signal.
- Anomaly output is a within-dataset rank (about 2% of accounts will always sit above the 0.98 percentile); it is not a fraud probability.

## 6. Architecture
```
CSV / generator -> validator -> SQLite (datasets, transactions)
                      pipeline.run_analysis:
   features (account + transaction) -> graph (cycles, chains, communities) -> rules -> ML -> risk scores
                      -> SQLite (alerts, alert_transactions, account_risk, model_runs)
Streamlit pages read SQLite; analyst actions write alert_status / notes / audit_log (all parameterised SQL in repository.py)
```
Dependencies point one way: `ui -> pipeline -> detection/features/graph -> data/database`. Re-running analysis keeps analyst statuses and notes because alert ids are deterministic hashes.

## 7. Security measures
Parameterised SQL only; schema CHECK/FOREIGN KEY constraints; upload size/row/binary checks; CSV parsed as text; spreadsheet formula-injection neutralised on export;
HTML in banners escaped; user-safe error messages (stack traces go to the log); audit log for imports, analysis runs, status changes, notes, settings;
joblib artefacts loaded only from `models/` after SHA-256 verification (joblib uses pickle: never load files from elsewhere); no external calls; no credentials.
This is a single-user local prototype: **there is no authentication**, and the analyst name is a self-declared label.

## 8. Datasets and scientific honesty
- Bundled/generated data is **synthetic**; generator-provided `is_fraud` labels describe the simulation, not real fraud.
- **PaySim** and **AMLSim** are synthetic/simulated resources; their labels and patterns do not represent verified real-world mule accounts. They can be loaded via the CSV importer after mapping columns (not bundled, not tested here).
- A sanity check on the seed-42 demo (page *Data Quality & Models -> Synthetic ground-truth check*): in one run, rules + ML flagged about 180 accounts, recovering roughly 90% of the designed mule accounts while only ~38% of flagged accounts were designed mules (the rest are mostly ordinary accounts, victims, merchants and payroll accounts - intentional hard negatives). Treat as a demo of the workflow, not a benchmark; it is measured against the generator's own design.

## 9. Known limitations
1. No real data tested; synthetic patterns are simple and rule-friendly, so results flatter the detectors.
2. Supervised labels are transaction-level; account labels are derived ("received a fraud-labelled transaction"). Features are computed over the full period, so the chronological split only approximates future deployment.
3. Account age is unknown, so "new account" signals are not used; R010 only describes the *observed* activity window.
4. Rapid-forward matching uses only the most recent prior incoming transfer; split/merge flows are partly missed. Cycle search is bounded and approximate (edge time span = first..last transfer).
5. Thresholds are heuristics and need tuning per institution; merchants and payroll accounts legitimately trigger fan-in/fan-out rules.
6. No onboarding/KYC, device, IP, or cross-institution data, which matter in real programmes.
7. Single-user, no authentication, SQLite; whole dataset is loaded in memory (practical up to a few hundred thousand rows).
8. The Streamlit/Plotly front end could not be run in the environment where this project was authored (see verification below); expect small cosmetic fixes on first launch.

## 10. Verification status (what was actually run)
- **Run and passing:** 64 automated tests (`unittest` runner; pytest-compatible but pytest was not installable offline) covering validation, features, rules, graph analytics, scoring, database persistence/restart, ML with/without labels, tiny/empty datasets, exports, case reports.
- **Run end-to-end:** synthetic generation -> ingest -> full analysis (about 5 s for ~9k transactions) -> persisted results.
- **Run with stand-ins only:** all six UI pages were executed against a stub of `streamlit`/`plotly` to catch Python/pandas errors on empty, unanalysed and analysed datasets. This does **not** verify real rendering, widget behaviour or layout. Please launch the app and report anything odd.

| # | Acceptance item | Status |
|---|---|---|
| 1-3 | Starts, DB auto-init, reproducible synthetic data | DB/generator tested; app start by stub only |
| 4-5 | Load demo, upload + validation feedback | validator tested; upload widget UI not run live |
| 6, 20 | Dashboard from active data, nothing hardcoded | metrics tested (changes between datasets) |
| 7-10 | Features, rules, ML, unlabelled data | tested |
| 11-12 | Interactive graph, account neighbourhood | graph logic tested; Plotly rendering not run live |
| 13-14 | Explanations, search/filters | scoring tested; filter UI via stub |
| 15-16 | Persistence after restart, case export | tested |
| 17-19 | Tests, README, empty/invalid inputs | done |

## 11. Roadmap
1. Launch-test the UI and fix cosmetics; add screenshots.
2. Temporal graph analysis (time-respecting paths, burst detection, sliding-window features without full-period leakage).
3. Importers/mappers for PaySim and AMLSim; cross-dataset evaluation.
4. Better supervised models (gradient boosting, time-aware CV, probability calibration), SHAP-style explanations.
5. Graph ML (node embeddings, GNN) as optional modules behind the existing score interface.
6. More realistic synthetic scenarios (agent networks, account takeover, mixed legitimate overlap).
7. Authentication/roles, analyst assignment, SLA tracking.
8. Drift dashboards over time; privacy-preserving analysis; carefully validated near-real-time scoring.
