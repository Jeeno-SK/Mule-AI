"""End-to-end analysis orchestration: features -> graph -> rules -> ML -> risk scores -> persistence."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config.settings import deep_merge, get_settings, load_rules_config
from src.database.repository import Repository
from src.detection import ml_detector, risk_scoring, rule_engine
from src.features import graph_features, transaction_features as tf
from src.graph import graph_analytics as ga
from src.graph.network_builder import build_graph
from src.utils.exceptions import AnalysisError
from src.utils.logging_config import get_logger

log = get_logger(__name__)
OVERRIDE_KEY = "rules_config_overrides"


def get_effective_config(repo: Repository) -> dict:
    """YAML defaults merged with analyst overrides stored in SQLite."""
    return deep_merge(load_rules_config(), repo.get_setting(OVERRIDE_KEY, {}))


def run_analysis(repo: Repository, dataset_id: int, config: dict | None = None) -> dict:
    """Run the full pipeline for a dataset, persist results and return a summary."""
    ds = repo.get_dataset(dataset_id)
    if ds is None:
        raise AnalysisError("Dataset not found.")
    tx = repo.get_transactions(dataset_id)
    if tx.empty:
        raise AnalysisError("The dataset has no transactions to analyse.")
    cfg = config or get_effective_config(repo)
    fcfg, gcfg = cfg["features"], cfg["graph"]
    try:
        acct = tf.compute_account_features(tx, fcfg["window_hours"], fcfg["rapid_forward_hours"],
                                           fcfg["forward_ratio_min"], fcfg["forward_ratio_max"])
        txf = tf.add_transaction_features(tx)
        pairs = tf.compute_rapid_forward_pairs(tx, fcfg["rapid_forward_hours"], fcfg["forward_ratio_min"], fcfg["forward_ratio_max"])
        G = build_graph(tx)
        r8 = cfg["rules"].get("R008", {}).get("params", {})
        cycles = ga.find_cycles(G, gcfg["max_cycle_length"], gcfg["max_cycles"], min_len=3,
                                min_edge_value=r8.get("min_cycle_value", 1000.0), max_span_hours=gcfg.get("max_cycle_span_hours", 72.0))
        r7 = cfg["rules"].get("R007", {}).get("params", {})
        chains, chain_warning = ga.find_rapid_chains(tx, gcfg["max_chain_gap_hours"], gcfg["chain_ratio_min"], gcfg["chain_ratio_max"],
                                                     r7.get("min_hops", 3), gcfg["max_chains"])
        gfeat = graph_features.compute_graph_features(G, cycles)
        rr = rule_engine.run_rules(tx, txf, acct, pairs, chains, cycles, G, cfg)

        merged = acct.merge(gfeat, on="account", how="left")
        info = {"dataset_id": dataset_id, "name": ds["name"], "origin": ds["origin"], "sha256": ds["sha256"],
                "label_status": ds["label_status"], "n_transactions": int(len(tx)), "n_accounts": int(len(acct))}
        try:
            ml = ml_detector.train_and_score(merged, tx, cfg["ml"], info, get_settings().models_dir)
        except Exception:  # ML is optional; rules must keep working
            log.exception("ML stage failed")
            ml = ml_detector.MLResult(pd.DataFrame(columns=["account", "ml_supervised_score", "ml_anomaly_percentile", "ml_in_training"]),
                                      {"supervised": {"status": "failed"}, "unsupervised": {"status": "failed"}},
                                      ["ML stage failed; rule-based detection only."])
        mla = ml_detector.ml_alerts(ml, cfg["ml"]) if ml.available else pd.DataFrame(columns=rule_engine.ALERT_COLUMNS)
        alerts = pd.concat([rr.alerts, mla], ignore_index=True).drop_duplicates("alert_id")
        risk = risk_scoring.compute_risk_scores(acct, gfeat, alerts, ml, tx, cfg["scoring"])
    except AnalysisError:
        raise
    except Exception as exc:
        log.exception("Analysis failed")
        raise AnalysisError("Analysis could not be completed. Check the application log for technical details.") from exc

    params = {"config": cfg, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "n_chains": int(len(chains)), "n_cycles": len(cycles), "chain_warning": chain_warning,
              "n_communities": int(gfeat["community_id"].nunique()) if len(gfeat) else 0,
              "n_components": int(gfeat["component_id"].nunique()) if len(gfeat) else 0, "ml_messages": ml.messages}
    repo.save_analysis(dataset_id, alerts, rr.alert_txs, risk, params)
    repo.save_model_run(dataset_id, "ml_baseline", ml.metadata)
    flagged = int((risk["risk_score"] >= cfg["scoring"]["review_min_score"]).sum())
    return {"alerts": int(len(alerts)), "flagged_accounts": flagged, "chains": int(len(chains)), "cycles": len(cycles),
            "ml_available": ml.available, "ml_messages": ml.messages, "chain_warning": chain_warning}
