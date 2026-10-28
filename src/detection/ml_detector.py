"""Baseline machine-learning detection at account level.

* Supervised (RandomForest) only when dataset-provided labels give enough positive AND negative accounts.
  Account label = account RECEIVED at least one transaction labelled fraud (positive); negative = has labelled
  transactions and received none labelled fraud; accounts with no labelled transactions are UNKNOWN and excluded
  (unlabelled is never treated as legitimate). Labels are never model inputs.
* Unsupervised (IsolationForest) when enough accounts exist. Output is a within-dataset anomaly percentile,
  NOT a probability of fraud: by construction ~5% of accounts always exceed the 0.95 percentile.
* If neither is possible the result states why and rule-based detection continues unaffected.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.detection.rule_engine import ALERT_COLUMNS, make_alert_id
from src.features.transaction_features import ML_FEATURES


@dataclass
class MLResult:
    scores: pd.DataFrame                      # account, ml_supervised_score, ml_anomaly_percentile, ml_in_training
    metadata: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(len(self.scores)) and (self.metadata.get("supervised", {}).get("status") == "trained"
                                           or self.metadata.get("unsupervised", {}).get("status") == "trained")


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def _pipeline(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("log", FunctionTransformer(_log1p)),
                     ("scale", StandardScaler()), ("model", model)])


def build_account_labels(tx: pd.DataFrame) -> pd.Series:
    """1 = received >=1 fraud-labelled tx, 0 = labelled but none received as fraud, NaN = unknown."""
    lab = tx[tx["is_fraud"].notna()]
    if lab.empty:
        return pd.Series(dtype=float)
    incident = pd.concat([lab["source_account"], lab["destination_account"]]).unique()
    pos = set(lab.loc[lab["is_fraud"] == 1, "destination_account"])
    return pd.Series({a: (1.0 if a in pos else 0.0) for a in incident})


def _metrics(y_true, y_prob, thr=0.5) -> dict:
    pred = (np.asarray(y_prob) >= thr).astype(int)
    out = {"threshold": thr, "n_test": int(len(y_true)), "n_test_positive": int(np.sum(y_true)),
           "precision": float(precision_score(y_true, pred, zero_division=0)),
           "recall": float(recall_score(y_true, pred, zero_division=0)),
           "f1": float(f1_score(y_true, pred, zero_division=0)),
           "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()}
    out["pr_auc"] = float(average_precision_score(y_true, y_prob)) if np.sum(y_true) > 0 else None
    return out


def train_and_score(account_df: pd.DataFrame, tx: pd.DataFrame, ml_cfg: dict, dataset_info: dict | None = None,
                    models_dir: Path | None = None) -> MLResult:
    """Train available models, evaluate when labels permit, and score every account."""
    feats = [f for f in ML_FEATURES if f in account_df.columns]
    X = account_df[feats].astype(float)
    accounts = account_df["account"].to_numpy()
    scores = pd.DataFrame({"account": accounts, "ml_supervised_score": np.nan, "ml_anomaly_percentile": np.nan, "ml_in_training": False})
    meta = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sklearn_version": sklearn.__version__,
            "features": feats, "dataset": dataset_info or {}, "random_state": ml_cfg["random_state"],
            "note": "ml_supervised_score is an uncalibrated model score, not a probability of fraud. "
                    "ml_anomaly_percentile is a within-dataset rank, not a fraud probability."}
    msgs: list[str] = []
    rs = ml_cfg["random_state"]

    # ---- unsupervised ----
    if len(X) >= ml_cfg["min_accounts_unsupervised"]:
        iso = _pipeline(IsolationForest(n_estimators=200, contamination="auto", random_state=rs))
        iso.fit(X)
        raw = -iso.decision_function(X)
        scores["ml_anomaly_percentile"] = pd.Series(raw).rank(pct=True).to_numpy()
        meta["unsupervised"] = {"status": "trained", "model": "IsolationForest", "n_accounts": int(len(X))}
        iso_pipe = iso
    else:
        iso_pipe = None
        meta["unsupervised"] = {"status": "skipped", "reason": f"Only {len(X)} accounts; need >= {ml_cfg['min_accounts_unsupervised']}."}
        msgs.append(meta["unsupervised"]["reason"])

    # ---- supervised ----
    sup_model = None
    labels = build_account_labels(tx) if "is_fraud" in tx.columns else pd.Series(dtype=float)
    y = account_df["account"].map(labels) if len(labels) else pd.Series(np.nan, index=account_df.index)
    known = y.notna().to_numpy()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not len(labels):
        meta["supervised"] = {"status": "skipped", "reason": "Dataset has no fraud labels; unlabelled rows are not assumed legitimate."}
    elif n_pos < ml_cfg["min_positive_accounts"] or n_neg < ml_cfg["min_negative_accounts"]:
        meta["supervised"] = {"status": "skipped", "n_positive_accounts": n_pos, "n_negative_accounts": n_neg,
                              "reason": f"Need >= {ml_cfg['min_positive_accounts']} positive and >= {ml_cfg['min_negative_accounts']} negative labelled accounts."}
    else:
        Xk, yk = X[known], y[known].astype(int)
        idx = np.arange(len(Xk))
        tr, te = train_test_split(idx, test_size=0.3, stratify=yk, random_state=rs)
        rf = lambda: _pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=rs, n_jobs=1))
        sup_model = rf().fit(Xk.iloc[tr], yk.iloc[tr])
        prob_te = sup_model.predict_proba(Xk.iloc[te])[:, 1]
        evals = {"stratified_split": {**_metrics(yk.iloc[te].to_numpy(), prob_te), "n_train": int(len(tr))}}
        # chronological: earliest-seen 70% of labelled accounts train, latest 30% test
        order = account_df.loc[known, "first_seen"].reset_index(drop=True).sort_values(kind="stable").index.to_numpy()
        cut = int(len(order) * 0.7)
        ctr, cte = order[:cut], order[cut:]
        if yk.iloc[ctr].nunique() == 2 and yk.iloc[cte].sum() > 0 and (yk.iloc[cte] == 0).any():
            m2 = rf().fit(Xk.iloc[ctr], yk.iloc[ctr])
            evals["chronological_split"] = {**_metrics(yk.iloc[cte].to_numpy(), m2.predict_proba(Xk.iloc[cte])[:, 1]), "n_train": int(len(ctr)),
                                            "caveat": "Features are computed over the full period, so this only approximates a future-deployment test."}
        else:
            evals["chronological_split"] = {"status": "skipped", "reason": "Training or test period lacks both classes."}
        scores["ml_supervised_score"] = sup_model.predict_proba(X)[:, 1]
        in_train = np.zeros(len(X), dtype=bool)
        in_train[np.where(known)[0][tr]] = True
        scores["ml_in_training"] = in_train
        meta["supervised"] = {"status": "trained", "model": "RandomForestClassifier", "n_labelled_accounts": int(known.sum()),
                              "n_positive_accounts": n_pos, "n_negative_accounts": n_neg, "evaluation": evals,
                              "label_definition": "positive = received >=1 transaction labelled fraud; accounts without labelled transactions excluded"}
    if not any(meta[k]["status"] == "trained" for k in ("supervised", "unsupervised")):
        msgs.append("No ML model could be trained; rule-based detection only.")
    for k in ("supervised", "unsupervised"):
        if meta[k]["status"] == "skipped":
            msgs.append(f"{k.capitalize()} model skipped: {meta[k]['reason']}")

    if models_dir is not None and (sup_model is not None or iso_pipe is not None):
        meta["artifact"] = _save_artifacts(models_dir, {"supervised": sup_model, "unsupervised": iso_pipe, "features": feats}, meta)
    return MLResult(scores, meta, msgs)


def _save_artifacts(models_dir: Path, bundle: dict, meta: dict) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"muleshield_models_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {"file": path.name, "sha256": digest}
    path.with_suffix(".json").write_text(json.dumps({**meta, "artifact": info}, indent=2, default=str), encoding="utf-8")
    return info


def load_artifacts(path: Path, expected_sha256: str, models_dir: Path) -> dict:
    """Load a locally saved bundle only if it lives in models_dir and its checksum matches (joblib uses pickle)."""
    path = Path(path).resolve()
    if models_dir.resolve() not in path.parents:
        raise ValueError("Model artefacts may only be loaded from the models directory.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Model artefact checksum mismatch; refusing to load.")
    return joblib.load(path)


def ml_alerts(ml: MLResult, ml_cfg: dict) -> pd.DataFrame:
    """Model-based indicators as alerts (rule ids ML001 / ML002). Not proof; not probabilities."""
    rows = []
    s = ml.scores
    for r in s[s["ml_anomaly_percentile"] >= ml_cfg["anomaly_alert_percentile"]].itertuples(index=False):
        rows.append(_ml_row("ML001", "Unsupervised anomaly indicator", r.account,
                            f"Behavioural profile is more unusual than {r.ml_anomaly_percentile:.1%} of accounts in this dataset (IsolationForest rank; not a fraud probability).",
                            {"anomaly_percentile": round(float(r.ml_anomaly_percentile), 4)}, {"min_percentile": ml_cfg["anomaly_alert_percentile"]}))
    hold = s[(s["ml_supervised_score"] >= ml_cfg["supervised_alert_score"])]
    for r in hold.itertuples(index=False):
        note = " (account was in the training set: score is optimistic)" if r.ml_in_training else ""
        rows.append(_ml_row("ML002", "Supervised model indicator", r.account,
                            f"RandomForest score {r.ml_supervised_score:.2f} (uncalibrated, not a probability){note}.",
                            {"supervised_score": round(float(r.ml_supervised_score), 4), "in_training": bool(r.ml_in_training)},
                            {"min_score": ml_cfg["supervised_alert_score"]}))
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)


def _ml_row(rid, name, account, evidence, observed, threshold) -> dict:
    return {"alert_id": make_alert_id(rid, account), "rule_id": rid, "rule_name": name, "severity": "medium", "account": account,
            "transaction_id": "", "timestamp": None, "description": "Model-derived indicator for human review; verify with transaction evidence.",
            "observed_json": json.dumps(observed), "threshold_json": json.dumps(threshold), "evidence": evidence}
