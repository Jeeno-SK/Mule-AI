"""Configurable thresholds and scoring weights (stored as overrides in SQLite; YAML file stays the default)."""
from __future__ import annotations

import copy

import streamlit as st

from config.settings import load_rules_config
from src.investigations.audit_log import log_action
from src.pipeline import OVERRIDE_KEY, run_analysis
from src.ui.common import AppContext, banner, safe_run


def _num(label: str, value, key: str):
    if isinstance(value, bool):
        return st.checkbox(label, value, key=key)
    if isinstance(value, int):
        return int(st.number_input(label, value=value, step=1, key=key))
    if isinstance(value, float):
        return float(st.number_input(label, value=value, key=key, format="%.4f" if abs(value) < 10 else "%.2f"))
    return value


def render(ctx: AppContext) -> None:
    st.title("Detection settings")
    banner(ctx)
    st.caption("Thresholds are tunable heuristics, not scientifically validated standards. Changes are stored locally and applied the next time analysis runs.")
    cfg = copy.deepcopy(ctx.config)
    with st.form("settings_form"):
        st.subheader("Feature windows")
        cols = st.columns(len(cfg["features"]))
        for col, (k, v) in zip(cols, cfg["features"].items()):
            with col:
                cfg["features"][k] = _num(k, v, f"f_{k}")
        st.subheader("Rules")
        for rid, r in cfg["rules"].items():
            with st.expander(f"{rid} - {r['name']}", expanded=False):
                st.caption(r["description"])
                c1, c2 = st.columns(2)
                r["enabled"] = c1.checkbox("Enabled", r["enabled"], key=f"en_{rid}")
                r["severity"] = c2.selectbox("Severity", ["low", "medium", "high"], ["low", "medium", "high"].index(r["severity"]), key=f"sev_{rid}")
                pcols = st.columns(max(1, len(r["params"])))
                for col, (k, v) in zip(pcols, r["params"].items()):
                    with col:
                        r["params"][k] = _num(k, v, f"p_{rid}_{k}")
        st.subheader("Risk scoring")
        s = cfg["scoring"]
        c = st.columns(4)
        for col, k in zip(c, ["low", "medium", "high", "critical"]):
            with col:
                s["bands"][k] = _num(f"{k} band minimum score", s["bands"][k], f"b_{k}")
        s["review_min_score"] = _num("Flag-for-review minimum score", s["review_min_score"], "review_min")
        c = st.columns(3)
        for col, k in zip(c, ["rules", "graph", "ml"]):
            with col:
                s["weights"][k] = _num(f"weight: {k}", s["weights"][k], f"w_{k}")
        c = st.columns(3)
        for col, k in zip(c, ["low", "medium", "high"]):
            with col:
                s["severity_points"][k] = _num(f"points: {k} severity", s["severity_points"][k], f"pts_{k}")
        st.subheader("Machine learning")
        mcols = st.columns(len(cfg["ml"]))
        for col, (k, v) in zip(mcols, cfg["ml"].items()):
            with col:
                cfg["ml"][k] = _num(k, v, f"ml_{k}")
        rerun = st.checkbox("Re-run analysis on the active dataset after saving", True)
        saved = st.form_submit_button("Save settings", type="primary")
    if saved:
        ctx.repo.set_setting(OVERRIDE_KEY, cfg)
        log_action(ctx.repo, ctx.actor, "settings_saved", ctx.dataset_id, "detection_config", {})
        if rerun and ctx.dataset_id:
            with st.spinner("Re-running analysis..."):
                ok, out = safe_run(run_analysis, ctx.repo, ctx.dataset_id, cfg)
            if ok:
                st.session_state["flash"] = f"Settings saved. Analysis re-run: {out['alerts']} alerts, {out['flagged_accounts']} accounts flagged."
        else:
            st.session_state["flash"] = "Settings saved."
        st.rerun()
    if st.button("Reset all settings to config/rules.yaml defaults"):
        ctx.repo.delete_setting(OVERRIDE_KEY)
        log_action(ctx.repo, ctx.actor, "settings_reset", ctx.dataset_id, "detection_config", {})
        st.session_state["flash"] = "Settings reset to defaults. Re-run analysis to apply."
        st.rerun()
    with st.expander("Scoring formula"):
        st.markdown("""
`score = 100 x (rule_points + graph_points + ml_points) / available_max`

- **rule_points**: sum over distinct triggered rules of severity points, capped (derived rule R009 does not double count).
- **graph_points**: share of direct counterparties that also have rule alerts, only for accounts that already have their own rule evidence.
- **ml_points**: supervised score if a model was trained, otherwise the top-decile anomaly percentile rescaled to 0-1. Dropped from `available_max` when no model ran.
- The result is a **prioritisation heuristic**. It is not a calibrated probability of fraud.
""")
