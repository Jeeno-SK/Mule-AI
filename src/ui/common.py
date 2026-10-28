"""Shared UI helpers: styling, context loading, banners, chart styling."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from config.settings import Settings
from src.database.repository import Repository
from src.pipeline import get_effective_config
from src.utils.exceptions import MuleShieldError

BAND_ORDER = ["none", "low", "medium", "high", "critical"]
BAND_COLORS = {"none": "#64748B", "low": "#38BDF8", "medium": "#FBBF24", "high": "#FB923C", "critical": "#EF4444", "unscored": "#475569"}
ACCENT = "#22B8CF"

CSS = """
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0E1626; border-right: 1px solid #1E2A44; }
[data-testid="stMetric"] { background: #121B2E; border: 1px solid #1E2A44; border-radius: 10px; padding: 14px 16px; }
[data-testid="stMetricLabel"] { color: #94A3B8; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ms-banner { border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.9rem; border: 1px solid; }
.ms-synth { background: #1B2A3F; border-color: #22B8CF; color: #BEEFF7; }
.ms-imported { background: #1D2433; border-color: #475569; color: #CBD5E1; }
.ms-note { color: #94A3B8; font-size: 0.85rem; }
.ms-title { font-size: 1.25rem; font-weight: 700; color: #E6EDF7; }
.ms-sub { font-size: 0.72rem; color: #7C8DA8; line-height: 1.3; }
</style>
"""


@dataclass
class AppContext:
    repo: Repository
    settings: Settings
    actor: str
    dataset: dict | None
    config: dict
    tx: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    alert_txs: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def dataset_id(self) -> int | None:
        return None if self.dataset is None else int(self.dataset["dataset_id"])

    @property
    def analysed(self) -> bool:
        return bool(self.dataset and self.dataset.get("analyzed_at"))

    @property
    def review_min_score(self) -> float:
        return float(self.config["scoring"]["review_min_score"])

    @property
    def detection_basis(self) -> str:
        if not self.analysed:
            return "not analysed"
        return "rule-based" + (" + " + self.risk["basis"].iloc[0].replace("rules + ", "") if len(self.risk) and self.risk["basis"].iloc[0] != "rules only" else "")


@st.cache_data(show_spinner=False)
def _load_static(db_path: str, dataset_id: int, analyzed_at: str | None):
    repo = Repository(db_path)
    return repo.get_transactions(dataset_id), repo.get_account_risk(dataset_id), repo.get_alert_transactions(dataset_id)


def build_context(repo: Repository, settings: Settings, dataset_id: int | None, actor: str) -> AppContext:
    config = get_effective_config(repo)
    dataset = repo.get_dataset(dataset_id) if dataset_id else None
    ctx = AppContext(repo, settings, actor, dataset, config)
    if dataset:
        ctx.tx, ctx.risk, ctx.alert_txs = _load_static(repo.db_path, dataset_id, dataset.get("analyzed_at"))
        ctx.alerts = repo.get_alerts(dataset_id)  # not cached: statuses change during triage
    return ctx


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def banner(ctx: AppContext) -> None:
    d = ctx.dataset
    if not d:
        return
    if d["origin"] == "synthetic":
        st.markdown(f'<div class="ms-banner ms-synth"><b>SIMULATED DATA</b> &nbsp;|&nbsp; Dataset "{_esc(d["name"])}" is synthetic. '
                    'Patterns shown are generated for demonstration and are not real banking evidence.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ms-banner ms-imported"><b>IMPORTED DATA</b> &nbsp;|&nbsp; Dataset "{_esc(d["name"])}". '
                    'Alerts are unverified risk indicators for human review, not findings of wrongdoing.</div>', unsafe_allow_html=True)


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def empty_state(message: str, hint: str | None = None) -> None:
    st.info(message + (f"\n\n{hint}" if hint else ""))


def require_analysis(ctx: AppContext) -> bool:
    """Show an empty state and return False if there is no analysed dataset."""
    if ctx.dataset is None:
        empty_state("No dataset is loaded.", "Open **Data & Transactions** to generate a synthetic dataset or upload a CSV.")
        return False
    if not ctx.analysed:
        empty_state("This dataset has not been analysed yet.", "Open **Data & Transactions** and press **Run analysis**.")
        return False
    return True


def style_fig(fig, height: int = 320):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(color="#CBD5E1"), legend=dict(orientation="h", y=-0.2))
    return fig


def safe_run(fn, *args, **kwargs):
    """Run an action; show user-safe errors instead of stack traces. Returns (ok, result)."""
    try:
        return True, fn(*args, **kwargs)
    except MuleShieldError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - technical details are logged, not shown
        from src.utils.logging_config import get_logger
        get_logger(__name__).exception("Unexpected UI error")
        st.error("Something went wrong. Technical details were written to the application log.")
    return False, None


def money(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"
