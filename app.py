"""MuleShield AI - single entry point.  Run with:  streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from src.database.repository import Repository
from src.ui import dashboard, detection_settings, graph_view, investigations, model_monitor, transactions
from src.ui.common import build_context, inject_css

SETTINGS = get_settings()
st.set_page_config(page_title=SETTINGS.app_name, page_icon=":shield:", layout="wide", initial_sidebar_state="expanded")

PAGES = {
    "Dashboard": dashboard.render,
    "Data & Transactions": transactions.render,
    "Alerts & Investigations": investigations.render,
    "Network Graph": graph_view.render,
    "Detection Settings": detection_settings.render,
    "Data Quality & Models": model_monitor.render,
}


@st.cache_resource
def get_repo() -> Repository:
    return Repository(SETTINGS.db_path)  # creates the SQLite file and schema on first run


def main() -> None:
    inject_css()
    repo = get_repo()
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"
    with st.sidebar:
        st.markdown(f'<div class="ms-title">{SETTINGS.app_name}</div><div class="ms-sub">{SETTINGS.subtitle}</div>', unsafe_allow_html=True)
        st.divider()
        st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        lib = repo.list_datasets()
        active = repo.get_active_dataset_id()
        if not lib.empty:
            ids = lib["dataset_id"].tolist()
            labels = lib.set_index("dataset_id")
            choice = st.selectbox("Active dataset", ids, index=ids.index(active) if active in ids else 0,
                                  format_func=lambda i: f"{i}: {labels.at[i, 'name']} [{labels.at[i, 'origin']}]")
            if choice != active:
                repo.set_active_dataset(int(choice))
                st.rerun()
            active = int(choice)
        actor = st.text_input("Analyst name (recorded in audit log)", SETTINGS.default_analyst, max_chars=40).strip() or "analyst"
        st.caption("Prototype for authorised research and demonstration. Uses no external services; no real accounts are contacted.")

    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    ctx = build_context(repo, SETTINGS, active, actor)
    try:
        PAGES[st.session_state["nav"]](ctx)
    except Exception:  # noqa: BLE001 - never show a stack trace to the analyst
        from src.utils.logging_config import get_logger
        get_logger("app").exception("Unhandled error on page %s", st.session_state["nav"])
        st.error("This page hit an unexpected error. Technical details were written to the application log.")


main()
