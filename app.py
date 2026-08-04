"""
SQLMind AI — Streamlit entry point.

Deploy on Streamlit Community Cloud with:
  Main file path: app.py
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from ui.gate import is_llm_ready
from ui.layouts.sidebar import render_sidebar
from ui.pages import (
    _handle_user_question,
    render_about,
    render_bookmarks,
    render_chat,
    render_exports,
    render_history,
    render_home,
    render_runtime_ops,
    render_schema,
    render_settings,
)
from ui.session import init_session_state, inject_minimal_css


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state(settings)
    inject_minimal_css()

    page = render_sidebar(settings)

    # Legacy Dashboard nav → Home (merged)
    if page == "Dashboard" or st.session_state.get("nav_page") == "Dashboard":
        page = "Home"
        st.session_state.nav_page = "Home"

    pending = st.session_state.pop("_pending_question", None)
    if pending:
        ok, reason = is_llm_ready(settings)
        if not ok or not st.session_state.get("connected"):
            st.session_state.nav_page = "Settings" if not ok else "Home"
            st.toast(reason if not ok else "Connect a database first.", icon="🔒")
            page = st.session_state.nav_page
        else:
            page = "Chat"
            st.session_state.nav_page = "Chat"
            st.session_state["_run_question"] = pending

    routes = {
        "Home": render_home,
        "Chat": render_chat,
        "Schema": render_schema,
        "Runtime": render_runtime_ops,
        "History": render_history,
        "Bookmarks": render_bookmarks,
        "Exports": render_exports,
        "Settings": render_settings,
        "About": render_about,
    }

    if page == "Chat":
        run_q = st.session_state.pop("_run_question", None)
        render_chat(settings)
        if run_q:
            _handle_user_question(run_q, settings)
    else:
        handler = routes.get(page, render_home)
        handler(settings)


if __name__ == "__main__":
    main()
