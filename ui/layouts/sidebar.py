"""Sidebar — lightweight grouped navigation."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from ui import theme as T
from ui.components import brand_block, connection_status_card
from ui.ui_helpers import inject_html
from ui.widgets.connection import render_connection_panel, render_llm_panel


def render_sidebar(settings: Settings) -> str:
    with st.sidebar:
        brand_block(settings.app_name, "Multi-agent SQL analytics")

        connected = bool(st.session_state.get("connected"))
        cfg = st.session_state.get("db_config")
        schema = st.session_state.get("schema")
        tables = len(getattr(schema, "tables", []) or []) if schema else "—"

        connection_status_card(
            connected=connected,
            name=cfg.display_name if cfg else "",
            engine=cfg.dialect if cfg else "",
            tables=tables,
        )

        page = _render_nav()

        from ui.gate import is_llm_ready

        llm_ok, _ = is_llm_ready(settings)
        inject_html(
            f'<div class="sq-side-foot">{"● LLM ready" if llm_ok else "○ Configure LLM"}</div>'
        )

        st.divider()
        inject_html('<div class="sq-nav-label">Connect</div>')
        with st.expander("Database", expanded=not connected):
            render_connection_panel(settings, key_prefix="side", framed=False)
        with st.expander("LLM provider", expanded=not llm_ok):
            render_llm_panel(settings, key_prefix="side", framed=False)

        st.divider()
        inject_html(f'<div class="sq-side-foot">v{settings.app_version}</div>')
    return page


def _render_nav() -> str:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"

    current = st.session_state.nav_page
    for group_label, items in T.NAV_GROUPS:
        inject_html(f'<div class="sq-nav-label">{group_label}</div>')
        for name, icon, _short, desc in items:
            active = current == name
            clicked = st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                type="primary" if active else "secondary",
                use_container_width=True,
                help=desc,
            )
            if clicked and not active:
                st.session_state.nav_page = name
                st.rerun()
    return st.session_state.nav_page
