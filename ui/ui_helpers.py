"""Shared UI helpers (HTML-safe rendering, inject once)."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def inject_html(markup: str) -> None:
    """Render HTML without Streamlit markdown sanitization leaks."""
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


def once(key: str) -> bool:
    """Return True the first time per session for a key."""
    flag = f"_once_{key}"
    if st.session_state.get(flag):
        return False
    st.session_state[flag] = True
    return True
