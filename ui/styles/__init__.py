"""Centralized CSS design system injector.

Streamlit rebuilds the DOM every rerun, so styles must be injected each run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

_STYLE_DIR = Path(__file__).resolve().parent
_CSS_FILES = (
    "variables.css",
    "typography.css",
    "layout.css",
    "cards.css",
    "buttons.css",
    "sidebar.css",
    "forms.css",
    "tables.css",
    "chat.css",
    "responsive.css",
)


@lru_cache(maxsize=1)
def _load_css() -> str:
    parts: list[str] = []
    for name in _CSS_FILES:
        path = _STYLE_DIR / name
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def inject_styles() -> None:
    """Inject the full design system (cached file read, injected each Streamlit run)."""
    payload = f"<style>{_load_css()}</style>"
    try:
        st.html(payload)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        import streamlit.components.v1 as components

        components.html(payload, height=0, width=0)
        return
    except Exception:  # noqa: BLE001
        pass
    st.markdown(payload, unsafe_allow_html=True)
