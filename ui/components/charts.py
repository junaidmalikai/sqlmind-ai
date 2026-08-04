"""Plotly chart rendering with stable unique Streamlit keys."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

_DEFAULT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "sqlmind_chart"},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _safe_key_part(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "")).strip("_").lower()
    return text or "chart"


def plotly_chart_key(prefix: str, label: str, index: int, *, extra: str = "") -> str:
    """Build a stable unique key for one plotly_chart element."""
    parts = [_safe_key_part(prefix), _safe_key_part(label), str(index)]
    if extra:
        parts.append(_safe_key_part(extra))
    return "plotly_" + "_".join(parts)


def render_plotly_chart(
    fig: Any,
    *,
    key: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Single chart — always keyed to avoid StreamlitDuplicateElementId."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config=config if config is not None else _DEFAULT_CONFIG,
    )


def render_plotly_chart_tabs(
    charts: list[tuple[str, Any]],
    *,
    key_prefix: str,
    config: dict[str, Any] | None = None,
) -> None:
    """Render chart tabs; each plotly_chart and the tabs widget get unique keys."""
    if not charts:
        return
    labels = [str(label) for label, _ in charts]
    tabs = st.tabs(labels, key=f"{_safe_key_part(key_prefix)}_chart_tabs")
    cfg = config if config is not None else _DEFAULT_CONFIG
    for i, (tab, (label, fig)) in enumerate(zip(tabs, charts)):
        with tab:
            render_plotly_chart(
                fig,
                key=plotly_chart_key(key_prefix, label, i),
                config=cfg,
            )
