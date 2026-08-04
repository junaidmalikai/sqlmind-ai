"""Agents package — import from ``agents.nodes`` / ``agents.react_sql``."""

from __future__ import annotations

__all__ = [
    "fail_node",
    "finalize_node",
    "make_sql_agent",
    "make_supervisor_agent",
    "make_visualization_agent",
    "make_sql_react_agent",
    "make_validation_node",
    "make_execution_node",
    "make_export_node",
]


def __getattr__(name: str):
    if name in __all__ or name.startswith("make_") or name.endswith("_agent") or name.endswith("_node"):
        from agents import nodes as _nodes
        from agents import react_sql as _react

        if hasattr(_nodes, name):
            return getattr(_nodes, name)
        if hasattr(_react, name):
            return getattr(_react, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
