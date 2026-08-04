"""Graph package — import submodules directly to avoid circular imports.

Preferred:
  from graph.state import GraphState, empty_state
  from graph.workflow import SQLMindOrchestrator, build_sqlmind_graph
"""

__all__ = [
    "GraphState",
    "SQLMindOrchestrator",
    "build_sqlmind_graph",
    "empty_state",
    "normalize_plan",
]


def __getattr__(name: str):
    if name in {"GraphState", "empty_state", "normalize_plan"}:
        from graph import state as _state

        return getattr(_state, name)
    if name in {"SQLMindOrchestrator", "build_sqlmind_graph"}:
        from graph import workflow as _workflow

        return getattr(_workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
