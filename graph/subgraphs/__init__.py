"""True LangGraph subgraphs — Planning / Execution / Analytics / Export /
Reflection / Recovery / Memory.

Supervisor (parent graph) orchestrates these compiled subgraphs.
Existing flat nodes remain registered for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from graph.state import GraphState


NodeFn = Callable[[GraphState], dict[str, Any]]


def _compile(builder: StateGraph) -> Any:
    return builder.compile()


def build_memory_subgraph(
    *,
    memory_agent: NodeFn,
    goal_tracking: NodeFn | None = None,
) -> Any:
    """Memory Graph — load hybrid memory (+ optional goal tracking bootstrap)."""
    g = StateGraph(GraphState)
    g.add_node("memory_agent", memory_agent)
    if goal_tracking is not None:
        g.add_node("goal_tracking", goal_tracking)
        g.add_edge(START, "memory_agent")
        g.add_edge("memory_agent", "goal_tracking")
        g.add_edge("goal_tracking", END)
    else:
        g.add_edge(START, "memory_agent")
        g.add_edge("memory_agent", END)
    return _compile(g)


def build_planning_subgraph(
    *,
    goal_understanding: NodeFn,
    planner: NodeFn,
    task_decomposition: NodeFn,
    goal_tracking: NodeFn | None = None,
) -> Any:
    """Planning Graph — Goal → Planner → Decomposition (+ goal tracking).

    If goal_understanding requests clarify/approval, subgraph ends so the
    parent graph can route to HITL nodes.
    """
    g = StateGraph(GraphState)
    g.add_node("goal_understanding", goal_understanding)
    g.add_node("planner", planner)
    g.add_node("task_decomposition", task_decomposition)
    g.add_edge(START, "goal_understanding")

    def after_goal(state: GraphState) -> str:
        nxt = state.get("next_agent") or "planner"
        if nxt in {"clarify", "approval_gate"}:
            return END
        return "planner"

    g.add_conditional_edges(
        "goal_understanding",
        after_goal,
        {"planner": "planner", END: END},
    )
    g.add_edge("planner", "task_decomposition")
    if goal_tracking is not None:
        g.add_node("goal_tracking", goal_tracking)
        g.add_edge("task_decomposition", "goal_tracking")
        g.add_edge("goal_tracking", END)
    else:
        g.add_edge("task_decomposition", END)
    return _compile(g)


def build_execution_subgraph(
    *,
    coordinator: NodeFn,
    supervisor: NodeFn,
    schema_agent: NodeFn,
    sql_agent: NodeFn,
    validation_node: NodeFn,
    execution_node: NodeFn,
) -> Any:
    """Execution Graph — Coordinator / Supervisor + SQL security chain.

    Soft routing returns to END with next_agent set; parent graph continues.
    Hard SQL path remains fixed inside this subgraph.
    """
    g = StateGraph(GraphState)
    g.add_node("execution_coordinator", coordinator)
    g.add_node("supervisor", supervisor)
    g.add_node("schema_agent", schema_agent)
    g.add_node("sql_agent", sql_agent)
    g.add_node("validation_node", validation_node)
    g.add_node("execution_node", execution_node)

    def _entry(state: GraphState) -> dict[str, Any]:
        # Passthrough — parent sets next via initial routing into subgraph
        return {}

    g.add_node("_entry", _entry)
    g.add_edge(START, "_entry")

    def route_entry(state: GraphState) -> str:
        nxt = state.get("next_agent") or "execution_coordinator"
        if nxt in {
            "execution_coordinator",
            "supervisor",
            "schema_agent",
            "sql_agent",
            "validation_node",
            "execution_node",
        }:
            return nxt
        return "execution_coordinator"

    g.add_conditional_edges(
        "_entry",
        route_entry,
        {
            "execution_coordinator": "execution_coordinator",
            "supervisor": "supervisor",
            "schema_agent": "schema_agent",
            "sql_agent": "sql_agent",
            "validation_node": "validation_node",
            "execution_node": "execution_node",
        },
    )

    # Coordinator / supervisor / schema exit to parent
    for n in ("execution_coordinator", "supervisor", "schema_agent"):
        g.add_edge(n, END)

    g.add_edge("sql_agent", "validation_node")

    def after_val(state: GraphState) -> str:
        return "execution_node" if state.get("sql_valid") else END

    g.add_conditional_edges(
        "validation_node",
        after_val,
        {"execution_node": "execution_node", END: END},
    )
    g.add_edge("execution_node", END)
    return _compile(g)


def build_analytics_subgraph(
    *,
    visualization_agent: NodeFn,
    insight_agent: NodeFn,
    optimization_agent: NodeFn,
    dashboard_agent: NodeFn,
    summary_agent: NodeFn,
) -> Any:
    """Analytics Graph — viz / insight / optimization / dashboard / summary."""
    g = StateGraph(GraphState)
    g.add_node("visualization_agent", visualization_agent)
    g.add_node("insight_agent", insight_agent)
    g.add_node("optimization_agent", optimization_agent)
    g.add_node("dashboard_agent", dashboard_agent)
    g.add_node("summary_agent", summary_agent)

    def _entry(state: GraphState) -> dict[str, Any]:
        return {}

    g.add_node("_entry", _entry)
    g.add_edge(START, "_entry")

    def route(state: GraphState) -> str:
        nxt = state.get("next_agent") or "insight_agent"
        allowed = {
            "visualization_agent",
            "insight_agent",
            "optimization_agent",
            "dashboard_agent",
            "summary_agent",
        }
        return nxt if nxt in allowed else "insight_agent"

    g.add_conditional_edges(
        "_entry",
        route,
        {
            "visualization_agent": "visualization_agent",
            "insight_agent": "insight_agent",
            "optimization_agent": "optimization_agent",
            "dashboard_agent": "dashboard_agent",
            "summary_agent": "summary_agent",
        },
    )
    for n in (
        "visualization_agent",
        "insight_agent",
        "optimization_agent",
        "dashboard_agent",
        "summary_agent",
    ):
        g.add_edge(n, END)
    return _compile(g)


def build_export_subgraph(*, export_node: NodeFn) -> Any:
    g = StateGraph(GraphState)
    g.add_node("export_node", export_node)
    g.add_edge(START, "export_node")
    g.add_edge("export_node", END)
    return _compile(g)


def build_reflection_subgraph(*, reflection_agent: NodeFn) -> Any:
    g = StateGraph(GraphState)
    g.add_node("reflection_agent", reflection_agent)
    g.add_edge(START, "reflection_agent")
    g.add_edge("reflection_agent", END)
    return _compile(g)


def build_recovery_subgraph(
    *,
    retry_agent: NodeFn,
    replan_agent: NodeFn | None = None,
    recovery_controller: NodeFn | None = None,
) -> Any:
    """Recovery Graph — policy controller → retry / replan / escalate paths.

    Actions: Recover | Retry | Fallback | Abort | Escalate (via recovery_controller).
    """
    g = StateGraph(GraphState)
    has_controller = recovery_controller is not None
    has_replan = replan_agent is not None

    if has_controller:
        g.add_node("recovery_controller", recovery_controller)
    g.add_node("retry_agent", retry_agent)
    if has_replan:
        g.add_node("replan_agent", replan_agent)

    g.add_edge(START, "recovery_controller" if has_controller else "retry_agent")

    if has_controller:

        def after_controller(state: GraphState) -> str:
            nxt = state.get("next_agent") or "retry_agent"
            action = (state.get("recovery_decision") or {}).get("action")
            if (nxt == "replan_agent" or action == "recover") and has_replan:
                return "replan_agent"
            if nxt in {"sql_agent", "retry_agent"} or action == "retry":
                return "retry_agent"
            return END

        targets: dict[str, Any] = {"retry_agent": "retry_agent", END: END}
        if has_replan:
            targets["replan_agent"] = "replan_agent"
        g.add_conditional_edges("recovery_controller", after_controller, targets)

    if has_replan:

        def after_retry(state: GraphState) -> str:
            if state.get("next_agent") == "replan_agent" or state.get("retry_next_action") == "replan":
                return "replan_agent"
            return END

        g.add_conditional_edges(
            "retry_agent",
            after_retry,
            {"replan_agent": "replan_agent", END: END},
        )
        g.add_edge("replan_agent", END)
    else:
        g.add_edge("retry_agent", END)

    return _compile(g)


class SubgraphBundle:
    """Collection of compiled subgraphs for supervisor orchestration."""

    def __init__(
        self,
        *,
        memory: Any | None = None,
        planning: Any | None = None,
        execution: Any | None = None,
        analytics: Any | None = None,
        export: Any | None = None,
        reflection: Any | None = None,
        recovery: Any | None = None,
    ) -> None:
        self.memory = memory
        self.planning = planning
        self.execution = execution
        self.analytics = analytics
        self.export = export
        self.reflection = reflection
        self.recovery = recovery

    def as_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "memory_graph": self.memory,
                "planning_graph": self.planning,
                "execution_graph": self.execution,
                "analytics_graph": self.analytics,
                "export_graph": self.export,
                "reflection_graph": self.reflection,
                "recovery_graph": self.recovery,
            }.items()
            if v is not None
        }

    @property
    def node_count(self) -> int:
        """Count of subgraph entry nodes exposed to the parent."""
        return len(self.as_dict())
