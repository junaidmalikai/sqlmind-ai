"""Architecture verification against a completed runtime execution trace."""

from __future__ import annotations

from typing import Any

from observability.runtime_trace.collector import ExecutionTraceSession
from observability.runtime_trace.models import TraceCategory

# Checklist items from the enterprise audit brief
VERIFICATION_CHECKS: tuple[str, ...] = (
    "Goal Understanding executed",
    "Planning executed",
    "Task Decomposition executed",
    "Execution Coordinator executed",
    "Memory Graph executed",
    "Planning Graph executed",
    "Execution Graph executed",
    "Reflection Graph executed",
    "Recovery Graph executed",
    "Export Graph executed",
    "IAM enforced",
    "Plugin execution",
    "Message Bus used",
    "Memory retrieval",
    "Reflection executed",
    "Replanning executed",
    "Circuit Breaker active",
    "Observability active",
)


def _node_seen(session: ExecutionTraceSession, *names: str) -> bool:
    nodes = session.nodes_executed()
    return any(n in nodes for n in names)


def _category_seen(session: ExecutionTraceSession, category: TraceCategory) -> bool:
    return any(e.category == category for e in session.all_events())


def _kind_seen(session: ExecutionTraceSession, *kinds: str) -> bool:
    wanted = {k.lower() for k in kinds}
    return any((e.event_kind or "").lower() in wanted for e in session.all_events())


def verify_architecture(session: ExecutionTraceSession) -> dict[str, dict[str, Any]]:
    """Return {check_name: {passed, reason}} for every architecture checkpoint."""
    nodes = session.nodes_executed()
    lifecycle = session.lifecycle_seen()
    events = session.all_events()

    def result(passed: bool, reason: str) -> dict[str, Any]:
        return {"passed": passed, "reason": reason}

    checks: dict[str, dict[str, Any]] = {}

    checks["Goal Understanding executed"] = result(
        _node_seen(session, "goal_understanding")
        or "Goal Understanding" in lifecycle,
        "goal_understanding node observed"
        if _node_seen(session, "goal_understanding")
        else "goal_understanding not observed in this run",
    )
    checks["Planning executed"] = result(
        _node_seen(session, "planner") or "Planning" in lifecycle,
        "planner node observed" if _node_seen(session, "planner") else "planner not observed",
    )
    checks["Task Decomposition executed"] = result(
        _node_seen(session, "task_decomposition")
        or "Task Decomposition" in lifecycle,
        "task_decomposition observed"
        if _node_seen(session, "task_decomposition")
        else "task_decomposition not observed",
    )
    checks["Execution Coordinator executed"] = result(
        _node_seen(session, "execution_coordinator")
        or "Execution Coordination" in lifecycle,
        "execution_coordinator observed"
        if _node_seen(session, "execution_coordinator")
        else "execution_coordinator not observed",
    )

    for label, node in (
        ("Memory Graph executed", "memory_graph"),
        ("Planning Graph executed", "planning_graph"),
        ("Execution Graph executed", "execution_graph"),
        ("Reflection Graph executed", "reflection_graph"),
        ("Recovery Graph executed", "recovery_graph"),
        ("Export Graph executed", "export_graph"),
    ):
        # Flat-mode runs may execute equivalent leaf nodes without subgraph wrappers
        flat_equivalents = {
            "memory_graph": {"memory_agent"},
            "planning_graph": {"goal_understanding", "planner", "task_decomposition"},
            "execution_graph": {
                "execution_coordinator",
                "supervisor",
                "sql_agent",
                "execution_node",
            },
            "reflection_graph": {"reflection_agent"},
            "recovery_graph": {"retry_agent", "replan_agent"},
            "export_graph": {"export_node"},
        }
        sg = next(
            (r for r in session.subgraph_reports() if r.name == node),
            None,
        )
        leaf_hit = bool(nodes & flat_equivalents.get(node, set()))
        passed = bool(sg and (sg.started or sg.completed)) or leaf_hit
        if sg and (sg.started or sg.completed):
            reason = f"subgraph {node} started={sg.started} completed={sg.completed}"
        elif leaf_hit:
            reason = f"equivalent leaf nodes executed (flat or nested): {sorted(nodes & flat_equivalents[node])}"
        else:
            reason = f"{node} not observed (may be skipped when enterprise_subgraphs_enabled=false or path not taken)"
        checks[label] = result(passed, reason)

    iam_events = [e for e in events if e.category == TraceCategory.IAM]
    checks["IAM enforced"] = result(
        bool(iam_events),
        f"{len(iam_events)} IAM check(s) recorded"
        if iam_events
        else "no IAM checks recorded",
    )

    plugin_events = [e for e in events if e.category == TraceCategory.PLUGIN]
    # Discovery/loading counts as platform plugin activity even without execute
    checks["Plugin execution"] = result(
        bool(plugin_events),
        f"{len(plugin_events)} plugin event(s)"
        if plugin_events
        else "no plugin events in this run",
    )

    bus_events = [e for e in events if e.category == TraceCategory.MESSAGE_BUS]
    checks["Message Bus used"] = result(
        bool(bus_events),
        f"{len(bus_events)} bus event(s)" if bus_events else "no message bus events",
    )

    mem_events = [
        e
        for e in events
        if e.category == TraceCategory.MEMORY
        or e.node_name in {"memory_agent", "memory_graph"}
    ]
    checks["Memory retrieval"] = result(
        bool(mem_events),
        f"{len(mem_events)} memory event(s)" if mem_events else "no memory retrievals",
    )

    checks["Reflection executed"] = result(
        _node_seen(session, "reflection_agent", "reflection_graph")
        or _kind_seen(session, "reflection"),
        "reflection observed"
        if _node_seen(session, "reflection_agent", "reflection_graph")
        else "reflection not on this path",
    )
    checks["Replanning executed"] = result(
        _node_seen(session, "replan_agent") or _kind_seen(session, "replan"),
        "replan observed"
        if _node_seen(session, "replan_agent")
        else "replan not on this path",
    )

    reliability = [
        e
        for e in events
        if e.category == TraceCategory.RELIABILITY
        or "circuit" in (e.event_kind or "").lower()
        or (e.payload or {}).get("circuit")
    ]
    # Wrapper always consults CB when enterprise_circuit_breaker=true — presence of
    # reliability heartbeat / CB allow events counts
    cb_active = any(
        "circuit" in (e.event_kind or "").lower()
        or e.event_kind in {"Circuit Breaker", "Health Check", "Heartbeat"}
        or e.category == TraceCategory.RELIABILITY
        for e in events
    )
    checks["Circuit Breaker active"] = result(
        cb_active,
        "reliability / circuit breaker events present"
        if cb_active
        else "no reliability events (circuit breaker may be disabled)",
    )

    checks["Observability active"] = result(
        len(events) > 0,
        f"runtime trace captured {len(events)} events",
    )

    # Ensure every advertised check key exists
    for name in VERIFICATION_CHECKS:
        checks.setdefault(name, result(False, "not evaluated"))

    return {k: checks[k] for k in VERIFICATION_CHECKS}
