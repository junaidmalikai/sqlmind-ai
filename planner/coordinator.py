"""Execution Coordinator — schedule, dependencies, parallel, progress, failures.

Does not call specialist agents directly. Sets ``next_agent`` / ``Send`` intents
that LangGraph edges honor. SQL security subgraph remains fixed.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from graph.state import GRAPH_NODES, GraphState
from planner.messages import (
    decision_message,
    get_message_bus,
    observation_message,
    result_message,
    task_request_message,
)
from planner.models import (
    AdaptivePlan,
    AutonDecision,
    ExecutionProgress,
    TaskGraph,
    TaskSpec,
)
from planner.registry import AgentRegistry
from planner.selection import bind_task_to_agent, select_agent_for_task
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Artifacts that indicate a task likely completed (heuristic, soft)
_PROVIDES_TO_STATE: dict[str, str] = {
    "schema_text": "schema_text",
    "suggested_questions": "suggested_questions",
    "sql": "sql",
    "sql_explanation": "sql_explanation",
    "insights": "insights",
    "insight_structured": "insight_structured",
    "chart_type": "chart_type",
    "chart_spec": "chart_spec",
    "dashboard_spec": "dashboard_spec",
    "database_summary": "database_summary",
    "optimization_tips": "optimization_tips",
    "export_paths": "export_paths",
    "final_response": "final_response",
    "reflection_verdict": "reflection_verdict",
}


def _log(message: str, status: str = "ok", detail: Any = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "agent": "execution_coordinator",
        "message": message,
        "status": status,
    }
    if detail is not None:
        entry["detail"] = detail
    return entry


def _task_satisfied(task: TaskSpec, state: GraphState) -> bool:
    """Heuristic: task complete if expected artifacts appear in state."""
    provides = task.provides_any or []
    if not provides:
        # If we routed to this node already and returned, mark by route history
        node = task.responsible_graph_node
        if node and node in (state.get("route_history") or []):
            return True
        return False
    for p in provides:
        key = _PROVIDES_TO_STATE.get(p, p)
        val = state.get(key)  # type: ignore[arg-type]
        if val is None or val == "" or val == {} or val == []:
            return False
        if p in {"chart_type"} and val == "none":
            return False
    return True


def _sync_task_statuses(graph: TaskGraph, state: GraphState, progress: ExecutionProgress) -> None:
    for task in graph.tasks:
        if task.status in {"completed", "skipped", "failed"}:
            continue
        if task.id in progress.failed_task_ids:
            task.status = "failed"
            continue
        if _task_satisfied(task, state) or task.id in progress.completed_task_ids:
            task.status = "completed"
            if task.id not in progress.completed_task_ids:
                progress.completed_task_ids.append(task.id)


def _estimate_parallel_group(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Return the largest parallel-safe subset among ready tasks (same dependency wave)."""
    if not tasks:
        return []
    parallel = [t for t in tasks if t.parallel_safe]
    if len(parallel) >= 2:
        return parallel
    return [tasks[0]]


class ExecutionCoordinator:
    """Schedules TaskGraph work against the Agent Registry."""

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self.registry = agent_registry

    def step(self, state: GraphState) -> dict[str, Any]:
        plan = AdaptivePlan.from_state(state.get("adaptive_plan"))  # type: ignore[arg-type]
        graph = TaskGraph.from_state(state.get("task_graph"))  # type: ignore[arg-type]
        progress = ExecutionProgress.from_state(state.get("execution_progress"))  # type: ignore[arg-type]

        # Fast path: no plan / supervisor fallback → existing Supervisor
        if plan is None or plan.use_supervisor_fallback or graph is None or not graph.tasks:
            decision = AutonDecision(
                decision_type="supervisor_handoff",
                chosen="supervisor",
                rationale="No executable plan or explicit supervisor fallback",
                confidence=1.0,
            )
            return {
                "next_agent": "supervisor",
                "plan_active": False,
                "execution_progress": progress.to_state_dict(),
                "auton_decisions": [decision.model_dump(mode="json")],
                "agent_messages": [
                    decision_message("execution_coordinator", decision.model_dump()).to_state_dict()
                ],
                "agent_logs": [_log("Handoff → supervisor (fast/fallback path)")],
                "route_history": ["supervisor"],
            }

        _sync_task_statuses(graph, state, progress)

        # Hard failure: query failed and SQL task was active → replan
        if state.get("sql_error") and not state.get("query_success"):
            sql_tasks = [
                t
                for t in graph.tasks
                if t.status == "running"
                or (t.responsible_graph_node == "sql_agent" and t.status not in {"completed", "skipped"})
            ]
            if sql_tasks and int(state.get("retry_count") or 0) >= int(state.get("max_retries") or 3):
                return self._to_replan(graph, plan, progress, "SQL retries exhausted")

        if graph.all_done():
            failed = [t.id for t in graph.tasks if t.status == "failed"]
            if failed and plan.replan_count < plan.max_replans:
                return self._to_replan(graph, plan, progress, f"Tasks failed: {failed}")
            # Prefer reflection when we have query results; else finalize
            next_node = "reflection_agent" if state.get("query_success") else "finalize"
            if next_node not in GRAPH_NODES:
                next_node = "finalize"
            plan.status = "completed"
            decision = AutonDecision(
                decision_type="finalize" if next_node == "finalize" else "reflect",
                chosen=next_node,
                rationale="All plan tasks finished",
                confidence=0.9,
            )
            progress.timeline.append({"event": "plan_complete", "next": next_node})
            return {
                "next_agent": next_node,
                "plan_active": False,
                "adaptive_plan": plan.to_state_dict(),
                "task_graph": graph.to_state_dict(),
                "execution_progress": progress.to_state_dict(),
                "auton_decisions": [decision.model_dump(mode="json")],
                "agent_messages": [
                    result_message(
                        "execution_coordinator",
                        {"status": "plan_complete", "next": next_node},
                    ).to_state_dict()
                ],
                "agent_logs": [_log(f"Plan complete → {next_node}")],
                "route_history": [next_node],
            }

        ready = graph.ready_tasks()
        if not ready:
            # Deadlock / waiting — hand to supervisor for soft recovery
            decision = AutonDecision(
                decision_type="supervisor_handoff",
                chosen="supervisor",
                rationale="No ready tasks (dependency deadlock or empty wave)",
                confidence=0.5,
            )
            return {
                "next_agent": "supervisor",
                "plan_active": True,
                "task_graph": graph.to_state_dict(),
                "execution_progress": progress.to_state_dict(),
                "auton_decisions": [decision.model_dump(mode="json")],
                "agent_logs": [_log("No ready tasks → supervisor", "warn")],
                "route_history": ["supervisor"],
            }

        # Bind agents dynamically from registry
        bound: list[TaskSpec] = []
        decisions: list[AutonDecision] = []
        for task in ready:
            match, decision = select_agent_for_task(self.registry, task)
            decisions.append(decision)
            if match is None or not match.descriptor.graph_node:
                task.status = "failed"
                progress.failed_task_ids.append(task.id)
                progress.timeline.append(
                    {"event": "selection_failed", "task_id": task.id}
                )
                continue
            updated = bind_task_to_agent(task, match)
            # Persist binding onto graph
            for i, t in enumerate(graph.tasks):
                if t.id == task.id:
                    graph.tasks[i] = updated
                    bound.append(updated)
                    break

        if not bound:
            if plan.replan_count < plan.max_replans:
                return self._to_replan(graph, plan, progress, "Agent selection failed for ready tasks")
            return {
                "next_agent": "fail",
                "error": "Execution coordinator could not select agents for ready tasks",
                "plan_active": False,
                "task_graph": graph.to_state_dict(),
                "execution_progress": progress.to_state_dict(),
                "agent_logs": [_log("Selection failed → fail", "error")],
                "route_history": ["fail"],
            }

        group = _estimate_parallel_group(bound)
        plan.status = "executing"
        progress.decisions.extend(decisions)

        # Parallel fan-out via Send when ≥2 parallel-safe and not security-gated SQL
        if (
            len(group) >= 2
            and all(t.parallel_safe for t in group)
            and all(t.responsible_graph_node not in {"sql_agent"} for t in group)
        ):
            decision = AutonDecision(
                decision_type="parallel",
                chosen=",".join(t.responsible_graph_node or "" for t in group),
                rationale="Parallel-safe ready wave",
                confidence=0.85,
            )
            progress.decisions.append(decision)
            for t in group:
                t.status = "running"
                progress.active_task_ids.append(t.id)
            progress.timeline.append(
                {
                    "event": "parallel_dispatch",
                    "tasks": [t.id for t in group],
                    "nodes": [t.responsible_graph_node for t in group],
                }
            )
            # LangGraph conditional edge emits Send list via route_coordinator_dispatch
            return {
                "next_agent": "parallel_dispatch",
                "parallel_sends": [
                    {"node": t.responsible_graph_node, "task_id": t.id} for t in group
                ],
                "plan_active": True,
                "adaptive_plan": plan.to_state_dict(),
                "task_graph": graph.to_state_dict(),
                "execution_progress": progress.to_state_dict(),
                "auton_decisions": [d.model_dump(mode="json") for d in decisions]
                + [decision.model_dump(mode="json")],
                "agent_messages": [
                    observation_message(
                        "execution_coordinator",
                        {"parallel": [t.id for t in group]},
                    ).to_state_dict()
                ],
                "agent_logs": [
                    _log(
                        f"Parallel dispatch → {[t.responsible_graph_node for t in group]}",
                        detail=[t.id for t in group],
                    )
                ],
                "route_history": [t.responsible_graph_node or "" for t in group],
                "_send_targets": [t.responsible_graph_node or "" for t in group],
            }

        # Sequential: pick highest-priority (first ready after selection)
        chosen = group[0]
        chosen.status = "running"
        for i, t in enumerate(graph.tasks):
            if t.id == chosen.id:
                graph.tasks[i] = chosen
                break
        node = chosen.responsible_graph_node or "supervisor"
        if node not in GRAPH_NODES:
            node = "supervisor"
        progress.active_task_ids = [chosen.id]
        progress.timeline.append(
            {"event": "dispatch", "task_id": chosen.id, "node": node}
        )
        plugin_cap = ""
        if node == "plugin_runtime_agent":
            plugin_cap = str(chosen.responsible_agent_id or "")
        # Live P2P: coordinator → specialist task_request on the message bus
        bus_payload: dict[str, Any] = {
            "task_id": chosen.id,
            "node": node,
            "description": chosen.description,
        }
        if plugin_cap:
            bus_payload["plugin_capability_id"] = plugin_cap
            bus_payload["capability_id"] = plugin_cap
        bus_msg = task_request_message(
            "execution_coordinator",
            bus_payload,
            recipient=node,
        )
        try:
            get_message_bus().publish(bus_msg)
            get_message_bus().heartbeat("execution_coordinator", active_task=chosen.id)
        except Exception:  # noqa: BLE001
            pass
        out: dict[str, Any] = {
            "next_agent": node,
            "active_task_id": chosen.id,
            "plan_active": True,
            "adaptive_plan": plan.to_state_dict(),
            "task_graph": graph.to_state_dict(),
            "execution_progress": progress.to_state_dict(),
            "auton_decisions": [d.model_dump(mode="json") for d in decisions],
            "agent_messages": [
                observation_message(
                    "execution_coordinator",
                    {"dispatch": chosen.id, "node": node},
                ).to_state_dict(),
                bus_msg.to_state_dict(),
            ],
            "agent_logs": [_log(f"Dispatch {chosen.id} → {node}")],
            "route_history": [node],
        }
        if plugin_cap:
            out["plugin_capability_id"] = plugin_cap
            out["selected_plugin"] = plugin_cap
        return out

    def _to_replan(
        self,
        graph: TaskGraph,
        plan: AdaptivePlan,
        progress: ExecutionProgress,
        reason: str,
    ) -> dict[str, Any]:
        plan.status = "replanning"
        plan.replan_count += 1
        decision = AutonDecision(
            decision_type="replan",
            chosen="replan_agent",
            rationale=reason,
            confidence=0.7,
        )
        progress.timeline.append({"event": "replan", "reason": reason})
        progress.decisions.append(decision)
        return {
            "next_agent": "replan_agent",
            "plan_active": True,
            "adaptive_plan": plan.to_state_dict(),
            "task_graph": graph.to_state_dict(),
            "execution_progress": progress.to_state_dict(),
            "replan_reason": reason,
            "auton_decisions": [decision.model_dump(mode="json")],
            "agent_messages": [
                decision_message("execution_coordinator", decision.model_dump()).to_state_dict()
            ],
            "agent_logs": [_log(f"Replan triggered: {reason}", "warn")],
            "route_history": ["replan_agent"],
        }


def make_execution_coordinator(agent_registry: AgentRegistry):
    """LangGraph node factory."""
    coordinator = ExecutionCoordinator(agent_registry)

    def execution_coordinator(state: GraphState) -> dict[str, Any]:
        try:
            return coordinator.step(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Execution coordinator failed")
            return {
                "next_agent": "supervisor",
                "plan_active": False,
                "error": f"Coordinator error (falling back to supervisor): {exc}",
                "agent_logs": [_log(str(exc), "error")],
                "route_history": ["supervisor"],
            }

    return execution_coordinator


def route_coordinator_dispatch(state: GraphState) -> Any:
    """Conditional edge after coordinator — may emit parallel Send list."""
    if state.get("next_agent") == "parallel_dispatch":
        targets = state.get("_send_targets") or []
        if isinstance(targets, list) and len(targets) >= 2:
            payload_base = dict(state)
            payload_base["parallel_job"] = True
            payload_base["next_agent"] = "join_post_query"
            try:
                from observability.parallel_metrics import record_parallel_send

                record_parallel_send(
                    [str(n) for n in targets],
                    source="execution_coordinator",
                    session_id=str(state.get("session_id") or ""),
                )
            except Exception:  # noqa: BLE001
                pass
            return [
                Send(str(node), {**payload_base, "active_task_hint": str(node)})
                for node in targets
                if str(node) in GRAPH_NODES
            ]
        return "supervisor"
    nxt = state.get("next_agent") or "supervisor"
    if nxt == "export_agent":
        nxt = "export_node"
    return nxt if nxt in GRAPH_NODES else "supervisor"
