"""Goal Tracking Agent — monitors progress, blocked/failed goals, notifies Planner."""

from __future__ import annotations

from typing import Any

from graph.state import GraphState
from planner.goal_models import (
    GoalObjectiveProgress,
    GoalStatusUpdate,
    GoalTrackingRecord,
)
from planner.goal_store import GoalStore
from planner.messages import status_update_message
from planner.models import ExecutionProgress, GoalSpec, TaskGraph
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _log(agent: str, message: str, status: str = "ok", detail: Any = None) -> dict:
    entry: dict[str, Any] = {"agent": agent, "message": message, "status": status}
    if detail is not None:
        entry["detail"] = detail
    return entry


def _goal_spec(state: GraphState) -> GoalSpec | None:
    raw = state.get("goal_spec")
    if isinstance(raw, dict) and raw:
        try:
            return GoalSpec.model_validate(raw)
        except Exception:  # noqa: BLE001
            return None
    return None


def _from_state_or_create(state: GraphState) -> GoalTrackingRecord:
    existing = GoalTrackingRecord.from_state(state.get("goal_tracking"))
    if existing is not None:
        return existing

    spec = _goal_spec(state)
    objectives = []
    if spec:
        objectives = [
            GoalObjectiveProgress(objective=o) for o in (spec.objectives or [])
        ]
        if not objectives and spec.success_criteria:
            objectives = [
                GoalObjectiveProgress(objective=c) for c in spec.success_criteria
            ]

    title = (spec.goal if spec else None) or state.get("question") or "Untitled goal"
    record = GoalTrackingRecord(
        session_id=state.get("session_id") or "",
        title=title,
        description=title,
        status="created",
        objectives=objectives,
        total_objectives=len(objectives),
        plan_id=(state.get("adaptive_plan") or {}).get("plan_id") or "",
        metadata={"intent": state.get("intent") or ""},
    )
    record.estimate_completion()
    return record


def _sync_from_progress(record: GoalTrackingRecord, state: GraphState) -> None:
    """Update objectives / blocked / failed from TaskGraph + ExecutionProgress."""
    progress = ExecutionProgress.from_state(state.get("execution_progress"))
    graph = TaskGraph.from_state(state.get("task_graph"))

    if graph and graph.tasks:
        # Map completed tasks onto objectives by index when counts align
        completed = set(progress.completed_task_ids)
        failed = set(progress.failed_task_ids)
        if record.objectives and len(record.objectives) == len(graph.tasks):
            for obj, task in zip(record.objectives, graph.tasks):
                if task.id in completed or task.status == "completed":
                    obj.status = "completed"
                elif task.id in failed or task.status == "failed":
                    obj.status = "failed"
                elif task.status == "running":
                    obj.status = "in_progress"
                elif task.status == "skipped":
                    obj.status = "completed"
        record.related_task_ids = [t.id for t in graph.tasks]

    if progress.failed_task_ids and record.status == "running":
        record.blocked = True
        record.block_reason = f"Failed tasks: {', '.join(progress.failed_task_ids[:5])}"

    # Terminal signals from state
    if state.get("status") == "failed" or state.get("error"):
        if record.status not in {"completed", "cancelled", "failed"}:
            try:
                record.transition("failed", reason=str(state.get("error") or "run failed"))
            except ValueError:
                record.status = "failed"
                record.failure_reason = str(state.get("error") or "run failed")
    elif state.get("final_response") and not state.get("error"):
        if record.status not in {"completed", "cancelled", "failed"}:
            # Mark remaining objectives complete when we finalize successfully
            for obj in record.objectives:
                if obj.status not in {"failed", "blocked"}:
                    obj.status = "completed"
            try:
                record.transition("completed")
            except ValueError:
                record.status = "completed"
                record.completion_pct = 1.0

    record.estimate_completion()


def make_goal_tracking_agent(goal_store: GoalStore | None = None):
    """Factory for the Goal Tracking Agent node.

    Responsibilities:
    - Track execution progress
    - Monitor completed objectives
    - Detect blocked / failed goals
    - Estimate completion
    - Update Goal Status
    - Notify Planner when replanning is warranted
    """

    def goal_tracking_agent(state: GraphState) -> dict[str, Any]:
        record = _from_state_or_create(state)
        previous = record.status
        notify_planner = False
        notify_reason = ""

        # Lifecycle advances based on where we are in the graph
        next_hint = state.get("next_agent") or ""
        plan_active = bool(state.get("plan_active"))
        has_plan = bool((state.get("adaptive_plan") or {}).get("plan_id"))
        has_goal = bool(state.get("goal_spec"))

        try:
            if record.status == "created" and has_goal:
                record.transition("planning")
            if record.status == "planning" and has_plan and (
                plan_active or next_hint in {"execution_coordinator", "supervisor", "sql_agent"}
            ):
                record.transition("running")
            if record.status == "running" and (
                state.get("needs_clarification")
                or state.get("needs_approval")
                or next_hint in {"clarify", "approval_gate"}
            ):
                reason = "Awaiting human clarification or approval"
                record.transition("waiting", reason=reason)
                notify_planner = True
                notify_reason = reason
            if record.status == "waiting" and not state.get("needs_clarification") and not state.get(
                "needs_approval"
            ):
                if has_plan:
                    record.transition("running")
                else:
                    record.transition("planning")
        except ValueError as exc:
            logger.debug("Goal transition skipped: %s", exc)

        _sync_from_progress(record, state)

        # Detect blocked goals mid-run
        if record.blocked and record.status == "running":
            try:
                record.transition("waiting", reason=record.block_reason or "blocked")
                notify_planner = True
                notify_reason = record.block_reason or "Goal blocked"
            except ValueError:
                pass

        # Failed task graph → notify planner to replan
        progress = ExecutionProgress.from_state(state.get("execution_progress"))
        if progress.failed_task_ids and record.status in {"running", "waiting"}:
            notify_planner = True
            notify_reason = notify_reason or (
                f"Task failures: {', '.join(progress.failed_task_ids[:5])}"
            )

        record.notify_planner = notify_planner
        record.notify_reason = notify_reason
        record.estimate_completion()

        if goal_store is not None:
            try:
                goal_store.upsert(record)
                if previous != record.status:
                    goal_store.record_event(
                        record.goal_id,
                        previous_status=previous,
                        new_status=record.status,
                        reason=notify_reason or "",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Goal store persist failed: %s", exc)

        update = GoalStatusUpdate(
            goal_id=record.goal_id,
            previous_status=previous,  # type: ignore[arg-type]
            new_status=record.status,  # type: ignore[arg-type]
            completion_pct=record.completion_pct,
            blocked=record.blocked,
            notify_planner=notify_planner,
            reason=notify_reason,
        )

        out: dict[str, Any] = {
            "goal_tracking": record.to_state_dict(),
            "goal_status_update": update.to_state_dict(),
            "agent_messages": [
                status_update_message(
                    "goal_tracking",
                    update.to_state_dict(),
                    recipient="planner" if notify_planner else "observability",
                ).to_state_dict()
            ],
            "agent_logs": [
                _log(
                    "goal_tracking",
                    f"{record.goal_id} {previous}→{record.status} "
                    f"pct={record.completion_pct:.0%} notify={notify_planner}",
                    detail=update.to_state_dict(),
                )
            ],
            "route_history": ["goal_tracking"],
        }

        # Default routing: continue to planner after create, else preserve next_agent
        if not state.get("next_agent") or state.get("next_agent") == "goal_tracking":
            if record.status in {"created", "planning"} and not has_plan:
                out["next_agent"] = "planner"
            elif notify_planner and record.status in {"waiting", "failed"}:
                out["next_agent"] = "replan_agent"
            else:
                out["next_agent"] = state.get("next_agent") or "execution_coordinator"
        return out

    return goal_tracking_agent
