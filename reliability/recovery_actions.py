"""Recovery action catalog — Recover / Retry / Fallback / Abort / Escalate.

Extends the existing Recovery Graph without replacing retry_agent or DLQ.
Every action is structured, logged to agent_logs + enterprise_events + metrics.
"""

from __future__ import annotations

import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.logging_config import get_logger

logger = get_logger(__name__)

RecoveryAction = Literal["recover", "retry", "fallback", "abort", "escalate"]
RecoveryTrigger = Literal[
    "tool_failure",
    "sql_failure",
    "plugin_failure",
    "timeout",
    "agent_exception",
    "validation_failure",
    "circuit_open",
    "unknown",
]


class RecoveryDecision(BaseModel):
    """Structured recovery outcome for graph state + audit trail."""

    action_id: str = Field(default_factory=lambda: f"rcv-{uuid4().hex[:10]}")
    action: RecoveryAction
    trigger: RecoveryTrigger = "unknown"
    reason: str = ""
    source_node: str = ""
    next_agent: str = "fail"
    fallback_agent: str = "supervisor"
    attempts: int = 0
    max_attempts: int = 3
    logged_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        return {
            "kind": "recovery_action",
            "action_id": self.action_id,
            "action": self.action,
            "trigger": self.trigger,
            "reason": self.reason,
            "source_node": self.source_node,
            "next_agent": self.next_agent,
            "ts": self.logged_at,
            **self.metadata,
        }


def classify_trigger(state: dict[str, Any]) -> RecoveryTrigger:
    """Infer recovery trigger from GraphState failure signals."""
    if state.get("sql_error") or (
        state.get("query_success") is False and state.get("sql")
    ):
        return "sql_failure"
    if state.get("validation_errors"):
        return "validation_failure"
    err = str(state.get("error") or "").lower()
    if "timeout" in err:
        return "timeout"
    if "plugin" in err:
        return "plugin_failure"
    if "tool" in err:
        return "tool_failure"
    if state.get("reliability_meta", {}).get("circuit_open"):
        return "circuit_open"
    if err:
        return "agent_exception"
    return "unknown"


def decide_recovery(state: dict[str, Any]) -> RecoveryDecision:
    """Policy: map failure context → Recover/Retry/Fallback/Abort/Escalate."""
    trigger = classify_trigger(state)
    retries = int(state.get("retry_count") or 0)
    max_retries = int(state.get("max_retries") or 3)
    replan_count = int(
        (state.get("adaptive_plan") or {}).get("replan_count")
        or state.get("runtime_replan_count")
        or 0
    )
    max_replans = int(
        (state.get("adaptive_plan") or {}).get("max_replans")
        or state.get("max_runtime_replans")
        or 2
    )
    source = str(
        (state.get("route_history") or ["unknown"])[-1]
        if state.get("route_history")
        else state.get("next_agent") or "unknown"
    )
    reason = (
        state.get("replan_reason")
        or state.get("sql_error")
        or state.get("error")
        or state.get("retry_diagnosis")
        or f"Recovery triggered ({trigger})"
    )

    # Escalate when human approval may help (sensitive / repeated failures)
    if retries >= max_retries and replan_count >= max_replans:
        return RecoveryDecision(
            action="escalate",
            trigger=trigger,
            reason=str(reason),
            source_node=source,
            next_agent="approval_gate"
            if state.get("approval_request") or True
            else "fail",
            attempts=retries,
            max_attempts=max_retries,
            metadata={"replan_count": replan_count, "escalate": True},
        )

    if trigger == "validation_failure" and retries < max_retries:
        return RecoveryDecision(
            action="retry",
            trigger=trigger,
            reason=str(reason),
            source_node=source,
            next_agent="sql_agent",
            attempts=retries,
            max_attempts=max_retries,
        )

    if trigger in {"sql_failure", "tool_failure", "timeout"} and retries < max_retries:
        return RecoveryDecision(
            action="retry",
            trigger=trigger,
            reason=str(reason),
            source_node=source,
            next_agent="sql_agent",
            attempts=retries,
            max_attempts=max_retries,
        )

    if replan_count < max_replans and (
        (state.get("adaptive_plan") or {}).get("plan_id") or state.get("plan_active")
    ):
        return RecoveryDecision(
            action="recover",
            trigger=trigger,
            reason=str(reason),
            source_node=source,
            next_agent="replan_agent",
            attempts=retries,
            max_attempts=max_retries,
            metadata={"via": "replan"},
        )

    if trigger in {"plugin_failure", "circuit_open", "agent_exception"}:
        return RecoveryDecision(
            action="fallback",
            trigger=trigger,
            reason=str(reason),
            source_node=source,
            next_agent="supervisor",
            fallback_agent="supervisor",
            attempts=retries,
            max_attempts=max_retries,
        )

    return RecoveryDecision(
        action="abort",
        trigger=trigger,
        reason=str(reason),
        source_node=source,
        next_agent="fail",
        attempts=retries,
        max_attempts=max_retries,
    )


def make_recovery_controller():
    """LangGraph node — Recovery Graph policy engine (before/around retry)."""

    def recovery_controller(state: dict[str, Any]) -> dict[str, Any]:
        decision = decide_recovery(state)
        logger.info(
            "Recovery action=%s trigger=%s next=%s",
            decision.action,
            decision.trigger,
            decision.next_agent,
        )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_recovery(decision.action, decision.trigger)
        except Exception:  # noqa: BLE001
            pass
        try:
            from observability.runtime_trace import safe_trace

            safe_trace(
                "reliability",
                event="recovery_action",
                action=decision.action,
                trigger=decision.trigger,
                next_agent=decision.next_agent,
                detail=decision.reason[:300],
            )
        except Exception:  # noqa: BLE001
            pass

        # Map escalate to clarify/approval when gate unavailable
        next_agent = decision.next_agent
        if decision.action == "escalate" and next_agent == "approval_gate":
            # Prefer clarify if no pending approval payload
            if not state.get("approval_request") and not state.get("needs_approval"):
                next_agent = "clarify"

        out: dict[str, Any] = {
            "next_agent": next_agent,
            "recovery_decision": decision.model_dump(mode="json"),
            "retry_next_action": (
                "regenerate_sql"
                if decision.action == "retry"
                else ("replan" if decision.action == "recover" else decision.action)
            ),
            "replan_reason": decision.reason if decision.action == "recover" else state.get("replan_reason"),
            "enterprise_events": [decision.to_event()],
            "agent_logs": [
                {
                    "agent": "recovery_controller",
                    "message": f"action={decision.action} → {next_agent}",
                    "status": "ok" if decision.action != "abort" else "error",
                    "detail": decision.reason,
                }
            ],
            "route_history": ["recovery_controller"],
        }
        if decision.action == "abort":
            out["error"] = decision.reason
            out["status"] = "failed"
            out["final_response"] = (
                f"Recovery aborted after {decision.trigger}: {decision.reason}"
            )
        if decision.action == "escalate":
            out["needs_clarification"] = True
            out["clarification_question"] = (
                "The runtime exhausted automatic recovery. "
                "Please clarify how to proceed, or approve an elevated path."
            )
        return out

    return recovery_controller
