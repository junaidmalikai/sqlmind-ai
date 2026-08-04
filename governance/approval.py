"""Human approval workflow — approval gates beyond clarification HITL.

Planner pauses on high-risk actions (DELETE/DROP/ALTER patterns, sensitive export,
DB connection changes, long-running tasks) and resumes after human approval.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from graph.state import GraphState
from planner.messages import approval_request_message
from utils.logging_config import get_logger

logger = get_logger(__name__)

ApprovalDecision = Literal["pending", "approved", "denied", "expired"]
ApprovalRisk = Literal["low", "medium", "high", "critical"]

# Patterns that always require approval when write/DDL is attempted
_HIGH_RISK_SQL = re.compile(
    r"\b(DELETE|DROP|ALTER|TRUNCATE|UPDATE\s+\w+\s+SET|INSERT\s+INTO|CREATE\s+|GRANT|REVOKE|REPLACE\s+INTO)\b",
    re.IGNORECASE,
)

ApprovalReason = Literal[
    "high_risk_sql",
    "delete",
    "drop",
    "alter",
    "export_sensitive",
    "database_connection",
    "long_running_task",
    "policy",
    "manual",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _approval_id() -> str:
    return f"appr-{uuid4().hex[:12]}"


class ApprovalRequest(BaseModel):
    """Structured approval gate request."""

    approval_id: str = Field(default_factory=_approval_id)
    reason: ApprovalReason
    risk: ApprovalRisk = "high"
    title: str
    detail: str = ""
    sql: str = ""
    action: str = ""
    actor: str = ""
    tenant_id: str = "default"
    session_id: str = ""
    policy_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: ApprovalDecision = "pending"
    decided_by: str = ""
    decision_note: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    decided_at: datetime | None = None

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state(cls, data: dict[str, Any] | None) -> ApprovalRequest | None:
        if not data:
            return None
        return cls.model_validate(data)


class ApprovalPolicy(BaseModel):
    """Declarative rules that trigger approval gates."""

    require_write_sql_approval: bool = True
    require_sensitive_export_approval: bool = True
    require_connection_approval: bool = True
    long_running_seconds: float = 120.0
    sensitive_export_formats: list[str] = Field(
        default_factory=lambda: ["pdf", "xlsx", "csv"]
    )
    always_approve_roles: list[str] = Field(default_factory=list)
    deny_roles_for_export: list[str] = Field(default_factory=list)


def classify_sql_risk(sql: str) -> tuple[ApprovalReason | None, ApprovalRisk]:
    if not sql or not sql.strip():
        return None, "low"
    upper = sql.upper()
    if re.search(r"\bDROP\b", upper):
        return "drop", "critical"
    if re.search(r"\bDELETE\b", upper):
        return "delete", "critical"
    if re.search(r"\bALTER\b", upper):
        return "alter", "critical"
    if _HIGH_RISK_SQL.search(sql):
        return "high_risk_sql", "high"
    return None, "low"


def evaluate_approval_need(
    state: GraphState,
    policy: ApprovalPolicy | None = None,
) -> ApprovalRequest | None:
    """Return an ApprovalRequest if the current state requires a human gate."""
    policy = policy or ApprovalPolicy()
    sql = state.get("sql") or ""
    session_id = state.get("session_id") or ""

    if policy.require_write_sql_approval:
        reason, risk = classify_sql_risk(sql)
        if reason is not None:
            return ApprovalRequest(
                reason=reason,
                risk=risk,
                title=f"Approve {reason.replace('_', ' ')} SQL",
                detail="High-risk SQL detected — human approval required before proceed.",
                sql=sql[:2000],
                action="execute_sql",
                session_id=session_id,
            )

    # Sensitive export
    if policy.require_sensitive_export_approval:
        intent = (state.get("intent") or "").lower()
        export_paths = state.get("export_paths") or {}
        if intent == "export" or state.get("next_agent") == "export_node":
            if not export_paths:  # about to export
                return ApprovalRequest(
                    reason="export_sensitive",
                    risk="high",
                    title="Approve sensitive data export",
                    detail="Export of query results may contain sensitive data.",
                    action="export",
                    session_id=session_id,
                    metadata={"formats": policy.sensitive_export_formats},
                )

    # Long-running signal
    exec_time = float(state.get("execution_time") or 0)
    if exec_time >= policy.long_running_seconds and state.get("needs_approval") is not False:
        # Only raise if explicitly flagged for continuation approval
        if state.get("request_long_running_approval"):
            return ApprovalRequest(
                reason="long_running_task",
                risk="medium",
                title="Approve continuation of long-running task",
                detail=f"Task already ran for {exec_time:.1f}s.",
                action="continue",
                session_id=session_id,
            )

    if state.get("request_connection_approval") and policy.require_connection_approval:
        return ApprovalRequest(
            reason="database_connection",
            risk="high",
            title="Approve database connection change",
            detail=str(state.get("connection_approval_detail") or ""),
            action="connect",
            session_id=session_id,
        )

    return None


def make_approval_gate_node(policy: ApprovalPolicy | None = None):
    """LangGraph HITL approval gate using interrupt() — pause / resume."""

    policy = policy or ApprovalPolicy()

    def approval_gate(state: GraphState) -> dict[str, Any]:
        from langgraph.types import interrupt

        existing = ApprovalRequest.from_state(state.get("approval_request"))
        request = existing or evaluate_approval_need(state, policy)
        if request is None:
            return {
                "needs_approval": False,
                "approval_decision": "approved",
                "next_agent": state.get("approval_resume_agent") or "execution_coordinator",
                "agent_logs": [
                    {
                        "agent": "approval_gate",
                        "message": "No approval required",
                        "status": "ok",
                    }
                ],
                "route_history": ["approval_gate"],
            }

        # Pause — resume value should be dict {decision, note, decided_by} or "approved"/"denied"
        payload = {
            "type": "approval",
            "approval_id": request.approval_id,
            "title": request.title,
            "detail": request.detail,
            "reason": request.reason,
            "risk": request.risk,
            "sql": request.sql[:500] if request.sql else "",
            "action": request.action,
        }
        user_reply = interrupt(payload)

        decision: ApprovalDecision = "denied"
        note = ""
        decided_by = "user"
        if isinstance(user_reply, dict):
            raw = str(user_reply.get("decision") or user_reply.get("value") or "").lower()
            note = str(user_reply.get("note") or "")
            decided_by = str(user_reply.get("decided_by") or "user")
        else:
            raw = str(user_reply).lower().strip()

        if raw in {"approved", "approve", "yes", "y", "true", "1"}:
            decision = "approved"
        elif raw in {"denied", "deny", "no", "n", "false", "0"}:
            decision = "denied"
        else:
            # Treat free-text non-empty as approved clarification-style only if starts with approve
            decision = "approved" if raw.startswith("approve") else "denied"

        request.status = decision
        request.decided_by = decided_by
        request.decision_note = note
        request.decided_at = _utc_now()

        out: dict[str, Any] = {
            "approval_request": request.to_state_dict(),
            "approval_decision": decision,
            "needs_approval": False,
            "agent_messages": [
                approval_request_message(
                    "approval_gate",
                    {**request.to_state_dict(), "decision": decision},
                ).to_state_dict()
            ],
            "agent_logs": [
                {
                    "agent": "approval_gate",
                    "message": f"{request.approval_id} → {decision}",
                    "status": "ok" if decision == "approved" else "warn",
                    "detail": note,
                }
            ],
            "route_history": ["approval_gate"],
        }

        if decision == "approved":
            out["next_agent"] = state.get("approval_resume_agent") or "execution_node"
        else:
            out["next_agent"] = "fail"
            out["error"] = f"Approval denied: {request.title}"
            out["status"] = "failed"
        return out

    return approval_gate


def maybe_require_approval(state: GraphState, policy: ApprovalPolicy | None = None) -> dict[str, Any]:
    """Helper for validation/export nodes — sets needs_approval + next_agent."""
    request = evaluate_approval_need(state, policy)
    if request is None:
        return {}
    return {
        "needs_approval": True,
        "approval_request": request.to_state_dict(),
        "approval_resume_agent": state.get("next_agent") or "execution_node",
        "next_agent": "approval_gate",
        "agent_messages": [
            approval_request_message("policy", request.to_state_dict()).to_state_dict()
        ],
    }
