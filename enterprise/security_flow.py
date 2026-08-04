"""Security flow helpers — session extraction, permission asserts, audit."""

from __future__ import annotations

from typing import Any, Literal

from iam import IAMService, SessionToken
from utils.logging_config import get_logger

logger = get_logger(__name__)

ActionKind = Literal[
    "agent_invoke",
    "tool_call",
    "sql_execute",
    "memory_access",
    "plugin_execute",
    "export",
]


# Map graph nodes → IAM resource identifiers
NODE_TO_RESOURCE: dict[str, str] = {
    "supervisor": "agent.supervisor",
    "schema_agent": "agent.schema",
    "sql_agent": "agent.sql",
    "validation_node": "sql.read",
    "execution_node": "sql.execute",
    "visualization_agent": "agent.visualization",
    "insight_agent": "agent.insight",
    "optimization_agent": "agent.optimization",
    "dashboard_agent": "agent.dashboard",
    "summary_agent": "agent.summary",
    "export_node": "export.*",
    "reflection_agent": "agent.reflection",
    "retry_agent": "agent.retry",
    "clarify": "agent.clarify",
    "finalize": "agent.finalize",
    "fail": "agent.fail",
    "join_post_query": "agent.supervisor",
    "memory_agent": "memory.read",
    "goal_understanding": "agent.planner",
    "planner": "agent.planner",
    "task_decomposition": "agent.planner",
    "execution_coordinator": "agent.coordinator",
    "replan_agent": "agent.planner",
    "goal_tracking": "agent.planner",
    "approval_gate": "approval.decide",
    "memory_graph": "memory.read",
    "planning_graph": "agent.planner",
    "execution_graph": "agent.coordinator",
    "analytics_graph": "agent.*",
    "export_graph": "export.*",
    "reflection_graph": "agent.reflection",
    "recovery_graph": "agent.retry",
}

ACTION_TO_RESOURCE: dict[ActionKind, str] = {
    "agent_invoke": "agent.*",
    "tool_call": "tool.*",
    "sql_execute": "sql.execute",
    "memory_access": "memory.read",
    "plugin_execute": "plugin.execute",
    "export": "export.*",
}


def session_from_state(state: dict[str, Any]) -> SessionToken | None:
    """Rehydrate SessionToken from GraphState ``iam_session``."""
    raw = state.get("iam_session") or {}
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return SessionToken.model_validate(raw)
    except Exception:  # noqa: BLE001
        return None


def resolve_security_context(state: dict[str, Any]) -> dict[str, Any]:
    """Build a serializable security context snapshot for audit / state."""
    session = session_from_state(state)
    if session is None:
        return {
            "authenticated": False,
            "principal_id": "",
            "tenant_id": state.get("tenant_id") or "default",
            "workspace_id": state.get("workspace_id") or "default",
            "roles": [],
        }
    return {
        "authenticated": True,
        "principal_id": session.principal_id,
        "tenant_id": session.tenant_id,
        "workspace_id": session.workspace_id,
        "roles": list(session.roles),
        "session_id": session.session_id,
        "api_key_id": session.api_key_id,
    }


def resource_for_node(node_name: str) -> str:
    return NODE_TO_RESOURCE.get(node_name, f"agent.{node_name}")


def assert_action_permission(
    iam: IAMService,
    session: SessionToken | None,
    action: ActionKind,
    *,
    resource: str | None = None,
    attributes: dict[str, Any] | None = None,
    on_deny: Literal["reject", "approval"] = "reject",
) -> dict[str, Any]:
    """Enforce permission for a high-level action.

    Returns empty dict on allow. On deny returns a GraphState patch that either
    rejects (fail) or requests approval — never silently bypasses.
    """
    res = resource or ACTION_TO_RESOURCE[action]

    def _trace_iam(
        *,
        allowed: bool,
        approval_required: bool = False,
        detail: Any = None,
    ) -> None:
        try:
            from observability.runtime_trace import safe_trace

            safe_trace(
                "iam_check",
                action=action,
                resource=res,
                allowed=allowed,
                decision="allowed" if allowed else ("approval_required" if approval_required else "denied"),
                approval_required=approval_required,
                auth_type="Authorization",
                detail=detail,
            )
        except Exception:  # noqa: BLE001
            pass

    if session is None:
        iam.audit(
            "anonymous",
            "default",
            "authz.runtime",
            resource=res,
            decision="deny",
            detail={"reason": "unauthenticated", "action": action},
        )
        if on_deny == "approval":
            _trace_iam(
                allowed=False,
                approval_required=True,
                detail="unauthenticated",
            )
            return {
                "needs_approval": True,
                "next_agent": "approval_gate",
                "approval_request": {
                    "reason": "policy",
                    "risk": "high",
                    "title": f"Unauthenticated {action}",
                    "detail": f"Authentication required for {res}",
                    "action": action,
                    "status": "pending",
                },
            }
        _trace_iam(allowed=False, detail="unauthenticated")
        return {
            "error": f"Unauthenticated: denied {res}",
            "status": "failed",
            "next_agent": "fail",
            "agent_logs": [
                {
                    "agent": "iam",
                    "message": f"deny unauthenticated {action} → {res}",
                    "status": "error",
                }
            ],
        }

    attrs = {
        "resource_tenant_id": session.tenant_id,
        **(attributes or {}),
    }
    try:
        iam.assert_permission(session, res, attributes=attrs)
        _trace_iam(allowed=True, detail={"principal": session.principal_id, "rbac": True, "abac": True})
        return {}
    except PermissionError as exc:
        logger.warning("IAM deny action=%s resource=%s: %s", action, res, exc)
        if on_deny == "approval":
            _trace_iam(allowed=False, approval_required=True, detail=str(exc))
            return {
                "needs_approval": True,
                "next_agent": "approval_gate",
                "approval_request": {
                    "reason": "policy",
                    "risk": "high",
                    "title": f"Authorize {action}",
                    "detail": str(exc),
                    "action": action,
                    "actor": session.principal_id,
                    "tenant_id": session.tenant_id,
                    "status": "pending",
                },
                "approval_resume_agent": "execution_coordinator",
            }
        _trace_iam(allowed=False, detail=str(exc))
        return {
            "error": str(exc),
            "status": "failed",
            "next_agent": "fail",
            "agent_logs": [
                {
                    "agent": "iam",
                    "message": f"deny {action} → {res}",
                    "status": "error",
                }
            ],
        }
