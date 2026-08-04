"""Enterprise node wrapper — IAM + circuit breaker + DLQ + bus + metrics + runtime trace."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from enterprise.security_flow import (
    assert_action_permission,
    resource_for_node,
    resolve_security_context,
    session_from_state,
)
from iam import IAMService, SessionToken
from planner.messages import (
    AgentMessageBus,
    get_message_bus,
    status_update_message,
    task_result_message,
)
from reliability import RecoveryManager, get_breakers, get_dlq, get_health_monitor
from utils.logging_config import get_logger

logger = get_logger(__name__)

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

# Re-export for callers
NODE_RESOURCE_MAP = {
    # populated via security_flow.NODE_TO_RESOURCE — keep alias
}


@dataclass
class SecurityContext:
    """Runtime security context bound to a graph execution."""

    session: SessionToken | None = None
    tenant_id: str = "default"
    workspace_id: str = "default"
    actor: str = "anonymous"
    authenticated: bool = False
    roles: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_state_dict(self) -> dict[str, Any]:
        if self.session is None:
            return {
                "authenticated": False,
                "tenant_id": self.tenant_id,
                "workspace_id": self.workspace_id,
                "actor": self.actor,
                "roles": self.roles,
            }
        return {
            **self.session.model_dump(mode="json"),
            "authenticated": True,
            "actor": self.actor,
        }

    @classmethod
    def from_session(
        cls,
        session: SessionToken,
        *,
        actor: str = "",
    ) -> SecurityContext:
        return cls(
            session=session,
            tenant_id=session.tenant_id,
            workspace_id=session.workspace_id,
            actor=actor or session.principal_id,
            authenticated=True,
            roles=list(session.roles),
        )


@dataclass
class EnterpriseRuntime:
    """Shared enterprise services injected into every wrapped node."""

    iam: IAMService
    recovery: RecoveryManager
    message_bus: AgentMessageBus | None = None
    enforce_iam: bool = True
    enforce_circuit_breaker: bool = True
    publish_bus: bool = True
    node_timeout_seconds: float = 0.0  # 0 = disabled soft timeout
    approval_on_deny: bool = False

    def __post_init__(self) -> None:
        if self.message_bus is None:
            self.message_bus = get_message_bus()

    @property
    def breakers(self):
        return self.recovery.breakers

    @property
    def dlq(self):
        return self.recovery.dlq

    @property
    def health(self):
        return get_health_monitor()


def _trace(method: str, **kwargs: Any) -> None:
    """Best-effort runtime trace — never affects business logic."""
    try:
        from observability.runtime_trace import safe_trace

        safe_trace(method, **kwargs)
    except Exception:  # noqa: BLE001
        pass


def _drain_inbox(bus: AgentMessageBus, node_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Merge actionable bus inbox messages into a shallow state copy for the node."""
    try:
        messages = bus.drain(node_name)
    except Exception:  # noqa: BLE001
        return state
    if not messages:
        return state
    enriched = dict(state)
    inbox = [m.to_state_dict() if hasattr(m, "to_state_dict") else m for m in messages]
    enriched["inbox_messages"] = list(state.get("inbox_messages") or []) + inbox
    # Apply high-priority task_request / command fields
    for m in messages:
        kind = getattr(m, "kind", None) or (m.get("kind") if isinstance(m, dict) else None)
        payload = getattr(m, "payload", None) or (m.get("payload") if isinstance(m, dict) else {}) or {}
        if kind in {"task_request", "command"} and isinstance(payload, dict):
            if payload.get("plugin_capability_id") or payload.get("capability_id"):
                enriched["plugin_capability_id"] = (
                    payload.get("plugin_capability_id")
                    or payload.get("capability_id")
                )
            if payload.get("task_id"):
                enriched["active_task_id"] = payload["task_id"]
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_queue("bus_deliver", kind or "message")
        except Exception:  # noqa: BLE001
            pass
        _trace(
            "message_bus",
            bus_event="Deliver",
            sender=getattr(m, "sender", "") or "",
            receiver=node_name,
            msg_type=str(kind or ""),
            payload_summary={"keys": list(payload.keys())[:8]} if isinstance(payload, dict) else None,
            direction="received",
        )
    return enriched


def _ack_inbox(bus: AgentMessageBus, node_name: str, *, ok: bool, detail: Any = None) -> None:
    try:
        bus.publish(
            status_update_message(
                node_name,
                {"ack": ok, "status": "completed" if ok else "failed", "detail": detail},
            )
        )
        from observability.metrics import get_metrics

        get_metrics().observe_queue("bus_ack" if ok else "bus_nack", node_name)
    except Exception:  # noqa: BLE001
        pass


def wrap_enterprise_node(
    node_name: str,
    fn: NodeFn,
    runtime: EnterpriseRuntime,
    *,
    action: str = "agent_invoke",
    skip_iam: bool = False,
) -> NodeFn:
    """Wrap a LangGraph node with IAM, circuit breaker, DLQ, bus, and metrics.

    Fully wired into the execution path — not a stub. Failures are audited,
    enqueued to DLQ, and circuit-breaker state is updated.
    Runtime execution tracing is additive only.
    """

    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        from observability.otel_setup import start_span

        started = time.perf_counter()
        session = session_from_state(state)
        resource = resource_for_node(node_name)
        bus = runtime.message_bus or get_message_bus()
        meta: dict[str, Any] = dict(state.get("reliability_meta") or {})
        events: list[dict[str, Any]] = []

        with start_span(
            f"sqlmind.node.{node_name}",
            attributes={
                "sqlmind.node": node_name,
                "sqlmind.session_id": str(state.get("session_id") or ""),
                "sqlmind.action": action,
            },
        ):
            _trace("node_enter", node_name=node_name, state=state)
            _trace(
                "reliability_event",
                kind="Heartbeat",
                node_name=node_name,
                detail={"session_id": state.get("session_id") or ""},
            )

            # --- Heartbeat / discovery presence ---
            try:
                runtime.health.beat(node_name, session_id=state.get("session_id") or "")
            except Exception:  # noqa: BLE001
                pass

            # --- Live A2A: drain inbox before node work ---
            work_state = _drain_inbox(bus, node_name, state)

            # --- IAM enforcement ---
            if runtime.enforce_iam and not skip_iam:
                deny_patch = assert_action_permission(
                    runtime.iam,
                    session,
                    action if action in {
                        "agent_invoke",
                        "tool_call",
                        "sql_execute",
                        "memory_access",
                        "plugin_execute",
                        "export",
                    } else "agent_invoke",  # type: ignore[arg-type]
                    resource=resource,
                    attributes={
                        "resource_tenant_id": work_state.get("tenant_id") or (
                            session.tenant_id if session else "default"
                        ),
                        "workspace_id": work_state.get("workspace_id") or "default",
                    },
                    on_deny="approval" if runtime.approval_on_deny else "reject",
                )
                if deny_patch:
                    deny_patch.setdefault(
                        "reliability_meta",
                        {**meta, "iam_denied": True, "node": node_name, "resource": resource},
                    )
                    deny_patch.setdefault(
                        "enterprise_events",
                        [
                            {
                                "type": "iam",
                                "action": "deny",
                                "node": node_name,
                                "resource": resource,
                            }
                        ],
                    )
                    _record_metrics(node_name, "denied", started)
                    _ack_inbox(bus, node_name, ok=False, detail="iam_deny")
                    _trace(
                        "node_exit",
                        node_name=node_name,
                        state_patch=deny_patch,
                        status="denied",
                        decision="iam_deny",
                        error=str(deny_patch.get("error") or "IAM denied"),
                    )
                    return deny_patch

            # --- Circuit breaker ---
            if runtime.enforce_circuit_breaker:
                br = runtime.breakers.get(node_name)
                allowed = br.allow()
                _trace(
                    "reliability_event",
                    kind="Circuit Breaker",
                    node_name=node_name,
                    status="allow" if allowed else "open",
                    detail={"checked": True},
                )
                if not allowed:
                    fallback = runtime.recovery.fallback_strategy(node_name)
                    dlq_id = runtime.dlq.enqueue(
                        kind="circuit_open",
                        error=f"Circuit open for {node_name}",
                        payload={"next": fallback, "state_keys": list(work_state.keys())[:40]},
                        session_id=work_state.get("session_id") or "",
                        node=node_name,
                    )
                    logger.warning(
                        "Circuit open node=%s → fallback=%s dlq=%s",
                        node_name,
                        fallback,
                        dlq_id,
                    )
                    _record_metrics(node_name, "circuit_open", started)
                    if runtime.publish_bus:
                        bus.publish(
                            status_update_message(
                                node_name,
                                {"status": "circuit_open", "fallback": fallback, "dlq_id": dlq_id},
                            )
                        )
                    patch = {
                        "next_agent": fallback,
                        "error": f"Circuit breaker open for {node_name}",
                        "reliability_meta": {
                            **meta,
                            "circuit_open": True,
                            "node": node_name,
                            "fallback": fallback,
                            "dlq_id": dlq_id,
                        },
                        "enterprise_events": [
                            {
                                "type": "reliability",
                                "action": "circuit_open",
                                "node": node_name,
                                "dlq_id": dlq_id,
                            }
                        ],
                        "agent_logs": [
                            {
                                "agent": node_name,
                                "message": f"circuit open → {fallback}",
                                "status": "warn",
                            }
                        ],
                        "agent_messages": [
                            status_update_message(
                                node_name,
                                {"status": "circuit_open", "fallback": fallback},
                            ).to_state_dict()
                        ],
                    }
                    _ack_inbox(bus, node_name, ok=False, detail="circuit_open")
                    _trace(
                        "reliability_event",
                        kind="DLQ",
                        node_name=node_name,
                        status="enqueued",
                        detail={"dlq_id": dlq_id, "fallback": fallback},
                    )
                    _trace(
                        "node_exit",
                        node_name=node_name,
                        state_patch=patch,
                        status="circuit_open",
                        decision=fallback,
                        error=patch["error"],
                    )
                    return patch

            # --- Pre-execution bus status ---
            if runtime.publish_bus:
                try:
                    bus.publish(
                        status_update_message(
                            node_name,
                            {
                                "status": "started",
                                "session_id": work_state.get("session_id") or "",
                                "security": resolve_security_context(work_state),
                            },
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass

            # --- Execute node ---
            try:
                result = fn(work_state)
                if not isinstance(result, dict):
                    result = {}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Node failure node=%s: %s", node_name, exc)
                dlq_id = runtime.recovery.on_node_failure(
                    node_name,
                    str(exc),
                    session_id=work_state.get("session_id") or "",
                    payload={
                        "question": (work_state.get("question") or "")[:500],
                        "sql": (work_state.get("sql") or "")[:500],
                    },
                )
                fallback = runtime.recovery.fallback_strategy(node_name)
                _record_metrics(node_name, "error", started)
                if runtime.publish_bus:
                    bus.publish(
                        task_result_message(
                            node_name,
                            {"ok": False, "error": str(exc), "dlq_id": dlq_id},
                            recipient="execution_coordinator",
                        )
                    )
                patch = {
                    "error": str(exc),
                    "status": "failed",
                    "next_agent": fallback if fallback != node_name else "fail",
                    "reliability_meta": {
                        **meta,
                        "failed": True,
                        "node": node_name,
                        "dlq_id": dlq_id,
                        "error": str(exc)[:500],
                    },
                    "enterprise_events": [
                        {
                            "type": "reliability",
                            "action": "node_failure",
                            "node": node_name,
                            "dlq_id": dlq_id,
                        }
                    ],
                    "agent_logs": [
                        {
                            "agent": node_name,
                            "message": f"failure → DLQ {dlq_id}",
                            "status": "error",
                            "detail": str(exc)[:300],
                        }
                    ],
                }
                _ack_inbox(bus, node_name, ok=False, detail=str(exc)[:200])
                _trace(
                    "reliability_event",
                    kind="Recovery",
                    node_name=node_name,
                    status="error",
                    detail={"dlq_id": dlq_id, "error": str(exc)[:300]},
                )
                _trace(
                    "node_exit",
                    node_name=node_name,
                    state_patch=patch,
                    status="error",
                    error=str(exc),
                )
                return patch

            # --- Success path ---
            if runtime.enforce_circuit_breaker:
                runtime.breakers.get(node_name).record_success()

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.setdefault("reliability_meta", {})
            if isinstance(result.get("reliability_meta"), dict):
                prior_nodes = {}
                if isinstance(meta.get("nodes"), dict):
                    prior_nodes = dict(meta["nodes"])
                result["reliability_meta"] = {
                    **meta,
                    **result["reliability_meta"],
                    "last_node": node_name,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "nodes": {
                        **prior_nodes,
                        node_name: {"elapsed_ms": round(elapsed_ms, 2), "status": "ok"},
                    },
                }

            if work_state.get("inbox_messages") and not result.get("inbox_messages"):
                # Do not re-emit drained messages; clear for next hop
                pass

            events.append(
                {
                    "type": "node",
                    "action": "completed",
                    "node": node_name,
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            existing_events = result.get("enterprise_events")
            if existing_events:
                result["enterprise_events"] = list(existing_events) + events
            else:
                result["enterprise_events"] = events

            if runtime.publish_bus:
                try:
                    bus.publish(
                        status_update_message(
                            node_name,
                            {
                                "status": "completed",
                                "elapsed_ms": round(elapsed_ms, 2),
                                "next_agent": result.get("next_agent"),
                            },
                        )
                    )
                    msgs = result.get("agent_messages")
                    if isinstance(msgs, list):
                        result["agent_messages"] = list(msgs) + [
                            status_update_message(
                                node_name, {"status": "completed"}
                            ).to_state_dict()
                        ]
                    else:
                        result["agent_messages"] = [
                            status_update_message(
                                node_name, {"status": "completed"}
                            ).to_state_dict()
                        ]
                except Exception:  # noqa: BLE001
                    pass

            _ack_inbox(bus, node_name, ok=True)
            _record_metrics(node_name, "ok", started)
            try:
                runtime.health.beat(
                    node_name,
                    session_id=work_state.get("session_id") or "",
                    elapsed_ms=elapsed_ms,
                )
            except Exception:  # noqa: BLE001
                pass

            if result.get("parallel_job") and node_name in {
                "execution_node",
                "execution_coordinator",
            }:
                _trace(
                    "parallel_send",
                    targets=["visualization_agent", "insight_agent"],
                    source=node_name,
                )

            _trace(
                "node_exit",
                node_name=node_name,
                state_patch=result,
                status="ok",
                decision=str(result.get("next_agent") or ""),
            )
            return result

    wrapped.__name__ = f"enterprise_{node_name}"
    wrapped.__qualname__ = f"enterprise_{node_name}"
    return wrapped


def _record_metrics(node_name: str, status: str, started: float) -> None:
    try:
        from observability.metrics import get_metrics

        elapsed = time.perf_counter() - started
        get_metrics().observe_node(node_name, status=status, duration_seconds=elapsed)
    except Exception:  # noqa: BLE001
        pass
