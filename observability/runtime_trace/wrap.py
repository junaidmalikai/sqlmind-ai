"""Lightweight LangGraph node wrapper — runtime trace only (no IAM/CB changes)."""

from __future__ import annotations

from typing import Any, Callable

from observability.otel_setup import start_span
from observability.runtime_trace import safe_trace

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def wrap_trace_node(node_name: str, fn: NodeFn) -> NodeFn:
    """Instrument a node with enter/exit tracing + inbox drain without altering returns."""

    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        work_state: dict[str, Any] = state
        try:
            from planner.messages import get_message_bus

            messages = get_message_bus().drain(node_name)
            if messages:
                work_state = dict(state)
                inbox = [
                    m.to_state_dict() if hasattr(m, "to_state_dict") else m
                    for m in messages
                ]
                work_state["inbox_messages"] = list(state.get("inbox_messages") or []) + inbox
                for m in messages:
                    payload = getattr(m, "payload", None) or {}
                    if isinstance(payload, dict) and (
                        payload.get("plugin_capability_id") or payload.get("capability_id")
                    ):
                        work_state["plugin_capability_id"] = (
                            payload.get("plugin_capability_id")
                            or payload.get("capability_id")
                        )
        except Exception:  # noqa: BLE001
            work_state = state

        safe_trace("node_enter", node_name=node_name, state=work_state)
        try:
            with start_span(
                f"sqlmind.node.{node_name}",
                attributes={"sqlmind.node": node_name},
            ):
                result = fn(work_state)
            if not isinstance(result, dict):
                result = {}
            status = "error" if result.get("error") else "ok"
            safe_trace(
                "node_exit",
                node_name=node_name,
                state_patch=result,
                status=status,
                decision=str(result.get("next_agent") or ""),
                error=str(result.get("error") or ""),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            safe_trace(
                "node_exit",
                node_name=node_name,
                status="error",
                error=str(exc),
            )
            raise

    wrapped.__name__ = f"trace_{node_name}"
    wrapped.__qualname__ = f"trace_{node_name}"
    return wrapped
