"""Real plugin runtime — invocation, permissions, isolation, routing, metrics.

Complements PluginMarketplace discovery/loading with graph-callable execution.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import uuid4

from utils.logging_config import get_logger

logger = get_logger(__name__)


class PluginPermissionError(PermissionError):
    pass


class PluginRuntime:
    """Runtime façade over PluginMarketplace with IAM-aware invocation."""

    def __init__(
        self,
        marketplace: Any,
        *,
        default_timeout: float = 15.0,
        require_permission: bool = True,
    ) -> None:
        self.marketplace = marketplace
        self.default_timeout = default_timeout
        self.require_permission = require_permission
        self._lock = threading.RLock()
        self._invocations: list[dict[str, Any]] = []
        self._metrics: dict[str, Any] = {
            "invocations": 0,
            "successes": 0,
            "failures": 0,
            "timeouts": 0,
            "denied": 0,
            "latencies_ms": [],
        }

    def list_capabilities(self) -> list[dict[str, Any]]:
        try:
            return list(self.marketplace.list_plugins())
        except Exception:  # noqa: BLE001
            return []

    def health(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            for p in self.marketplace.list_plugins():
                out.append(
                    {
                        "plugin_id": p.get("id"),
                        "version": p.get("version"),
                        "health": p.get("health"),
                        "enabled": p.get("enabled"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Plugin health snapshot failed: %s", exc)
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            lats = list(self._metrics["latencies_ms"])
            return {
                **{k: v for k, v in self._metrics.items() if k != "latencies_ms"},
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0.0,
                "recent": list(self._invocations[-10:]),
            }

    def check_permission(
        self,
        capability_id: str,
        *,
        iam_session: dict[str, Any] | None = None,
        action: str = "plugin:execute",
    ) -> bool:
        if not self.require_permission:
            return True
        if not iam_session:
            return True
        denied = (iam_session.get("denied_plugins") or []) if isinstance(iam_session, dict) else []
        if capability_id in denied:
            return False
        allowed = iam_session.get("allowed_plugins") if isinstance(iam_session, dict) else None
        if isinstance(allowed, list) and allowed and capability_id not in allowed:
            return False
        return True

    def invoke(
        self,
        capability_id: str,
        *args: Any,
        iam_session: dict[str, Any] | None = None,
        timeout: float | None = None,
        isolation: bool = True,
        **kwargs: Any,
    ) -> Any:
        from observability.otel_setup import start_span

        started = time.perf_counter()
        invocation_id = f"pinv-{uuid4().hex[:10]}"
        with self._lock:
            self._metrics["invocations"] += 1

        with start_span(
            "sqlmind.plugin.invoke",
            attributes={
                "sqlmind.plugin.capability_id": capability_id,
                "sqlmind.plugin.invocation_id": invocation_id,
            },
        ):
            if not self.check_permission(capability_id, iam_session=iam_session):
                with self._lock:
                    self._metrics["denied"] += 1
                self._record(invocation_id, capability_id, "denied", 0.0)
                raise PluginPermissionError(
                    f"Permission denied for plugin capability {capability_id}"
                )

            timeout_s = timeout if timeout is not None else self.default_timeout
            try:
                if isolation and hasattr(self.marketplace, "sandbox"):
                    handler = self.marketplace.get_handler(capability_id)
                    if handler is None:
                        raise KeyError(f"Unknown plugin capability: {capability_id}")
                    sandbox = self.marketplace.sandbox
                    prev = getattr(sandbox, "timeout_seconds", timeout_s)
                    try:
                        sandbox.timeout_seconds = timeout_s
                        result = sandbox.invoke(handler, *args, **kwargs)
                    finally:
                        sandbox.timeout_seconds = prev
                else:
                    result = self.marketplace.execute_capability(
                        capability_id, *args, **kwargs
                    )
                elapsed = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._metrics["successes"] += 1
                    self._metrics["latencies_ms"].append(elapsed)
                self._record(invocation_id, capability_id, "ok", elapsed)
                try:
                    from observability.metrics import get_metrics

                    get_metrics().observe_plugin("execute", capability_id)
                except Exception:  # noqa: BLE001
                    pass
                return result
            except TimeoutError as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._metrics["timeouts"] += 1
                    self._metrics["failures"] += 1
                self._record(invocation_id, capability_id, "timeout", elapsed, str(exc))
                self._enqueue_failure(capability_id, str(exc), iam_session)
                raise
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._metrics["failures"] += 1
                self._record(invocation_id, capability_id, "error", elapsed, str(exc))
                self._enqueue_failure(capability_id, str(exc), iam_session)
                try:
                    from observability.metrics import get_metrics

                    get_metrics().observe_plugin(
                        "execute_error", capability_id
                    )
                except Exception:  # noqa: BLE001
                    pass
                raise

    def _enqueue_failure(
        self,
        capability_id: str,
        error: str,
        iam_session: dict[str, Any] | None,
    ) -> None:
        try:
            from reliability.enterprise_queue import get_enterprise_queue

            get_enterprise_queue().enqueue_retry(
                topic="plugin_failure",
                payload={
                    "capability_id": capability_id,
                    "error": error[:500],
                    "tenant_id": (iam_session or {}).get("tenant_id", "default"),
                },
                error=error[:500],
                node="plugin_runtime",
            )
        except Exception:  # noqa: BLE001
            pass

    def _record(
        self,
        invocation_id: str,
        capability_id: str,
        status: str,
        latency_ms: float,
        error: str = "",
    ) -> None:
        entry = {
            "invocation_id": invocation_id,
            "capability_id": capability_id,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "error": error[:300],
            "ts": time.time(),
        }
        with self._lock:
            self._invocations.append(entry)
            if len(self._invocations) > 200:
                self._invocations = self._invocations[-200:]
        try:
            from observability.runtime_trace import safe_trace

            safe_trace(
                "plugin_event",
                kind="Plugin Execution" if status == "ok" else "Plugin Failure",
                plugin_id=capability_id,
                status=status,
                output=error or None,
                detail={"latency_ms": latency_ms, "invocation_id": invocation_id},
            )
        except Exception:  # noqa: BLE001
            pass


def _failure_next_agent() -> str:
    try:
        from config.settings import get_settings

        settings = get_settings()
        if getattr(settings, "enterprise_enabled", True) and getattr(
            settings, "enterprise_subgraphs_enabled", True
        ):
            return "recovery_graph"
        if getattr(settings, "enterprise_enabled", True) and getattr(
            settings, "recovery_controller_enabled", True
        ):
            return "recovery_controller"
    except Exception:  # noqa: BLE001
        pass
    return "retry_agent"


def make_plugin_runtime_agent(plugin_runtime: PluginRuntime | None):
    """LangGraph node — invoke a selected plugin capability from state / inbox."""

    def plugin_runtime_agent(state: dict[str, Any]) -> dict[str, Any]:
        if plugin_runtime is None:
            return {
                "next_agent": "supervisor",
                "agent_logs": [
                    {
                        "agent": "plugin_runtime_agent",
                        "message": "No plugin runtime configured",
                        "status": "skip",
                    }
                ],
            }
        cap = (
            state.get("plugin_capability_id")
            or state.get("selected_plugin")
            or ""
        )
        # Inbox / bus task_request payload
        if not cap:
            for m in reversed(list(state.get("inbox_messages") or []) + list(state.get("agent_messages") or [])):
                if not isinstance(m, dict):
                    continue
                if m.get("kind") in {"task_request", "command"}:
                    payload = m.get("payload") or {}
                    cap = (
                        payload.get("plugin_capability_id")
                        or payload.get("capability_id")
                        or ""
                    )
                    if cap:
                        break
        if not cap:
            # Explicit health probe only when requested
            if state.get("plugin_health_probe"):
                cap = "skill.echo"
            else:
                return {
                    "next_agent": "supervisor",
                    "agent_logs": [
                        {
                            "agent": "plugin_runtime_agent",
                            "message": "No plugin_capability_id selected",
                            "status": "skip",
                        }
                    ],
                    "route_history": ["plugin_runtime_agent"],
                }
        try:
            result = plugin_runtime.invoke(
                str(cap),
                {
                    "question": state.get("question"),
                    "sql": state.get("sql"),
                    "row_count": state.get("row_count"),
                    "session_id": state.get("session_id"),
                },
                iam_session=state.get("iam_session"),
            )
            return {
                "plugin_result": result if isinstance(result, dict) else {"result": result},
                "plugin_capability_id": str(cap),
                "next_agent": "supervisor"
                if not state.get("plan_active")
                else "execution_coordinator",
                "agent_logs": [
                    {
                        "agent": "plugin_runtime_agent",
                        "message": f"Invoked {cap}",
                        "status": "ok",
                        "detail": str(result)[:300],
                    }
                ],
                "route_history": ["plugin_runtime_agent"],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "error": f"Plugin invocation failed: {exc}",
                "replan_reason": f"plugin failure: {cap}",
                "retry_next_action": "replan",
                "next_agent": _failure_next_agent(),
                "agent_logs": [
                    {
                        "agent": "plugin_runtime_agent",
                        "message": str(exc),
                        "status": "error",
                    }
                ],
                "route_history": ["plugin_runtime_agent"],
            }

    return plugin_runtime_agent
