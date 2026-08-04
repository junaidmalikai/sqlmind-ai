"""Prometheus + in-process metrics for SQLMind enterprise observability.

Exposes counters/histograms for agents, goals, tools, memory, planner, SQL,
retries, reflection, plugins, and exports. Works with ``prometheus_client``
when installed; otherwise uses a compatible in-memory registry that still
renders Prometheus text exposition.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)

_LOCK = threading.RLock()
_METRICS: "SQLMindMetrics | None" = None


class _Counter:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._values: dict[tuple[str, ...], float] = defaultdict(float)

    def labels(self, **labels: str) -> "_CounterChild":
        key = tuple(labels.get(n, "") for n in self.labelnames)
        return _CounterChild(self, key)

    def inc(self, amount: float = 1.0) -> None:
        self.labels().inc(amount)

    def collect_lines(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} counter",
        ]
        for key, val in self._values.items():
            if self.labelnames:
                label_str = ",".join(
                    f'{n}="{v}"' for n, v in zip(self.labelnames, key, strict=True)
                )
                lines.append(f"{self.name}{{{label_str}}} {val}")
            else:
                lines.append(f"{self.name} {val}")
        return lines


class _CounterChild:
    def __init__(self, parent: _Counter, key: tuple[str, ...]) -> None:
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        with _LOCK:
            self._parent._values[self._key] += amount


class _Histogram:
    """Simple cumulative histogram with fixed buckets + sum + count."""

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        float("inf"),
    )

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple[str, ...], list[float]] = {}
        self._sums: dict[tuple[str, ...], float] = defaultdict(float)
        self._totals: dict[tuple[str, ...], float] = defaultdict(float)

    def labels(self, **labels: str) -> "_HistogramChild":
        key = tuple(labels.get(n, "") for n in self.labelnames)
        return _HistogramChild(self, key)

    def observe(self, value: float) -> None:
        self.labels().observe(value)

    def collect_lines(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.documentation}",
            f"# TYPE {self.name} histogram",
        ]
        for key, counts in self._counts.items():
            label_prefix = ""
            if self.labelnames:
                label_prefix = ",".join(
                    f'{n}="{v}"' for n, v in zip(self.labelnames, key, strict=True)
                )
            cumulative = 0.0
            for boundary, count in zip(self.buckets, counts, strict=True):
                cumulative += count
                b = "+Inf" if boundary == float("inf") else str(boundary)
                if label_prefix:
                    lines.append(
                        f'{self.name}_bucket{{{label_prefix},le="{b}"}} {cumulative}'
                    )
                else:
                    lines.append(f'{self.name}_bucket{{le="{b}"}} {cumulative}')
            if label_prefix:
                lines.append(f"{self.name}_sum{{{label_prefix}}} {self._sums[key]}")
                lines.append(f"{self.name}_count{{{label_prefix}}} {self._totals[key]}")
            else:
                lines.append(f"{self.name}_sum {self._sums[key]}")
                lines.append(f"{self.name}_count {self._totals[key]}")
        return lines


class _HistogramChild:
    def __init__(self, parent: _Histogram, key: tuple[str, ...]) -> None:
        self._parent = parent
        self._key = key

    def observe(self, value: float) -> None:
        with _LOCK:
            if self._key not in self._parent._counts:
                self._parent._counts[self._key] = [0.0] * len(self._parent.buckets)
            for i, boundary in enumerate(self._parent.buckets):
                if value <= boundary:
                    self._parent._counts[self._key][i] += 1.0
                    break
            self._parent._sums[self._key] += value
            self._parent._totals[self._key] += 1.0


class SQLMindMetrics:
    """Enterprise metrics registry — Prometheus text + timeline helpers."""

    def __init__(self) -> None:
        self._prom = None
        try:
            from prometheus_client import Counter, Histogram, REGISTRY  # type: ignore

            self._prom = {"Counter": Counter, "Histogram": Histogram, "registry": REGISTRY}
        except Exception:  # noqa: BLE001
            self._prom = None

        self.node_runs = _Counter(
            "sqlmind_node_runs_total",
            "LangGraph node executions",
            ("node", "status"),
        )
        self.node_duration = _Histogram(
            "sqlmind_node_duration_seconds",
            "LangGraph node duration",
            ("node",),
        )
        self.agent_runs = _Counter(
            "sqlmind_agent_runs_total",
            "AI agent invocations",
            ("agent", "status"),
        )
        self.goal_events = _Counter(
            "sqlmind_goal_events_total",
            "Goal lifecycle events",
            ("event",),
        )
        self.tool_calls = _Counter(
            "sqlmind_tool_calls_total",
            "Tool invocations",
            ("tool", "status"),
        )
        self.memory_ops = _Counter(
            "sqlmind_memory_ops_total",
            "Memory fabric operations",
            ("op", "status"),
        )
        self.planner_events = _Counter(
            "sqlmind_planner_events_total",
            "Planner timeline events",
            ("event",),
        )
        self.sql_events = _Counter(
            "sqlmind_sql_events_total",
            "SQL validation/execution events",
            ("event", "status"),
        )
        self.retry_events = _Counter(
            "sqlmind_retry_events_total",
            "Retry timeline events",
            ("action",),
        )
        self.reflection_events = _Counter(
            "sqlmind_reflection_events_total",
            "Reflection timeline events",
            ("verdict",),
        )
        self.plugin_events = _Counter(
            "sqlmind_plugin_events_total",
            "Plugin marketplace events",
            ("event", "plugin"),
        )
        self.export_events = _Counter(
            "sqlmind_export_events_total",
            "Export events",
            ("format", "status"),
        )
        self.iam_decisions = _Counter(
            "sqlmind_iam_decisions_total",
            "IAM authorization decisions",
            ("decision", "resource"),
        )
        self.dlq_events = _Counter(
            "sqlmind_dlq_events_total",
            "Dead letter queue enqueues",
            ("kind",),
        )
        self.circuit_events = _Counter(
            "sqlmind_circuit_events_total",
            "Circuit breaker state changes",
            ("node", "state"),
        )
        self.replan_events = _Counter(
            "sqlmind_replan_events_total",
            "Runtime replan events",
            ("strategy",),
        )
        self.recovery_events = _Counter(
            "sqlmind_recovery_events_total",
            "Recovery graph actions",
            ("action", "trigger"),
        )
        self.parallel_events = _Counter(
            "sqlmind_parallel_waves_total",
            "Parallel Send fan-out waves",
            ("source",),
        )
        self.parallel_workers = _Histogram(
            "sqlmind_parallel_workers",
            "Workers per parallel wave",
            ("source",),
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, float("inf")),
        )
        self.queue_events = _Counter(
            "sqlmind_queue_events_total",
            "Distributed / enterprise queue events",
            ("event", "kind"),
        )
        self.worker_events = _Counter(
            "sqlmind_worker_events_total",
            "Distributed worker lifecycle events",
            ("event", "worker"),
        )
        self.llm_latency = _Histogram(
            "sqlmind_llm_latency_seconds",
            "LLM call latency",
            ("provider",),
        )
        self.llm_tokens = _Counter(
            "sqlmind_llm_tokens_total",
            "LLM token usage",
            ("direction",),
        )
        self.plugin_latency = _Histogram(
            "sqlmind_plugin_latency_seconds",
            "Plugin invocation latency",
            ("plugin",),
        )
        self.memory_latency = _Histogram(
            "sqlmind_memory_retrieval_seconds",
            "Memory retrieval latency",
            ("backend",),
        )
        self.sql_latency = _Histogram(
            "sqlmind_sql_execution_seconds",
            "SQL execution latency",
            ("status",),
        )
        self.gauge_values: dict[str, float] = {
            "queue_length": 0.0,
            "worker_count": 0.0,
            "export_count": 0.0,
            "recovery_count": 0.0,
            "replan_count": 0.0,
            "success_rate": 1.0,
            "circuit_open_count": 0.0,
        }
        self._timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._started_at = time.time()
        self._run_successes = 0
        self._run_failures = 0

    def observe_node(self, node: str, *, status: str, duration_seconds: float) -> None:
        self.node_runs.labels(node=node, status=status).inc()
        self.node_duration.labels(node=node).observe(max(duration_seconds, 0.0))
        if "agent" in node or node in {
            "supervisor",
            "planner",
            "goal_understanding",
            "task_decomposition",
            "execution_coordinator",
            "replan_agent",
            "goal_tracking",
            "memory_agent",
            "clarify",
        }:
            self.agent_runs.labels(agent=node, status=status).inc()
        self.timeline("agent", {"node": node, "status": status, "duration": duration_seconds})

    def observe_goal(self, event: str, **payload: Any) -> None:
        self.goal_events.labels(event=event).inc()
        self.timeline("goal", {"event": event, **payload})

    def observe_tool(self, tool: str, *, status: str = "ok") -> None:
        self.tool_calls.labels(tool=tool, status=status).inc()
        self.timeline("tool", {"tool": tool, "status": status})

    def observe_memory(self, op: str, *, status: str = "ok") -> None:
        self.memory_ops.labels(op=op, status=status).inc()
        self.timeline("memory", {"op": op, "status": status})

    def observe_planner(self, event: str, **payload: Any) -> None:
        self.planner_events.labels(event=event).inc()
        self.timeline("planner", {"event": event, **payload})

    def observe_sql(self, event: str, *, status: str = "ok") -> None:
        self.sql_events.labels(event=event, status=status).inc()
        self.timeline("sql", {"event": event, "status": status})

    def observe_retry(self, action: str) -> None:
        self.retry_events.labels(action=action).inc()
        self.timeline("retry", {"action": action})

    def observe_reflection(self, verdict: str) -> None:
        self.reflection_events.labels(verdict=verdict).inc()
        self.timeline("reflection", {"verdict": verdict})

    def observe_plugin(self, event: str, plugin: str = "") -> None:
        self.plugin_events.labels(event=event, plugin=plugin).inc()
        self.timeline("plugin", {"event": event, "plugin": plugin})

    def observe_export(self, fmt: str, *, status: str = "ok") -> None:
        self.export_events.labels(format=fmt, status=status).inc()
        self.timeline("export", {"format": fmt, "status": status})

    def observe_iam(self, decision: str, resource: str) -> None:
        self.iam_decisions.labels(decision=decision, resource=resource[:80]).inc()

    def observe_dlq(self, kind: str) -> None:
        self.dlq_events.labels(kind=kind).inc()

    def observe_circuit(self, node: str, state: str) -> None:
        self.circuit_events.labels(node=node, state=state).inc()
        if state == "open":
            self.gauge_values["circuit_open_count"] = (
                self.gauge_values.get("circuit_open_count", 0.0) + 1.0
            )

    def observe_replan(self, strategy: str = "revise") -> None:
        self.replan_events.labels(strategy=strategy).inc()
        self.gauge_values["replan_count"] = self.gauge_values.get("replan_count", 0.0) + 1.0
        self.timeline("replan", {"strategy": strategy})

    def observe_recovery(self, action: str, trigger: str = "unknown") -> None:
        self.recovery_events.labels(action=action, trigger=trigger).inc()
        self.gauge_values["recovery_count"] = (
            self.gauge_values.get("recovery_count", 0.0) + 1.0
        )
        self.timeline("recovery", {"action": action, "trigger": trigger})

    def observe_parallel(self, *, source: str, workers: int, wave_id: str = "") -> None:
        self.parallel_events.labels(source=source).inc()
        self.parallel_workers.labels(source=source).observe(float(workers))
        self.timeline(
            "parallel",
            {"source": source, "workers": workers, "wave_id": wave_id},
        )

    def observe_queue(self, event: str, kind: str = "") -> None:
        self.queue_events.labels(event=event, kind=kind).inc()

    def observe_worker(self, event: str, worker: str = "") -> None:
        self.worker_events.labels(event=event, worker=worker).inc()

    def observe_llm_latency(self, provider: str, seconds: float) -> None:
        self.llm_latency.labels(provider=provider).observe(max(seconds, 0.0))

    def observe_llm_tokens(self, *, prompt: int = 0, completion: int = 0) -> None:
        if prompt:
            self.llm_tokens.labels(direction="prompt").inc(prompt)
        if completion:
            self.llm_tokens.labels(direction="completion").inc(completion)

    def observe_plugin_latency(self, plugin: str, seconds: float) -> None:
        self.plugin_latency.labels(plugin=plugin).observe(max(seconds, 0.0))

    def observe_memory_latency(self, backend: str, seconds: float) -> None:
        self.memory_latency.labels(backend=backend).observe(max(seconds, 0.0))

    def observe_sql_latency(self, seconds: float, *, status: str = "ok") -> None:
        self.sql_latency.labels(status=status).observe(max(seconds, 0.0))

    def set_gauge(self, name: str, value: float) -> None:
        self.gauge_values[name] = float(value)

    def record_run_outcome(self, *, success: bool) -> None:
        if success:
            self._run_successes += 1
        else:
            self._run_failures += 1
        total = self._run_successes + self._run_failures
        if total:
            self.gauge_values["success_rate"] = self._run_successes / total

    def timeline(self, kind: str, payload: dict[str, Any]) -> None:
        with _LOCK:
            bucket = self._timelines[kind]
            bucket.append({"ts": time.time(), **payload})
            if len(bucket) > 500:
                self._timelines[kind] = bucket[-500:]

    def get_timeline(self, kind: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with _LOCK:
            return list(self._timelines.get(kind, []))[-limit:]

    def all_timelines(self) -> dict[str, list[dict[str, Any]]]:
        with _LOCK:
            return {k: list(v)[-50:] for k, v in self._timelines.items()}

    def render_prometheus(self) -> str:
        """Prometheus text exposition format."""
        collectors = [
            self.node_runs,
            self.node_duration,
            self.agent_runs,
            self.goal_events,
            self.tool_calls,
            self.memory_ops,
            self.planner_events,
            self.sql_events,
            self.retry_events,
            self.reflection_events,
            self.plugin_events,
            self.export_events,
            self.iam_decisions,
            self.dlq_events,
            self.circuit_events,
            self.replan_events,
            self.recovery_events,
            self.parallel_events,
            self.parallel_workers,
            self.queue_events,
            self.worker_events,
            self.llm_latency,
            self.llm_tokens,
            self.plugin_latency,
            self.memory_latency,
            self.sql_latency,
        ]
        lines: list[str] = [
            f"# SQLMind metrics uptime_seconds {time.time() - self._started_at:.1f}",
        ]
        for c in collectors:
            lines.extend(c.collect_lines())
        lines.append("# HELP sqlmind_gauge Runtime gauges")
        lines.append("# TYPE sqlmind_gauge gauge")
        for name, val in self.gauge_values.items():
            lines.append(f'sqlmind_gauge{{name="{name}"}} {val}')
        # Optional: merge prometheus_client default registry if present
        if self._prom is not None:
            try:
                from prometheus_client import generate_latest  # type: ignore

                extra = generate_latest(self._prom["registry"]).decode("utf-8")
                lines.append(extra)
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(lines) + "\n"


def get_metrics() -> SQLMindMetrics:
    global _METRICS
    with _LOCK:
        if _METRICS is None:
            _METRICS = SQLMindMetrics()
        return _METRICS


def reset_metrics() -> SQLMindMetrics:
    global _METRICS
    with _LOCK:
        _METRICS = SQLMindMetrics()
        return _METRICS
