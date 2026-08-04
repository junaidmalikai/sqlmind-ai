"""Observability package — LangSmith, OpenTelemetry, Prometheus, correlation, events, runtime trace."""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from observability.events import EnterpriseEventTracker, get_event_tracker
from observability.langsmith_setup import configure_langsmith, langsmith_tags
from observability.metrics import SQLMindMetrics, get_metrics, reset_metrics
from observability.otel_setup import configure_otel, get_tracer, start_span
from observability.runtime_trace import (
    ExecutionTraceSession,
    execution_trace_session,
    finalize_and_export,
    get_active_trace,
    get_last_completed_trace,
    safe_trace,
)
from observability.tracing import (
    bind_trace_context,
    clear_trace_context,
    correlation_fields,
    ensure_trace_id,
    get_span_id,
    get_trace_id,
    run_metadata,
)
from utils.helpers import ensure_dirs

__all__ = [
    "EnterpriseEventTracker",
    "ExecutionTraceSession",
    "SQLMindMetrics",
    "bind_trace_context",
    "clear_trace_context",
    "configure_langsmith",
    "configure_observability",
    "configure_otel",
    "correlation_fields",
    "ensure_trace_id",
    "execution_trace_session",
    "finalize_and_export",
    "get_active_trace",
    "get_event_tracker",
    "get_last_completed_trace",
    "get_metrics",
    "get_span_id",
    "get_trace_id",
    "get_tracer",
    "langsmith_tags",
    "reset_metrics",
    "run_metadata",
    "safe_trace",
    "start_span",
]


def configure_observability(settings: Settings) -> None:
    """Bootstrap LangSmith, OpenTelemetry, Prometheus metrics, event tracker, and trace dir."""
    configure_langsmith(settings)
    configure_otel(settings)
    log_path = getattr(settings, "enterprise_events_log_path", None)
    get_event_tracker(log_path=log_path)
    if getattr(settings, "prometheus_enabled", True):
        get_metrics()
    if getattr(settings, "runtime_trace_enabled", True):
        ensure_dirs(Path(getattr(settings, "runtime_trace_dir", "data/runtime_traces")))
