"""Trace / correlation ID helpers shared by logs, LangSmith, and OpenTelemetry."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator
from uuid import uuid4

_TRACE_ID: ContextVar[str | None] = ContextVar("sqlmind_trace_id", default=None)
_SPAN_ID: ContextVar[str | None] = ContextVar("sqlmind_span_id", default=None)


def new_trace_id() -> str:
    """Generate a 32-char hex trace id (OTel-compatible width)."""
    return uuid4().hex


def new_span_id() -> str:
    """Generate a 16-char hex span id."""
    return uuid4().hex[:16]


def get_trace_id() -> str | None:
    """Return the active SQLMind trace id, falling back to the current OTel span."""
    tid = _TRACE_ID.get()
    if tid:
        return tid
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:  # noqa: BLE001 — observability must never crash callers
        pass
    return None


def get_span_id() -> str | None:
    """Return the active local or OTel span id."""
    sid = _SPAN_ID.get()
    if sid:
        return sid
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and ctx.is_valid:
            return format(ctx.span_id, "016x")
    except Exception:  # noqa: BLE001
        pass
    return None


def set_trace_id(trace_id: str | None) -> Token[str | None]:
    """Set the active trace id; returns a context token for reset."""
    return _TRACE_ID.set(trace_id)


def set_span_id(span_id: str | None) -> Token[str | None]:
    """Set the active span id; returns a context token for reset."""
    return _SPAN_ID.set(span_id)


def reset_span_id(token: Token[str | None]) -> None:
    """Restore a previous span id from ``set_span_id``."""
    _SPAN_ID.reset(token)


def ensure_trace_id() -> str:
    """Return an existing trace id or create and bind a new one."""
    existing = get_trace_id()
    if existing:
        if _TRACE_ID.get() is None:
            _TRACE_ID.set(existing)
        return existing
    tid = new_trace_id()
    _TRACE_ID.set(tid)
    return tid


def clear_trace_context() -> None:
    """Clear local trace/span context vars."""
    _TRACE_ID.set(None)
    _SPAN_ID.set(None)


@contextmanager
def bind_trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id for the duration of the block."""
    tid = trace_id or new_trace_id()
    token = _TRACE_ID.set(tid)
    try:
        yield tid
    finally:
        _TRACE_ID.reset(token)


def sync_from_otel_span() -> None:
    """Copy OTel span context into local context vars when a valid span is active."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is None or not ctx.is_valid:
            return
        if _TRACE_ID.get() is None:
            _TRACE_ID.set(format(ctx.trace_id, "032x"))
        _SPAN_ID.set(format(ctx.span_id, "016x"))
    except Exception:  # noqa: BLE001
        pass


def correlation_fields() -> dict[str, str]:
    """Fields suitable for log records and audit JSONL."""
    return {
        "trace_id": get_trace_id() or "-",
        "span_id": get_span_id() or "-",
    }


def run_metadata(
    *,
    session_id: str,
    provider: str,
    dialect: str,
    database: str = "",
    tenant_id: str = "",
    actor: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build LangSmith / LangChain runnable metadata for a graph run."""
    meta: dict[str, Any] = {
        "session_id": session_id,
        "provider": provider,
        "dialect": dialect,
        "database": database,
        "tenant_id": tenant_id or "default",
        "actor": actor or "anonymous",
        "trace_id": ensure_trace_id(),
    }
    if extra:
        meta.update(extra)
    return meta
