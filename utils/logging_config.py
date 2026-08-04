"""Logging configuration and audit helpers with request correlation."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _ContextFilter(logging.Filter):
    """Inject request / tenant / trace correlation fields into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from core.context import get_request_context

            ctx = get_request_context()
        except Exception:  # noqa: BLE001 — logging must never crash
            ctx = None
        record.request_id = getattr(ctx, "request_id", "-") if ctx else "-"  # type: ignore[attr-defined]
        record.session_id = getattr(ctx, "session_id", "-") if ctx else "-"  # type: ignore[attr-defined]
        record.tenant_id = getattr(ctx, "tenant_id", "-") if ctx else "-"  # type: ignore[attr-defined]
        record.actor = getattr(ctx, "actor", "-") if ctx else "-"  # type: ignore[attr-defined]

        trace_id, span_id = "-", "-"
        try:
            from observability.tracing import get_span_id, get_trace_id

            trace_id = get_trace_id() or "-"
            span_id = get_span_id() or "-"
        except Exception:  # noqa: BLE001
            pass
        record.trace_id = trace_id  # type: ignore[attr-defined]
        record.span_id = span_id  # type: ignore[attr-defined]
        return True


def setup_logging(level: str = "INFO", audit_log_path: str | None = None) -> None:
    """Configure root logging once for the application."""
    root = logging.getLogger()
    if root.handlers:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "req=%(request_id)s sess=%(session_id)s tenant=%(tenant_id)s "
            "trace=%(trace_id)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(_ContextFilter())
    root.addHandler(stream)

    if audit_log_path:
        path = Path(audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_ContextFilter())
        file_handler.setLevel(logging.INFO)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def write_audit_event(
    event_type: str,
    payload: dict[str, Any],
    audit_log_path: str,
) -> None:
    """Append a structured audit event as JSONL (includes request context)."""
    path = Path(audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ctx_payload: dict[str, Any] = {}
    try:
        from core.context import get_request_context

        ctx = get_request_context()
        if ctx:
            ctx_payload = ctx.as_dict()
    except Exception:  # noqa: BLE001
        pass

    try:
        from observability.tracing import correlation_fields

        ctx_payload.update(correlation_fields())
    except Exception:  # noqa: BLE001
        ctx_payload.setdefault("trace_id", "-")
        ctx_payload.setdefault("span_id", "-")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **ctx_payload,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
