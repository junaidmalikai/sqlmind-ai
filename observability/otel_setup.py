"""OpenTelemetry bootstrap and span helpers for SQLMind AI."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Token
from typing import Any, Iterator

from config.settings import Settings
from observability.tracing import (
    clear_trace_context,
    ensure_trace_id,
    new_span_id,
    reset_span_id,
    set_span_id,
    sync_from_otel_span,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

_CONFIGURED = False


def configure_otel(settings: Settings) -> None:
    """Initialize the global TracerProvider when OTel is enabled.

    Safe to call multiple times. No-ops when disabled or when packages are missing.
    Export target:
      - ``otel_exporter=console`` → ConsoleSpanExporter (default, local debug)
      - ``otel_exporter=otlp`` → OTLP/HTTP to ``otel_endpoint``
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not getattr(settings, "otel_enabled", False):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry enabled but packages missing; "
            "install opentelemetry-api/sdk/exporter-otlp-proto-http"
        )
        return

    service_name = getattr(settings, "otel_service_name", None) or "SQLMind-AI"
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": getattr(settings, "app_version", "1.0.0"),
            "deployment.environment": getattr(settings, "environment", "development"),
        }
    )
    provider = TracerProvider(resource=resource)

    exporter_kind = str(getattr(settings, "otel_exporter", "console") or "console").lower()
    try:
        if exporter_kind == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = getattr(settings, "otel_endpoint", "") or ""
            kwargs: dict[str, Any] = {}
            if endpoint:
                kwargs["endpoint"] = endpoint
            exporter: Any = OTLPSpanExporter(**kwargs)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                "OpenTelemetry OTLP exporter configured endpoint=%s",
                endpoint or "(default)",
            )
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            logger.info(
                "OpenTelemetry console exporter configured service=%s", service_name
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenTelemetry exporter setup failed: %s", exc)
        return

    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer(name: str = "sqlmind"):
    """Return an OTel tracer only after ``configure_otel`` succeeded."""
    if not _CONFIGURED:
        return None
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:  # noqa: BLE001
        return None


def _set_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    if span is None or not attributes:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            if isinstance(value, (bool, int, float, str)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
        except Exception:  # noqa: BLE001
            pass


def _mark_error(span: Any, exc: BaseException) -> None:
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def start_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
) -> Iterator[Any]:
    """Start an OTel span (or a local correlation span) and sync trace/span ids.

    Always ensures a ``trace_id`` is bound so logs/audits correlate even when
    OTel is disabled.
    """
    ensure_trace_id()
    tracer = get_tracer()
    if tracer is None:
        token: Token[str | None] = set_span_id(new_span_id())
        try:
            yield None
        finally:
            reset_span_id(token)
        return

    with tracer.start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        sync_from_otel_span()
        try:
            yield span
        except Exception as exc:
            if record_exception:
                _mark_error(span, exc)
            raise


def reset_otel_for_tests() -> None:
    """Reset module configuration flag (unit tests only)."""
    global _CONFIGURED
    _CONFIGURED = False
    clear_trace_context()
