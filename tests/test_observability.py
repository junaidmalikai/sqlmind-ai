"""Unit tests for observability helpers (no network required)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.settings import Settings
from observability.langsmith_setup import (
    configure_langsmith,
    langsmith_tags,
    reset_langsmith_for_tests,
)
from observability.otel_setup import configure_otel, reset_otel_for_tests, start_span
from observability.tracing import (
    bind_trace_context,
    clear_trace_context,
    correlation_fields,
    ensure_trace_id,
    get_trace_id,
    run_metadata,
)
from utils.logging_config import setup_logging, write_audit_event


@pytest.fixture(autouse=True)
def _reset_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_langsmith_for_tests()
    reset_otel_for_tests()
    clear_trace_context()
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    yield
    reset_langsmith_for_tests()
    reset_otel_for_tests()
    clear_trace_context()


def test_ensure_trace_id_is_stable_within_context() -> None:
    clear_trace_context()
    first = ensure_trace_id()
    second = ensure_trace_id()
    assert first == second
    assert len(first) == 32


def test_bind_trace_context_sets_and_restores() -> None:
    clear_trace_context()
    with bind_trace_context("abc123def456abc123def456abc123de") as tid:
        assert tid == "abc123def456abc123def456abc123de"
        assert get_trace_id() == tid
    assert get_trace_id() is None


def test_correlation_fields_include_trace() -> None:
    with bind_trace_context("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
        fields = correlation_fields()
    assert fields["trace_id"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert "span_id" in fields


def test_run_metadata_contains_session_and_trace() -> None:
    clear_trace_context()
    meta = run_metadata(
        session_id="sess-1",
        provider="ollama",
        dialect="sqlite",
        database="retail",
        tenant_id="t1",
        actor="tester",
    )
    assert meta["session_id"] == "sess-1"
    assert meta["provider"] == "ollama"
    assert meta["dialect"] == "sqlite"
    assert len(meta["trace_id"]) == 32


def test_langsmith_tags() -> None:
    tags = langsmith_tags(provider="openai", dialect="postgres", extra=["canary"])
    assert "sqlmind" in tags
    assert "provider:openai" in tags
    assert "dialect:postgres" in tags
    assert "canary" in tags


def test_configure_langsmith_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
    settings = Settings(
        langchain_api_key="test-key-not-real",
        langchain_project="SQLMind-Test",
        langchain_tracing_v2=True,
    )
    configure_langsmith(settings)
    import os

    assert os.environ.get("LANGCHAIN_API_KEY") == "test-key-not-real"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_PROJECT") == "SQLMind-Test"


def test_configure_langsmith_noop_without_key() -> None:
    settings = Settings(langchain_api_key="")
    configure_langsmith(settings)
    # Should not raise; tracing remains unset by this call
    assert True


def test_configure_otel_disabled_is_noop() -> None:
    settings = Settings(otel_enabled=False)
    configure_otel(settings)
    # start_span still works without OTel provider (local correlation)
    with start_span("sqlmind.test.noop") as span:
        assert span is None
        assert get_trace_id() is not None


def test_configure_otel_console_exporter() -> None:
    pytest.importorskip("opentelemetry.sdk")
    reset_otel_for_tests()
    settings = Settings(
        otel_enabled=True,
        otel_exporter="console",
        otel_service_name="SQLMind-Test",
    )
    configure_otel(settings)
    with start_span(
        "sqlmind.test.span",
        attributes={"sqlmind.unit_test": True},
    ) as span:
        assert span is not None
        assert get_trace_id() is not None


def test_log_formatter_includes_trace_id(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    setup_logging("INFO")

    with bind_trace_context("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"):
        logging.getLogger("sqlmind.test.obs").info("hello observability")

    captured = capsys.readouterr()
    assert "trace=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in captured.out
    assert "hello observability" in captured.out


def test_audit_event_includes_trace_id(tmp_path: Path) -> None:
    audit = tmp_path / "audit.log"
    with bind_trace_context("cccccccccccccccccccccccccccccccc"):
        write_audit_event("unit_test", {"ok": True}, str(audit))
    line = audit.read_text(encoding="utf-8").strip()
    assert "cccccccccccccccccccccccccccccccc" in line
    assert '"event": "unit_test"' in line or '"event":"unit_test"' in line
