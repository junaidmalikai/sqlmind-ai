"""Tests for request context and audit enrichment."""

from __future__ import annotations

import json
from pathlib import Path

from core.context import RequestContext, clear_request_context, get_request_context, set_request_context
from utils.logging_config import write_audit_event


def test_request_context_roundtrip() -> None:
    clear_request_context()
    assert get_request_context() is None
    ctx = RequestContext(session_id="s1", actor="alice", tenant_id="t1", database="retail")
    set_request_context(ctx)
    got = get_request_context()
    assert got is not None
    assert got.actor == "alice"
    assert got.as_dict()["tenant_id"] == "t1"
    clear_request_context()


def test_audit_includes_context(tmp_path: Path) -> None:
    path = tmp_path / "audit.log"
    set_request_context(
        RequestContext(session_id="sess", actor="bob", tenant_id="acme", question="top cities")
    )
    try:
        write_audit_event("sql_executed", {"sql": "SELECT 1", "rows": 1}, str(path))
    finally:
        clear_request_context()
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "sql_executed"
    assert record["actor"] == "bob"
    assert record["tenant_id"] == "acme"
    assert record["session_id"] == "sess"
