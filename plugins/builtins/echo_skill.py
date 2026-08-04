"""Builtin example plugin package (optional demo exporter skill)."""

from __future__ import annotations

from typing import Any


def echo_skill(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sample skill entrypoint for plugin marketplace smoke tests."""
    return {"ok": True, "payload": payload or {}, "skill": "echo"}
