"""Enterprise observability events — every goal/plan/agent/tool/SQL/retry/…"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from observability.otel_setup import start_span
from utils.helpers import ensure_dirs
from utils.logging_config import get_logger

logger = get_logger(__name__)

EnterpriseEventType = Literal[
    "goal",
    "plan",
    "agent",
    "tool_call",
    "sql",
    "retry",
    "reflection",
    "replan",
    "memory_retrieval",
    "approval",
    "plugin",
    "graph_transition",
    "heartbeat",
    "dlq",
    "learning",
]


class EnterpriseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    event_type: EnterpriseEventType
    name: str
    session_id: str = ""
    tenant_id: str = "default"
    actor: str = ""
    agent: str = ""
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EnterpriseEventTracker:
    """Structured event stream integrated with logs + optional OTel spans."""

    def __init__(self, log_path: str | None = None) -> None:
        self.log_path = log_path
        if log_path:
            ensure_dirs(Path(log_path).parent)
        self._lock = threading.RLock()
        self._buffer: list[EnterpriseEvent] = []
        self._max_buffer = 1000

    def emit(
        self,
        event_type: EnterpriseEventType,
        name: str,
        *,
        session_id: str = "",
        tenant_id: str = "default",
        actor: str = "",
        agent: str = "",
        status: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> EnterpriseEvent:
        event = EnterpriseEvent(
            event_type=event_type,
            name=name,
            session_id=session_id,
            tenant_id=tenant_id,
            actor=actor,
            agent=agent,
            status=status,
            payload=payload or {},
        )
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self._max_buffer:
                self._buffer = self._buffer[-self._max_buffer :]

        logger.info(
            "enterprise_event type=%s name=%s agent=%s status=%s",
            event.event_type,
            event.name,
            event.agent,
            event.status,
            extra={"enterprise_event": event.model_dump()},
        )

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.model_dump(), default=str) + "\n")
            except Exception as exc:  # noqa: BLE001
                logger.debug("event log write failed: %s", exc)

        try:
            with start_span(
                f"sqlmind.event.{event_type}",
                attributes={
                    "sqlmind.event.name": name,
                    "sqlmind.event.agent": agent,
                    "sqlmind.session_id": session_id,
                    "sqlmind.tenant_id": tenant_id,
                },
            ):
                pass
        except Exception:  # noqa: BLE001
            pass

        return event

    def recent(
        self, *, event_type: EnterpriseEventType | None = None, limit: int = 50
    ) -> list[EnterpriseEvent]:
        with self._lock:
            items = list(self._buffer)
        if event_type:
            items = [e for e in items if e.event_type == event_type]
        return items[-limit:]

    def track_graph_transition(
        self, source: str, dest: str, *, session_id: str = ""
    ) -> EnterpriseEvent:
        return self.emit(
            "graph_transition",
            f"{source}→{dest}",
            session_id=session_id,
            agent=source,
            payload={"source": source, "dest": dest},
        )


_TRACKER: EnterpriseEventTracker | None = None


def get_event_tracker(log_path: str | None = None) -> EnterpriseEventTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = EnterpriseEventTracker(log_path=log_path)
    return _TRACKER
