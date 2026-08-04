"""Request / session correlation context for logs and audit trails."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class RequestContext:
    """Per-request identity attached to logs and audit events."""

    request_id: str = field(default_factory=lambda: uuid4().hex)
    session_id: str = "default"
    actor: str = "anonymous"
    tenant_id: str = "default"
    database: str = ""
    question: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "actor": self.actor,
            "tenant_id": self.tenant_id,
            "database": self.database,
            "question": self.question[:200] if self.question else "",
        }


_CTX: ContextVar[RequestContext | None] = ContextVar("sqlmind_request_ctx", default=None)


def set_request_context(ctx: RequestContext) -> None:
    _CTX.set(ctx)


def get_request_context() -> RequestContext | None:
    return _CTX.get()


def clear_request_context() -> None:
    _CTX.set(None)
