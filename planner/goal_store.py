"""Persistent goal lifecycle store — all goal states across sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from planner.goal_models import GoalLifecycleStatus, GoalTrackingRecord
from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)


class GoalStore:
    """SQLite persistence for GoalTrackingRecord lifecycle history."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        ensure_dirs(Path(db_path).parent)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    tenant_id TEXT,
                    workspace_id TEXT,
                    actor TEXT,
                    status TEXT NOT NULL,
                    title TEXT,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    reason TEXT,
                    payload_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_tenant ON goals(tenant_id, status)"
            )

    def upsert(self, record: GoalTrackingRecord) -> GoalTrackingRecord:
        record.estimate_completion()
        payload = json.dumps(record.to_state_dict(), default=str)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO goals
                (goal_id, session_id, tenant_id, workspace_id, actor, status, title,
                 record_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    tenant_id=excluded.tenant_id,
                    workspace_id=excluded.workspace_id,
                    actor=excluded.actor,
                    status=excluded.status,
                    title=excluded.title,
                    record_json=excluded.record_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.goal_id,
                    record.session_id,
                    record.tenant_id,
                    record.workspace_id,
                    record.actor,
                    record.status,
                    record.title,
                    payload,
                    record.created_at.isoformat()
                    if hasattr(record.created_at, "isoformat")
                    else str(record.created_at),
                    now,
                ),
            )
        return record

    def get(self, goal_id: str) -> GoalTrackingRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        if not row:
            return None
        return GoalTrackingRecord.model_validate(json.loads(row["record_json"]))

    def list_by_session(
        self, session_id: str, *, limit: int = 50
    ) -> list[GoalTrackingRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT record_json FROM goals
                WHERE session_id = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            GoalTrackingRecord.model_validate(json.loads(r["record_json"])) for r in rows
        ]

    def list_by_status(
        self,
        status: GoalLifecycleStatus,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[GoalTrackingRecord]:
        with self._connect() as conn:
            if tenant_id:
                rows = conn.execute(
                    """
                    SELECT record_json FROM goals
                    WHERE status = ? AND tenant_id = ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (status, tenant_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT record_json FROM goals
                    WHERE status = ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
        return [
            GoalTrackingRecord.model_validate(json.loads(r["record_json"])) for r in rows
        ]

    def record_event(
        self,
        goal_id: str,
        *,
        previous_status: str,
        new_status: str,
        reason: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO goal_events
                (goal_id, previous_status, new_status, reason, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    previous_status,
                    new_status,
                    reason,
                    json.dumps(payload or {}, default=str),
                    utc_now_iso(),
                ),
            )

    def transition(
        self,
        goal_id: str,
        new_status: GoalLifecycleStatus,
        *,
        reason: str = "",
    ) -> GoalTrackingRecord:
        record = self.get(goal_id)
        if record is None:
            raise KeyError(f"Unknown goal_id={goal_id}")
        prev = record.status
        record.transition(new_status, reason=reason)
        self.upsert(record)
        self.record_event(
            goal_id,
            previous_status=prev,
            new_status=new_status,
            reason=reason,
            payload=record.to_state_dict(),
        )
        return record
