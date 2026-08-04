"""Long-term workflow memory — successful/failed plans, preferences, query patterns.

Planner reads this while planning. Complements episodic Q→SQL memory.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}", re.I)
WorkflowKind = Literal[
    "successful_workflow",
    "failed_workflow",
    "user_preference",
    "frequent_query",
    "preferred_visualization",
    "successful_plan",
]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


class WorkflowMemoryStore:
    """SQLite store for long-horizon autonomy memory."""

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
                CREATE TABLE IF NOT EXISTS workflow_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    session_id TEXT,
                    database_name TEXT,
                    question TEXT,
                    goal_summary TEXT,
                    plan_json TEXT,
                    preference_key TEXT,
                    preference_value TEXT,
                    success INTEGER,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wf_kind ON workflow_memory(kind)"
            )

    def record(
        self,
        *,
        kind: WorkflowKind,
        question: str = "",
        goal_summary: str = "",
        plan: dict[str, Any] | None = None,
        preference_key: str = "",
        preference_value: str = "",
        success: bool | None = None,
        session_id: str = "",
        database_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO workflow_memory
                (kind, session_id, database_name, question, goal_summary, plan_json,
                 preference_key, preference_value, success, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    session_id,
                    database_name,
                    question,
                    goal_summary,
                    json.dumps(plan or {}, default=str),
                    preference_key,
                    preference_value,
                    None if success is None else (1 if success else 0),
                    json.dumps(metadata or {}, default=str),
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def record_plan_outcome(
        self,
        *,
        question: str,
        goal_summary: str,
        plan: dict[str, Any] | None,
        success: bool,
        session_id: str = "",
        database_name: str = "",
    ) -> None:
        kind: WorkflowKind = "successful_workflow" if success else "failed_workflow"
        self.record(
            kind=kind,
            question=question,
            goal_summary=goal_summary,
            plan=plan,
            success=success,
            session_id=session_id,
            database_name=database_name,
        )
        if success and plan:
            self.record(
                kind="successful_plan",
                question=question,
                goal_summary=goal_summary,
                plan=plan,
                success=True,
                session_id=session_id,
                database_name=database_name,
            )

    def record_preference(
        self,
        key: str,
        value: str,
        *,
        session_id: str = "",
        database_name: str = "",
    ) -> None:
        self.record(
            kind="user_preference",
            preference_key=key,
            preference_value=value,
            session_id=session_id,
            database_name=database_name,
        )

    def record_visualization_pref(
        self,
        chart_type: str,
        *,
        question: str = "",
        session_id: str = "",
        database_name: str = "",
    ) -> None:
        if not chart_type or chart_type == "none":
            return
        self.record(
            kind="preferred_visualization",
            question=question,
            preference_key="chart_type",
            preference_value=chart_type,
            session_id=session_id,
            database_name=database_name,
            success=True,
        )

    def retrieve_for_planner(
        self,
        question: str,
        *,
        database_name: str = "",
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        q_tokens = _tokens(question)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM workflow_memory
                WHERE (? = '' OR database_name = ? OR database_name = '')
                ORDER BY id DESC
                LIMIT 120
                """,
                (database_name, database_name),
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            d = dict(row)
            blob = " ".join(
                [
                    d.get("question") or "",
                    d.get("goal_summary") or "",
                    d.get("preference_key") or "",
                    d.get("preference_value") or "",
                ]
            )
            overlap = len(q_tokens & _tokens(blob))
            kind_boost = {
                "successful_plan": 2.0,
                "successful_workflow": 1.5,
                "preferred_visualization": 1.2,
                "user_preference": 1.0,
                "frequent_query": 1.3,
                "failed_workflow": 0.8,
            }.get(d.get("kind") or "", 1.0)
            score = overlap * kind_boost
            if d.get("kind") in {"user_preference", "preferred_visualization"}:
                score += 0.5
            if score > 0 or d.get("kind") == "user_preference":
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def format_for_prompt(
        self,
        question: str,
        *,
        database_name: str = "",
        limit: int = 6,
    ) -> str:
        rows = self.retrieve_for_planner(
            question, database_name=database_name, limit=limit
        )
        if not rows:
            return "(no long-term workflow memory)"
        lines: list[str] = []
        for r in rows:
            kind = r.get("kind")
            if kind in {"successful_plan", "successful_workflow", "failed_workflow"}:
                lines.append(
                    f"- [{kind}] Q: {(r.get('question') or '')[:120]} | "
                    f"goal: {(r.get('goal_summary') or '')[:100]} | "
                    f"success={r.get('success')}"
                )
            elif kind in {"user_preference", "preferred_visualization"}:
                lines.append(
                    f"- [{kind}] {r.get('preference_key')}={r.get('preference_value')}"
                )
            else:
                lines.append(f"- [{kind}] {(r.get('question') or '')[:140]}")
        return "\n".join(lines)
