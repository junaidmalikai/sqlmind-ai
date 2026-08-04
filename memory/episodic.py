"""Episodic memory — persist successful Q→SQL→outcome and retrieve into LLM context."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}", re.I)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


class EpisodicMemoryStore:
    """SQLite episodic store with simple lexical retrieval (injected into prompts)."""

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
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    sql TEXT,
                    outcome TEXT,
                    row_count INTEGER,
                    success INTEGER,
                    database_name TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        question: str,
        sql: str,
        outcome: str,
        row_count: int = 0,
        success: bool = True,
        session_id: str = "",
        database_name: str = "",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodes
                (session_id, question, sql, outcome, row_count, success, database_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    sql,
                    outcome,
                    row_count,
                    1 if success else 0,
                    database_name,
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def retrieve(self, question: str, *, limit: int = 5, database_name: str = "") -> list[dict[str, Any]]:
        q_tokens = _tokens(question)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM episodes
                WHERE success = 1
                  AND (? = '' OR database_name = ? OR database_name = '')
                ORDER BY id DESC
                LIMIT 100
                """,
                (database_name, database_name),
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            d = dict(row)
            overlap = len(q_tokens & _tokens(d.get("question", "")))
            if overlap == 0 and q_tokens:
                continue
            score = float(overlap) + 0.01 * min(d.get("id", 0), 100)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def format_for_prompt(self, question: str, *, limit: int = 3, database_name: str = "") -> str:
        hits = self.retrieve(question, limit=limit, database_name=database_name)
        if not hits:
            return "(no prior episodes)"
        lines = []
        for h in hits:
            lines.append(
                f"- Q: {h.get('question')}\n  SQL: {(h.get('sql') or '')[:240]}\n"
                f"  Outcome: {(h.get('outcome') or '')[:160]}"
            )
        return "\n".join(lines)
