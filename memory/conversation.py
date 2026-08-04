"""Conversation memory and query history persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    role: str  # user | assistant | system
    content: str
    timestamp: str = field(default_factory=utc_now_iso)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryHistoryItem:
    question: str
    sql: str
    execution_time: float
    timestamp: str
    database: str
    row_count: int = 0
    success: bool = True
    exports: dict[str, str] = field(default_factory=dict)
    insights: str = ""
    chart_type: str = ""


class ConversationMemory:
    """In-session conversation buffer with optional SQLite persistence."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self.messages: list[ChatMessage] = []

    def add(self, role: str, content: str, **meta: Any) -> None:
        self.messages.append(ChatMessage(role=role, content=content, meta=meta))
        # Keep last N user/assistant pairs roughly
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-(self.max_turns * 2) :]

    def as_dicts(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()


class HistoryStore:
    """SQLite-backed query history, bookmarks, and saved queries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        ensure_dirs(Path(db_path).parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    sql TEXT,
                    execution_time REAL,
                    timestamp TEXT NOT NULL,
                    database_name TEXT,
                    row_count INTEGER,
                    success INTEGER,
                    exports_json TEXT,
                    insights TEXT,
                    chart_type TEXT
                );

                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    sql TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS saved_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    sql TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_history(self, item: QueryHistoryItem) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO query_history
                (question, sql, execution_time, timestamp, database_name,
                 row_count, success, exports_json, insights, chart_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.question,
                    item.sql,
                    item.execution_time,
                    item.timestamp,
                    item.database,
                    item.row_count,
                    1 if item.success else 0,
                    json.dumps(item.exports),
                    item.insights,
                    item.chart_type,
                ),
            )
            return int(cur.lastrowid)

    def list_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            d["exports"] = json.loads(d.pop("exports_json") or "{}")
            d["success"] = bool(d["success"])
            result.append(d)
        return result

    def delete_history(self, history_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM query_history WHERE id = ?", (history_id,))

    def add_bookmark(self, title: str, question: str, sql: str = "") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO bookmarks (title, question, sql, created_at) VALUES (?, ?, ?, ?)",
                (title, question, sql, utc_now_iso()),
            )
            return int(cur.lastrowid)

    def list_bookmarks(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bookmarks ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_bookmark(self, bookmark_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))

    def save_query(self, name: str, question: str, sql: str = "") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO saved_queries (name, question, sql, created_at) VALUES (?, ?, ?, ?)",
                (name, question, sql, utc_now_iso()),
            )
            return int(cur.lastrowid)

    def list_saved_queries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM saved_queries ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_saved_query(self, query_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM saved_queries WHERE id = ?", (query_id,))

    def save_connection(self, name: str, config: dict[str, Any]) -> None:
        # Never persist raw passwords in plain text in production; store redacted.
        safe = {**config, "password": "***" if config.get("password") else ""}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO connections (name, config_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    config_json=excluded.config_json,
                    created_at=excluded.created_at
                """,
                (name, json.dumps(safe), utc_now_iso()),
            )

    def list_connections(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM connections ORDER BY name"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            out.append(d)
        return out
