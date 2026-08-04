"""LangGraph checkpointer factory (MemorySaver + SQLite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from utils.helpers import ensure_dirs
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Keep strong refs so connections are not closed by GC / context-manager exit.
_OPEN_CHECKPOINT_CONNS: list[sqlite3.Connection] = []


def build_checkpointer(checkpoint_db_path: str | None = None) -> Any:
    """Return a durable SqliteSaver when possible, else in-memory MemorySaver.

    IMPORTANT: Do not use ``SqliteSaver.from_conn_string(...).__enter__()`` and
    discard the context manager. That context uses ``closing(conn)``, so when the
    generator is garbage-collected the SQLite connection is closed and LangGraph
    later fails with ``Cannot operate on a closed database``.
    """
    if not checkpoint_db_path:
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(checkpoint_db_path)
        ensure_dirs(path.parent)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        _OPEN_CHECKPOINT_CONNS.append(conn)
        saver = SqliteSaver(conn)
        saver.setup()
        logger.info("Using SqliteSaver at %s", path)
        return saver
    except Exception as exc:  # noqa: BLE001
        logger.warning("SqliteSaver unavailable (%s); using MemorySaver", exc)
        return MemorySaver()
