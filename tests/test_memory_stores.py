"""Episodic memory retrieval feeds distinct context into prompts."""

from __future__ import annotations

import gc

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver

from memory.checkpointing import build_checkpointer
from memory.episodic import EpisodicMemoryStore


def test_sqlite_checkpointer_survives_gc(tmp_path) -> None:
    """from_conn_string + discarded CM used to close the DB after GC."""
    cp = build_checkpointer(str(tmp_path / "ckpt.sqlite3"))
    assert not isinstance(cp, MemorySaver)
    config = {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
    config = cp.put(config, empty_checkpoint(), {}, {})
    gc.collect()
    config = cp.put(config, empty_checkpoint(), {}, {})
    assert cp.get_tuple(config) is not None


def test_episodic_retrieve_and_format(tmp_path) -> None:
    store = EpisodicMemoryStore(str(tmp_path / "ep.sqlite3"))
    store.record(
        question="Top customers by spend",
        sql="SELECT name, SUM(amount) FROM orders GROUP BY name",
        outcome="Alice leads",
        row_count=10,
        success=True,
        database_name="retail",
    )
    store.record(
        question="Products by category",
        sql="SELECT category, COUNT(*) FROM products GROUP BY category",
        outcome="Electronics dominant",
        row_count=5,
        success=True,
        database_name="retail",
    )
    hits = store.retrieve("customer spend ranking", database_name="retail")
    assert hits
    assert "customers" in hits[0]["question"].lower() or "spend" in hits[0]["question"].lower()
    text = store.format_for_prompt("customer spend", database_name="retail")
    assert "SQL:" in text
    assert "(no prior episodes)" not in text
