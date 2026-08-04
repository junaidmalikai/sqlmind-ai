"""Enterprise queue reliability — Retry / Delayed / DLQ / Poison / Replay.

Extends the existing DeadLetterQueue without replacing it. Adds retry policies,
exponential backoff, delayed delivery, poison detection, and DLQ replay.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

QueueKind = Literal["retry", "delayed", "dead", "poison"]


@dataclass
class RetryPolicy:
    max_retries: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    multiplier: float = 2.0
    poison_after: int = 5

    def delay_for_attempt(self, attempt: int) -> float:
        delay = self.base_delay_seconds * (self.multiplier ** max(0, attempt - 1))
        return min(delay, self.max_delay_seconds)


class EnterpriseQueue:
    """Unified retry / delayed / dead-letter / poison message plane."""

    def __init__(
        self,
        db_path: str = "data/sqlmind_enterprise_queue.sqlite3",
        *,
        policy: RetryPolicy | None = None,
    ) -> None:
        self.db_path = db_path
        self.policy = policy or RetryPolicy()
        ensure_dirs(Path(db_path).parent)
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_messages (
                    id TEXT PRIMARY KEY,
                    queue_kind TEXT NOT NULL,
                    topic TEXT,
                    payload_json TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    max_retries INTEGER,
                    available_at REAL,
                    status TEXT DEFAULT 'open',
                    session_id TEXT,
                    node TEXT,
                    correlation_id TEXT,
                    failure_metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eq_kind_avail "
                "ON queue_messages(queue_kind, status, available_at)"
            )

    def enqueue_retry(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        error: str = "",
        attempts: int = 0,
        session_id: str = "",
        node: str = "",
        correlation_id: str = "",
        failure_metadata: dict[str, Any] | None = None,
    ) -> str:
        attempts = max(attempts, 1)
        if attempts >= self.policy.poison_after:
            return self.enqueue_poison(
                topic=topic,
                payload=payload,
                error=error or "poison_threshold",
                attempts=attempts,
                session_id=session_id,
                node=node,
                correlation_id=correlation_id,
                failure_metadata=failure_metadata,
            )
        if attempts > self.policy.max_retries:
            return self.enqueue_dead(
                topic=topic,
                payload=payload,
                error=error or "max_retries_exceeded",
                attempts=attempts,
                session_id=session_id,
                node=node,
                correlation_id=correlation_id,
                failure_metadata=failure_metadata,
            )
        delay = self.policy.delay_for_attempt(attempts)
        return self._insert(
            queue_kind="delayed" if delay > 0 else "retry",
            topic=topic,
            payload=payload,
            error=error,
            attempts=attempts,
            available_at=time.time() + delay,
            session_id=session_id,
            node=node,
            correlation_id=correlation_id,
            failure_metadata=failure_metadata,
        )

    def enqueue_delayed(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        delay_seconds: float,
        session_id: str = "",
        node: str = "",
        correlation_id: str = "",
    ) -> str:
        return self._insert(
            queue_kind="delayed",
            topic=topic,
            payload=payload,
            error="",
            attempts=0,
            available_at=time.time() + max(0.0, delay_seconds),
            session_id=session_id,
            node=node,
            correlation_id=correlation_id,
        )

    def enqueue_dead(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        error: str,
        attempts: int = 0,
        session_id: str = "",
        node: str = "",
        correlation_id: str = "",
        failure_metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = self._insert(
            queue_kind="dead",
            topic=topic,
            payload=payload,
            error=error,
            attempts=attempts,
            available_at=time.time(),
            session_id=session_id,
            node=node,
            correlation_id=correlation_id,
            failure_metadata=failure_metadata,
        )
        # Mirror into legacy DLQ for existing monitoring
        try:
            from reliability import get_dlq

            get_dlq().enqueue(
                kind=f"enterprise_dead:{topic}",
                error=error,
                payload={**(payload or {}), "enterprise_queue_id": item_id},
                session_id=session_id,
                node=node,
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_dlq("enterprise_dead")
        except Exception:  # noqa: BLE001
            pass
        return item_id

    def enqueue_poison(
        self,
        *,
        topic: str,
        payload: dict[str, Any],
        error: str,
        attempts: int = 0,
        session_id: str = "",
        node: str = "",
        correlation_id: str = "",
        failure_metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = {**(failure_metadata or {}), "poison": True}
        item_id = self._insert(
            queue_kind="poison",
            topic=topic,
            payload=payload,
            error=error,
            attempts=attempts,
            available_at=time.time(),
            session_id=session_id,
            node=node,
            correlation_id=correlation_id,
            failure_metadata=meta,
        )
        logger.error("Poison message detected id=%s topic=%s", item_id, topic)
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_dlq("poison")
        except Exception:  # noqa: BLE001
            pass
        return item_id

    def _insert(
        self,
        *,
        queue_kind: QueueKind,
        topic: str,
        payload: dict[str, Any],
        error: str,
        attempts: int,
        available_at: float,
        session_id: str = "",
        node: str = "",
        correlation_id: str = "",
        failure_metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = f"eq-{uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queue_messages (
                    id, queue_kind, topic, payload_json, error, attempts,
                    max_retries, available_at, status, session_id, node,
                    correlation_id, failure_metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    queue_kind,
                    topic,
                    json.dumps(payload or {}, default=str),
                    error,
                    attempts,
                    self.policy.max_retries,
                    available_at,
                    session_id,
                    node,
                    correlation_id or uuid4().hex[:12],
                    json.dumps(failure_metadata or {}, default=str),
                    now,
                    now,
                ),
            )
        return item_id

    def claim_ready(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Claim messages from retry/delayed queues whose available_at has passed."""
        now = time.time()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM queue_messages
                WHERE status='open' AND queue_kind IN ('retry', 'delayed')
                  AND available_at <= ?
                ORDER BY available_at ASC LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                conn.execute(
                    "UPDATE queue_messages SET status='claimed', updated_at=? WHERE id=?",
                    (utc_now_iso(), r["id"]),
                )
                out.append(self._row_dict(r))
            return out

    def mark_claimed(
        self,
        item_id: str,
        *,
        status: str = "completed",
        error: str = "",
    ) -> None:
        """Finalize a claimed retry/delayed message."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_messages
                SET status=?, error=COALESCE(NULLIF(?, ''), error), updated_at=?
                WHERE id=?
                """,
                (status, error[:500], utc_now_iso(), item_id),
            )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_queue(status, "claimed")
        except Exception:  # noqa: BLE001
            pass

    def list_dlq(self, *, include_poison: bool = True, limit: int = 50) -> list[dict[str, Any]]:
        kinds = ("dead", "poison") if include_poison else ("dead",)
        placeholders = ",".join("?" for _ in kinds)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM queue_messages
                WHERE queue_kind IN ({placeholders}) AND status='open'
                ORDER BY created_at DESC LIMIT ?
                """,
                (*kinds, limit),
            ).fetchall()
        return [self._row_dict(r) for r in rows]

    def replay_dlq(
        self,
        item_id: str,
        *,
        handler: Callable[[dict[str, Any]], Any] | None = None,
        to_retry: bool = True,
    ) -> dict[str, Any]:
        """Replay a dead/poison message — optionally via handler, else re-queue as retry."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM queue_messages WHERE id=?", (item_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown queue message: {item_id}")
            data = self._row_dict(row)
            if handler is not None:
                try:
                    result = handler(data)
                    conn.execute(
                        "UPDATE queue_messages SET status='replayed', updated_at=? WHERE id=?",
                        (utc_now_iso(), item_id),
                    )
                    return {"id": item_id, "status": "replayed", "result": result}
                except Exception as exc:  # noqa: BLE001
                    conn.execute(
                        """
                        UPDATE queue_messages SET attempts=attempts+1, error=?, updated_at=?
                        WHERE id=?
                        """,
                        (str(exc), utc_now_iso(), item_id),
                    )
                    raise
            if to_retry:
                conn.execute(
                    "UPDATE queue_messages SET status='replayed', updated_at=? WHERE id=?",
                    (utc_now_iso(), item_id),
                )
        if to_retry:
            new_id = self.enqueue_retry(
                topic=data.get("topic") or "replay",
                payload=data.get("payload") or {},
                error="replay_from_dlq",
                attempts=0,
                session_id=data.get("session_id") or "",
                node=data.get("node") or "",
                correlation_id=data.get("correlation_id") or "",
                failure_metadata={"replayed_from": item_id},
            )
            return {"id": item_id, "status": "requeued", "retry_id": new_id}
        return {"id": item_id, "status": "noop"}

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT queue_kind, status, COUNT(*) AS c
                FROM queue_messages GROUP BY queue_kind, status
                """
            ).fetchall()
        out: dict[str, Any] = {"by_kind_status": {}, "open_dead": 0, "open_poison": 0, "ready": 0}
        now = time.time()
        for r in rows:
            key = f"{r['queue_kind']}:{r['status']}"
            out["by_kind_status"][key] = int(r["c"])
            if r["queue_kind"] == "dead" and r["status"] == "open":
                out["open_dead"] = int(r["c"])
            if r["queue_kind"] == "poison" and r["status"] == "open":
                out["open_poison"] = int(r["c"])
        with self._connect() as conn:
            ready = conn.execute(
                """
                SELECT COUNT(*) AS c FROM queue_messages
                WHERE status='open' AND queue_kind IN ('retry','delayed')
                  AND available_at <= ?
                """,
                (now,),
            ).fetchone()
            out["ready"] = int(ready["c"] if ready else 0)
        return out

    @staticmethod
    def _row_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d.pop("payload_json", None) or "{}")
        d["failure_metadata"] = json.loads(d.pop("failure_metadata_json", None) or "{}")
        return d


_EQ: EnterpriseQueue | None = None
_EQ_LOCK = threading.Lock()


def get_enterprise_queue(
    db_path: str = "data/sqlmind_enterprise_queue.sqlite3",
) -> EnterpriseQueue:
    global _EQ
    with _EQ_LOCK:
        if _EQ is None:
            _EQ = EnterpriseQueue(db_path)
        return _EQ


_CLAIM_STOP: threading.Event | None = None
_CLAIM_THREAD: threading.Thread | None = None


def process_claimed_message(
    item: dict[str, Any],
    *,
    handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> Any:
    """Dispatch a claimed queue item by topic to a handler registry."""
    handlers = handlers or {}
    topic = str(item.get("topic") or "")
    handler = handlers.get(topic) or handlers.get("*")
    if handler is None:
        logger.debug("No claim handler for topic=%s id=%s", topic, item.get("id"))
        return {"skipped": True, "topic": topic}
    return handler(item)


def start_claim_consumer(
    queue: EnterpriseQueue | None = None,
    *,
    handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    poll_interval: float = 1.0,
    limit: int = 5,
) -> threading.Thread:
    """Background loop: claim_ready → handlers → mark_claimed.

    Idempotent — reuses a single process-wide thread.
    """
    global _CLAIM_STOP, _CLAIM_THREAD
    q = queue or get_enterprise_queue()
    if _CLAIM_THREAD is not None and _CLAIM_THREAD.is_alive():
        return _CLAIM_THREAD
    stop = threading.Event()
    _CLAIM_STOP = stop

    def _loop() -> None:
        logger.info("EnterpriseQueue claim consumer started")
        while not stop.is_set():
            try:
                items = q.claim_ready(limit=limit)
                for item in items:
                    item_id = str(item.get("id") or "")
                    try:
                        process_claimed_message(item, handlers=handlers)
                        if item_id:
                            q.mark_claimed(item_id, status="completed")
                        try:
                            from observability.metrics import get_metrics

                            get_metrics().observe_queue("claim_ok", item.get("topic") or "")
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Claim handler failed id=%s: %s", item_id, exc
                        )
                        if item_id:
                            q.mark_claimed(
                                item_id, status="failed", error=str(exc)
                            )
                        try:
                            from observability.metrics import get_metrics

                            get_metrics().observe_queue(
                                "claim_fail", item.get("topic") or ""
                            )
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                logger.debug("Claim loop error: %s", exc)
            stop.wait(poll_interval)
        logger.info("EnterpriseQueue claim consumer stopped")

    thread = threading.Thread(
        target=_loop, name="sqlmind-enterprise-claim", daemon=True
    )
    _CLAIM_THREAD = thread
    thread.start()
    return thread


def stop_claim_consumer() -> None:
    global _CLAIM_STOP
    if _CLAIM_STOP is not None:
        _CLAIM_STOP.set()
