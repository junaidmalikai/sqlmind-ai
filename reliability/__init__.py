"""Enterprise reliability — DLQ, circuit breaker, timeouts, heartbeat, recovery."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

CircuitState = Literal["closed", "open", "half_open"]


# ---------------------------------------------------------------------------
# Dead Letter Queue
# ---------------------------------------------------------------------------

class DeadLetterQueue:
    """Persist failed tasks/nodes for later recovery."""

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
                CREATE TABLE IF NOT EXISTS dead_letters (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    session_id TEXT,
                    node TEXT,
                    payload_json TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

    def enqueue(
        self,
        *,
        kind: str,
        error: str,
        payload: dict[str, Any] | None = None,
        session_id: str = "",
        node: str = "",
    ) -> str:
        item_id = f"dlq-{uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dead_letters
                (id, kind, session_id, node, payload_json, error, attempts, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, 'open', ?, ?)
                """,
                (
                    item_id,
                    kind,
                    session_id,
                    node,
                    json.dumps(payload or {}, default=str),
                    error,
                    now,
                    now,
                ),
            )
        logger.warning("DLQ enqueue id=%s kind=%s node=%s", item_id, kind, node)
        return item_id

    def list_open(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dead_letters WHERE status = 'open'
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_resolved(self, item_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE dead_letters SET status='resolved', updated_at=? WHERE id=?",
                (utc_now_iso(), item_id),
            )

    def bump_attempt(self, item_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dead_letters
                SET attempts = attempts + 1, updated_at = ?
                WHERE id = ?
                """,
                (utc_now_iso(), item_id),
            )


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_successes: int = 2
    state: CircuitState = "closed"
    failures: int = 0
    successes: int = 0
    opened_at: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def allow(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if time.time() - self.opened_at >= self.recovery_timeout:
                    self.state = "half_open"
                    self.successes = 0
                    return True
                return False
            return True  # half_open — probe

    def record_success(self) -> None:
        with self._lock:
            if self.state == "half_open":
                self.successes += 1
                if self.successes >= self.half_open_successes:
                    self.state = "closed"
                    self.failures = 0
            else:
                self.failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.time()


class CircuitBreakerRegistry:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                )
            return self._breakers[name]


# ---------------------------------------------------------------------------
# Timeout + Fallback
# ---------------------------------------------------------------------------

def run_with_timeout(
    fn: Callable[[], Any],
    *,
    timeout_seconds: float,
    fallback: Callable[[], Any] | None = None,
) -> Any:
    """Run callable with a soft timeout using a worker thread."""
    result: dict[str, Any] = {"value": None, "error": None, "done": False}

    def _target() -> None:
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            result["done"] = True

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if not result["done"]:
        logger.warning("Timeout after %.1fs — invoking fallback", timeout_seconds)
        if fallback is not None:
            return fallback()
        raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
    if result["error"] is not None:
        if fallback is not None:
            return fallback()
        raise result["error"]
    return result["value"]


# ---------------------------------------------------------------------------
# Heartbeat / Agent Health
# ---------------------------------------------------------------------------

@dataclass
class AgentHeartbeat:
    agent_id: str
    last_beat: float = field(default_factory=time.time)
    status: Literal["healthy", "stale", "dead"] = "healthy"
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentHealthMonitor:
    def __init__(self, *, stale_after: float = 60.0, dead_after: float = 300.0) -> None:
        self.stale_after = stale_after
        self.dead_after = dead_after
        self._agents: dict[str, AgentHeartbeat] = {}
        self._lock = threading.RLock()

    def beat(self, agent_id: str, **metadata: Any) -> None:
        with self._lock:
            hb = self._agents.get(agent_id) or AgentHeartbeat(agent_id=agent_id)
            hb.last_beat = time.time()
            hb.status = "healthy"
            hb.metadata.update(metadata)
            self._agents[agent_id] = hb

    def refresh(self) -> list[AgentHeartbeat]:
        now = time.time()
        with self._lock:
            for hb in self._agents.values():
                age = now - hb.last_beat
                if age >= self.dead_after:
                    hb.status = "dead"
                elif age >= self.stale_after:
                    hb.status = "stale"
                else:
                    hb.status = "healthy"
            return list(self._agents.values())


# ---------------------------------------------------------------------------
# Task / Node / Checkpoint Recovery
# ---------------------------------------------------------------------------

class RecoveryManager:
    """Coordinates DLQ + circuit breakers for task/node/checkpoint recovery."""

    def __init__(self, dlq: DeadLetterQueue, breakers: CircuitBreakerRegistry | None = None) -> None:
        self.dlq = dlq
        self.breakers = breakers or CircuitBreakerRegistry()

    def on_node_failure(
        self,
        node: str,
        error: str,
        *,
        session_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> str:
        br = self.breakers.get(node)
        br.record_failure()
        item_id = self.dlq.enqueue(
            kind="node_failure",
            error=error,
            payload=payload,
            session_id=session_id,
            node=node,
        )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_dlq("node_failure")
            get_metrics().observe_circuit(node, br.state)
        except Exception:  # noqa: BLE001
            pass
        return item_id

    def on_task_failure(
        self,
        task_id: str,
        error: str,
        *,
        session_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> str:
        return self.dlq.enqueue(
            kind="task_failure",
            error=error,
            payload={**(payload or {}), "task_id": task_id},
            session_id=session_id,
            node=task_id,
        )

    def on_checkpoint_recovery(self, session_id: str, detail: dict[str, Any] | None = None) -> str:
        return self.dlq.enqueue(
            kind="checkpoint_recovery",
            error="checkpoint_resume_required",
            payload=detail or {},
            session_id=session_id,
            node="checkpointer",
        )

    def fallback_strategy(self, node: str) -> str:
        """Return a safe fallback next_agent when circuit is open or node fails.

        Prefers recovery_graph / recovery_controller when enterprise features are on.
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
            if getattr(settings, "enterprise_enabled", True) and getattr(
                settings, "enterprise_subgraphs_enabled", True
            ):
                return "recovery_graph"
            if getattr(settings, "enterprise_enabled", True) and getattr(
                settings, "recovery_controller_enabled", True
            ):
                return "recovery_controller"
        except Exception:  # noqa: BLE001
            pass
        br = self.breakers.get(node)
        if not br.allow():
            if node == "sql_agent":
                return "retry_agent"
            if node.startswith("visualization") or node.startswith("insight"):
                return "supervisor"
            return "fail"
        if node == "sql_agent":
            return "retry_agent"
        return "retry_agent"


# Process singletons for convenience
_DLQ: DeadLetterQueue | None = None
_BREAKERS: CircuitBreakerRegistry | None = None
_HEALTH = AgentHealthMonitor()


def get_dlq(db_path: str = "data/sqlmind_dlq.sqlite3") -> DeadLetterQueue:
    global _DLQ
    if _DLQ is None:
        _DLQ = DeadLetterQueue(db_path)
    return _DLQ


def get_breakers(
    *,
    failure_threshold: int | None = None,
    recovery_timeout: float | None = None,
) -> CircuitBreakerRegistry:
    global _BREAKERS
    if _BREAKERS is None:
        _BREAKERS = CircuitBreakerRegistry(
            failure_threshold=failure_threshold or 5,
            recovery_timeout=recovery_timeout or 30.0,
        )
    return _BREAKERS


def get_health_monitor() -> AgentHealthMonitor:
    return _HEALTH
