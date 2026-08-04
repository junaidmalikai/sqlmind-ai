"""Distributed execution plane — Task Queue, Workers, Leases, Locks.

In-process default with adapter stubs for Celery / Ray / Temporal / K8s Jobs.
Does not replace the LangGraph Execution Coordinator — it sits underneath as
an optional task execution backend.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

TaskStatus = Literal[
    "pending",
    "leased",
    "running",
    "completed",
    "failed",
    "cancelled",
    "retry",
    "dead",
]


class BackendKind(str, Enum):
    LOCAL = "local"
    CELERY = "celery"
    RAY = "ray"
    TEMPORAL = "temporal"
    K8S = "kubernetes"


@dataclass
class DistributedTask:
    task_id: str
    kind: str
    payload: dict[str, Any]
    status: TaskStatus = "pending"
    owner: str = ""
    lease_until: float = 0.0
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "status": self.status,
            "owner": self.owner,
            "lease_until": self.lease_until,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
        }


@dataclass
class WorkerInfo:
    worker_id: str
    backend: str = BackendKind.LOCAL.value
    last_heartbeat: float = field(default_factory=time.time)
    status: Literal["healthy", "stale", "dead"] = "healthy"
    leased_tasks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


class DistributedLock:
    """SQLite-backed advisory lock for cross-process future compatibility."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        ensure_dirs(Path(db_path).parent)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS distributed_locks (
                    lock_name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at TEXT
                )
                """
            )

    def acquire(
        self, name: str, owner: str, *, ttl_seconds: float = 30.0
    ) -> bool:
        now = time.time()
        expires = now + ttl_seconds
        with self._connect() as conn:
            row = conn.execute(
                "SELECT owner, expires_at FROM distributed_locks WHERE lock_name=?",
                (name,),
            ).fetchone()
            if row and float(row["expires_at"]) > now and row["owner"] != owner:
                return False
            conn.execute(
                """
                INSERT INTO distributed_locks (lock_name, owner, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner=excluded.owner,
                    expires_at=excluded.expires_at
                """,
                (name, owner, expires, utc_now_iso()),
            )
        return True

    def release(self, name: str, owner: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM distributed_locks WHERE lock_name=? AND owner=?",
                (name, owner),
            )


class TaskQueue:
    """Priority task queue with lease-based ownership and retry."""

    def __init__(self, db_path: str = "data/sqlmind_task_queue.sqlite3") -> None:
        self.db_path = db_path
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
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT,
                    status TEXT,
                    owner TEXT,
                    lease_until REAL,
                    attempts INTEGER,
                    max_attempts INTEGER,
                    priority INTEGER,
                    created_at REAL,
                    updated_at REAL,
                    error TEXT,
                    result_json TEXT,
                    session_id TEXT,
                    correlation_id TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status_pri "
                "ON tasks(status, priority DESC, created_at)"
            )

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 5,
        max_attempts: int = 3,
        session_id: str = "",
        correlation_id: str = "",
        task_id: str | None = None,
    ) -> DistributedTask:
        task = DistributedTask(
            task_id=task_id or f"task-{uuid4().hex[:12]}",
            kind=kind,
            payload=payload or {},
            priority=priority,
            max_attempts=max_attempts,
            session_id=session_id,
            correlation_id=correlation_id or uuid4().hex[:12],
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, kind, payload_json, status, owner, lease_until,
                    attempts, max_attempts, priority, created_at, updated_at,
                    error, result_json, session_id, correlation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.kind,
                    json.dumps(task.payload, default=str),
                    task.status,
                    "",
                    0.0,
                    0,
                    task.max_attempts,
                    task.priority,
                    task.created_at,
                    task.updated_at,
                    "",
                    "{}",
                    task.session_id,
                    task.correlation_id,
                ),
            )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_queue("enqueue", kind)
        except Exception:  # noqa: BLE001
            pass
        return task

    def lease(
        self,
        worker_id: str,
        *,
        kinds: list[str] | None = None,
        lease_seconds: float = 60.0,
    ) -> DistributedTask | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            # Reclaim expired leases
            conn.execute(
                """
                UPDATE tasks SET status='pending', owner='', lease_until=0
                WHERE status='leased' AND lease_until < ?
                """,
                (now,),
            )
            sql = """
                SELECT * FROM tasks
                WHERE status IN ('pending', 'retry')
            """
            params: list[Any] = []
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                sql += f" AND kind IN ({placeholders})"
                params.extend(kinds)
            sql += " ORDER BY priority DESC, created_at ASC LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if not row:
                return None
            lease_until = now + lease_seconds
            conn.execute(
                """
                UPDATE tasks SET status='leased', owner=?, lease_until=?,
                    updated_at=?, attempts=attempts+1
                WHERE task_id=?
                """,
                (worker_id, lease_until, now, row["task_id"]),
            )
            return self._row_to_task(dict(row), owner=worker_id, lease_until=lease_until)

    def complete(self, task_id: str, result: dict[str, Any] | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks SET status='completed', result_json=?, updated_at=?,
                    owner='', lease_until=0
                WHERE task_id=?
                """,
                (json.dumps(result or {}, default=str), time.time(), task_id),
            )

    def fail(
        self,
        task_id: str,
        error: str,
        *,
        retry: bool = True,
        delay_seconds: float = 0.0,
    ) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if not row:
                return
            attempts = int(row["attempts"])
            max_attempts = int(row["max_attempts"])
            now = time.time()
            if retry and attempts < max_attempts:
                status = "retry"
                lease_until = now + delay_seconds
            else:
                status = "dead"
                lease_until = 0.0
                try:
                    from reliability import get_dlq

                    get_dlq().enqueue(
                        kind="distributed_task_dead",
                        error=error,
                        payload={"task_id": task_id},
                        node="distributed_queue",
                    )
                except Exception:  # noqa: BLE001
                    pass
            conn.execute(
                """
                UPDATE tasks SET status=?, error=?, updated_at=?,
                    owner='', lease_until=?
                WHERE task_id=?
                """,
                (status, error, now, lease_until, task_id),
            )

    def cancel(self, task_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE tasks SET status='cancelled', updated_at=?, owner='', lease_until=0
                WHERE task_id=? AND status NOT IN ('completed', 'cancelled')
                """,
                (time.time(), task_id),
            )

    def queue_length(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS c FROM tasks GROUP BY status"
            ).fetchall()
        return {str(r["status"]): int(r["c"]) for r in rows}

    def list_tasks(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_task(dict(r)).to_dict() for r in rows]

    @staticmethod
    def _row_to_task(
        row: dict[str, Any],
        *,
        owner: str | None = None,
        lease_until: float | None = None,
    ) -> DistributedTask:
        return DistributedTask(
            task_id=row["task_id"],
            kind=row["kind"],
            payload=json.loads(row.get("payload_json") or "{}"),
            status=row.get("status") or "pending",  # type: ignore[arg-type]
            owner=owner if owner is not None else (row.get("owner") or ""),
            lease_until=lease_until if lease_until is not None else float(row.get("lease_until") or 0),
            attempts=int(row.get("attempts") or 0),
            max_attempts=int(row.get("max_attempts") or 3),
            priority=int(row.get("priority") or 5),
            created_at=float(row.get("created_at") or 0),
            updated_at=float(row.get("updated_at") or 0),
            error=row.get("error") or "",
            result=json.loads(row.get("result_json") or "{}"),
            session_id=row.get("session_id") or "",
            correlation_id=row.get("correlation_id") or "",
        )


class WorkerRegistry:
    """Worker registration, heartbeat, and health."""

    def __init__(self, *, stale_after: float = 30.0, dead_after: float = 120.0) -> None:
        self.stale_after = stale_after
        self.dead_after = dead_after
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = threading.RLock()

    def register(
        self,
        worker_id: str | None = None,
        *,
        backend: str = BackendKind.LOCAL.value,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerInfo:
        wid = worker_id or f"worker-{uuid4().hex[:8]}"
        info = WorkerInfo(worker_id=wid, backend=backend, metadata=metadata or {})
        with self._lock:
            self._workers[wid] = info
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_worker("register", wid)
        except Exception:  # noqa: BLE001
            pass
        return info

    def heartbeat(self, worker_id: str, **metadata: Any) -> None:
        with self._lock:
            w = self._workers.get(worker_id)
            if w is None:
                w = WorkerInfo(worker_id=worker_id)
                self._workers[worker_id] = w
            w.last_heartbeat = time.time()
            w.status = "healthy"
            w.metadata.update(metadata)

    def refresh(self) -> list[WorkerInfo]:
        now = time.time()
        with self._lock:
            for w in self._workers.values():
                age = now - w.last_heartbeat
                if age >= self.dead_after:
                    w.status = "dead"
                elif age >= self.stale_after:
                    w.status = "stale"
                else:
                    w.status = "healthy"
            return list(self._workers.values())

    def worker_count(self, *, healthy_only: bool = False) -> int:
        workers = self.refresh()
        if healthy_only:
            return sum(1 for w in workers if w.status == "healthy")
        return len(workers)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "worker_id": w.worker_id,
                "backend": w.backend,
                "status": w.status,
                "last_heartbeat": w.last_heartbeat,
                "leased_tasks": list(w.leased_tasks),
                "metadata": dict(w.metadata),
            }
            for w in self.refresh()
        ]


TaskHandler = Callable[[DistributedTask], dict[str, Any]]


class LocalWorker:
    """In-process worker loop — Celery/Ray-compatible shape."""

    def __init__(
        self,
        queue: TaskQueue,
        registry: WorkerRegistry,
        *,
        worker_id: str | None = None,
        handlers: dict[str, TaskHandler] | None = None,
    ) -> None:
        self.queue = queue
        self.registry = registry
        self.info = registry.register(worker_id, backend=BackendKind.LOCAL.value)
        self.handlers = handlers or {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register_handler(self, kind: str, handler: TaskHandler) -> None:
        self.handlers[kind] = handler

    def start(self, *, poll_interval: float = 0.5) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                self.registry.heartbeat(self.info.worker_id)
                task = self.queue.lease(self.info.worker_id)
                if task is None:
                    time.sleep(poll_interval)
                    continue
                handler = self.handlers.get(task.kind)
                try:
                    if handler is None:
                        raise KeyError(f"No handler for kind={task.kind}")
                    result = handler(task)
                    self.queue.complete(task.task_id, result)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Worker task failed: %s", exc)
                    delay = min(60.0, 2.0 ** max(task.attempts, 1))
                    self.queue.fail(task.task_id, str(exc), retry=True, delay_seconds=delay)

        self._thread = threading.Thread(target=_loop, daemon=True, name=self.info.worker_id)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


class DistributedExecutor:
    """Façade: Coordinator → Queue → Workers.

    ``backend`` selects local execution today; adapters document future
    Celery / Ray / Temporal / Kubernetes Jobs compatibility.
    """

    def __init__(
        self,
        *,
        db_path: str = "data/sqlmind_task_queue.sqlite3",
        lock_db_path: str = "data/sqlmind_locks.sqlite3",
        backend: BackendKind = BackendKind.LOCAL,
        worker_count: int = 2,
    ) -> None:
        self.backend = backend
        self.queue = TaskQueue(db_path)
        self.locks = DistributedLock(lock_db_path)
        self.registry = WorkerRegistry()
        self.workers: list[LocalWorker] = []
        self._worker_count = max(1, worker_count)
        if backend == BackendKind.LOCAL:
            for i in range(self._worker_count):
                w = LocalWorker(self.queue, self.registry, worker_id=f"local-{i}")
                self.workers.append(w)

    def start(self) -> None:
        if self.backend != BackendKind.LOCAL:
            logger.info(
                "Distributed backend=%s — use adapter.submit(); local workers not started",
                self.backend.value,
            )
            return
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()

    def register_handler(self, kind: str, handler: TaskHandler) -> None:
        for w in self.workers:
            w.register_handler(kind, handler)

    def submit(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistributedTask:
        if self.backend != BackendKind.LOCAL:
            return BackendAdapter(self.backend).submit(kind, payload or {}, **kwargs)
        return self.queue.enqueue(kind, payload, **kwargs)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "queue_length": self.queue.queue_length(),
            "worker_count": self.registry.worker_count(),
            "healthy_workers": self.registry.worker_count(healthy_only=True),
            "workers": self.registry.snapshot(),
        }


class BackendNotConfiguredError(RuntimeError):
    """Raised when a remote distributed backend is selected but not configured."""


class BackendAdapter:
    """Contract for future Celery / Ray / Temporal / K8s backends.

    Production default is LOCAL (``DistributedExecutor`` + ``LocalWorker``).
    Remote adapters do **not** pretend to succeed — they raise until configured.
    """

    def __init__(self, kind: BackendKind) -> None:
        self.kind = kind

    def submit(
        self,
        task_kind: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> DistributedTask:
        if self.kind == BackendKind.LOCAL:
            raise BackendNotConfiguredError(
                "Use DistributedExecutor.submit for LOCAL backend"
            )
        raise BackendNotConfiguredError(
            f"Distributed backend '{self.kind.value}' is not configured. "
            f"Use LOCAL workers or implement {self.kind.value} adapter. "
            f"Guidance: {self.describe().get('guidance')}"
        )

    def describe(self) -> dict[str, str]:
        mapping = {
            BackendKind.CELERY: "Map TaskQueue → Celery broker + workers",
            BackendKind.RAY: "Map TaskQueue → Ray remote tasks",
            BackendKind.TEMPORAL: "Map TaskQueue → Temporal workflows/activities",
            BackendKind.K8S: "Map TaskQueue → Kubernetes Job / CronJob",
            BackendKind.LOCAL: "In-process LocalWorker pool (production default)",
        }
        return {"backend": self.kind.value, "guidance": mapping.get(self.kind, "")}


_EXECUTOR: DistributedExecutor | None = None
_EXEC_LOCK = threading.Lock()


def get_distributed_executor(
    *,
    db_path: str = "data/sqlmind_task_queue.sqlite3",
    worker_count: int = 2,
    autostart: bool = True,
) -> DistributedExecutor:
    global _EXECUTOR
    with _EXEC_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = DistributedExecutor(
                db_path=db_path, worker_count=worker_count
            )
            if autostart:
                _EXECUTOR.start()
        return _EXECUTOR
