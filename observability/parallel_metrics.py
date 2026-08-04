"""Accurate parallel execution metrics for LangGraph Send fan-outs.

Tracks concurrent branches, start/end times, max parallelism, critical path,
average workers, and worker timelines — fixing Parallel Executions = 0 when
Send is used but not explicitly recorded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from utils.logging_config import get_logger

logger = get_logger(__name__)

_LOCK = threading.RLock()
_COLLECTOR: "ParallelMetricsCollector | None" = None


@dataclass
class WorkerSpan:
    worker_id: str
    node: str
    wave_id: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, (end - self.started_at) * 1000.0)


@dataclass
class ParallelWave:
    wave_id: str
    source: str
    targets: list[str]
    started_at: float
    ended_at: float | None = None
    workers: dict[str, WorkerSpan] = field(default_factory=dict)

    @property
    def concurrency(self) -> int:
        return len(self.targets)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return max(0.0, (end - self.started_at) * 1000.0)

    @property
    def critical_path_ms(self) -> float:
        if not self.workers:
            return self.duration_ms
        return max(w.duration_ms for w in self.workers.values())


class ParallelMetricsCollector:
    """Process-wide parallel branch detector + aggregator."""

    def __init__(self) -> None:
        self._waves: list[ParallelWave] = []
        self._active: dict[str, ParallelWave] = {}
        self._session_waves: dict[str, list[str]] = {}
        self.total_parallel_executions = 0
        self.max_parallelism = 0
        self._concurrency_samples: list[int] = []
        self._lock = threading.RLock()

    def begin_wave(
        self,
        targets: list[str],
        *,
        source: str = "",
        session_id: str = "",
        wave_id: str | None = None,
    ) -> str:
        tid = wave_id or f"wave-{uuid4().hex[:10]}"
        now = time.time()
        wave = ParallelWave(
            wave_id=tid,
            source=source,
            targets=[str(t) for t in targets],
            started_at=now,
        )
        for node in wave.targets:
            wid = f"{tid}:{node}"
            wave.workers[wid] = WorkerSpan(
                worker_id=wid, node=node, wave_id=tid, started_at=now
            )
        with self._lock:
            self._waves.append(wave)
            self._active[tid] = wave
            self.total_parallel_executions += 1
            conc = wave.concurrency
            self._concurrency_samples.append(conc)
            if conc > self.max_parallelism:
                self.max_parallelism = conc
            if session_id:
                self._session_waves.setdefault(session_id, []).append(tid)
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_parallel(
                source=source or "unknown",
                workers=conc,
                wave_id=tid,
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from observability.runtime_trace import safe_trace

            safe_trace(
                "parallel_send",
                targets=list(wave.targets),
                source=source,
            )
        except Exception:  # noqa: BLE001
            pass
        logger.debug(
            "Parallel wave %s source=%s workers=%s", tid, source, wave.targets
        )
        return tid

    def worker_exit(
        self,
        node: str,
        *,
        wave_id: str | None = None,
        status: str = "ok",
    ) -> None:
        with self._lock:
            wave = None
            if wave_id and wave_id in self._active:
                wave = self._active[wave_id]
            else:
                # Match most recent active wave containing this node
                for w in reversed(list(self._active.values())):
                    if any(ws.node == node for ws in w.workers.values()):
                        wave = w
                        break
            if wave is None:
                return
            now = time.time()
            for ws in wave.workers.values():
                if ws.node == node and ws.ended_at is None:
                    ws.ended_at = now
                    ws.status = status
            if all(ws.ended_at is not None for ws in wave.workers.values()):
                wave.ended_at = now
                self._active.pop(wave.wave_id, None)

    def snapshot(self, *, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            waves = list(self._waves)
            if session_id:
                ids = set(self._session_waves.get(session_id, []))
                waves = [w for w in waves if w.wave_id in ids]
            samples = list(self._concurrency_samples)
            max_p = self.max_parallelism
            total = self.total_parallel_executions

        avg_workers = (sum(samples) / len(samples)) if samples else 0.0
        timelines: list[dict[str, Any]] = []
        critical_paths: list[float] = []
        for w in waves[-50:]:
            critical_paths.append(w.critical_path_ms)
            timelines.append(
                {
                    "wave_id": w.wave_id,
                    "source": w.source,
                    "targets": w.targets,
                    "concurrency": w.concurrency,
                    "started_at": w.started_at,
                    "ended_at": w.ended_at,
                    "duration_ms": round(w.duration_ms, 2),
                    "critical_path_ms": round(w.critical_path_ms, 2),
                    "workers": [
                        {
                            "worker_id": ws.worker_id,
                            "node": ws.node,
                            "started_at": ws.started_at,
                            "ended_at": ws.ended_at,
                            "duration_ms": round(ws.duration_ms, 2),
                            "status": ws.status,
                        }
                        for ws in w.workers.values()
                    ],
                }
            )
        return {
            "total_parallel_executions": total if session_id is None else len(waves),
            "max_parallelism": max_p,
            "average_parallel_workers": round(avg_workers, 3),
            "critical_path_duration_ms": round(
                max(critical_paths) if critical_paths else 0.0, 2
            ),
            "active_waves": len(self._active),
            "worker_timeline": timelines,
            "concurrency_count": samples[-1] if samples else 0,
        }

    def reset(self) -> None:
        with self._lock:
            self._waves.clear()
            self._active.clear()
            self._session_waves.clear()
            self.total_parallel_executions = 0
            self.max_parallelism = 0
            self._concurrency_samples.clear()


def get_parallel_metrics() -> ParallelMetricsCollector:
    global _COLLECTOR
    with _LOCK:
        if _COLLECTOR is None:
            _COLLECTOR = ParallelMetricsCollector()
        return _COLLECTOR


def record_parallel_send(
    targets: list[str],
    *,
    source: str = "",
    session_id: str = "",
) -> str:
    """Public helper used by routing edges and enterprise wrappers."""
    if len(targets) < 2:
        return ""
    return get_parallel_metrics().begin_wave(
        targets, source=source, session_id=session_id
    )
