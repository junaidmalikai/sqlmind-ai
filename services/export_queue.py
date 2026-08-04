"""Async export graph support — queue, streaming, progress, metadata.

Extends ExportService / export_node without replacing synchronous exporters.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4

import pandas as pd

from utils.helpers import ensure_dirs, utc_now_iso
from utils.logging_config import get_logger

logger = get_logger(__name__)

ExportFormat = Literal["csv", "excel", "json", "markdown", "html", "pdf", "chart", "image"]
ExportStatus = Literal["queued", "running", "streaming", "completed", "failed", "cancelled"]


@dataclass
class ExportJob:
    job_id: str
    formats: list[str]
    status: ExportStatus = "queued"
    progress: float = 0.0
    label: str = "export"
    session_id: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    rows_total: int = 0
    rows_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "formats": self.formats,
            "status": self.status,
            "progress": round(self.progress, 3),
            "label": self.label,
            "session_id": self.session_id,
            "paths": self.paths,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "rows_total": self.rows_total,
            "rows_processed": self.rows_processed,
        }


def stream_dataframe_chunks(
    df: pd.DataFrame, *, chunk_size: int = 500
) -> Iterator[pd.DataFrame]:
    """Yield dataframe slices for large-dataset streaming exports."""
    n = len(df)
    if n == 0:
        yield df
        return
    for start in range(0, n, max(1, chunk_size)):
        yield df.iloc[start : start + chunk_size]


def export_html(
    df: pd.DataFrame,
    export_dir: Path,
    label: str,
    *,
    question: str = "",
    insights: str = "",
    meta: dict[str, Any] | None = None,
) -> str:
    """Write a standalone HTML report (table + narrative)."""
    ensure_dirs(export_dir)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:60]
    path = export_dir / f"{safe}_{uuid4().hex[:8]}.html"
    table_html = df.head(500).to_html(index=False, classes="data", border=0)
    meta = meta or {}
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>SQLMind Export — {safe}</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; background: #f7f4ef; color: #1a1a1a; }}
h1 {{ font-size: 1.6rem; }} table.data {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
table.data th, table.data td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
table.data th {{ background: #e8e2d6; }} .meta {{ color: #555; font-size: 0.9rem; }}
</style></head><body>
<h1>SQLMind AI Export</h1>
<p class="meta">Generated {utc_now_iso()} · rows={meta.get('rows', len(df))}</p>
{f'<p><strong>Question:</strong> {question}</p>' if question else ''}
{f'<section><h2>Insights</h2><pre>{insights[:8000]}</pre></section>' if insights else ''}
<section><h2>Data</h2>{table_html}</section>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return str(path)


class ExportQueue:
    """Async export job queue with progress tracking."""

    def __init__(
        self,
        export_dir: str = "exports",
        *,
        max_workers: int = 2,
        chunk_size: int = 500,
    ) -> None:
        self.export_dir = Path(export_dir)
        ensure_dirs(self.export_dir)
        self.chunk_size = chunk_size
        self._jobs: dict[str, ExportJob] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="export")

    def enqueue(
        self,
        df: pd.DataFrame,
        *,
        formats: list[str] | None = None,
        label: str = "query",
        question: str = "",
        sql: str = "",
        insights: str = "",
        recommendations: str = "",
        meta: dict[str, Any] | None = None,
        session_id: str = "",
        fig: Any | None = None,
        include_images: bool = True,
    ) -> ExportJob:
        fmts = formats or ["csv", "excel", "json", "markdown", "html", "pdf"]
        job = ExportJob(
            job_id=f"exp-{uuid4().hex[:10]}",
            formats=list(fmts),
            label=label,
            session_id=session_id,
            rows_total=len(df),
            metadata={
                "question": question,
                "sql": sql,
                "insights_len": len(insights or ""),
                "formats": list(fmts),
                **(meta or {}),
            },
        )
        with self._lock:
            self._jobs[job.job_id] = job
        # Capture frame for worker thread
        frame = df.copy()
        self._pool.submit(
            self._run_job,
            job.job_id,
            frame,
            question,
            sql,
            insights,
            recommendations,
            meta or {},
            fig,
            include_images,
        )
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_export("queue", status="queued")
        except Exception:  # noqa: BLE001
            pass
        return job

    def _run_job(
        self,
        job_id: str,
        df: pd.DataFrame,
        question: str,
        sql: str,
        insights: str,
        recommendations: str,
        meta: dict[str, Any],
        fig: Any,
        include_images: bool,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        self._update(job_id, status="running", progress=0.05)
        paths: dict[str, str] = {}
        try:
            from services.export_service import ExportService

            svc = ExportService(str(self.export_dir))
            total = max(len(job.formats), 1)
            # Stream large frames through chunked CSV first for progress evidence
            if len(df) > self.chunk_size and "csv" in job.formats:
                self._update(job_id, status="streaming", progress=0.1)
                csv_path = self._stream_csv(df, job.label, job_id)
                paths["csv"] = csv_path
                self._update(
                    job_id,
                    progress=0.25,
                    rows_processed=len(df),
                    paths=dict(paths),
                )

            done = 1 if "csv" in paths else 0
            for fmt in job.formats:
                if fmt == "csv" and "csv" in paths:
                    continue
                if fmt == "html":
                    paths["html"] = export_html(
                        df,
                        self.export_dir,
                        job.label,
                        question=question,
                        insights=insights,
                        meta={**meta, "rows": len(df)},
                    )
                elif fmt == "csv":
                    paths["csv"] = svc.to_csv(df, job.label)
                elif fmt in {"excel", "xlsx"}:
                    paths["excel"] = svc.to_excel_report(
                        df,
                        job.label,
                        question=question,
                        sql=sql,
                        insights=insights,
                        recommendations=recommendations,
                        meta=meta,
                        fig=fig,
                    )
                elif fmt == "json":
                    paths["json"] = svc.to_json(df, job.label)
                elif fmt in {"markdown", "md"}:
                    paths["markdown"] = svc.to_markdown(df, job.label)
                elif fmt == "pdf":
                    paths["pdf"] = svc.build_pdf_report(
                        title=job.label,
                        question=question,
                        sql=sql,
                        insights=insights,
                        df=df,
                        meta=meta,
                        fig=fig,
                        recommendations=recommendations,
                    )
                elif fmt in {"chart", "image"} and include_images and fig is not None:
                    img_path = self.export_dir / f"{job.label}_{uuid4().hex[:8]}.png"
                    try:
                        fig.write_image(str(img_path))
                        paths[fmt] = str(img_path)
                    except Exception as exc:  # noqa: BLE001
                        logger.info("Chart image export skipped: %s", exc)
                done += 1
                self._update(
                    job_id,
                    progress=min(0.95, done / total),
                    paths=dict(paths),
                    rows_processed=len(df),
                )
                try:
                    from observability.metrics import get_metrics

                    get_metrics().observe_export(fmt, status="ok")
                except Exception:  # noqa: BLE001
                    pass

            # Persist job metadata sidecar
            meta_path = self.export_dir / f"{job_id}.meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        **job.to_dict(),
                        "paths": paths,
                        "status": "completed",
                        "progress": 1.0,
                        "completed_at": utc_now_iso(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            self._update(
                job_id,
                status="completed",
                progress=1.0,
                paths=paths,
                metadata={**job.metadata, "meta_path": str(meta_path)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Export job %s failed", job_id)
            self._update(job_id, status="failed", error=str(exc), progress=job.progress)
            try:
                from observability.metrics import get_metrics

                get_metrics().observe_export("job", status="failed")
            except Exception:  # noqa: BLE001
                pass

    def _stream_csv(self, df: pd.DataFrame, label: str, job_id: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:60]
        path = self.export_dir / f"{safe}_{job_id}_stream.csv"
        first = True
        processed = 0
        for chunk in stream_dataframe_chunks(df, chunk_size=self.chunk_size):
            chunk.to_csv(path, mode="w" if first else "a", header=first, index=False)
            first = False
            processed += len(chunk)
            self._update(job_id, rows_processed=processed, status="streaming")
        return str(path)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                if k == "paths" and isinstance(v, dict):
                    job.paths.update(v)
                elif k == "metadata" and isinstance(v, dict):
                    job.metadata.update(v)
                elif hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = time.time()

    def get(self, job_id: str) -> ExportJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs[:limit]]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        by_status: dict[str, int] = {}
        for j in jobs:
            by_status[j.status] = by_status.get(j.status, 0) + 1
        return {
            "total_jobs": len(jobs),
            "by_status": by_status,
            "completed": by_status.get("completed", 0),
            "failed": by_status.get("failed", 0),
            "queued": by_status.get("queued", 0) + by_status.get("running", 0),
        }


_EXPORT_Q: ExportQueue | None = None
_EXPORT_LOCK = threading.Lock()


def get_export_queue(export_dir: str = "exports") -> ExportQueue:
    global _EXPORT_Q
    with _EXPORT_LOCK:
        if _EXPORT_Q is None:
            _EXPORT_Q = ExportQueue(export_dir)
        return _EXPORT_Q
