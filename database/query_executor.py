"""Safe, timed, read-only SQL execution."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import get_settings
from observability.otel_setup import start_span
from utils.logging_config import get_logger, write_audit_event
from utils.security import SQLSecurityError, SQLSecurityGuard

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Structured result of a SQL execution."""

    success: bool
    dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_count: int = 0
    execution_time: float = 0.0
    sql: str = ""
    error: str | None = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_records(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.dataframe.empty:
            return []
        return self.dataframe.head(limit).to_dict(orient="records")


class QueryExecutor:
    """Executes validated SQL with timeout and row limits."""

    def __init__(
        self,
        engine: Engine,
        *,
        security: SQLSecurityGuard | None = None,
        timeout: int | None = None,
        max_rows: int | None = None,
        database_label: str = "",
    ) -> None:
        settings = get_settings()
        self.engine = engine
        self.timeout = timeout or settings.query_timeout_seconds
        self.max_rows = max_rows or settings.max_rows
        self.security = security or SQLSecurityGuard(
            read_only=settings.read_only_mode,
            max_rows=self.max_rows,
        )
        self.database_label = database_label
        self.audit_path = settings.audit_log_path

    async def aexecute(self, sql: str) -> QueryResult:
        """Async wrapper around ``execute`` (DB I/O runs in a worker thread)."""
        import asyncio

        return await asyncio.to_thread(self.execute, sql)

    def execute(self, sql: str) -> QueryResult:
        """Validate and execute SQL; return QueryResult. Never skip the security gate."""
        with start_span(
            "sqlmind.query.execute",
            attributes={
                "db.system": self.engine.dialect.name if self.engine else "",
                "sqlmind.database": self.database_label,
                "sqlmind.sql_chars": len(sql or ""),
            },
        ) as span:
            return self._execute_inner(sql, span)

    def _execute_inner(self, sql: str, span: Any = None) -> QueryResult:
        validation = self.security.validate(sql)
        if not validation.is_safe:
            if span is not None:
                try:
                    span.set_attribute("sqlmind.rejected", True)
                    span.set_attribute(
                        "sqlmind.validation_errors", "; ".join(validation.errors)[:500]
                    )
                except Exception:  # noqa: BLE001
                    pass
            write_audit_event(
                "sql_rejected",
                {"sql": sql, "errors": validation.errors, "db": self.database_label},
                self.audit_path,
            )
            return QueryResult(
                success=False,
                sql=sql,
                error="; ".join(validation.errors),
                warnings=validation.warnings,
            )

        safe_sql = validation.sql
        start = time.perf_counter()

        try:
            df = self._run_with_timeout(safe_sql)
            elapsed = time.perf_counter() - start
            truncated = len(df) >= self.max_rows
            if truncated:
                df = df.head(self.max_rows)

            if span is not None:
                try:
                    span.set_attribute("sqlmind.row_count", len(df))
                    span.set_attribute("sqlmind.elapsed_s", elapsed)
                    span.set_attribute("sqlmind.truncated", truncated)
                except Exception:  # noqa: BLE001
                    pass

            write_audit_event(
                "sql_executed",
                {
                    "sql": safe_sql,
                    "rows": len(df),
                    "elapsed": elapsed,
                    "db": self.database_label,
                },
                self.audit_path,
            )
            return QueryResult(
                success=True,
                dataframe=df,
                row_count=len(df),
                execution_time=elapsed,
                sql=safe_sql,
                truncated=truncated,
                warnings=validation.warnings,
            )
        except FuturesTimeout:
            elapsed = time.perf_counter() - start
            msg = f"Query timed out after {self.timeout}s"
            logger.error(msg)
            write_audit_event(
                "sql_timeout",
                {"sql": safe_sql, "db": self.database_label},
                self.audit_path,
            )
            return QueryResult(
                success=False,
                sql=safe_sql,
                execution_time=elapsed,
                error=msg,
                warnings=validation.warnings,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            logger.exception("Query execution failed")
            write_audit_event(
                "sql_error",
                {"sql": safe_sql, "error": str(exc), "db": self.database_label},
                self.audit_path,
            )
            return QueryResult(
                success=False,
                sql=safe_sql,
                execution_time=elapsed,
                error=str(exc),
                warnings=validation.warnings,
            )

    def _run_with_timeout(self, sql: str) -> pd.DataFrame:
        def _query() -> pd.DataFrame:
            with self.engine.connect() as conn:
                # Attempt read-only session settings where supported
                try:
                    dialect = conn.dialect.name
                    if dialect == "postgresql":
                        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
                    elif dialect == "mysql":
                        conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
                except Exception:  # noqa: BLE001
                    pass
                return pd.read_sql(text(sql), conn)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_query)
            return future.result(timeout=self.timeout)
