"""LangChain tools wrapping deterministic business operations.

Tools perform work. When bound via ``LLMService.bind_tools`` + ``ToolNode``,
the LLM emits ``tool_calls`` that select which tool runs. Security-critical
validation and execution remain rule/AST based *inside* these tools — the LLM
chooses *when* to call them, never whether validation is skipped.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from database.query_executor import QueryExecutor, QueryResult
from database.schema_inspector import SchemaSnapshot
from memory.conversation import HistoryStore
from services.export_service import ExportService
from utils.security import SQLSecurityGuard


class QueryToolInput(BaseModel):
    sql: str = Field(description="Read-only SQL to execute")


class ValidateToolInput(BaseModel):
    sql: str = Field(description="SQL to validate deterministically")


class StatsToolInput(BaseModel):
    max_rows: int = Field(default=50, description="Sample size for stats")


class ExportToolInput(BaseModel):
    label: str = Field(default="query", description="Export file label")


class HistoryToolInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)


class BookmarkToolInput(BaseModel):
    title: str
    question: str
    sql: str = ""


class ExplainToolInput(BaseModel):
    sql: str = Field(description="SQL to EXPLAIN")


def build_toolbelt(
    *,
    schema: SchemaSnapshot,
    executor: QueryExecutor,
    security: SQLSecurityGuard,
    export_dir: str,
    history_store: HistoryStore | None = None,
    get_last_df: Callable[[], pd.DataFrame | None] | None = None,
) -> list[StructuredTool]:
    """Construct the standard SQLMind toolbelt for agent binding."""

    def schema_tool() -> str:
        return schema.to_prompt_text()

    def validate_tool(sql: str) -> str:
        result = security.validate(sql)
        return json.dumps(
            {
                "is_safe": result.is_safe,
                "sql": result.sql,
                "errors": result.errors,
                "warnings": result.warnings,
                "tables": result.tables,
                "columns": result.columns,
                "joins": result.joins,
                "statement_type": result.statement_type,
            }
        )

    def query_tool(sql: str) -> str:
        result: QueryResult = executor.execute(sql)
        payload = {
            "success": result.success,
            "row_count": result.row_count,
            "execution_time": result.execution_time,
            "sql": result.sql,
            "error": result.error,
            "truncated": result.truncated,
            "columns": list(result.dataframe.columns) if result.success else [],
            "sample": result.to_records(20),
        }
        return json.dumps(payload, default=str)

    def stats_tool(max_rows: int = 50) -> str:
        df = get_last_df() if get_last_df else None
        if df is None or df.empty:
            return json.dumps({"error": "No result dataframe available"})
        sample = df.head(max_rows)
        numeric = sample.select_dtypes(include="number")
        stats = numeric.describe().to_dict() if not numeric.empty else {}
        return json.dumps(
            {
                "columns": list(df.columns),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "row_count": len(df),
                "statistics": stats,
                "sample": sample.to_dict(orient="records"),
            },
            default=str,
        )

    def export_tool(label: str = "query") -> str:
        df = get_last_df() if get_last_df else None
        if df is None or df.empty:
            return json.dumps({"error": "No data to export"})
        paths = ExportService(export_dir).export_all(df, label)
        return json.dumps(paths)

    def history_tool(limit: int = 10) -> str:
        if history_store is None:
            return json.dumps([])
        return json.dumps(history_store.list_history(limit=limit), default=str)

    def bookmark_tool(title: str, question: str, sql: str = "") -> str:
        if history_store is None:
            return json.dumps({"error": "History store unavailable"})
        bid = history_store.add_bookmark(title, question, sql)
        return json.dumps({"bookmark_id": bid})

    def explain_tool(sql: str) -> str:
        dialect = schema.dialect.lower()
        # Deterministic EXPLAIN — never mutate
        if dialect in {"postgresql", "postgres"}:
            explain_sql = f"EXPLAIN (FORMAT TEXT) {sql}"
        elif dialect == "mysql":
            explain_sql = f"EXPLAIN {sql}"
        else:
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        # Validate first
        validation = security.validate(sql)
        if not validation.is_safe:
            return json.dumps({"error": validation.errors, "plan": None})
        result = executor.execute(explain_sql)
        if not result.success:
            # Some dialects reject EXPLAIN via allow-list — try raw engine path
            return json.dumps({"error": result.error, "plan": None, "sql": explain_sql})
        return json.dumps(
            {
                "sql": explain_sql,
                "plan_rows": result.to_records(100),
                "columns": list(result.dataframe.columns),
            },
            default=str,
        )

    tools = [
        StructuredTool.from_function(
            func=schema_tool,
            name="schema_tool",
            description="Return database schema text",
        ),
        StructuredTool.from_function(
            func=validate_tool,
            name="validate_sql_tool",
            description="Deterministically validate SQL (AST + security). Never skip.",
            args_schema=ValidateToolInput,
        ),
        StructuredTool.from_function(
            func=query_tool,
            name="query_tool",
            description="Execute a validated read-only SQL query and return sample rows",
            args_schema=QueryToolInput,
        ),
        StructuredTool.from_function(
            func=stats_tool,
            name="statistics_tool",
            description="Compute column dtypes and numeric statistics for the last result",
            args_schema=StatsToolInput,
        ),
        StructuredTool.from_function(
            func=export_tool,
            name="export_tool",
            description="Export the last result set to CSV/Excel/JSON/Markdown",
            args_schema=ExportToolInput,
        ),
        StructuredTool.from_function(
            func=history_tool,
            name="history_tool",
            description="List recent query history",
            args_schema=HistoryToolInput,
        ),
        StructuredTool.from_function(
            func=bookmark_tool,
            name="bookmark_tool",
            description="Bookmark a question/SQL pair",
            args_schema=BookmarkToolInput,
        ),
        StructuredTool.from_function(
            func=explain_tool,
            name="explain_tool",
            description="Run EXPLAIN / EXPLAIN QUERY PLAN for a SQL statement",
            args_schema=ExplainToolInput,
        ),
    ]
    return tools


def tools_by_name(tools: list[StructuredTool]) -> dict[str, StructuredTool]:
    return {t.name: t for t in tools}
