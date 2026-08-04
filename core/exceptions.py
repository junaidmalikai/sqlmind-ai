"""Typed application exceptions — never use bare except for control flow."""

from __future__ import annotations


class SQLMindError(Exception):
    """Base error for SQLMind AI."""

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConfigurationError(SQLMindError):
    """Invalid or missing configuration."""


class DatabaseConnectionError(SQLMindError):
    """Failed to connect or configure a database engine."""


class ValidationError(SQLMindError):
    """Input / schema validation failure (non-SQL)."""


class SQLSecurityError(SQLMindError):
    """SQL rejected by deterministic security policy."""


class QueryExecutionError(SQLMindError):
    """Validated SQL failed at the database or timed out."""


class LLMProviderError(SQLMindError):
    """LLM provider invocation or structured-output failure."""


class ExportError(SQLMindError):
    """Export / report generation failure."""
