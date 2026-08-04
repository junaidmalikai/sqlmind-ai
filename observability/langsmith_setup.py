"""LangSmith / LangChain tracing bootstrap."""

from __future__ import annotations

import os

from config.settings import Settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

_CONFIGURED = False


def configure_langsmith(settings: Settings) -> None:
    """Enable LangSmith tracing when an API key is configured.

    Sets LangChain env vars used by LangSmith auto-tracing. Safe to call
    repeatedly; subsequent calls are no-ops after the first successful configure.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    api_key = getattr(settings, "langchain_api_key", "") or os.getenv(
        "LANGCHAIN_API_KEY", ""
    )
    if not api_key:
        return

    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    tracing = str(getattr(settings, "langchain_tracing_v2", True)).lower()
    os.environ["LANGCHAIN_TRACING_V2"] = (
        "true" if tracing in {"1", "true", "yes"} else "false"
    )
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        getattr(settings, "langchain_project", None) or "SQLMind-AI",
    )
    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT",
        getattr(settings, "langchain_endpoint", None)
        or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )
    _CONFIGURED = True
    logger.info(
        "LangSmith tracing enabled project=%s tracing_v2=%s",
        os.environ.get("LANGCHAIN_PROJECT"),
        os.environ.get("LANGCHAIN_TRACING_V2"),
    )


def reset_langsmith_for_tests() -> None:
    """Reset configure guard (unit tests only)."""
    global _CONFIGURED
    _CONFIGURED = False


def langsmith_tags(
    *,
    provider: str,
    dialect: str,
    extra: list[str] | None = None,
) -> list[str]:
    """Stable tags attached to LangGraph / LangChain runs."""
    tags = ["sqlmind", f"provider:{provider}", f"dialect:{dialect}"]
    if extra:
        tags.extend(extra)
    return tags
