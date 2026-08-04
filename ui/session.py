"""Streamlit session bootstrap."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import Settings, get_settings
from memory.conversation import ConversationMemory, HistoryStore
from ui.styles import inject_styles
from utils.helpers import ensure_dirs
from utils.logging_config import setup_logging


def init_session_state(settings: Settings | None = None) -> Settings:
    if not st.session_state.get("_secrets_applied"):
        _apply_streamlit_secrets()
        st.session_state["_secrets_applied"] = True
    settings = settings or get_settings()
    ensure_dirs(settings.data_dir, settings.export_dir)
    setup_logging(settings.log_level, settings.audit_log_path)

    defaults: dict[str, Any] = {
        "connected": False,
        "connector": None,
        "engine": None,
        "schema": None,
        "executor": None,
        "orchestrator": None,
        "db_config": None,
        "memory": ConversationMemory(),
        "history_store": HistoryStore(settings.history_db_path),
        "chat_messages": [],
        "last_result": None,
        "last_df": None,
        "agent_logs": [],
        "suggested_questions": [],
        "ai_summary": "",
        "dashboard_spec": {},
        "memory_summary": "",
        "session_id": "",
        "active_page": "Home",
        "nav_page": "Home",
        "llm_ready": False,
        "connections_registry": {},
        "active_connection_name": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return settings


def _apply_streamlit_secrets() -> None:
    import os

    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001
        return

    mapping = [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "ANTHROPIC_API_KEY",
        "CLAUDE_MODEL",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "QUERY_TIMEOUT_SECONDS",
        "MAX_ROWS",
        "MAX_SQL_RETRIES",
        "READ_ONLY_MODE",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
        "OTEL_ENABLED",
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER",
        "OTEL_ENDPOINT",
    ]
    for key in mapping:
        if key in secrets and key not in os.environ:
            os.environ[key] = str(secrets[key])
    get_settings.cache_clear()


def inject_minimal_css() -> None:
    inject_styles()


def get_orchestrator():
    """Return the active SQLMindOrchestrator from Streamlit session, if any."""
    return st.session_state.get("orchestrator")
