"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal["ollama", "openai", "gemini", "claude", "groq"]


class Settings(BaseSettings):
    """Central configuration for SQLMind AI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "SQLMind AI"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    # LLM Provider
    llm_provider: ProviderName = "ollama"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_timeout: int = 120

    # Provider API keys / endpoints
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Security / query limits
    query_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_rows: int = Field(default=1000, ge=1, le=100_000)
    max_sql_retries: int = Field(default=3, ge=1, le=10)
    read_only_mode: bool = True

    # Paths
    export_dir: str = "exports"
    data_dir: str = "data"
    audit_log_path: str = "data/audit.log"
    history_db_path: str = "data/sqlmind_history.sqlite3"
    checkpoint_db_path: str = "data/sqlmind_checkpoints.sqlite3"

    # LangSmith / observability
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "SQLMind-AI"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # OpenTelemetry (optional; disabled by default)
    otel_enabled: bool = False
    otel_service_name: str = "SQLMind-AI"
    otel_exporter: Literal["console", "otlp"] = "console"
    otel_endpoint: str = ""  # e.g. http://localhost:4318/v1/traces

    # Memory
    memory_max_turns: int = Field(default=24, ge=4, le=200)
    memory_summarize_after: int = Field(default=12, ge=4, le=100)

    # Multi-tenant / actor defaults (Streamlit local uses these)
    default_tenant_id: str = "default"
    default_actor: str = "local-user"

    # Phase 2 autonomy — goal planner preamble (False = legacy START→supervisor)
    autonomy_planning_enabled: bool = True
    goal_clarify_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    workflow_memory_db_path: str = ""

    # Phase 3 — Enterprise Agentic Platform
    enterprise_enabled: bool = True
    enterprise_subgraphs_enabled: bool = True  # nest true LangGraph subgraphs
    enterprise_iam_enforcement: bool = True
    enterprise_circuit_breaker: bool = True
    vector_memory_enabled: bool = True
    vector_memory_db_path: str = ""
    embedding_provider: str = "hashing"  # hashing|ollama|openai|sentence_transformers|bge|e5|nomic
    embedding_model: str = ""
    embedding_dim: int = 256
    memory_ttl_seconds: float = 0.0  # 0 = no expiration
    goal_store_db_path: str = ""
    iam_db_path: str = ""
    iam_username: str = ""
    iam_password: str = ""
    iam_api_key: str = ""
    learning_db_path: str = ""
    dlq_db_path: str = ""
    plugin_dirs: str = "plugins"
    plugin_hot_reload: bool = True
    plugin_require_signature: bool = False
    plugin_signing_secret: str = "sqlmind-plugin-dev-secret"
    approval_gates_enabled: bool = True
    require_write_sql_approval: bool = True
    require_sensitive_export_approval: bool = True
    enterprise_events_log_path: str = "data/enterprise_events.jsonl"
    agent_heartbeat_stale_seconds: float = 60.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0
    prometheus_enabled: bool = True
    metrics_export_path: str = "data/sqlmind_metrics.prom"
    metrics_http_enabled: bool = True
    metrics_http_host: str = "127.0.0.1"
    metrics_http_port: int = 9108

    # Runtime Execution Trace System (enterprise audit artifacts)
    runtime_trace_enabled: bool = True
    runtime_trace_dir: str = "data/runtime_traces"
    runtime_trace_print_summary: bool = True

    # Phase 4 — Production enterprise extensions
    max_runtime_replans: int = Field(default=2, ge=0, le=10)
    recovery_controller_enabled: bool = True
    export_async_enabled: bool = True
    plugin_runtime_enabled: bool = True
    plugin_timeout_seconds: float = 15.0
    distributed_execution_enabled: bool = True
    distributed_worker_count: int = Field(default=2, ge=1, le=32)
    distributed_queue_db_path: str = ""
    enterprise_queue_db_path: str = ""

    # UI
    default_theme: str = "light"
    page_title: str = "SQLMind AI"
    page_icon: str = "🧠"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    def resolve_model(self) -> str:
        """Return the active model name for the selected provider."""
        mapping = {
            "ollama": self.llm_model,
            "openai": self.openai_model or self.llm_model,
            "gemini": self.gemini_model or self.llm_model,
            "claude": self.claude_model or self.llm_model,
            "groq": self.groq_model or self.llm_model,
        }
        return mapping.get(self.llm_provider, self.llm_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
