"""Database connection + LLM settings widgets (presentation wrappers around existing logic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from config.settings import Settings
from database.connector import DatabaseConfig, DatabaseConnector
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaInspector
from graph.workflow import SQLMindOrchestrator
from sample_data.seed import create_sample_database
from services.llm_service import LLMService
from services.provider_health import (
    PROVIDER_DEFAULT_MODELS,
    list_ollama_models,
    validate_provider,
)
from utils.logging_config import get_logger
from utils.security import SQLSecurityGuard

logger = get_logger(__name__)


def render_connection_panel(
    settings: Settings, *, key_prefix: str = "side", framed: bool = True
) -> None:
    """Render DB connection controls. ``key_prefix`` must be unique per page instance.

    When ``framed`` is False, skip the outer expander (caller already provides one).
    """
    p = key_prefix
    container = (
        st.expander("Database connection", expanded=not st.session_state.get("connected"))
        if framed
        else st.container()
    )
    with container:
        dialect = st.selectbox(
            "Engine", ["sqlite", "postgresql", "mysql"], key=f"{p}_ui_dialect"
        )
        if dialect == "sqlite":
            use_sample = st.checkbox(
                "Use built-in sample database", value=True, key=f"{p}_ui_use_sample"
            )
            uploaded = st.file_uploader(
                "Or upload SQLite file",
                type=["db", "sqlite", "sqlite3"],
                key=f"{p}_ui_sqlite_upload",
            )
            if st.button("Connect SQLite", type="primary", use_container_width=True, key=f"{p}_btn_connect_sqlite"):
                try:
                    if uploaded is not None:
                        path = DatabaseConnector.save_uploaded_sqlite(
                            uploaded.getvalue(), uploaded.name, settings.data_dir
                        )
                    elif use_sample:
                        path = create_sample_database(Path(settings.data_dir) / "sample_retail.db")
                    else:
                        st.error("Upload a SQLite file or enable the sample database.")
                        return
                    config = DatabaseConfig(
                        dialect="sqlite", sqlite_path=path, database=Path(path).name
                    )
                    connect_database(config, settings)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection failed: {exc}")
                    logger.exception("SQLite connect failed")
        else:
            host = st.text_input("Host", value="localhost", key=f"{p}_ui_host")
            default_port = 5432 if dialect == "postgresql" else 3306
            port = st.number_input(
                "Port", min_value=1, max_value=65535, value=default_port, key=f"{p}_ui_port"
            )
            username = st.text_input("Username", key=f"{p}_ui_user")
            password = st.text_input("Password", type="password", key=f"{p}_ui_pass")
            database = st.text_input("Database name", key=f"{p}_ui_db")
            display = st.text_input(
                "Connection name", value=database or dialect, key=f"{p}_ui_conn_name"
            )
            if st.button(
                f"Connect {dialect.title()}",
                type="primary",
                use_container_width=True,
                key=f"{p}_btn_connect_db",
            ):
                if not database:
                    st.error("Database name is required.")
                    return
                config = DatabaseConfig(
                    dialect=dialect,  # type: ignore[arg-type]
                    host=host,
                    port=int(port),
                    username=username,
                    password=password,
                    database=database,
                    display_name=display or database,
                )
                try:
                    connect_database(config, settings)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Connection failed: {exc}")
                    logger.exception("DB connect failed")

        if st.session_state.get("connected") and st.button(
            "Disconnect", use_container_width=True, key=f"{p}_btn_disconnect"
        ):
            disconnect_database()


def render_llm_panel(
    settings: Settings, *, key_prefix: str = "side", framed: bool = True
) -> None:
    """Render LLM settings. ``key_prefix`` must be unique per page instance.

    When ``framed`` is False, skip the outer expander (caller already provides one).
    """
    p = key_prefix
    container = st.expander("LLM provider", expanded=False) if framed else st.container()
    with container:
        providers = ["ollama", "openai", "gemini", "claude", "groq"]
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(settings.llm_provider) if settings.llm_provider in providers else 0,
            key=f"{p}_ui_llm_provider",
        )
        prev_key = f"_{p}_llm_provider_prev"
        model_key = f"{p}_ui_llm_model"
        prev_provider = st.session_state.get(prev_key)
        if prev_provider != provider:
            st.session_state[prev_key] = provider
            if provider == "ollama":
                st.session_state[model_key] = settings.llm_model or PROVIDER_DEFAULT_MODELS["ollama"]
            else:
                attr = {
                    "openai": "openai_model",
                    "gemini": "gemini_model",
                    "claude": "claude_model",
                    "groq": "groq_model",
                }[provider]
                st.session_state[model_key] = getattr(settings, attr) or PROVIDER_DEFAULT_MODELS[provider]

        settings.llm_provider = provider  # type: ignore[assignment]

        if provider == "ollama":
            settings.ollama_base_url = st.text_input(
                "Ollama Base URL", value=settings.ollama_base_url, key=f"{p}_ui_ollama_url"
            )
            ollama_models: list[str] = []
            try:
                ollama_models = list_ollama_models(settings.ollama_base_url)
            except Exception:  # noqa: BLE001
                ollama_models = []
            default_model = (
                settings.llm_model
                if settings.llm_model in ollama_models
                else (
                    PROVIDER_DEFAULT_MODELS["ollama"]
                    if PROVIDER_DEFAULT_MODELS["ollama"] in ollama_models
                    else None
                )
            )
            if ollama_models:
                options = ollama_models + ["(custom)"]
                current = st.session_state.get(model_key, default_model or ollama_models[0])
                idx = ollama_models.index(current) if current in ollama_models else len(options) - 1
                choice = st.selectbox("Model", options, index=idx, key=f"{p}_ui_ollama_model_pick")
                if choice == "(custom)":
                    model = st.text_input(
                        "Custom model name",
                        value=current if current not in ollama_models else ollama_models[0],
                        key=model_key,
                    )
                else:
                    model = choice
                    st.session_state[model_key] = model
            else:
                st.caption("Could not list Ollama models. Enter a model name manually.")
                model = st.text_input(
                    "Model",
                    value=st.session_state.get(
                        model_key, settings.llm_model or PROVIDER_DEFAULT_MODELS["ollama"]
                    ),
                    key=model_key,
                )
            settings.llm_model = model
        else:
            model = st.text_input(
                "Model",
                value=st.session_state.get(
                    model_key, PROVIDER_DEFAULT_MODELS.get(provider, settings.resolve_model())
                ),
                key=model_key,
            )
            if provider == "openai":
                settings.openai_model = model
                settings.openai_api_key = st.text_input(
                    "OpenAI API Key",
                    value=settings.openai_api_key,
                    type="password",
                    key=f"{p}_ui_openai_key",
                )
            elif provider == "gemini":
                settings.gemini_model = model
                settings.gemini_api_key = st.text_input(
                    "Gemini API Key",
                    value=settings.gemini_api_key,
                    type="password",
                    key=f"{p}_ui_gemini_key",
                )
            elif provider == "claude":
                settings.claude_model = model
                settings.anthropic_api_key = st.text_input(
                    "Anthropic API Key",
                    value=settings.anthropic_api_key,
                    type="password",
                    key=f"{p}_ui_claude_key",
                )
            elif provider == "groq":
                settings.groq_model = model
                settings.groq_api_key = st.text_input(
                    "Groq API Key",
                    value=settings.groq_api_key,
                    type="password",
                    key=f"{p}_ui_groq_key",
                )

        if st.button("Apply LLM Settings", use_container_width=True, key=f"{p}_btn_apply_llm"):
            try:
                check = validate_provider(settings)
                if not check.ok:
                    st.session_state["llm_ready"] = False
                    st.error(check.message)
                    return
                llm = LLMService(settings)
                st.session_state["llm_service"] = llm
                st.session_state["llm_ready"] = True
                if st.session_state.get("connected") and st.session_state.get("schema"):
                    rebuild_orchestrator(settings, llm)
                st.success(check.message)
            except Exception as exc:  # noqa: BLE001
                st.session_state["llm_ready"] = False
                st.error(f"LLM setup failed: {exc}")


def connect_database(config: DatabaseConfig, settings: Settings) -> None:
    connector = DatabaseConnector(config)
    ok, msg = connector.test_connection()
    if not ok:
        st.error(msg)
        return

    engine = connector.connect()
    inspector = SchemaInspector(engine, config.dialect, config.display_name)
    schema = inspector.discover()
    guard = SQLSecurityGuard(
        read_only=True,
        max_rows=settings.max_rows,
        dialect=config.dialect,
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
    )
    executor = QueryExecutor(
        engine,
        security=guard,
        timeout=settings.query_timeout_seconds,
        max_rows=settings.max_rows,
        database_label=config.display_name,
    )
    llm = st.session_state.get("llm_service") or LLMService(settings)
    session_id = st.session_state.get("session_id") or str(uuid4())
    st.session_state.session_id = session_id
    orchestrator = SQLMindOrchestrator(
        schema,
        executor,
        llm,
        settings,
        security=guard,
        history_store=st.session_state.history_store,
        session_id=session_id,
    )

    st.session_state.connected = True
    st.session_state.connector = connector
    st.session_state.engine = engine
    st.session_state.schema = schema
    st.session_state.executor = executor
    st.session_state.orchestrator = orchestrator
    st.session_state.db_config = config
    st.session_state.llm_service = llm
    st.session_state.suggested_questions = []
    st.session_state.ai_summary = ""
    st.session_state.dashboard_spec = {}
    st.session_state.memory_summary = ""
    st.session_state.active_connection_name = config.display_name
    st.session_state.connections_registry[config.display_name] = config

    store: Any = st.session_state.history_store
    store.save_connection(config.display_name, config.to_safe_dict())

    with st.spinner("Generating AI database summary…"):
        try:
            from models.structured import DatabaseSummaryModel, SuggestedQuestionsModel
            from prompts.templates import suggest_prompt, summary_prompt

            summary = llm.invoke_structured(
                summary_prompt(),
                DatabaseSummaryModel,
                {
                    "database_name": schema.database_name,
                    "dialect": schema.dialect,
                    "schema_text": schema.to_prompt_text(),
                },
            )
            st.session_state.ai_summary = summary.model_dump()
            st.session_state.suggested_questions = summary.suggested_questions
            if not st.session_state.suggested_questions:
                sug = llm.invoke_structured(
                    suggest_prompt(),
                    SuggestedQuestionsModel,
                    {
                        "database_name": schema.database_name,
                        "dialect": schema.dialect,
                        "schema_text": schema.to_prompt_text(),
                    },
                )
                st.session_state.suggested_questions = sug.questions
        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-connect AI summary skipped: %s", exc)

    st.success(f"Connected to {config.display_name} · {len(schema.tables)} tables discovered")
    st.rerun()


def rebuild_orchestrator(settings: Settings, llm: LLMService) -> None:
    schema = st.session_state.schema
    executor = st.session_state.executor
    guard = getattr(executor, "security", None)
    st.session_state.orchestrator = SQLMindOrchestrator(
        schema,
        executor,
        llm,
        settings,
        security=guard,
        history_store=st.session_state.history_store,
        session_id=st.session_state.get("session_id") or "default",
    )


def disconnect_database() -> None:
    connector = st.session_state.get("connector")
    if connector is not None:
        connector.dispose()
    for key in (
        "connected",
        "connector",
        "engine",
        "schema",
        "executor",
        "orchestrator",
        "db_config",
        "last_result",
        "last_df",
        "ai_summary",
    ):
        if key == "connected":
            st.session_state[key] = False
        else:
            st.session_state[key] = None if key != "ai_summary" else ""
    st.rerun()
