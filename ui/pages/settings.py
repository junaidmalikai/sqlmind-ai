"""Settings — enterprise configuration console (presentation only)."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from ui.components import (
    card,
    health_pills,
    kv_card,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.gate import is_llm_ready, provider_api_key
from ui.ui_helpers import esc, inject_html
from ui.widgets.connection import render_connection_panel, render_llm_panel


def render_settings(settings: Settings) -> None:
    page_header(
        "Settings",
        "Configure providers, database, security, and workspace",
        settings=settings,
    )

    llm_ok, llm_msg = is_llm_ready(settings)
    connected = bool(st.session_state.get("connected"))
    cfg = st.session_state.get("db_config")
    model = "—"
    try:
        model = settings.resolve_model()
    except Exception:  # noqa: BLE001
        pass

    key_ok = bool(provider_api_key(settings) or settings.llm_provider == "ollama")

    health_pills(
        [
            ("Database", cfg.display_name if cfg and connected else "Offline", "ok" if connected else "off"),
            ("LLM", "Ready" if llm_ok else "Setup required", "ok" if llm_ok else "warn"),
            ("Provider", settings.llm_provider, "accent" if llm_ok else "off"),
            ("Security", "Read-only" if settings.read_only_mode else "Open", "ok" if settings.read_only_mode else "warn"),
        ]
    )

    kind = "ok" if llm_ok else "warn"
    msg_html = (
        f'<p class="sq-card__body sq-card__body--spaced">{esc(llm_msg)}</p>'
        if (not llm_ok and llm_msg)
        else ""
    )
    inject_html(
        f"""
        <div class="sq-card sq-card--{kind}">
          <div class="sq-card__title">LLM status</div>
          <div class="sq-card__heading">{
            "Ready — AI features available" if llm_ok else "Setup required — apply LLM settings"
          }</div>
          <div class="sq-kv-grid">
            <div><span class="sq-kv-label">Provider</span><strong>{esc(settings.llm_provider)}</strong></div>
            <div><span class="sq-kv-label">Model</span><strong>{esc(model)}</strong></div>
            <div><span class="sq-kv-label">API key</span><strong>{"Configured" if key_ok else "Missing"}</strong></div>
            <div><span class="sq-kv-label">Environment</span><strong>{esc(settings.environment)}</strong></div>
          </div>
          {msg_html}
        </div>
        """
    )

    tab_conn, tab_sec, tab_ws = st.tabs(["Connection", "Security", "Workspace"])

    with tab_conn:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            section_title("LLM provider", "Required before Chat and AI actions")
            render_llm_panel(settings, key_prefix="settings", framed=False)
        with c2:
            section_title("Database", "SQLite, PostgreSQL, or MySQL")
            render_connection_panel(settings, key_prefix="settings", framed=False)

        registry = st.session_state.get("connections_registry") or {}
        if registry:
            section_title("Saved connections", "Session registry")
            for name, conn_cfg in registry.items():
                safe: dict = {}
                try:
                    safe = conn_cfg.to_safe_dict() if hasattr(conn_cfg, "to_safe_dict") else {}
                except Exception:  # noqa: BLE001
                    safe = {}
                dialect = getattr(conn_cfg, "dialect", safe.get("dialect", "—"))
                host = safe.get("host") or getattr(conn_cfg, "host", "—") or "—"
                database = safe.get("database") or getattr(conn_cfg, "database", name) or name
                active = bool(
                    connected and cfg and getattr(cfg, "display_name", "") == name
                )
                badge = (
                    status_badge("Active", kind="ok")
                    if active
                    else status_badge(str(dialect).upper(), kind="info")
                )
                inject_html(
                    f"""
                    <div class="sq-card">
                      <div class="sq-card__title">Connection</div>
                      <div class="sq-card__heading">{esc(name)} · {badge}</div>
                      <div class="sq-kv-grid">
                        <div><span class="sq-kv-label">Engine</span><strong>{esc(dialect)}</strong></div>
                        <div><span class="sq-kv-label">Database</span><strong>{esc(database)}</strong></div>
                        <div><span class="sq-kv-label">Host</span><strong>{esc(host)}</strong></div>
                        <div><span class="sq-kv-label">Status</span><strong>{"Connected" if active else "Saved"}</strong></div>
                      </div>
                    </div>
                    """
                )

    with tab_sec:
        section_title("Security posture", "Read-only enforcement")
        kv_card(
            "Enforcement",
            [
                ("App", f"{settings.app_name} v{settings.app_version}"),
                ("Read-only", settings.read_only_mode),
                ("Query timeout", f"{settings.query_timeout_seconds}s"),
                ("Max rows", settings.max_rows),
            ],
        )
        inject_html(
            """
            <div class="sq-card">
              <div class="sq-card__title">Guardrails</div>
              <ul class="sq-list">
                <li>Only SELECT / WITH / EXPLAIN / SHOW / DESCRIBE</li>
                <li>Destructive DDL/DML blocked by sqlglot AST validation</li>
                <li>Multi-statement payloads rejected</li>
              </ul>
            </div>
            """
        )
        section_title("Runtime limits", "From configuration")
        kv_card(
            "Limits",
            [
                ("Query timeout", f"{settings.query_timeout_seconds}s"),
                ("Max rows", settings.max_rows),
                ("Read only", settings.read_only_mode),
                ("Environment", settings.environment),
            ],
        )
        section_title("Audit")
        inject_html(
            f"""
            <div class="sq-card">
              <div class="sq-card__title">Audit log</div>
              <div class="sq-mono-block">{esc(settings.audit_log_path)}</div>
            </div>
            """
        )

    with tab_ws:
        section_title("Conversation memory", "Session-scoped chat history")
        card(
            title="Session memory",
            heading="Clear chat for this session",
            body=(
                "Removes chat messages and the compressed conversation summary. "
                "Long-term stores and history database are unchanged."
            ),
            kind="accent",
        )
        if st.button("Clear chat memory", use_container_width=True, key="settings_clear_memory"):
            st.session_state.memory.clear()
            st.session_state.chat_messages = []
            st.session_state.memory_summary = ""
            st.success("Chat memory cleared.")

        section_title("Export defaults", "Formats available after analysis")
        inject_html(
            f"""
            <div class="sq-card">
              <div class="sq-card__title">Formats</div>
              <div class="sq-stack-row sq-stack-row--tight">
                {status_badge("CSV", kind="default")}
                {status_badge("Excel", kind="info")}
                {status_badge("PDF", kind="accent")}
                {status_badge("JSON", kind="default")}
                {status_badge("Markdown", kind="ok")}
                {status_badge("SQL", kind="warn")}
              </div>
              <p class="sq-card__body sq-card__body--spaced">
                Downloads are available from Chat and the Exports page after a successful analysis.
              </p>
            </div>
            """
        )

        section_title("Paths")
        kv_card(
            "Workspace paths",
            [
                ("Audit log", settings.audit_log_path),
                ("Export dir", getattr(settings, "export_dir", "—")),
                ("Data dir", getattr(settings, "data_dir", "—")),
            ],
            columns=2,
        )

    page_footer(settings.app_version)
