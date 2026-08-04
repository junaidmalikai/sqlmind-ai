"""LLM / activity gating helpers."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from ui.components import card, empty_state


def provider_api_key(settings: Settings) -> str:
    p = settings.llm_provider
    if p == "openai":
        return (settings.openai_api_key or "").strip()
    if p == "gemini":
        return (settings.gemini_api_key or "").strip()
    if p == "claude":
        return (settings.anthropic_api_key or "").strip()
    if p == "groq":
        return (settings.groq_api_key or "").strip()
    return ""  # ollama has no API key


def is_llm_ready(settings: Settings) -> tuple[bool, str]:
    """Return (ok, reason). Blocks AI activity until provider is applied successfully."""
    provider = settings.llm_provider
    if provider != "ollama" and not provider_api_key(settings):
        return (
            False,
            f"Add your {provider.upper()} API key in the sidebar or Settings, then click Apply LLM Settings.",
        )
    if not st.session_state.get("llm_ready") and not st.session_state.get("llm_service"):
        if provider == "ollama":
            return (
                False,
                "Configure Ollama and click Apply LLM Settings before using Chat or other AI features.",
            )
        return (
            False,
            "Click Apply LLM Settings to validate your API key before using Chat or other AI features.",
        )
    return True, ""


def require_llm(settings: Settings, *, action: str = "continue") -> bool:
    """Show a gate UI and return False if LLM is not ready."""
    ok, reason = is_llm_ready(settings)
    if ok:
        return True
    card(
        title="LLM setup required",
        heading=f"Configure a provider to {action}",
        body=reason,
        kind="warn",
    )
    if st.button("Open Settings", type="primary", use_container_width=True, key=f"gate_settings_{action}"):
        st.session_state.nav_page = "Settings"
        st.rerun()
    return False


def require_db_and_llm(settings: Settings, *, action: str = "analyze") -> bool:
    if not st.session_state.get("connected"):
        empty_state(
            "Connect a database",
            "Open Workspace → Database connection in the sidebar. A built-in sample SQLite database is available for demos.",
            icon="DB",
        )
        return False
    return require_llm(settings, action=action)
