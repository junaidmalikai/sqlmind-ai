"""Unit tests for provider API-key / Ollama model validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import Settings
from services.provider_health import (
    ProviderCheckResult,
    _model_installed,
    validate_provider,
)


def test_model_installed_exact_and_base() -> None:
    installed = ["qwen3:8b", "minimax-m2.5:cloud"]
    assert _model_installed("qwen3:8b", installed)
    assert _model_installed("qwen3", installed)
    assert not _model_installed("llama3.2", installed)


def test_ollama_missing_model() -> None:
    settings = Settings(llm_provider="ollama", llm_model="llama3.2", ollama_base_url="http://localhost:11434")
    with patch(
        "services.provider_health.list_ollama_models",
        return_value=["qwen3:8b", "qwen3.5:cloud"],
    ):
        result = validate_provider(settings)
    assert isinstance(result, ProviderCheckResult)
    assert result.ok is False
    assert "llama3.2" in result.message
    assert "qwen3:8b" in result.message


def test_ollama_ok() -> None:
    settings = Settings(llm_provider="ollama", llm_model="qwen3:8b")
    with patch(
        "services.provider_health.list_ollama_models",
        return_value=["qwen3:8b"],
    ):
        result = validate_provider(settings)
    assert result.ok is True


def test_openai_missing_key() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="", openai_model="gpt-4o-mini")
    with pytest.raises(Exception) as exc:
        validate_provider(settings)
    assert "required" in str(exc.value).lower()


def test_openai_revoked_key() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-revoked",
        openai_model="gpt-4o-mini",
    )
    with patch(
        "services.provider_health._http_json",
        return_value=(401, {"error": {"message": "Incorrect API key provided"}}),
    ):
        result = validate_provider(settings)
    assert result.ok is False
    assert "invalid or revoked" in result.message.lower()


def test_openai_valid_key() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-live",
        openai_model="gpt-4o-mini",
    )
    with patch(
        "services.provider_health._http_json",
        return_value=(200, {"data": [{"id": "gpt-4o-mini"}]}),
    ):
        result = validate_provider(settings)
    assert result.ok is True
    assert "valid" in result.message.lower()


def test_claude_revoked_key() -> None:
    settings = Settings(
        llm_provider="claude",
        anthropic_api_key="sk-ant-bad",
        claude_model="claude-sonnet-4-20250514",
    )
    with patch(
        "services.provider_health._http_json",
        return_value=(401, {"error": {"message": "invalid x-api-key"}}),
    ):
        result = validate_provider(settings)
    assert result.ok is False
    assert "revoked" in result.message.lower() or "invalid" in result.message.lower()


def test_gemini_valid_key() -> None:
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="AIza-test",
        gemini_model="gemini-2.0-flash",
    )
    with patch(
        "services.provider_health._http_json",
        return_value=(200, {"models": [{"name": "models/gemini-2.0-flash"}]}),
    ):
        result = validate_provider(settings)
    assert result.ok is True
    assert "gemini-2.0-flash" in result.available_models
