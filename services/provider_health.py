"""Validate LLM providers: API keys (live / revoked) and Ollama model presence."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings
from core.exceptions import LLMProviderError


@dataclass
class ProviderCheckResult:
    ok: bool
    message: str
    available_models: list[str] = field(default_factory=list)


def _http_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200)
            try:
                return status, json.loads(body) if body else {}
            except json.JSONDecodeError:
                return status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            payload: Any = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = body
        return exc.code, payload
    except urllib.error.URLError as exc:
        raise LLMProviderError(f"Cannot reach provider endpoint: {exc.reason}") from exc


def _auth_failure_message(provider: str, status: int, payload: Any) -> str:
    detail = ""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("status") or "")
        elif isinstance(err, str):
            detail = err
        else:
            detail = str(payload.get("message") or "")
    detail = detail.strip()
    if status in {401, 403}:
        base = (
            f"{provider} API key is invalid or revoked (HTTP {status}). "
            "Check the key in the provider console."
        )
        return f"{base} Details: {detail}" if detail else base
    return f"{provider} API check failed (HTTP {status})" + (f": {detail}" if detail else "")


def _require_key(label: str, key: str) -> None:
    if not (key or "").strip():
        raise LLMProviderError(f"{label} API key is required.")


def list_ollama_models(base_url: str, *, timeout: float = 10.0) -> list[str]:
    """Return installed Ollama model names (e.g. qwen3:8b)."""
    url = base_url.rstrip("/") + "/api/tags"
    status, payload = _http_json(url, timeout=timeout)
    if status != 200 or not isinstance(payload, dict):
        raise LLMProviderError(f"Ollama is not reachable at {base_url} (HTTP {status}).")
    models = []
    for item in payload.get("models") or []:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            models.append(str(name))
    return sorted(models)


def _model_installed(requested: str, installed: list[str]) -> bool:
    req = requested.strip()
    if not req:
        return False
    if req in installed or f"{req}:latest" in installed:
        return True
    # "qwen3" matches any installed "qwen3:…" tag
    if ":" not in req:
        return any(name == req or name.startswith(req + ":") for name in installed)
    return False


def validate_provider(settings: Settings) -> ProviderCheckResult:
    """Live-check the configured provider credentials / local model."""
    provider = settings.llm_provider
    model = settings.resolve_model()

    if provider == "ollama":
        installed = list_ollama_models(settings.ollama_base_url)
        if not installed:
            return ProviderCheckResult(
                ok=False,
                message="Ollama is running but no models are installed. Pull one with `ollama pull qwen3:8b`.",
                available_models=[],
            )
        if not _model_installed(model, installed):
            preview = ", ".join(installed[:8])
            more = f" (+{len(installed) - 8} more)" if len(installed) > 8 else ""
            return ProviderCheckResult(
                ok=False,
                message=(
                    f"Model `{model}` is not installed in Ollama. "
                    f"Available: {preview}{more}. "
                    f"Run `ollama pull {model}` or pick an installed model."
                ),
                available_models=installed,
            )
        return ProviderCheckResult(
            ok=True,
            message=f"Ollama OK — model `{model}` is available.",
            available_models=installed,
        )

    if provider == "openai":
        _require_key("OpenAI", settings.openai_api_key)
        status, payload = _http_json(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {settings.openai_api_key.strip()}"},
        )
        if status in {401, 403}:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("OpenAI", status, payload))
        if status != 200:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("OpenAI", status, payload))
        names = _extract_openai_style_models(payload)
        return ProviderCheckResult(
            ok=True,
            message=f"OpenAI API key is valid — using `{model}`.",
            available_models=names,
        )

    if provider == "claude":
        _require_key("Anthropic", settings.anthropic_api_key)
        status, payload = _http_json(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": settings.anthropic_api_key.strip(),
                "anthropic-version": "2023-06-01",
            },
        )
        if status in {401, 403}:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Claude", status, payload))
        if status != 200:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Claude", status, payload))
        names = _extract_openai_style_models(payload)
        return ProviderCheckResult(
            ok=True,
            message=f"Claude API key is valid — using `{model}`.",
            available_models=names,
        )

    if provider == "gemini":
        _require_key("Gemini", settings.gemini_api_key)
        key = settings.gemini_api_key.strip()
        status, payload = _http_json(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        )
        if status in {401, 403}:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Gemini", status, payload))
        if status != 200:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Gemini", status, payload))
        names: list[str] = []
        if isinstance(payload, dict):
            for item in payload.get("models") or []:
                if isinstance(item, dict) and item.get("name"):
                    # "models/gemini-2.0-flash" -> "gemini-2.0-flash"
                    raw = str(item["name"])
                    names.append(raw.split("/", 1)[-1])
        return ProviderCheckResult(
            ok=True,
            message=f"Gemini API key is valid — using `{model}`.",
            available_models=sorted(names),
        )

    if provider == "groq":
        _require_key("Groq", settings.groq_api_key)
        status, payload = _http_json(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.groq_api_key.strip()}"},
        )
        if status in {401, 403}:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Groq", status, payload))
        if status != 200:
            return ProviderCheckResult(ok=False, message=_auth_failure_message("Groq", status, payload))
        names = _extract_openai_style_models(payload)
        return ProviderCheckResult(
            ok=True,
            message=f"Groq API key is valid — using `{model}`.",
            available_models=names,
        )

    raise LLMProviderError(f"Unsupported LLM provider: {provider}")


def _extract_openai_style_models(payload: Any) -> list[str]:
    names: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                names.append(str(item["id"]))
    return sorted(names)


# Sensible defaults when switching provider in the UI
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3:8b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-sonnet-4-20250514",
    "groq": "llama-3.3-70b-versatile",
}
