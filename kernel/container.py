"""Lightweight dependency injection container for kernel services."""

from __future__ import annotations

from typing import Any, TypeVar

from kernel.exceptions import ContainerError

T = TypeVar("T")

# Well-known service keys
KEY_REGISTRY = "capability_registry"
KEY_MATCHER = "capability_matcher"
KEY_SETTINGS = "settings"
KEY_LLM = "llm_service"
KEY_SECURITY = "sql_security_guard"


class ServiceContainer:
    """Simple typed service locator used during graph bootstrap.

    Prefer constructor injection at call sites; the container holds process-scoped
    singletons (registry, matcher) so Streamlit / API / workers share one catalog.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, key: str, instance: Any, *, overwrite: bool = False) -> None:
        if key in self._services and not overwrite:
            raise ContainerError(f"Service already registered: {key}")
        self._services[key] = instance

    def resolve(self, key: str) -> Any:
        if key not in self._services:
            raise ContainerError(f"Service not found: {key}")
        return self._services[key]

    def resolve_typed(self, key: str, expected: type[T]) -> T:
        inst = self.resolve(key)
        if not isinstance(inst, expected):
            raise ContainerError(
                f"Service {key!r} expected {expected.__name__}, got {type(inst).__name__}"
            )
        return inst

    def has(self, key: str) -> bool:
        return key in self._services

    def keys(self) -> list[str]:
        return sorted(self._services.keys())
