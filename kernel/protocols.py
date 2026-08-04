"""Protocol interfaces for the autonomy kernel (SOLID — Interface Segregation)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kernel.models import (
    CapabilityDescriptor,
    CapabilityMatch,
    CapabilityRequirement,
    RegistrySnapshot,
)


@runtime_checkable
class CapabilityRegistryProtocol(Protocol):
    """Discoverable catalog of agents, tools, models, and other plugins."""

    @property
    def version(self) -> int: ...

    def register(
        self,
        descriptor: CapabilityDescriptor,
        *,
        handler: Any = None,
        overwrite: bool = False,
    ) -> CapabilityDescriptor: ...

    def unregister(self, capability_id: str) -> None: ...

    def get(self, capability_id: str) -> CapabilityDescriptor: ...

    def get_handler(self, capability_id: str) -> Any: ...

    def list(
        self,
        *,
        kind: str | None = None,
        enabled_only: bool = True,
        ai_routable_only: bool = False,
    ) -> list[CapabilityDescriptor]: ...

    def resolve_route_tool(self, route_tool_name: str) -> CapabilityDescriptor: ...

    def resolve_graph_node(self, graph_node: str) -> CapabilityDescriptor | None: ...

    def snapshot(self) -> RegistrySnapshot: ...

    def record_outcome(
        self,
        capability_id: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None: ...


@runtime_checkable
class CapabilityMatcherProtocol(Protocol):
    """Scores capabilities against a declarative requirement (no if-task branching)."""

    def match(
        self,
        requirement: CapabilityRequirement,
        registry: CapabilityRegistryProtocol,
    ) -> list[CapabilityMatch]: ...


@runtime_checkable
class ServiceContainerProtocol(Protocol):
    """Minimal DI container for kernel services."""

    def register(self, key: str, instance: Any, *, overwrite: bool = False) -> None: ...

    def resolve(self, key: str) -> Any: ...

    def has(self, key: str) -> bool: ...
