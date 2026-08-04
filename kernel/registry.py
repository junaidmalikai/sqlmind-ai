"""Thread-safe capability registry — runtime discovery and registration."""

from __future__ import annotations

import threading
from typing import Any

from kernel.enums import CapabilityKind
from kernel.exceptions import (
    CapabilityConflictError,
    CapabilityNotFoundError,
    CapabilityNotRoutableError,
)
from kernel.models import CapabilityDescriptor, RegistrySnapshot
from utils.logging_config import get_logger

logger = get_logger(__name__)


class CapabilityRegistry:
    """Enterprise capability catalog with versioned snapshots.

    Planners and supervisors discover capabilities here instead of hardcoding
    ``if task == …``. Security-gated nodes remain registered but ``ai_routable=False``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        self._handlers: dict[str, Any] = {}
        self._route_index: dict[str, str] = {}  # route_tool_name → capability_id
        self._node_index: dict[str, str] = {}  # graph_node → capability_id
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def register(
        self,
        descriptor: CapabilityDescriptor,
        *,
        handler: Any = None,
        overwrite: bool = False,
    ) -> CapabilityDescriptor:
        with self._lock:
            existing = self._descriptors.get(descriptor.id)
            if existing is not None:
                if existing.system_protected and not overwrite:
                    raise CapabilityConflictError(
                        f"Protected capability cannot be overwritten: {descriptor.id}"
                    )
                if not overwrite:
                    raise CapabilityConflictError(
                        f"Capability already registered: {descriptor.id}"
                    )
                self._drop_indexes(existing)

            if descriptor.route_tool_name:
                owner = self._route_index.get(descriptor.route_tool_name)
                if owner and owner != descriptor.id:
                    raise CapabilityConflictError(
                        f"Route tool {descriptor.route_tool_name!r} already bound to {owner}"
                    )

            if descriptor.graph_node:
                owner = self._node_index.get(descriptor.graph_node)
                if owner and owner != descriptor.id:
                    raise CapabilityConflictError(
                        f"Graph node {descriptor.graph_node!r} already bound to {owner}"
                    )

            stored = descriptor.touch() if existing else descriptor
            self._descriptors[stored.id] = stored
            if handler is not None:
                self._handlers[stored.id] = handler
            elif stored.id in self._handlers and overwrite:
                pass  # keep prior handler unless replaced
            self._index(stored)
            self._version += 1
            logger.debug(
                "Registered capability id=%s kind=%s node=%s route=%s v=%s",
                stored.id,
                stored.kind.value,
                stored.graph_node,
                stored.route_tool_name,
                self._version,
            )
            return stored

    def unregister(self, capability_id: str) -> None:
        with self._lock:
            desc = self._descriptors.get(capability_id)
            if desc is None:
                raise CapabilityNotFoundError(capability_id)
            if desc.system_protected:
                raise CapabilityConflictError(
                    f"Protected capability cannot be unregistered: {capability_id}"
                )
            self._drop_indexes(desc)
            del self._descriptors[capability_id]
            self._handlers.pop(capability_id, None)
            self._version += 1

    def get(self, capability_id: str) -> CapabilityDescriptor:
        with self._lock:
            desc = self._descriptors.get(capability_id)
            if desc is None:
                raise CapabilityNotFoundError(capability_id)
            return desc

    def get_handler(self, capability_id: str) -> Any:
        with self._lock:
            if capability_id not in self._descriptors:
                raise CapabilityNotFoundError(capability_id)
            return self._handlers.get(capability_id)

    def list(
        self,
        *,
        kind: CapabilityKind | str | None = None,
        enabled_only: bool = True,
        ai_routable_only: bool = False,
    ) -> list[CapabilityDescriptor]:
        with self._lock:
            items = list(self._descriptors.values())
        if kind is not None:
            kind_val = kind.value if isinstance(kind, CapabilityKind) else str(kind)
            items = [c for c in items if c.kind.value == kind_val]
        if enabled_only:
            items = [c for c in items if c.enabled]
        if ai_routable_only:
            items = [c for c in items if c.ai_routable]
        return sorted(items, key=lambda c: c.id)

    def resolve_route_tool(self, route_tool_name: str) -> CapabilityDescriptor:
        """Map a supervisor tool-call name → capability (and thus graph node)."""
        with self._lock:
            cap_id = self._route_index.get(route_tool_name)
            if not cap_id:
                raise CapabilityNotFoundError(f"Unknown route tool: {route_tool_name}")
            desc = self._descriptors[cap_id]
            if not desc.enabled:
                raise CapabilityNotFoundError(f"Capability disabled: {cap_id}")
            if not desc.ai_routable:
                raise CapabilityNotRoutableError(
                    f"Capability not AI-routable: {cap_id}"
                )
            return desc

    def resolve_graph_node(self, graph_node: str) -> CapabilityDescriptor | None:
        with self._lock:
            cap_id = self._node_index.get(graph_node)
            if not cap_id:
                return None
            return self._descriptors.get(cap_id)

    def route_tool_to_node_map(self) -> dict[str, str]:
        """Compatibility map for existing supervisor routing."""
        with self._lock:
            out: dict[str, str] = {}
            for tool_name, cap_id in self._route_index.items():
                desc = self._descriptors[cap_id]
                if desc.graph_node and desc.enabled and desc.ai_routable:
                    out[tool_name] = desc.graph_node
            return out

    def graph_nodes(self) -> set[str]:
        with self._lock:
            return {
                d.graph_node
                for d in self._descriptors.values()
                if d.graph_node and d.enabled
            }

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return RegistrySnapshot(
                version=self._version,
                capabilities=list(self._descriptors.values()),
            )

    def record_outcome(
        self,
        capability_id: str,
        *,
        success: bool,
        latency_ms: float = 0.0,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock:
            desc = self._descriptors.get(capability_id)
            if desc is None:
                return
            stats = desc.stats.model_copy(deep=True)
            stats.record(
                success=success,
                latency_ms=latency_ms,
                tokens=tokens,
                cost_usd=cost_usd,
            )
            self._descriptors[capability_id] = desc.model_copy(update={"stats": stats})

    def _index(self, desc: CapabilityDescriptor) -> None:
        if desc.route_tool_name:
            self._route_index[desc.route_tool_name] = desc.id
        if desc.graph_node:
            self._node_index[desc.graph_node] = desc.id

    def _drop_indexes(self, desc: CapabilityDescriptor) -> None:
        if desc.route_tool_name and self._route_index.get(desc.route_tool_name) == desc.id:
            del self._route_index[desc.route_tool_name]
        if desc.graph_node and self._node_index.get(desc.graph_node) == desc.id:
            del self._node_index[desc.graph_node]
