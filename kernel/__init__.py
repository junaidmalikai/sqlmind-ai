"""SQLMind Autonomy Kernel — capability registry, matching, DI, routing.

Phase 1 of the Autonomous Platform evolution. Existing agents/tools are
registered as discoverable capabilities; planners (Phase 2+) select via
matching instead of hardcoded task switches.
"""

from __future__ import annotations

from kernel.bootstrap import (
    bootstrap_builtin_capabilities,
    create_kernel_container,
    register_langchain_tools,
)
from kernel.container import (
    KEY_LLM,
    KEY_MATCHER,
    KEY_REGISTRY,
    KEY_SECURITY,
    KEY_SETTINGS,
    ServiceContainer,
)
from kernel.enums import CapabilityKind, ExecutionStrategy, ModelTier, RiskClass
from kernel.exceptions import (
    CapabilityConflictError,
    CapabilityNotFoundError,
    CapabilityNotRoutableError,
    ContainerError,
    KernelError,
)
from kernel.matching import CapabilityMatcher
from kernel.models import (
    CapabilityDescriptor,
    CapabilityMatch,
    CapabilityRequirement,
    CapabilityStats,
    RegistrySnapshot,
)
from kernel.registry import CapabilityRegistry
from kernel.routing import (
    build_routing_tools_from_registry,
    describe_routable_catalog,
    legacy_route_map,
    resolve_next_node,
)

__all__ = [
    "KEY_LLM",
    "KEY_MATCHER",
    "KEY_REGISTRY",
    "KEY_SECURITY",
    "KEY_SETTINGS",
    "CapabilityConflictError",
    "CapabilityDescriptor",
    "CapabilityKind",
    "CapabilityMatch",
    "CapabilityMatcher",
    "CapabilityNotFoundError",
    "CapabilityNotRoutableError",
    "CapabilityRegistry",
    "CapabilityRequirement",
    "CapabilityStats",
    "ContainerError",
    "ExecutionStrategy",
    "KernelError",
    "ModelTier",
    "RegistrySnapshot",
    "RiskClass",
    "ServiceContainer",
    "bootstrap_builtin_capabilities",
    "build_routing_tools_from_registry",
    "create_kernel_container",
    "describe_routable_catalog",
    "legacy_route_map",
    "register_langchain_tools",
    "resolve_next_node",
]
