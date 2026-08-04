"""Supervisor routing tools — LLM tool-calls select the next specialist.

Phase 1+: the capability registry is the source of truth. This module keeps the
legacy ``ROUTE_TOOL_TO_NODE`` / ``build_routing_tools`` API so existing imports
continue to work, backed by a process-level builtin registry bootstrap.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from kernel.bootstrap import bootstrap_builtin_capabilities
from kernel.registry import CapabilityRegistry
from kernel.routing import (
    ClarifyArgs,
    EmptyArgs,
    RouteArgs,
    build_routing_tools_from_registry,
    legacy_route_map,
)

__all__ = [
    "ClarifyArgs",
    "EmptyArgs",
    "ROUTE_TOOL_TO_NODE",
    "RouteArgs",
    "build_routing_tools",
    "get_default_routing_registry",
]


@lru_cache(maxsize=1)
def get_default_routing_registry() -> CapabilityRegistry:
    """Process-level registry used when callers have not injected one."""
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg, tools=None, overwrite=True)
    return reg


# Backward-compatible map — derived from builtin capability catalog
ROUTE_TOOL_TO_NODE: dict[str, str] = legacy_route_map(get_default_routing_registry())


def build_routing_tools(
    registry: CapabilityRegistry | None = None,
) -> list[StructuredTool]:
    """Tools the Supervisor may call to choose the next node.

    When ``registry`` is provided (orchestrator path), tools reflect the live
    catalog including any runtime-registered plugins. Otherwise uses builtins.
    """
    return build_routing_tools_from_registry(registry or get_default_routing_registry())


# Re-export Field for any external importers that poked at pydantic models here
Field = Field
BaseModel = BaseModel
