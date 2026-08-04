"""Registry-driven supervisor routing tools (replaces hardcoded route table as source of truth)."""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from kernel.models import CapabilityDescriptor
from kernel.registry import CapabilityRegistry


class EmptyArgs(BaseModel):
    """No arguments — selecting the tool is the routing decision."""


class ClarifyArgs(BaseModel):
    question: str = Field(description="Clarifying question to ask the user")


class RouteArgs(BaseModel):
    reasoning: str = Field(
        default="",
        description="Brief reason for choosing this next step",
    )


def _noop(**_kwargs: object) -> str:
    return "routed"


def build_routing_tools_from_registry(registry: CapabilityRegistry) -> list[StructuredTool]:
    """Build supervisor ``bind_tools`` list from AI-routable agent capabilities.

    Order is stable (sorted by capability id) for prompt cache friendliness.
    """
    tools: list[StructuredTool] = []
    for desc in registry.list(kind="agent", enabled_only=True, ai_routable_only=True):
        if not desc.route_tool_name or not desc.graph_node:
            continue
        schema: type[BaseModel] = (
            ClarifyArgs if desc.route_tool_name == "ask_user_clarification" else RouteArgs
        )
        description = desc.route_tool_description or desc.description
        tools.append(
            StructuredTool.from_function(
                func=_noop,
                name=desc.route_tool_name,
                description=description,
                args_schema=schema,
            )
        )
    # Stable order
    tools.sort(key=lambda t: t.name)
    return tools


def resolve_next_node(registry: CapabilityRegistry, route_tool_name: str) -> str:
    """Resolve supervisor tool call → LangGraph node via registry."""
    desc = registry.resolve_route_tool(route_tool_name)
    if not desc.graph_node:
        raise ValueError(f"Capability {desc.id} has no graph_node")
    return desc.graph_node


def legacy_route_map(registry: CapabilityRegistry) -> dict[str, str]:
    """Export tool→node map for backward-compatible imports."""
    return registry.route_tool_to_node_map()


def describe_routable_catalog(registry: CapabilityRegistry) -> str:
    """Short catalog text for supervisor / future planner prompts."""
    return registry.snapshot().catalog_text(routable_only=True)


def routable_descriptors(registry: CapabilityRegistry) -> list[CapabilityDescriptor]:
    return [
        d
        for d in registry.list(kind="agent", enabled_only=True, ai_routable_only=True)
        if d.route_tool_name and d.graph_node
    ]
