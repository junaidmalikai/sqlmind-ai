"""Phase 1 — Autonomy Kernel: registry, matcher, DI, routing."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agents.nodes import make_supervisor_agent
from graph.state import empty_state
from kernel import (
    CapabilityConflictError,
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityMatcher,
    CapabilityNotFoundError,
    CapabilityNotRoutableError,
    CapabilityRegistry,
    CapabilityRequirement,
    RiskClass,
    bootstrap_builtin_capabilities,
    build_routing_tools_from_registry,
    create_kernel_container,
    resolve_next_node,
)
from kernel.container import KEY_MATCHER, KEY_REGISTRY
from kernel.enums import ExecutionStrategy
from tools.routing_tools import ROUTE_TOOL_TO_NODE, build_routing_tools


def test_bootstrap_registers_agents_and_tools():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    agents = reg.list(kind=CapabilityKind.AGENT)
    tools = reg.list(kind=CapabilityKind.TOOL)
    assert len(agents) >= 15
    assert len(tools) >= 8
    assert reg.get("agent.sql").graph_node == "sql_agent"
    assert reg.get("node.validation").ai_routable is False
    assert reg.get("node.execution").system_protected is True


def test_security_nodes_not_ai_routable():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    routable = {c.id for c in reg.list(kind=CapabilityKind.AGENT, ai_routable_only=True)}
    assert "node.validation" not in routable
    assert "node.execution" not in routable
    assert "agent.sql" in routable
    assert "agent.finalize" in routable

    # Explicitly non-routable capability with a route tool name is rejected
    reg.register(
        CapabilityDescriptor(
            id="agent.gated_demo",
            kind=CapabilityKind.AGENT,
            name="Gated",
            description="gated",
            graph_node="gated_demo",
            route_tool_name="route_to_gated_demo",
            ai_routable=False,
            system_protected=False,
        ),
        overwrite=True,
    )
    with pytest.raises(CapabilityNotRoutableError):
        resolve_next_node(reg, "route_to_gated_demo")
    reg.unregister("agent.gated_demo")


def test_route_resolution_from_registry():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    assert resolve_next_node(reg, "route_to_sql_agent") == "sql_agent"
    assert resolve_next_node(reg, "ask_user_clarification") == "clarify"
    with pytest.raises(CapabilityNotFoundError):
        resolve_next_node(reg, "route_to_unknown")


def test_routing_tools_built_from_registry():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    tools = build_routing_tools_from_registry(reg)
    names = {t.name for t in tools}
    assert "route_to_sql_agent" in names
    assert "route_to_finalize" in names
    assert "ask_user_clarification" in names
    # No tool for non-routable security nodes
    assert "route_to_validation_agent" not in names


def test_legacy_route_map_compatible():
    assert ROUTE_TOOL_TO_NODE["route_to_sql_agent"] == "sql_agent"
    assert ROUTE_TOOL_TO_NODE["route_to_visualization_agent"] == "visualization_agent"
    tools = build_routing_tools()
    assert len(tools) >= 10


def test_matcher_selects_sql_by_skills_not_if_task():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    matcher = CapabilityMatcher()
    matches = matcher.match(
        CapabilityRequirement(
            description="write a select query for revenue",
            required_skills=frozenset({"nl2sql"}),
            kind=CapabilityKind.AGENT,
        ),
        reg,
    )
    assert matches
    assert matches[0].capability_id == "agent.sql"
    assert matches[0].score > 0.4


def test_matcher_visualize_and_insights():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    matcher = CapabilityMatcher()
    viz = matcher.best(
        CapabilityRequirement(
            required_skills=frozenset({"visualize"}),
            kind=CapabilityKind.AGENT,
        ),
        reg,
    )
    assert viz is not None
    assert viz.capability_id == "agent.visualization"

    insight = matcher.best(
        CapabilityRequirement(
            provides_any=frozenset({"insights"}),
            kind=CapabilityKind.AGENT,
        ),
        reg,
    )
    assert insight is not None
    assert insight.capability_id == "agent.insight"


def test_matcher_respects_max_risk():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    matcher = CapabilityMatcher()
    # SQL agent is MEDIUM risk — blocked when max_risk=LOW
    matches = matcher.match(
        CapabilityRequirement(
            required_skills=frozenset({"nl2sql"}),
            max_risk=RiskClass.LOW,
            kind=CapabilityKind.AGENT,
        ),
        reg,
    )
    assert matches == []


def test_protected_capability_cannot_unregister():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    with pytest.raises(CapabilityConflictError):
        reg.unregister("agent.sql")


def test_runtime_plugin_registration():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    v0 = reg.version
    reg.register(
        CapabilityDescriptor(
            id="agent.custom_kpi",
            kind=CapabilityKind.AGENT,
            name="Custom KPI",
            description="Compute custom KPIs",
            skills=frozenset({"kpi", "custom"}),
            tags=frozenset({"plugin"}),
            graph_node="plugin_kpi_agent",
            route_tool_name="route_to_custom_kpi",
            route_tool_description="Run custom KPI capability",
            ai_routable=True,
            system_protected=False,
            provides=frozenset({"dashboard_spec"}),
        ),
        overwrite=False,
    )
    assert reg.version > v0
    assert resolve_next_node(reg, "route_to_custom_kpi") == "plugin_kpi_agent"
    tools = build_routing_tools_from_registry(reg)
    assert any(t.name == "route_to_custom_kpi" for t in tools)
    reg.unregister("agent.custom_kpi")


def test_duplicate_route_tool_conflict():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    with pytest.raises(CapabilityConflictError):
        reg.register(
            CapabilityDescriptor(
                id="agent.dup",
                kind=CapabilityKind.AGENT,
                name="Dup",
                description="Dup",
                graph_node="plugin_dup_node",
                route_tool_name="route_to_sql_agent",
                ai_routable=True,
            )
        )


def test_stats_recording():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    reg.record_outcome("agent.sql", success=True, latency_ms=120.0, tokens=50)
    reg.record_outcome("agent.sql", success=False, latency_ms=80.0)
    stats = reg.get("agent.sql").stats
    assert stats.invocations == 2
    assert stats.successes == 1
    assert stats.success_rate == 0.5


def test_container_di():
    c = create_kernel_container()
    assert isinstance(c.resolve(KEY_REGISTRY), CapabilityRegistry)
    assert isinstance(c.resolve(KEY_MATCHER), CapabilityMatcher)


def test_catalog_snapshot_text():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    text = reg.snapshot().catalog_text()
    assert "agent.sql" in text
    assert "node.validation" not in text  # not routable


def test_supervisor_resolves_via_registry():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)

    class FakeLLM:
        def history_messages(self, history, limit: int = 12):  # noqa: ANN001
            return []

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001
            def _invoke(messages):  # noqa: ANN001
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "route_to_sql_agent",
                            "args": {"reasoning": "need SQL"},
                        }
                    ],
                )

            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(_invoke)

    agent = make_supervisor_agent(FakeLLM(), registry=reg)  # type: ignore[arg-type]
    out = agent(empty_state("top customers"))
    assert out["next_agent"] == "sql_agent"
    assert "sql_agent" in out["route_history"]


def test_register_langchain_tool_handler():
    reg = CapabilityRegistry()

    def _fn(x: str = "") -> str:
        return "ok"

    tool = StructuredTool.from_function(func=_fn, name="schema_tool", description="schema")
    bootstrap_builtin_capabilities(reg, tools=[tool])
    handler = reg.get_handler("tool.schema_tool")
    assert handler is not None
    assert handler.name == "schema_tool"


def test_execution_strategy_on_sql():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    assert reg.get("agent.sql").execution_strategy == ExecutionStrategy.SECURITY_GATED
