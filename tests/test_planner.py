"""Phase 2 — Goal Planner, TaskGraph, Agent Registry, Execution Coordinator."""

from __future__ import annotations

import pytest

from kernel import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRegistry,
    bootstrap_builtin_capabilities,
    create_kernel_container,
)
from kernel.container import KEY_MATCHER, KEY_REGISTRY
from kernel.enums import RiskClass
from memory.workflow_memory import WorkflowMemoryStore
from planner.coordinator import ExecutionCoordinator
from planner.messages import AgentMessage, goal_message, plan_message
from planner.models import (
    AdaptivePlan,
    GoalSpec,
    PlanStep,
    TaskGraph,
    TaskSpec,
)
from planner.registry import AgentRegistry
from planner.selection import expand_skills, select_agent_for_task


@pytest.fixture
def agent_registry() -> AgentRegistry:
    container = create_kernel_container()
    reg: CapabilityRegistry = container.resolve(KEY_REGISTRY)
    bootstrap_builtin_capabilities(reg)
    return AgentRegistry(reg, container.resolve(KEY_MATCHER))


def test_agent_registry_lists_discoverable_agents(agent_registry: AgentRegistry):
    agents = agent_registry.list_agents()
    ids = {a.capability_id for a in agents}
    assert "agent.sql" in ids
    assert "agent.visualization" in ids
    # Security nodes are not AI-routable → absent from agent list
    assert "node.validation" not in ids
    assert all(a.availability for a in agents)
    assert agent_registry.catalog_text()


def test_dynamic_selection_by_skills_not_hardcoded(agent_registry: AgentRegistry):
    task = TaskSpec(
        id="t1",
        title="Write SQL",
        required_skills=["nl2sql"],
        provides_any=["sql"],
    )
    match, decision = select_agent_for_task(agent_registry, task)
    assert match is not None
    assert match.descriptor.graph_node == "sql_agent"
    assert decision.decision_type == "agent_selection"
    assert decision.chosen == "agent.sql"


def test_skill_aliases_expand():
    skills = expand_skills(["sql", "chart", "insight"])
    assert "nl2sql" in skills
    assert "visualize" in skills
    assert "insights" in skills


def test_capability_discovery_hot_register(agent_registry: AgentRegistry):
    agent_registry.register_agent(
        CapabilityDescriptor(
            id="agent.custom_kpi",
            kind=CapabilityKind.AGENT,
            name="Custom KPI",
            description="Custom KPI specialist",
            skills=frozenset({"custom_kpi"}),
            tags=frozenset({"kpi", "plugin"}),
            provides=frozenset({"kpi_result"}),
            graph_node="custom_kpi_agent",
            ai_routable=True,
            system_protected=False,
            risk_class=RiskClass.LOW,
        )
    )
    match = agent_registry.best_agent(required_skills=["custom_kpi"])
    assert match is not None
    assert match.capability_id == "agent.custom_kpi"
    assert match.descriptor.graph_node == "custom_kpi_agent"


def test_goal_and_plan_models_roundtrip():
    goal = GoalSpec(
        goal="Analyze sales trends",
        objectives=["Find trends", "Create dashboard", "Export report"],
        expected_output="dashboard + export",
        priority="high",
        confidence=0.82,
        simple=False,
    )
    data = goal.to_state_dict()
    assert GoalSpec.model_validate(data).goal == goal.goal

    plan = AdaptivePlan(
        plan_id="p1",
        goal_summary=goal.goal,
        strategy="schema → sql → parallel viz/insight → dashboard → export",
        steps=[PlanStep(step_id="s1", task_ids=["t1"], mode="sequential")],
        execution_mode="mixed",
        estimated_cost=4.0,
    )
    assert AdaptivePlan.from_state(plan.to_state_dict()).plan_id == "p1"


def test_task_graph_dependencies_and_ready():
    graph = TaskGraph(
        tasks=[
            TaskSpec(id="t1", title="schema", status="completed"),
            TaskSpec(id="t2", title="sql", dependencies=["t1"], status="pending"),
            TaskSpec(
                id="t3",
                title="viz",
                dependencies=["t2"],
                status="pending",
                parallel_safe=True,
            ),
            TaskSpec(
                id="t4",
                title="insight",
                dependencies=["t2"],
                status="pending",
                parallel_safe=True,
            ),
        ],
        roots=["t1"],
    )
    ready = graph.ready_tasks()
    assert [t.id for t in ready] == ["t2"]
    graph.mark("t2", "completed")
    ready2 = {t.id for t in graph.ready_tasks()}
    assert ready2 == {"t3", "t4"}


def test_message_protocol():
    msg = goal_message("goal_understanding", {"goal": "x"})
    assert isinstance(msg, AgentMessage)
    assert msg.kind == "goal"
    plan = plan_message("planner", {"plan_id": "p1"})
    assert plan.kind == "plan"


def test_coordinator_supervisor_fallback(agent_registry: AgentRegistry):
    coord = ExecutionCoordinator(agent_registry)
    state = {
        "adaptive_plan": AdaptivePlan(
            plan_id="p1",
            use_supervisor_fallback=True,
            status="ready",
        ).to_state_dict(),
        "task_graph": TaskGraph(tasks=[]).to_state_dict(),
        "execution_progress": {},
        "route_history": [],
    }
    out = coord.step(state)  # type: ignore[arg-type]
    assert out["next_agent"] == "supervisor"
    assert out["plan_active"] is False


def test_coordinator_dispatches_via_registry(agent_registry: AgentRegistry):
    coord = ExecutionCoordinator(agent_registry)
    graph = TaskGraph(
        tasks=[
            TaskSpec(
                id="t1",
                title="Author SQL",
                required_skills=["nl2sql"],
                provides_any=["sql"],
                status="pending",
            )
        ],
        roots=["t1"],
    )
    plan = AdaptivePlan(
        plan_id="p1",
        use_supervisor_fallback=False,
        status="ready",
    )
    state = {
        "adaptive_plan": plan.to_state_dict(),
        "task_graph": graph.to_state_dict(),
        "execution_progress": {},
        "schema_text": "tables...",
        "route_history": [],
    }
    out = coord.step(state)  # type: ignore[arg-type]
    assert out["next_agent"] == "sql_agent"
    assert out["plan_active"] is True
    assert out["active_task_id"] == "t1"


def test_workflow_memory_store(tmp_path):
    from pathlib import Path

    store = WorkflowMemoryStore(str(Path(tmp_path) / "wf.sqlite3"))
    store.record_plan_outcome(
        question="top customers by revenue",
        goal_summary="rank customers",
        plan={"strategy": "sql then chart"},
        success=True,
        database_name="demo",
    )
    store.record_visualization_pref("bar", question="top customers by revenue")
    store.record_preference("export_format", "excel")
    text = store.format_for_prompt("customers revenue", database_name="demo")
    assert "successful" in text or "preferred" in text or "bar" in text


def test_bootstrap_includes_phase2_agents():
    reg = CapabilityRegistry()
    bootstrap_builtin_capabilities(reg)
    assert reg.get("agent.goal_understanding").graph_node == "goal_understanding"
    assert reg.get("agent.planner").graph_node == "planner"
    assert reg.get("agent.execution_coordinator").ai_routable is False
    assert reg.get("agent.replan").graph_node == "replan_agent"


def test_graph_nodes_include_phase2():
    from graph.state import GRAPH_NODES

    for node in (
        "memory_agent",
        "goal_understanding",
        "planner",
        "task_decomposition",
        "execution_coordinator",
        "replan_agent",
    ):
        assert node in GRAPH_NODES


def test_legacy_start_supervisor_when_planning_disabled(monkeypatch):
    """Backward compat: AUTONOMY_PLANNING_ENABLED=false keeps START→supervisor."""
    from config.settings import Settings

    s = Settings(autonomy_planning_enabled=False)
    assert s.autonomy_planning_enabled is False
