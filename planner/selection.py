"""Dynamic agent selection — Planner/Coordinator query the registry, never hardcode."""

from __future__ import annotations

from typing import Any

from kernel.models import CapabilityMatch
from planner.messages import decision_message
from planner.models import AutonDecision, TaskSpec
from planner.registry import AgentRegistry


# Soft skill aliases so LLM-decomposed skills still match catalog skills
_SKILL_ALIASES: dict[str, frozenset[str]] = {
    "sql": frozenset({"nl2sql", "sql_react", "query_authoring"}),
    "nl2sql": frozenset({"nl2sql", "sql_react"}),
    "query": frozenset({"nl2sql", "sql_react"}),
    "chart": frozenset({"visualize", "chart_recommend"}),
    "visualize": frozenset({"visualize", "chart_recommend"}),
    "visualization": frozenset({"visualize", "chart_recommend"}),
    "insight": frozenset({"insights", "business_analysis"}),
    "insights": frozenset({"insights", "business_analysis"}),
    "dashboard": frozenset({"dashboard_design", "kpi"}),
    "kpi": frozenset({"dashboard_design", "kpi"}),
    "export": frozenset({"export", "export_graph"}),
    "plugin": frozenset({"plugin", "skill", "marketplace"}),
    "skill": frozenset({"plugin", "skill", "marketplace"}),
    "recovery": frozenset({"recovery", "retry", "fallback"}),
    "schema": frozenset({"schema_discovery", "suggest_questions"}),
    "summary": frozenset({"database_summary"}),
    "optimize": frozenset({"sql_optimization", "explain"}),
    "optimization": frozenset({"sql_optimization", "explain"}),
    "reflect": frozenset({"reflection", "self_critique"}),
    "reflection": frozenset({"reflection", "self_critique", "verification"}),
    "finalize": frozenset({"finalize", "answer_assembly"}),
    "clarify": frozenset({"clarification", "hitl"}),
}


def expand_skills(skills: list[str] | None) -> frozenset[str]:
    """Expand human/LLM skill labels into registry skill tokens."""
    out: set[str] = set()
    for s in skills or []:
        key = (s or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        out.add(key)
        if key in _SKILL_ALIASES:
            out |= set(_SKILL_ALIASES[key])
    return frozenset(out)


def select_agent_for_task(
    registry: AgentRegistry,
    task: TaskSpec,
) -> tuple[CapabilityMatch | None, AutonDecision]:
    """Choose the best capable agent from the registry for a task.

    Returns (match|None, decision record). Never uses if-task hardcoded routing.
    """
    skills = expand_skills(task.required_skills)
    tags = frozenset(t.lower() for t in (task.preferred_tags or []))
    provides = frozenset(task.provides_any or [])

    matches = registry.discover(
        required_skills=skills if skills else None,
        preferred_tags=tags if tags else None,
        provides_any=provides if provides else None,
        description=f"{task.title} {task.description} {task.expected_output}",
        limit=5,
    )

    # If strict skill match failed, relax to provides / description only
    if not matches and (provides or task.description or task.title):
        matches = registry.discover(
            provides_any=provides if provides else None,
            preferred_tags=tags if tags else None,
            description=f"{task.title} {task.description}",
            limit=5,
        )

    best = matches[0] if matches else None
    alternatives = [m.capability_id for m in matches[1:4]]
    decision = AutonDecision(
        decision_type="agent_selection",
        chosen=best.capability_id if best else "none",
        alternatives=alternatives,
        rationale=(
            f"Matched skills={sorted(skills)} provides={sorted(provides)} "
            f"reasons={best.reasons if best else []}"
        ),
        confidence=float(best.score) if best else 0.0,
    )
    return best, decision


def bind_task_to_agent(task: TaskSpec, match: CapabilityMatch | None) -> TaskSpec:
    """Fill responsible_agent_id / graph_node from a registry match."""
    if match is None:
        return task
    graph_node = match.descriptor.graph_node or ""
    cap_id = match.capability_id or ""
    tags = set(match.descriptor.tags or ())
    is_plugin = cap_id.startswith(("plugin.", "skill.")) or "plugin" in tags
    if is_plugin and graph_node not in {
        "plugin_runtime_agent",
        "schema_agent",
        "sql_agent",
        "visualization_agent",
        "insight_agent",
        "export_node",
        "export_graph",
    }:
        graph_node = "plugin_runtime_agent"
    update: dict[str, Any] = {
        "responsible_agent_id": match.capability_id,
        "responsible_graph_node": graph_node,
        "required_tools": sorted(
            set(task.required_tools) | set(match.descriptor.requires_tools)
        ),
    }
    # Stash capability id for plugin_runtime_agent (TaskSpec.notes / description safe)
    if graph_node == "plugin_runtime_agent" and hasattr(task, "expected_output"):
        # Encode capability into expected_output metadata via description suffix is fragile;
        # coordinator reads responsible_agent_id as plugin_capability_id.
        pass
    return task.model_copy(update=update)


def selection_message_payload(decision: AutonDecision) -> dict[str, Any]:
    return decision_message("agent_selector", decision.model_dump(mode="json")).to_state_dict()
