"""LangGraph agent state for tool-calling multi-agent orchestration."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _overwrite(left: Any, right: Any) -> Any:
    """Last-write-wins reducer for parallel Send workers."""
    return right if right is not None else left


def _merge_dicts(left: Any, right: Any) -> Any:
    """Shallow-merge dicts for parallel Send workers (e.g. reliability_meta)."""
    if left is None or left == {}:
        return right if right is not None else {}
    if right is None or right == {}:
        return left if isinstance(left, dict) else {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return right if right is not None else left
    merged = {**left, **right}
    # Keep per-node timing when both sides report elapsed_ms under nodes.*
    left_nodes = left.get("nodes") if isinstance(left.get("nodes"), dict) else {}
    right_nodes = right.get("nodes") if isinstance(right.get("nodes"), dict) else {}
    if left_nodes or right_nodes:
        merged["nodes"] = {**left_nodes, **right_nodes}
    return merged


class AgentLog(TypedDict, total=False):
    agent: str
    message: str
    status: str
    detail: Any


class GraphState(TypedDict, total=False):
    """Shared state across the dynamic LangGraph workflow."""

    # Identity / memory
    session_id: str
    question: str
    rewritten_question: str
    conversation_history: list[dict[str, str]]
    memory_summary: str
    episodic_context: str
    messages: Annotated[list[BaseMessage], add_messages]
    react_messages: list[BaseMessage]

    # Supervisor (tool-call driven)
    intent: str
    next_agent: Annotated[str, _overwrite]
    supervisor_reasoning: str
    supervisor_visit: int
    max_supervisor_visits: int
    needs_clarification: bool
    clarification_question: str
    route_history: Annotated[list[str], operator.add]

    # Schema
    schema_text: str
    schema_dict: dict[str, Any]
    suggested_questions: list[str]
    database_summary: dict[str, Any]
    dashboard_spec: dict[str, Any]

    # SQL loop
    sql: str
    sql_explanation: str
    sql_valid: bool
    validation_errors: list[str]
    validation_warnings: list[str]
    validation_meta: dict[str, Any]
    retry_count: int
    max_retries: int
    sql_error: str
    fix_hint: str
    retry_diagnosis: str
    revised_approach: str
    should_retry: bool
    retry_next_action: str

    # Execution
    dataframe_records: list[dict[str, Any]]
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    execution_time: float
    truncated: bool
    query_success: bool
    explain_plan: str

    # Visualization / insights / optimization
    chart_type: Annotated[str, _overwrite]
    chart_spec: Annotated[dict[str, Any], _overwrite]
    insights: Annotated[str, _overwrite]
    insight_structured: Annotated[dict[str, Any], _overwrite]
    optimization_tips: list[str]
    optimization_structured: dict[str, Any]
    export_paths: dict[str, str]

    # Reflection
    reflection_verdict: str
    reflection_notes: str
    reflection_count: int
    max_reflections: int
    parallel_job: Annotated[bool, _overwrite]

    # Meta
    database_name: str
    dialect: str
    final_response: str
    stream_tokens: str
    agent_logs: Annotated[list[AgentLog], operator.add]
    error: str
    status: str
    # Deprecated plan cursor fields (kept for checkpoint compatibility)
    plan: list[str]
    plan_index: int

    # Autonomy kernel (Phase 1+) — catalog version observed at run start
    capability_registry_version: int

    # Phase 2 — Goal / Plan / TaskGraph / Coordinator
    goal_spec: dict[str, Any]
    adaptive_plan: dict[str, Any]
    task_graph: dict[str, Any]
    execution_progress: dict[str, Any]
    planner_output: dict[str, Any]
    replan_decision: dict[str, Any]
    replan_reason: str
    plan_active: bool
    active_task_id: str
    workflow_memory_context: str
    agent_messages: Annotated[list[dict[str, Any]], operator.add]
    auton_decisions: Annotated[list[dict[str, Any]], operator.add]
    parallel_sends: list[dict[str, Any]]
    _send_targets: list[str]

    # Phase 3 — Enterprise Agentic Platform
    goal_tracking: dict[str, Any]
    goal_status_update: dict[str, Any]
    approval_request: dict[str, Any]
    approval_decision: str
    needs_approval: bool
    approval_resume_agent: str
    vector_memory_context: str
    learning_context: str
    iam_session: dict[str, Any]
    tenant_id: str
    workspace_id: str
    plugin_catalog_version: int
    reliability_meta: Annotated[dict[str, Any], _merge_dicts]
    enterprise_events: Annotated[list[dict[str, Any]], operator.add]

    # Phase 4 — Production enterprise extensions
    runtime_replan_count: int
    max_runtime_replans: int
    prior_execution_context: dict[str, Any]
    recovery_decision: dict[str, Any]
    plugin_capability_id: str
    plugin_result: dict[str, Any]
    export_job_id: str
    export_progress: dict[str, Any]
    parallel_metrics: dict[str, Any]


def empty_state(question: str = "", session_id: str = "default") -> GraphState:
    return GraphState(
        session_id=session_id,
        question=question,
        rewritten_question=question,
        conversation_history=[],
        memory_summary="",
        episodic_context="",
        messages=[],
        react_messages=[],
        intent="query",
        next_agent="supervisor",
        supervisor_reasoning="",
        supervisor_visit=0,
        max_supervisor_visits=12,
        needs_clarification=False,
        clarification_question="",
        route_history=[],
        schema_text="",
        schema_dict={},
        suggested_questions=[],
        database_summary={},
        dashboard_spec={},
        sql="",
        sql_explanation="",
        sql_valid=False,
        validation_errors=[],
        validation_warnings=[],
        validation_meta={},
        retry_count=0,
        max_retries=3,
        sql_error="",
        fix_hint="",
        retry_diagnosis="",
        revised_approach="",
        should_retry=False,
        retry_next_action="",
        dataframe_records=[],
        columns=[],
        dtypes={},
        row_count=0,
        execution_time=0.0,
        truncated=False,
        query_success=False,
        explain_plan="",
        chart_type="none",
        chart_spec={},
        insights="",
        insight_structured={},
        optimization_tips=[],
        optimization_structured={},
        export_paths={},
        reflection_verdict="",
        reflection_notes="",
        reflection_count=0,
        max_reflections=2,
        parallel_job=False,
        database_name="",
        dialect="",
        final_response="",
        stream_tokens="",
        agent_logs=[],
        error="",
        status="pending",
        plan=[],
        plan_index=0,
        capability_registry_version=0,
        goal_spec={},
        adaptive_plan={},
        task_graph={},
        execution_progress={},
        planner_output={},
        replan_decision={},
        replan_reason="",
        plan_active=False,
        active_task_id="",
        workflow_memory_context="",
        agent_messages=[],
        auton_decisions=[],
        parallel_sends=[],
        _send_targets=[],
        goal_tracking={},
        goal_status_update={},
        approval_request={},
        approval_decision="",
        needs_approval=False,
        approval_resume_agent="",
        vector_memory_context="",
        learning_context="",
        iam_session={},
        tenant_id="default",
        workspace_id="default",
        plugin_catalog_version=0,
        reliability_meta={},
        enterprise_events=[],
        runtime_replan_count=0,
        max_runtime_replans=2,
        prior_execution_context={},
        recovery_decision={},
        plugin_capability_id="",
        plugin_result={},
        export_job_id="",
        export_progress={},
        parallel_metrics={},
    )


def records_to_dataframe(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns or None)
    return pd.DataFrame(records)


# Nodes the outer graph may route to (AI agents + deterministic utilities)
GRAPH_NODES = {
    "memory_agent",
    "goal_understanding",
    "goal_tracking",
    "planner",
    "task_decomposition",
    "execution_coordinator",
    "replan_agent",
    "schema_agent",
    "sql_agent",
    "validation_node",
    "execution_node",
    "visualization_agent",
    "insight_agent",
    "optimization_agent",
    "dashboard_agent",
    "export_node",
    "summary_agent",
    "reflection_agent",
    "retry_agent",
    "clarify",
    "approval_gate",
    "finalize",
    "fail",
    "supervisor",
    "join_post_query",
    # Subgraph entry nodes (enterprise mode)
    "memory_graph",
    "planning_graph",
    "execution_graph",
    "analytics_graph",
    "export_graph",
    "reflection_graph",
    "recovery_graph",
    "recovery_controller",
    "plugin_runtime_agent",
}

# Backward-compatible alias
AGENT_NODES = GRAPH_NODES


def normalize_plan(plan: list[str] | None, intent: str = "query") -> list[str]:
    """Sanitize a route history list (no silent DEFAULT_PLANS injection).

    Kept for tests/backward compatibility. Empty input stays empty — callers must
    not treat this as an LLM decision substitute.
    """
    cleaned = [
        p
        for p in (plan or [])
        if p in GRAPH_NODES and p not in {"retry_agent", "fail", "supervisor"}
    ]
    return cleaned
