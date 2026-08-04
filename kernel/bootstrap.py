"""Bootstrap: register existing SQLMind agents/tools as discoverable capabilities.

Does NOT rewrite agents — only catalogs them for Planner → Matcher → Selection.
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.tools import BaseTool

from kernel.container import KEY_MATCHER, KEY_REGISTRY, ServiceContainer
from kernel.enums import CapabilityKind, ExecutionStrategy, ModelTier, RiskClass
from kernel.matching import CapabilityMatcher
from kernel.models import CapabilityDescriptor
from kernel.registry import CapabilityRegistry


def create_kernel_container(
    *,
    registry: CapabilityRegistry | None = None,
    matcher: CapabilityMatcher | None = None,
) -> ServiceContainer:
    """Build a DI container with registry + matcher singletons."""
    container = ServiceContainer()
    reg = registry or CapabilityRegistry()
    mat = matcher or CapabilityMatcher()
    container.register(KEY_REGISTRY, reg)
    container.register(KEY_MATCHER, mat)
    return container


def bootstrap_builtin_capabilities(
    registry: CapabilityRegistry,
    *,
    tools: Iterable[BaseTool] | None = None,
    overwrite: bool = True,
) -> CapabilityRegistry:
    """Register the current SQLMind agent/tool surface into the capability catalog.

    Security nodes (validation, execution) are registered but ``ai_routable=False``
    so planners cannot skip the deterministic gate.
    """
    for desc in _builtin_agent_descriptors():
        registry.register(desc, overwrite=overwrite)

    if tools is not None:
        register_langchain_tools(registry, tools, overwrite=overwrite)
    else:
        # Catalog tool *types* even before a live toolbelt exists (metadata only)
        for desc in _builtin_tool_descriptors():
            registry.register(desc, overwrite=overwrite)

    return registry


def register_langchain_tools(
    registry: CapabilityRegistry,
    tools: Iterable[BaseTool],
    *,
    overwrite: bool = True,
) -> None:
    """Register live LangChain tools with handlers for later dynamic selection."""
    meta_by_name = {d.metadata.get("tool_name") or d.id.split(".")[-1]: d for d in _builtin_tool_descriptors()}
    # Prefer id suffix matching
    meta_by_tool_name = {d.id.removeprefix("tool."): d for d in _builtin_tool_descriptors()}

    for tool in tools:
        name = getattr(tool, "name", None) or ""
        base = meta_by_tool_name.get(name) or meta_by_name.get(name)
        if base is not None:
            desc = base.model_copy(
                update={
                    "description": getattr(tool, "description", None) or base.description,
                    "enabled": True,
                }
            )
        else:
            desc = CapabilityDescriptor(
                id=f"tool.{name}",
                kind=CapabilityKind.TOOL,
                name=name,
                description=str(getattr(tool, "description", "") or name),
                tags=frozenset({"dynamic", "tool"}),
                skills=frozenset({name}),
                risk_class=RiskClass.MEDIUM,
                ai_routable=True,
                system_protected=False,
                metadata={"tool_name": name},
            )
        registry.register(desc, handler=tool, overwrite=overwrite)


def _agent(
    *,
    id: str,
    name: str,
    description: str,
    graph_node: str,
    route_tool_name: str | None,
    route_tool_description: str | None,
    skills: set[str],
    tags: set[str],
    provides: set[str],
    risk: RiskClass = RiskClass.LOW,
    model_tier: ModelTier = ModelTier.STANDARD,
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ai_routable: bool = True,
    system_protected: bool = True,
    requires_tools: set[str] | None = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        id=id,
        kind=CapabilityKind.AGENT,
        name=name,
        description=description,
        version="1.0.0",
        tags=frozenset(tags),
        skills=frozenset(skills),
        risk_class=risk,
        preferred_model_tier=model_tier,
        execution_strategy=strategy,
        graph_node=graph_node,
        route_tool_name=route_tool_name,
        route_tool_description=route_tool_description,
        ai_routable=ai_routable,
        system_protected=system_protected,
        requires_tools=frozenset(requires_tools or ()),
        provides=frozenset(provides),
        plugin_id="sqlmind.builtin",
    )


def _builtin_agent_descriptors() -> list[CapabilityDescriptor]:
    return [
        _agent(
            id="agent.schema",
            name="Schema Agent",
            description="Load/refresh database schema context and suggest analytical questions.",
            graph_node="schema_agent",
            route_tool_name="route_to_schema_agent",
            route_tool_description="Load/refresh database schema context and AI suggested questions.",
            skills={"schema_discovery", "suggest_questions"},
            tags={"schema", "discovery"},
            provides={"schema_text", "suggested_questions"},
            requires_tools={"schema_tool"},
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.sql",
            name="SQL ReAct Agent",
            description="Draft and refine read-only SQL via ReAct tool loop; submits SQL for gated validation/execution.",
            graph_node="sql_agent",
            route_tool_name="route_to_sql_agent",
            route_tool_description="Run the SQL ReAct specialist (schema tools, draft SQL, validate, submit).",
            skills={"nl2sql", "sql_react", "query_authoring"},
            tags={"sql", "analytics"},
            provides={"sql", "sql_explanation"},
            requires_tools={"schema_tool", "validate_sql_tool"},
            risk=RiskClass.MEDIUM,
            model_tier=ModelTier.SQL_OPTIMIZED,
            strategy=ExecutionStrategy.SECURITY_GATED,
        ),
        _agent(
            id="node.validation",
            name="SQL Validation",
            description="Deterministic sqlglot AST security validation. Not AI-routable.",
            graph_node="validation_node",
            route_tool_name=None,
            route_tool_description=None,
            skills={"sql_validation", "security_gate"},
            tags={"security", "deterministic"},
            provides={"sql_valid", "validation_errors"},
            risk=RiskClass.CRITICAL,
            strategy=ExecutionStrategy.DETERMINISTIC,
            ai_routable=False,
            system_protected=True,
        ),
        _agent(
            id="node.execution",
            name="SQL Execution",
            description="Deterministic query execution after validation. Not AI-routable.",
            graph_node="execution_node",
            route_tool_name=None,
            route_tool_description=None,
            skills={"sql_execution"},
            tags={"security", "deterministic", "database"},
            provides={"dataframe_records", "row_count", "query_success"},
            risk=RiskClass.CRITICAL,
            strategy=ExecutionStrategy.DETERMINISTIC,
            ai_routable=False,
            system_protected=True,
        ),
        _agent(
            id="agent.retry",
            name="Retry Agent",
            description="Diagnose SQL/validation/execution failures and choose regenerate, replan, or give up.",
            graph_node="retry_agent",
            route_tool_name=None,
            route_tool_description=None,
            skills={"failure_diagnosis", "recovery"},
            tags={"recovery"},
            provides={"retry_next_action", "fix_hint"},
            risk=RiskClass.MEDIUM,
            model_tier=ModelTier.REASONING,
            ai_routable=False,  # reached via security/execution failure edges
            system_protected=True,
        ),
        _agent(
            id="agent.visualization",
            name="Visualization Agent",
            description="Recommend a chart type and spec for the current query result set.",
            graph_node="visualization_agent",
            route_tool_name="route_to_visualization_agent",
            route_tool_description="Recommend a chart for the current query result set.",
            skills={"visualize", "chart_recommend"},
            tags={"viz", "analytics"},
            provides={"chart_type", "chart_spec"},
            strategy=ExecutionStrategy.PARALLEL_SAFE,
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.insight",
            name="Insight Agent",
            description="Generate business insights from the current result set.",
            graph_node="insight_agent",
            route_tool_name="route_to_insight_agent",
            route_tool_description="Generate business insights from the current result set.",
            skills={"insights", "business_analysis"},
            tags={"insights", "analytics"},
            provides={"insights", "insight_structured"},
            strategy=ExecutionStrategy.PARALLEL_SAFE,
            model_tier=ModelTier.STANDARD,
        ),
        _agent(
            id="agent.summary",
            name="Summary Agent",
            description="Produce an executive summary of the connected database.",
            graph_node="summary_agent",
            route_tool_name="route_to_summary_agent",
            route_tool_description="Produce an executive summary of the connected database.",
            skills={"database_summary"},
            tags={"summary"},
            provides={"database_summary"},
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.dashboard",
            name="Dashboard Agent",
            description="Design a KPI dashboard blueprint for this database.",
            graph_node="dashboard_agent",
            route_tool_name="route_to_dashboard_agent",
            route_tool_description="Design a KPI dashboard blueprint for this database.",
            skills={"dashboard_design", "kpi"},
            tags={"dashboard"},
            provides={"dashboard_spec"},
            model_tier=ModelTier.REASONING,
        ),
        _agent(
            id="agent.optimization",
            name="Optimization Agent",
            description="Analyze SQL with EXPLAIN and advise indexes/rewrites.",
            graph_node="optimization_agent",
            route_tool_name="route_to_optimization_agent",
            route_tool_description="Analyze SQL with EXPLAIN and advise indexes/rewrites.",
            skills={"sql_optimization", "explain"},
            tags={"performance"},
            provides={"optimization_tips", "explain_plan"},
            requires_tools={"explain_tool"},
            model_tier=ModelTier.SQL_OPTIMIZED,
        ),
        _agent(
            id="manager.export",
            name="Export Manager",
            description="Orchestrate Excel/PDF/CSV/JSON/Markdown exporters for the current result set.",
            graph_node="export_node",
            route_tool_name="route_to_export_node",
            route_tool_description="Export the current result set via specialized Excel/PDF/CSV/JSON/Markdown exporters.",
            skills={"export"},
            tags={"export", "io"},
            provides={"export_paths"},
            strategy=ExecutionStrategy.DETERMINISTIC,
            risk=RiskClass.LOW,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="manager.export_graph",
            name="Export Graph",
            description="Enter the nested Export Graph (CSV/Excel/JSON/Markdown/PDF pipeline).",
            graph_node="export_graph",
            route_tool_name="route_to_export_graph",
            route_tool_description="Run the nested export subgraph for the current result set.",
            skills={"export", "export_graph"},
            tags={"export", "io", "subgraph"},
            provides={"export_paths"},
            strategy=ExecutionStrategy.DETERMINISTIC,
            risk=RiskClass.LOW,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="agent.plugin_runtime",
            name="Plugin Runtime Agent",
            description="Invoke a registered plugin capability (skills/tools from the Plugin Marketplace).",
            graph_node="plugin_runtime_agent",
            route_tool_name="route_to_plugin_runtime_agent",
            route_tool_description="Invoke a plugin capability selected via plugin_capability_id or marketplace skill.",
            skills={"plugin", "skill", "marketplace"},
            tags={"plugin", "extensibility"},
            provides={"plugin_result"},
            strategy=ExecutionStrategy.SEQUENTIAL,
            risk=RiskClass.MEDIUM,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="agent.recovery_controller",
            name="Recovery Controller",
            description="Decide Recover/Retry/Fallback/Abort/Escalate after a runtime failure.",
            graph_node="recovery_controller",
            route_tool_name="route_to_recovery_controller",
            route_tool_description="Enter recovery policy after tool/SQL/plugin/timeout failures.",
            skills={"recovery", "retry", "fallback"},
            tags={"reliability"},
            provides={"recovery_decision"},
            strategy=ExecutionStrategy.DETERMINISTIC,
            risk=RiskClass.LOW,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="subgraph.recovery",
            name="Recovery Graph",
            description="Nested recovery subgraph: controller → retry → optional replan.",
            graph_node="recovery_graph",
            route_tool_name="route_to_recovery_graph",
            route_tool_description="Enter the nested Recovery Graph after a failed execution.",
            skills={"recovery", "recovery_graph"},
            tags={"reliability", "subgraph"},
            provides={"recovery_decision"},
            strategy=ExecutionStrategy.DETERMINISTIC,
            risk=RiskClass.LOW,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="agent.reflection",
            name="Reflection Agent",
            description="Critique SQL/results/insights quality before finishing; may request another pass.",
            graph_node="reflection_agent",
            route_tool_name="route_to_reflection_agent",
            route_tool_description="Critique SQL/results/insights quality before finishing; may request another pass.",
            skills={"reflection", "self_critique", "verification"},
            tags={"quality"},
            provides={"reflection_verdict", "reflection_notes"},
            model_tier=ModelTier.REASONING,
        ),
        _agent(
            id="agent.clarify",
            name="Clarify (HITL)",
            description="Pause and ask the user a clarifying question (human-in-the-loop).",
            graph_node="clarify",
            route_tool_name="ask_user_clarification",
            route_tool_description="Pause and ask the user a clarifying question (human-in-the-loop).",
            skills={"clarification", "hitl"},
            tags={"hitl"},
            provides={"clarification_question"},
            strategy=ExecutionStrategy.HITL,
            risk=RiskClass.LOW,
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.finalize",
            name="Finalize",
            description="Assemble the final user-facing answer from current state and end the run.",
            graph_node="finalize",
            route_tool_name="route_to_finalize",
            route_tool_description="Assemble the final user-facing answer from current state and end the run.",
            skills={"finalize", "answer_assembly"},
            tags={"terminal"},
            provides={"final_response"},
            strategy=ExecutionStrategy.DETERMINISTIC,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="agent.fail",
            name="Fail",
            description="Terminal failure sink when recovery is exhausted.",
            graph_node="fail",
            route_tool_name=None,
            route_tool_description=None,
            skills={"fail"},
            tags={"terminal"},
            provides={"error"},
            ai_routable=False,
            strategy=ExecutionStrategy.DETERMINISTIC,
        ),
        _agent(
            id="agent.supervisor",
            name="Supervisor",
            description="Outer router: selects next capability via bind_tools against the registry catalog.",
            graph_node="supervisor",
            route_tool_name=None,
            route_tool_description=None,
            skills={"routing", "orchestration"},
            tags={"orchestrator"},
            provides={"next_agent"},
            ai_routable=False,
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.join_post_query",
            name="Join Post Query",
            description="Barrier after parallel viz/insight fan-out.",
            graph_node="join_post_query",
            route_tool_name=None,
            route_tool_description=None,
            skills={"barrier"},
            tags={"infra"},
            provides=set(),
            ai_routable=False,
            strategy=ExecutionStrategy.DETERMINISTIC,
        ),
        # Phase 2 — autonomous planning layer (discoverable; not Supervisor-routable)
        _agent(
            id="agent.memory",
            name="Long-Term Memory Agent",
            description="Load workflow memory (successful plans, preferences, failures) for the Planner.",
            graph_node="memory_agent",
            route_tool_name=None,
            route_tool_description=None,
            skills={"workflow_memory", "preferences"},
            tags={"memory", "autonomy"},
            provides={"workflow_memory_context"},
            ai_routable=False,
            model_tier=ModelTier.LOCAL,
            strategy=ExecutionStrategy.DETERMINISTIC,
        ),
        _agent(
            id="agent.goal_understanding",
            name="Goal Understanding Agent",
            description="Extract structured Goal (objectives, constraints, confidence, ambiguity).",
            graph_node="goal_understanding",
            route_tool_name=None,
            route_tool_description=None,
            skills={"goal_understanding", "intent_extraction"},
            tags={"planning", "autonomy"},
            provides={"goal_spec"},
            ai_routable=False,
            model_tier=ModelTier.FAST,
        ),
        _agent(
            id="agent.planner",
            name="Planner Agent",
            description="Create AdaptivePlan from Goal using Agent Registry discovery.",
            graph_node="planner",
            route_tool_name=None,
            route_tool_description=None,
            skills={"planning", "strategy"},
            tags={"planning", "autonomy"},
            provides={"adaptive_plan"},
            ai_routable=False,
            model_tier=ModelTier.REASONING,
        ),
        _agent(
            id="agent.task_decomposition",
            name="Task Decomposition Agent",
            description="Decompose Goal/Plan into TaskGraph with skills, deps, expected outputs.",
            graph_node="task_decomposition",
            route_tool_name=None,
            route_tool_description=None,
            skills={"task_decomposition"},
            tags={"planning", "autonomy"},
            provides={"task_graph"},
            ai_routable=False,
            model_tier=ModelTier.REASONING,
        ),
        _agent(
            id="agent.execution_coordinator",
            name="Execution Coordinator",
            description="Schedule TaskGraph: dependencies, parallel dispatch, progress, failure→replan.",
            graph_node="execution_coordinator",
            route_tool_name=None,
            route_tool_description=None,
            skills={"coordination", "scheduling"},
            tags={"runtime", "autonomy"},
            provides={"next_agent", "execution_progress"},
            ai_routable=False,
            strategy=ExecutionStrategy.DETERMINISTIC,
            model_tier=ModelTier.LOCAL,
        ),
        _agent(
            id="agent.replan",
            name="Reflection & Replanning Agent",
            description="On failure, analyze and create a new strategy (true replan, not bare retry).",
            graph_node="replan_agent",
            route_tool_name=None,
            route_tool_description=None,
            skills={"replanning", "recovery", "strategy_revision"},
            tags={"recovery", "autonomy"},
            provides={"replan_decision", "adaptive_plan"},
            ai_routable=False,
            model_tier=ModelTier.REASONING,
        ),
        # Phase 3 — Enterprise Agentic Platform
        _agent(
            id="agent.goal_tracking",
            name="Goal Tracking Agent",
            description="Track goal lifecycle, progress, blocked/failed goals; notify Planner.",
            graph_node="goal_tracking",
            route_tool_name=None,
            route_tool_description=None,
            skills={"goal_tracking", "progress_monitoring"},
            tags={"planning", "autonomy", "enterprise"},
            provides={"goal_tracking", "goal_status_update"},
            ai_routable=False,
            model_tier=ModelTier.LOCAL,
            strategy=ExecutionStrategy.DETERMINISTIC,
        ),
        _agent(
            id="agent.approval_gate",
            name="Human Approval Gate",
            description="Pause for human approval on high-risk SQL, exports, connections, long tasks.",
            graph_node="approval_gate",
            route_tool_name=None,
            route_tool_description=None,
            skills={"approval", "hitl", "governance"},
            tags={"hitl", "governance", "enterprise"},
            provides={"approval_decision", "approval_request"},
            ai_routable=False,
            strategy=ExecutionStrategy.HITL,
            risk=RiskClass.HIGH,
            model_tier=ModelTier.LOCAL,
        ),
    ]


def _builtin_tool_descriptors() -> list[CapabilityDescriptor]:
    specs: list[dict[str, Any]] = [
        {
            "id": "tool.schema_tool",
            "name": "schema_tool",
            "description": "Return schema text.",
            "skills": {"schema_discovery"},
            "tags": {"schema"},
            "provides": {"schema_text"},
            "risk": RiskClass.LOW,
        },
        {
            "id": "tool.validate_sql_tool",
            "name": "validate_sql_tool",
            "description": "Deterministic AST validation of SQL (security).",
            "skills": {"sql_validation"},
            "tags": {"security"},
            "provides": {"validation"},
            "risk": RiskClass.CRITICAL,
            "ai_routable": True,  # SQL ReAct may call; cannot bypass gate path
        },
        {
            "id": "tool.query_tool",
            "name": "query_tool",
            "description": "Gated execute of validated read-only SQL.",
            "skills": {"sql_execution"},
            "tags": {"database", "security"},
            "provides": {"query_result"},
            "risk": RiskClass.CRITICAL,
        },
        {
            "id": "tool.statistics_tool",
            "name": "statistics_tool",
            "description": "Compute stats on the last result dataframe.",
            "skills": {"statistics"},
            "tags": {"analytics"},
            "provides": {"stats"},
            "risk": RiskClass.LOW,
        },
        {
            "id": "tool.export_tool",
            "name": "export_tool",
            "description": "Export results to files.",
            "skills": {"export"},
            "tags": {"io"},
            "provides": {"export_paths"},
            "risk": RiskClass.LOW,
        },
        {
            "id": "tool.history_tool",
            "name": "history_tool",
            "description": "Fetch recent query history.",
            "skills": {"history"},
            "tags": {"memory"},
            "provides": {"history"},
            "risk": RiskClass.LOW,
        },
        {
            "id": "tool.bookmark_tool",
            "name": "bookmark_tool",
            "description": "Save a bookmark for a question/SQL pair.",
            "skills": {"bookmark"},
            "tags": {"memory"},
            "provides": {"bookmark"},
            "risk": RiskClass.LOW,
        },
        {
            "id": "tool.explain_tool",
            "name": "explain_tool",
            "description": "Dialect-aware EXPLAIN after validation.",
            "skills": {"explain", "sql_optimization"},
            "tags": {"performance"},
            "provides": {"explain_plan"},
            "risk": RiskClass.MEDIUM,
        },
    ]
    out: list[CapabilityDescriptor] = []
    for s in specs:
        out.append(
            CapabilityDescriptor(
                id=s["id"],
                kind=CapabilityKind.TOOL,
                name=s["name"],
                description=s["description"],
                tags=frozenset(s["tags"]),
                skills=frozenset(s["skills"]),
                provides=frozenset(s["provides"]),
                risk_class=s["risk"],
                ai_routable=bool(s.get("ai_routable", True)),
                system_protected=True,
                plugin_id="sqlmind.builtin",
                metadata={"tool_name": s["name"]},
                execution_strategy=(
                    ExecutionStrategy.DETERMINISTIC
                    if s["risk"] in {RiskClass.CRITICAL, RiskClass.HIGH}
                    else ExecutionStrategy.SEQUENTIAL
                ),
            )
        )
    return out
