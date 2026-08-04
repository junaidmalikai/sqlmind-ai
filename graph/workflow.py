"""LangGraph workflow — Goal Planner preamble + Supervisor + SQL ReAct + security gate."""

from __future__ import annotations

from typing import Any, Iterator, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.nodes import (
    fail_node,
    finalize_node,
    make_clarify_node,
    make_dashboard_agent,
    make_execution_node,
    make_export_node,
    make_insight_agent,
    make_optimization_agent,
    make_reflection_agent,
    make_retry_agent,
    make_schema_agent,
    make_sql_agent,
    make_summary_agent,
    make_supervisor_agent,
    make_validation_node,
    make_visualization_agent,
)
from config.settings import Settings, get_settings
from core.context import RequestContext, clear_request_context, set_request_context
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaSnapshot
from graph.state import AGENT_NODES, GraphState, empty_state
from kernel.bootstrap import bootstrap_builtin_capabilities, create_kernel_container
from kernel.container import KEY_MATCHER, KEY_REGISTRY, ServiceContainer
from kernel.matching import CapabilityMatcher
from kernel.registry import CapabilityRegistry
from governance.approval import ApprovalPolicy, make_approval_gate_node
from graph.subgraphs import (
    SubgraphBundle,
    build_analytics_subgraph,
    build_execution_subgraph,
    build_export_subgraph,
    build_memory_subgraph,
    build_planning_subgraph,
    build_recovery_subgraph,
    build_reflection_subgraph,
)
from iam import IAMService
from learning import PlannerExperienceStore
from memory.checkpointing import build_checkpointer
from memory.embeddings import build_embedding_provider
from memory.episodic import EpisodicMemoryStore
from memory.summarizer import ConversationSummarizer
from memory.vector_store import MemoryFabric, VectorMemoryStore
from memory.workflow_memory import WorkflowMemoryStore
from observability import (
    clear_trace_context,
    configure_observability,
    ensure_trace_id,
    get_event_tracker,
    langsmith_tags,
    run_metadata,
    start_span,
)
from observability.metrics import get_metrics
from observability.runtime_trace import (
    SUBGRAPH_NAMES,
    execution_trace_session,
    finalize_and_export,
    get_active_trace,
    safe_trace,
    wrap_trace_node,
)
from enterprise.runtime import EnterpriseRuntime, wrap_enterprise_node
from planner.agents import (
    make_goal_understanding_agent,
    make_memory_agent,
    make_planner_agent,
    make_replan_agent,
    make_task_decomposition_agent,
    persist_workflow_outcome,
)
from planner.coordinator import make_execution_coordinator, route_coordinator_dispatch
from planner.goal_store import GoalStore
from planner.goal_tracker import make_goal_tracking_agent
from planner.messages import get_message_bus
from planner.registry import AgentRegistry
from plugins import PluginMarketplace
from reliability import (
    RecoveryManager,
    get_breakers,
    get_dlq,
    get_health_monitor,
)
from services.llm_service import LLMService
from tools.sqlmind_tools import build_toolbelt, tools_by_name
from utils.logging_config import get_logger
from utils.security import SQLSecurityGuard
from kernel.enums import CapabilityKind, RiskClass
from kernel.models import CapabilityDescriptor

logger = get_logger(__name__)


def route_next(state: GraphState) -> str:
    nxt = state.get("next_agent") or "finalize"
    if nxt == "export_agent":  # legacy alias
        nxt = "export_node"
    # Prefer nested export subgraph when enterprise subgraphs are enabled
    if nxt in {"export_node", "export_graph"}:
        try:
            from config.settings import get_settings

            settings = get_settings()
            if (
                getattr(settings, "enterprise_enabled", True)
                and getattr(settings, "enterprise_subgraphs_enabled", True)
            ):
                nxt = "export_graph"
            else:
                nxt = "export_node"
        except Exception:  # noqa: BLE001
            nxt = "export_node" if nxt == "export_graph" else nxt
    return nxt if nxt in AGENT_NODES else "finalize"


def route_failure(state: GraphState | None = None) -> str:
    """Unified failure entry: recovery_graph → recovery_controller → retry_agent.

    Preserves backward compatibility when enterprise subgraphs / controller are off.
    """
    try:
        from config.settings import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001
        settings = None

    subgraphs = bool(
        settings is not None
        and getattr(settings, "enterprise_enabled", True)
        and getattr(settings, "enterprise_subgraphs_enabled", True)
    )
    controller = bool(
        settings is not None
        and getattr(settings, "enterprise_enabled", True)
        and getattr(settings, "recovery_controller_enabled", True)
    )
    # State may carry an explicit override from recovery decisions
    if state and state.get("force_retry_agent"):
        return "retry_agent"
    if subgraphs:
        return "recovery_graph"
    if controller:
        return "recovery_controller"
    return "retry_agent"


def route_after_validation(state: GraphState) -> str:
    """Valid SQL always executes; invalid SQL goes to recovery — never LLM-bypass security."""
    if state.get("sql_valid"):
        return "execution_node"
    return route_failure(state)


def route_after_execution(state: GraphState) -> Any:
    """On success, fan-out viz+insight in parallel (Send); on failure, recovery path."""
    if not state.get("query_success"):
        return route_failure(state)
    # If plan is actively coordinating post-query tasks, return to coordinator
    if state.get("plan_active"):
        return "execution_coordinator"
    # Parallel map: both specialists receive a copy with parallel_job flag
    payload = {**state, "parallel_job": True, "next_agent": "join_post_query"}
    try:
        from observability.parallel_metrics import record_parallel_send

        record_parallel_send(
            ["visualization_agent", "insight_agent"],
            source="execution_node",
            session_id=str(state.get("session_id") or ""),
        )
    except Exception:  # noqa: BLE001
        pass
    return [
        Send("visualization_agent", payload),
        Send("insight_agent", payload),
    ]


def route_after_retry(
    state: GraphState,
) -> str:
    action = state.get("retry_next_action") or ""
    nxt = state.get("next_agent") or "fail"
    if action == "regenerate_sql" or nxt == "sql_agent":
        return "sql_agent"
    if action == "replan" or nxt in {"replan_agent", "supervisor"}:
        replan_count = int(
            (state.get("adaptive_plan") or {}).get("replan_count")
            or state.get("runtime_replan_count")
            or 0
        )
        max_replans = int(
            (state.get("adaptive_plan") or {}).get("max_replans")
            or state.get("max_runtime_replans")
            or 2
        )
        planning_live = True
        try:
            from config.settings import get_settings

            planning_live = bool(get_settings().autonomy_planning_enabled)
        except Exception:  # noqa: BLE001
            pass
        # Prefer true replanning under budget (even without a prior plan_id)
        if planning_live and replan_count < max_replans:
            if nxt == "replan_agent" or action == "replan" or nxt == "supervisor":
                return "replan_agent"
        has_plan = bool((state.get("adaptive_plan") or {}).get("plan_id")) or bool(
            state.get("plan_active")
        )
        if has_plan and nxt == "replan_agent":
            return "replan_agent"
        if has_plan and action == "replan":
            return "replan_agent"
        return "supervisor"
    if state.get("final_response") and not state.get("should_retry"):
        return "finalize"
    return "fail"


def route_after_specialist(state: GraphState) -> str:
    """Workers in a parallel job join; plan mode returns to coordinator; else Supervisor."""
    if state.get("parallel_job") or state.get("next_agent") == "join_post_query":
        return "join_post_query"
    if state.get("plan_active") and state.get("next_agent") not in {
        "clarify",
        "fail",
        "finalize",
        "replan_agent",
    }:
        return "execution_coordinator"
    return route_next(state)


def join_post_query(state: GraphState) -> dict[str, Any]:
    """Barrier after parallel viz/insight — coordinator if plan active, else Supervisor."""
    try:
        from observability.parallel_metrics import get_parallel_metrics

        pm = get_parallel_metrics()
        # Honest worker status from merged state (not unconditional ok)
        viz_status = "error" if state.get("error") and not state.get("chart_type") else "ok"
        if state.get("chart_type") or state.get("chart_spec") or state.get("plotly_fig"):
            viz_status = "ok"
        insight_status = "ok" if (state.get("insights") or "").strip() else (
            "error" if state.get("error") else "ok"
        )
        pm.worker_exit("visualization_agent", status=viz_status)
        pm.worker_exit("insight_agent", status=insight_status)
        parallel_snap = pm.snapshot(
            session_id=str(state.get("session_id") or "") or None
        )
    except Exception:  # noqa: BLE001
        parallel_snap = {}
    base: dict[str, Any] = {"parallel_job": False, "parallel_metrics": parallel_snap}
    if state.get("plan_active"):
        return {**base, "next_agent": "execution_coordinator"}
    return {**base, "next_agent": "supervisor"}


def route_after_goal(state: GraphState) -> str:
    nxt = state.get("next_agent") or "planner"
    if nxt == "clarify":
        return "clarify"
    if nxt == "approval_gate":
        return "approval_gate"
    return "planner"


def route_after_clarify(state: GraphState) -> str:
    """After HITL, resume planning if we came from goal understanding; else specialist path."""
    if state.get("goal_spec") and not state.get("adaptive_plan"):
        return "goal_understanding"
    if state.get("plan_active") or state.get("replan_decision"):
        return "planner"
    return route_after_specialist(state)


def route_after_validation_enterprise(state: GraphState) -> str:
    """Valid SQL → approval gate (if needed) or execution; invalid → recovery."""
    if not state.get("sql_valid"):
        return route_failure(state)
    if state.get("needs_approval") and state.get("approval_decision") != "approved":
        return "approval_gate"
    if state.get("approval_request") and state.get("approval_decision") != "approved":
        from governance.approval import classify_sql_risk

        reason, _risk = classify_sql_risk(state.get("sql") or "")
        if reason is not None:
            return "approval_gate"
    return "execution_node"


def route_after_approval(state: GraphState) -> str:
    nxt = state.get("next_agent") or "fail"
    if nxt in AGENT_NODES:
        return nxt
    return "fail"


def build_sqlmind_graph(
    schema: SchemaSnapshot,
    executor: QueryExecutor,
    llm: LLMService | None = None,
    settings: Settings | None = None,
    *,
    security: SQLSecurityGuard | None = None,
    checkpointer: Any | None = None,
    history_store: Any | None = None,
    get_last_df: Any | None = None,
    episodic_store: EpisodicMemoryStore | None = None,
    registry: CapabilityRegistry | None = None,
    kernel_container: ServiceContainer | None = None,
    agent_registry: AgentRegistry | None = None,
    workflow_memory: WorkflowMemoryStore | None = None,
    memory_fabric: MemoryFabric | None = None,
    goal_store: GoalStore | None = None,
    experience_store: PlannerExperienceStore | None = None,
    approval_policy: ApprovalPolicy | None = None,
    enterprise_runtime: EnterpriseRuntime | None = None,
    plugin_runtime: Any | None = None,
):
    """Compile the tool-calling multi-agent LangGraph app.

    Phase 1: ``registry`` capability catalog for Supervisor bind_tools.
    Phase 2: optional Goal→Planner→Decompose→Coordinator preamble.
    Phase 3: Goal Tracking, Approval gates, hybrid memory, enterprise nodes,
    IAM/CB/DLQ wrappers, production subgraphs when enabled.
    Phase 4: recovery controller, plugin runtime agent, accurate parallel metrics.
    Security edges remain fixed.
    """
    settings = settings or get_settings()
    llm = llm or LLMService(settings)
    security = security or SQLSecurityGuard(
        read_only=settings.read_only_mode,
        max_rows=settings.max_rows,
        dialect=schema.dialect,
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
    )

    tools = build_toolbelt(
        schema=schema,
        executor=executor,
        security=security,
        export_dir=settings.export_dir,
        history_store=history_store,
        get_last_df=get_last_df,
    )
    tool_map = tools_by_name(tools)

    # Autonomy kernel — catalog existing agents/tools (no rewrite)
    if registry is None:
        if kernel_container is not None and kernel_container.has(KEY_REGISTRY):
            registry = kernel_container.resolve(KEY_REGISTRY)
        else:
            registry = CapabilityRegistry()
            bootstrap_builtin_capabilities(registry, tools=tools, overwrite=True)
    else:
        bootstrap_builtin_capabilities(registry, tools=tools, overwrite=True)

    matcher = (
        kernel_container.resolve(KEY_MATCHER)
        if kernel_container is not None and kernel_container.has(KEY_MATCHER)
        else CapabilityMatcher()
    )
    agent_reg = agent_registry or AgentRegistry(registry, matcher)

    graph = StateGraph(GraphState)

    enterprise = bool(getattr(settings, "enterprise_enabled", True))
    approval_enabled = enterprise and bool(
        getattr(settings, "approval_gates_enabled", True)
    )
    subgraphs_enabled = enterprise and bool(
        getattr(settings, "enterprise_subgraphs_enabled", True)
    )
    policy = approval_policy or ApprovalPolicy(
        require_write_sql_approval=bool(
            getattr(settings, "require_write_sql_approval", True)
        ),
        require_sensitive_export_approval=bool(
            getattr(settings, "require_sensitive_export_approval", True)
        ),
    )

    def _wrap(name: str, fn: Any, *, action: str = "agent_invoke", skip_iam: bool = False) -> Any:
        if enterprise_runtime is not None and enterprise:
            return wrap_enterprise_node(
                name, fn, enterprise_runtime, action=action, skip_iam=skip_iam
            )
        # Trace-only wrap when enterprise middleware is off — no business logic change
        if getattr(settings, "runtime_trace_enabled", True):
            return wrap_trace_node(name, fn)
        return fn

    # --- Build raw node callables ---
    planning_enabled = bool(settings.autonomy_planning_enabled)

    memory_fn = None
    goal_fn = None
    planner_fn = None
    decompose_fn = None
    coordinator_fn = None
    replan_fn = None
    goal_track_fn = None

    if planning_enabled:
        memory_fn = _wrap(
            "memory_agent",
            make_memory_agent(
                workflow_memory,
                memory_fabric=memory_fabric,
                experience_store=experience_store,
            ),
            action="memory_access",
        )
        goal_fn = _wrap(
            "goal_understanding",
            make_goal_understanding_agent(
                llm, clarify_threshold=settings.goal_clarify_threshold
            ),
        )
        planner_fn = _wrap(
            "planner",
            make_planner_agent(llm, agent_reg, experience_store=experience_store),
        )
        decompose_fn = _wrap(
            "task_decomposition",
            make_task_decomposition_agent(llm, agent_reg),
        )
        coordinator_fn = _wrap(
            "execution_coordinator",
            make_execution_coordinator(agent_reg),
        )
        replan_fn = _wrap("replan_agent", make_replan_agent(llm, agent_reg))
        if enterprise:
            goal_track_fn = _wrap(
                "goal_tracking",
                make_goal_tracking_agent(goal_store),
            )

    supervisor_fn = _wrap("supervisor", make_supervisor_agent(llm, registry=registry))
    schema_fn = _wrap("schema_agent", make_schema_agent(schema, llm, tool_map))
    sql_fn = _wrap("sql_agent", make_sql_agent(llm, tool_map))
    validation_fn = _wrap(
        "validation_node",
        make_validation_node(security, approval_policy=policy),
        action="sql_execute",
        skip_iam=False,
    )
    execution_fn = _wrap(
        "execution_node",
        make_execution_node(executor, tool_map),
        action="sql_execute",
    )
    viz_fn = _wrap("visualization_agent", make_visualization_agent(llm))
    insight_fn = _wrap("insight_agent", make_insight_agent(llm))
    opt_fn = _wrap("optimization_agent", make_optimization_agent(llm, tool_map))
    dash_fn = _wrap("dashboard_agent", make_dashboard_agent(llm))
    summary_fn = _wrap("summary_agent", make_summary_agent(llm))
    export_fn = _wrap(
        "export_node",
        make_export_node(settings.export_dir, llm, tool_map),
        action="export",
    )
    reflection_fn = _wrap("reflection_agent", make_reflection_agent(llm))
    retry_fn = _wrap("retry_agent", make_retry_agent(llm))
    clarify_fn = _wrap("clarify", make_clarify_node(), skip_iam=True)
    approval_fn = (
        _wrap("approval_gate", make_approval_gate_node(policy), skip_iam=True)
        if approval_enabled
        else None
    )
    finalize_fn = _wrap("finalize", finalize_node, skip_iam=True)
    fail_fn = _wrap("fail", fail_node, skip_iam=True)
    join_fn = _wrap("join_post_query", join_post_query, skip_iam=True)

    recovery_ctrl_fn = None
    if enterprise and bool(getattr(settings, "recovery_controller_enabled", True)):
        from reliability.recovery_actions import make_recovery_controller

        recovery_ctrl_fn = _wrap(
            "recovery_controller",
            make_recovery_controller(),
            skip_iam=True,
        )

    plugin_agent_fn = None
    if enterprise and plugin_runtime is not None and bool(
        getattr(settings, "plugin_runtime_enabled", True)
    ):
        from plugins.runtime import make_plugin_runtime_agent

        plugin_agent_fn = _wrap(
            "plugin_runtime_agent",
            make_plugin_runtime_agent(plugin_runtime),
            action="plugin_execute",
        )

    # --- Register flat nodes (always — specialists + security chain) ---
    if planning_enabled:
        assert memory_fn and goal_fn and planner_fn and decompose_fn and coordinator_fn
        graph.add_node("memory_agent", memory_fn)
        graph.add_node("goal_understanding", goal_fn)
        graph.add_node("planner", planner_fn)
        graph.add_node("task_decomposition", decompose_fn)
        graph.add_node("execution_coordinator", coordinator_fn)
        assert replan_fn is not None
        graph.add_node("replan_agent", replan_fn)
        if goal_track_fn is not None:
            graph.add_node("goal_tracking", goal_track_fn)

    graph.add_node("supervisor", supervisor_fn)
    graph.add_node("schema_agent", schema_fn)
    graph.add_node("sql_agent", sql_fn)
    graph.add_node("validation_node", validation_fn)
    graph.add_node("execution_node", execution_fn)
    graph.add_node("visualization_agent", viz_fn)
    graph.add_node("insight_agent", insight_fn)
    graph.add_node("optimization_agent", opt_fn)
    graph.add_node("dashboard_agent", dash_fn)
    graph.add_node("summary_agent", summary_fn)
    graph.add_node("export_node", export_fn)
    graph.add_node("reflection_agent", reflection_fn)
    graph.add_node("retry_agent", retry_fn)
    graph.add_node("clarify", clarify_fn)
    if approval_fn is not None:
        graph.add_node("approval_gate", approval_fn)
    graph.add_node("finalize", finalize_fn)
    graph.add_node("fail", fail_fn)
    graph.add_node("join_post_query", join_fn)
    if recovery_ctrl_fn is not None:
        graph.add_node("recovery_controller", recovery_ctrl_fn)
    if plugin_agent_fn is not None:
        graph.add_node("plugin_runtime_agent", plugin_agent_fn)

    # --- Production subgraphs (reachable parent nodes) ---
    subgraph_bundle: SubgraphBundle | None = None
    if subgraphs_enabled and planning_enabled and memory_fn and goal_fn and planner_fn and decompose_fn:
        memory_sg = build_memory_subgraph(memory_agent=memory_fn)
        planning_sg = build_planning_subgraph(
            goal_understanding=goal_fn,
            planner=planner_fn,
            task_decomposition=decompose_fn,
            goal_tracking=goal_track_fn,
        )
        execution_sg = build_execution_subgraph(
            coordinator=coordinator_fn or supervisor_fn,
            supervisor=supervisor_fn,
            schema_agent=schema_fn,
            sql_agent=sql_fn,
            validation_node=validation_fn,
            execution_node=execution_fn,
        )
        analytics_sg = build_analytics_subgraph(
            visualization_agent=viz_fn,
            insight_agent=insight_fn,
            optimization_agent=opt_fn,
            dashboard_agent=dash_fn,
            summary_agent=summary_fn,
        )
        export_sg = build_export_subgraph(export_node=export_fn)
        reflection_sg = build_reflection_subgraph(reflection_agent=reflection_fn)
        recovery_sg = build_recovery_subgraph(
            retry_agent=retry_fn,
            replan_agent=replan_fn,
            recovery_controller=recovery_ctrl_fn,
        )
        subgraph_bundle = SubgraphBundle(
            memory=memory_sg,
            planning=planning_sg,
            execution=execution_sg,
            analytics=analytics_sg,
            export=export_sg,
            reflection=reflection_sg,
            recovery=recovery_sg,
        )
        for name, sg in subgraph_bundle.as_dict().items():
            graph.add_node(name, sg)
        logger.info(
            "Enterprise subgraphs registered: %s",
            list(subgraph_bundle.as_dict().keys()),
        )

    # --- Edges ---
    if planning_enabled and subgraphs_enabled and subgraph_bundle is not None:
        # Supervisor-orchestrated subgraph preamble
        graph.add_edge(START, "memory_graph")
        graph.add_edge("memory_graph", "planning_graph")

        def route_after_planning_subgraph(state: GraphState) -> str:
            nxt = state.get("next_agent") or "execution_coordinator"
            if nxt == "clarify":
                return "clarify"
            if nxt == "approval_gate" and approval_enabled:
                return "approval_gate"
            return "execution_coordinator"

        plan_targets: dict[str, str] = {
            "clarify": "clarify",
            "execution_coordinator": "execution_coordinator",
        }
        if approval_enabled:
            plan_targets["approval_gate"] = "approval_gate"
        graph.add_conditional_edges(
            "planning_graph",
            route_after_planning_subgraph,
            plan_targets,
        )
        graph.add_conditional_edges(
            "execution_coordinator",
            route_coordinator_dispatch,
        )
        graph.add_conditional_edges("replan_agent", route_next)
        graph.add_conditional_edges("clarify", route_after_clarify)
        # Subgraph aliases remain routable from supervisor / coordinator
        for sg_name in (
            "execution_graph",
            "analytics_graph",
            "export_graph",
            "reflection_graph",
            "recovery_graph",
        ):
            graph.add_conditional_edges(sg_name, route_after_specialist)
    elif planning_enabled:
        graph.add_edge(START, "memory_agent")
        graph.add_edge("memory_agent", "goal_understanding")
        goal_targets: dict[str, str] = {"clarify": "clarify", "planner": "planner"}
        if approval_enabled:
            goal_targets["approval_gate"] = "approval_gate"
        graph.add_conditional_edges(
            "goal_understanding",
            route_after_goal,
            goal_targets,
        )
        if enterprise and goal_track_fn is not None:
            graph.add_edge("planner", "task_decomposition")
            graph.add_edge("task_decomposition", "goal_tracking")
            graph.add_edge("goal_tracking", "execution_coordinator")
        else:
            graph.add_edge("planner", "task_decomposition")
            graph.add_edge("task_decomposition", "execution_coordinator")
        graph.add_conditional_edges(
            "execution_coordinator",
            route_coordinator_dispatch,
        )
        graph.add_conditional_edges("replan_agent", route_next)
        graph.add_conditional_edges("clarify", route_after_clarify)
    else:
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("clarify", route_after_specialist)

    graph.add_conditional_edges("supervisor", route_next)

    # Specialists: parallel workers → join; plan mode → coordinator; else Supervisor
    for node in (
        "schema_agent",
        "visualization_agent",
        "insight_agent",
        "optimization_agent",
        "dashboard_agent",
        "summary_agent",
        "export_node",
    ):
        graph.add_conditional_edges(node, route_after_specialist)
    if plugin_agent_fn is not None:
        graph.add_conditional_edges("plugin_runtime_agent", route_after_specialist)
    if recovery_ctrl_fn is not None:
        graph.add_conditional_edges("recovery_controller", route_after_specialist)

    graph.add_conditional_edges("reflection_agent", route_after_specialist)
    graph.add_conditional_edges(
        "join_post_query",
        route_next,
    )

    # Fixed security pipeline — not AI-skippable
    graph.add_edge("sql_agent", "validation_node")
    failure_targets: dict[str, str] = {
        "execution_node": "execution_node",
        "retry_agent": "retry_agent",
    }
    if subgraphs_enabled:
        failure_targets["recovery_graph"] = "recovery_graph"
    if recovery_ctrl_fn is not None:
        failure_targets["recovery_controller"] = "recovery_controller"
    if approval_enabled:
        failure_targets["approval_gate"] = "approval_gate"
        graph.add_conditional_edges(
            "validation_node",
            route_after_validation_enterprise,
            failure_targets,
        )
        graph.add_conditional_edges("approval_gate", route_after_approval)
    else:
        graph.add_conditional_edges(
            "validation_node",
            route_after_validation,
            {k: v for k, v in failure_targets.items() if k != "approval_gate"},
        )
    # Success → parallel Send(viz, insight) or coordinator; failure → recovery
    graph.add_conditional_edges("execution_node", route_after_execution)

    retry_targets: dict[str, str] = {
        "sql_agent": "sql_agent",
        "supervisor": "supervisor",
        "fail": "fail",
        "finalize": "finalize",
    }
    if planning_enabled:
        retry_targets["replan_agent"] = "replan_agent"
    if subgraphs_enabled:
        retry_targets["recovery_graph"] = "recovery_graph"
    if recovery_ctrl_fn is not None:
        retry_targets["recovery_controller"] = "recovery_controller"

    def _route_after_retry_safe(state: GraphState) -> str:
        dest = route_after_retry(state)
        if dest not in retry_targets:
            return "supervisor" if dest == "replan_agent" else "fail"
        return dest

    graph.add_conditional_edges(
        "retry_agent",
        _route_after_retry_safe,
        retry_targets,
    )

    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)

    cp = checkpointer if checkpointer is not None else build_checkpointer(
        settings.checkpoint_db_path
    )
    return graph.compile(checkpointer=cp)


class SQLMindOrchestrator:
    """Facade over the compiled dynamic graph with checkpointed sessions."""

    def __init__(
        self,
        schema: SchemaSnapshot,
        executor: QueryExecutor,
        llm: LLMService | None = None,
        settings: Settings | None = None,
        *,
        security: SQLSecurityGuard | None = None,
        history_store: Any | None = None,
        session_id: str = "default",
        episodic_store: EpisodicMemoryStore | None = None,
        workflow_memory: WorkflowMemoryStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_observability(self.settings)
        self.llm = llm or LLMService(self.settings)
        self.schema = schema
        self.executor = executor
        self.session_id = session_id
        self._last_df = None
        self.history_store = history_store
        self.episodic = episodic_store or EpisodicMemoryStore(
            self.settings.history_db_path.replace("sqlmind_history", "sqlmind_episodes")
            if "sqlmind_history" in self.settings.history_db_path
            else str(self.settings.data_dir) + "/sqlmind_episodes.sqlite3"
        )
        wf_path = self.settings.workflow_memory_db_path or (
            self.settings.history_db_path.replace("sqlmind_history", "sqlmind_workflows")
            if "sqlmind_history" in self.settings.history_db_path
            else str(self.settings.data_dir) + "/sqlmind_workflows.sqlite3"
        )
        self.workflow_memory = workflow_memory or WorkflowMemoryStore(wf_path)
        self.security = security or SQLSecurityGuard(
            read_only=self.settings.read_only_mode,
            max_rows=self.settings.max_rows,
            dialect=schema.dialect,
            known_tables=set(schema.table_names()),
            known_columns=schema.columns_map(),
        )
        self.checkpointer = build_checkpointer(self.settings.checkpoint_db_path)
        self.summarizer = ConversationSummarizer(self.llm, self.settings)
        self.tools = build_toolbelt(
            schema=schema,
            executor=executor,
            security=self.security,
            export_dir=self.settings.export_dir,
            history_store=history_store,
            get_last_df=lambda: self._last_df,
        )
        # Phase 1 Autonomy Kernel — capability catalog + DI
        self.kernel = create_kernel_container()
        self.registry: CapabilityRegistry = self.kernel.resolve(KEY_REGISTRY)
        bootstrap_builtin_capabilities(self.registry, tools=self.tools, overwrite=True)
        self.matcher: CapabilityMatcher = self.kernel.resolve(KEY_MATCHER)
        # Phase 2 — Agent Registry facade + planner
        self.agent_registry = AgentRegistry(self.registry, self.matcher)

        # Phase 3 — Enterprise platform services
        self.message_bus = get_message_bus()
        self.events = get_event_tracker(
            log_path=getattr(self.settings, "enterprise_events_log_path", None)
        )
        self.metrics = get_metrics()
        data_dir = str(self.settings.data_dir)
        vec_path = getattr(self.settings, "vector_memory_db_path", "") or (
            f"{data_dir}/sqlmind_vectors.sqlite3"
        )
        embedder = build_embedding_provider(
            getattr(self.settings, "embedding_provider", "hashing"),
            model=getattr(self.settings, "embedding_model", "") or "",
            api_key=self.settings.openai_api_key,
            base_url=self.settings.ollama_base_url,
            dim=int(getattr(self.settings, "embedding_dim", 256) or 256),
        )
        self.embedder = embedder
        self.vector_memory = VectorMemoryStore(
            vec_path,
            embedder=embedder,
            default_ttl_seconds=float(
                getattr(self.settings, "memory_ttl_seconds", 0.0) or 0.0
            ),
        )
        self.memory_fabric = MemoryFabric(
            self.vector_memory,
            workflow_memory=self.workflow_memory,
            episodic=self.episodic,
        )
        goal_path = getattr(self.settings, "goal_store_db_path", "") or (
            f"{data_dir}/sqlmind_goals.sqlite3"
        )
        self.goal_store = GoalStore(goal_path)
        learn_path = getattr(self.settings, "learning_db_path", "") or (
            f"{data_dir}/sqlmind_learning.sqlite3"
        )
        self.experience_store = PlannerExperienceStore(learn_path)
        iam_path = getattr(self.settings, "iam_db_path", "") or (
            f"{data_dir}/sqlmind_iam.sqlite3"
        )
        self.iam = IAMService(iam_path)
        dlq_path = getattr(self.settings, "dlq_db_path", "") or (
            f"{data_dir}/sqlmind_dlq.sqlite3"
        )
        self.dlq = get_dlq(dlq_path)
        self.breakers = get_breakers(
            failure_threshold=int(
                getattr(self.settings, "circuit_breaker_failure_threshold", 5) or 5
            ),
            recovery_timeout=float(
                getattr(self.settings, "circuit_breaker_recovery_timeout", 30.0) or 30.0
            ),
        )
        self.health = get_health_monitor()
        self.recovery = RecoveryManager(self.dlq, self.breakers)
        self.enterprise_runtime = EnterpriseRuntime(
            iam=self.iam,
            recovery=self.recovery,
            message_bus=self.message_bus,
            enforce_iam=bool(
                getattr(self.settings, "enterprise_iam_enforcement", True)
            ),
            enforce_circuit_breaker=bool(
                getattr(self.settings, "enterprise_circuit_breaker", True)
            ),
            publish_bus=True,
        )

        # Register agents on the live P2P message bus for discovery/heartbeat
        for agent_id in (
            "supervisor",
            "schema_agent",
            "sql_agent",
            "visualization_agent",
            "insight_agent",
            "optimization_agent",
            "dashboard_agent",
            "summary_agent",
            "export_node",
            "reflection_agent",
            "retry_agent",
            "memory_agent",
            "goal_understanding",
            "planner",
            "task_decomposition",
            "execution_coordinator",
            "replan_agent",
            "goal_tracking",
            "approval_gate",
        ):
            self.message_bus.register_agent(agent_id, capabilities=[agent_id])
            self.message_bus.subscribe(
                agent_id,
                lambda msg, aid=agent_id: self.health.beat(
                    aid, last_message=msg.kind, from_sender=msg.sender
                ),
            )

        # Plugin marketplace — auto-discover + register into capability catalog
        plugin_dirs = [
            p.strip()
            for p in str(getattr(self.settings, "plugin_dirs", "plugins")).split(";")
            if p.strip()
        ]

        def _on_plugin_register(manifest, cap, handler):  # type: ignore[no-untyped-def]
            kind_map = {
                "agent": CapabilityKind.AGENT,
                "tool": CapabilityKind.TOOL,
                "skill": CapabilityKind.SKILL,
                "exporter": CapabilityKind.TOOL,
                "memory": CapabilityKind.MEMORY,
                "bundle": CapabilityKind.WORKFLOW,
            }
            risk_map = {
                "none": RiskClass.NONE,
                "low": RiskClass.LOW,
                "medium": RiskClass.MEDIUM,
                "high": RiskClass.HIGH,
                "critical": RiskClass.CRITICAL,
            }
            desc = CapabilityDescriptor(
                id=cap.id
                if cap.id.startswith(("agent.", "tool.", "skill.", "plugin."))
                else f"plugin.{cap.id}",
                kind=kind_map.get(cap.kind, CapabilityKind.SKILL),
                name=cap.name,
                description=cap.description or cap.name,
                version=manifest.version,
                tags=frozenset(set(cap.tags or {"plugin"}) | {"plugin"}),
                skills=frozenset(cap.skills or ()),
                risk_class=risk_map.get(str(cap.risk_class).lower(), RiskClass.MEDIUM),
                graph_node=cap.graph_node or "plugin_runtime_agent",
                ai_routable=cap.kind in {"agent", "tool", "skill"},
                system_protected=False,
                provides=frozenset(cap.provides or ()),
                plugin_id=manifest.id,
                metadata={"plugin_version": manifest.version, **(cap.metadata or {})},
            )
            try:
                self.registry.register(desc, handler=handler, overwrite=True)
                self.metrics.observe_plugin("register", manifest.id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin capability register failed %s: %s", cap.id, exc)

        self.plugins = PluginMarketplace(
            plugin_dirs=plugin_dirs,
            on_register=_on_plugin_register,
            signing_secret=str(
                getattr(self.settings, "plugin_signing_secret", "sqlmind-plugin-dev-secret")
            ),
            require_signature=bool(
                getattr(self.settings, "plugin_require_signature", False)
            ),
        )
        try:
            self.plugins.load_all()
            if getattr(self.settings, "plugin_hot_reload", True):
                self.plugins.hot_reload()
            # Planner discovers plugins dynamically via catalog
            self.plugin_catalog = self.plugins.discover_for_planner()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plugin marketplace load failed: %s", exc)
            self.plugin_catalog = []

        # Phase 4 — Production enterprise extensions (additive)
        from plugins.runtime import PluginRuntime
        from reliability.enterprise_queue import get_enterprise_queue
        from services.export_queue import get_export_queue
        from observability.parallel_metrics import get_parallel_metrics

        self.plugin_runtime = PluginRuntime(
            self.plugins,
            default_timeout=float(
                getattr(self.settings, "plugin_timeout_seconds", 15.0) or 15.0
            ),
        )
        eq_path = getattr(self.settings, "enterprise_queue_db_path", "") or (
            f"{data_dir}/sqlmind_enterprise_queue.sqlite3"
        )
        self.enterprise_queue = get_enterprise_queue(eq_path)
        self.export_queue = get_export_queue(self.settings.export_dir)
        self.parallel_metrics = get_parallel_metrics()
        self.distributed = None
        if bool(getattr(self.settings, "distributed_execution_enabled", True)):
            try:
                from distributed import get_distributed_executor

                dq = getattr(self.settings, "distributed_queue_db_path", "") or (
                    f"{data_dir}/sqlmind_task_queue.sqlite3"
                )
                self.distributed = get_distributed_executor(
                    db_path=dq,
                    worker_count=int(
                        getattr(self.settings, "distributed_worker_count", 2) or 2
                    ),
                    autostart=True,
                )

                def _plugin_handler(task):  # type: ignore[no-untyped-def]
                    payload = task.payload or {}
                    cap = str(
                        payload.get("capability_id")
                        or payload.get("plugin_capability_id")
                        or ""
                    )
                    if not cap:
                        raise ValueError("plugin_invoke requires capability_id")
                    return {
                        "result": self.plugin_runtime.invoke(cap, payload),
                    }

                def _export_handler(task):  # type: ignore[no-untyped-def]
                    # Export jobs are owned by ExportQueue; this handler polls/status
                    job_id = str((task.payload or {}).get("job_id") or "")
                    job = self.export_queue.get(job_id) if job_id else None
                    return {
                        "job_id": job_id,
                        "status": job.status if job else "missing",
                        "paths": dict(job.paths) if job else {},
                    }

                def _enterprise_claim_handler(task):  # type: ignore[no-untyped-def]
                    from reliability.enterprise_queue import process_claimed_message

                    item = task.payload or {}
                    return process_claimed_message(
                        item,
                        handlers=self._enterprise_claim_handlers(),
                    )

                self.distributed.register_handler("plugin_invoke", _plugin_handler)
                self.distributed.register_handler("export_job", _export_handler)
                self.distributed.register_handler(
                    "enterprise_claim", _enterprise_claim_handler
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Distributed executor init failed: %s", exc)

        # Always-on enterprise retry/DLQ claim consumer
        try:
            from reliability.enterprise_queue import start_claim_consumer

            start_claim_consumer(
                self.enterprise_queue,
                handlers=self._enterprise_claim_handlers(),
                poll_interval=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enterprise claim consumer failed to start: %s", exc)

        if bool(getattr(self.settings, "metrics_http_enabled", True)):
            try:
                from observability.metrics_server import start_metrics_server

                self.metrics_url = start_metrics_server(
                    host=str(getattr(self.settings, "metrics_http_host", "127.0.0.1")),
                    port=int(getattr(self.settings, "metrics_http_port", 9108) or 9108),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metrics HTTP server failed to start: %s", exc)
                self.metrics_url = ""
        else:
            self.metrics_url = ""

        for agent_id in ("recovery_controller", "plugin_runtime_agent"):
            self.message_bus.register_agent(agent_id, capabilities=[agent_id])

        self.app = build_sqlmind_graph(
            schema,
            executor,
            self.llm,
            self.settings,
            security=self.security,
            checkpointer=self.checkpointer,
            history_store=history_store,
            get_last_df=lambda: self._last_df,
            episodic_store=self.episodic,
            registry=self.registry,
            kernel_container=self.kernel,
            agent_registry=self.agent_registry,
            workflow_memory=self.workflow_memory,
            memory_fabric=self.memory_fabric
            if getattr(self.settings, "vector_memory_enabled", True)
            else None,
            goal_store=self.goal_store,
            experience_store=self.experience_store,
            enterprise_runtime=self.enterprise_runtime,
            plugin_runtime=self.plugin_runtime
            if getattr(self.settings, "plugin_runtime_enabled", True)
            else None,
        )

    def _enterprise_claim_handlers(self) -> dict[str, Any]:
        """Topic → handler for EnterpriseQueue claim consumer / DLQ replay."""

        def _plugin_failure(item: dict[str, Any]) -> dict[str, Any]:
            payload = item.get("payload") or {}
            cap = str(payload.get("capability_id") or "")
            if not cap or self.plugin_runtime is None:
                return {"skipped": True}
            try:
                result = self.plugin_runtime.invoke(cap, payload)
                return {"ok": True, "result": result}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}

        def _sql_retry(item: dict[str, Any]) -> dict[str, Any]:
            # Persist evidence for ops; live graph resume uses checkpoints/HITL
            logger.info(
                "Enterprise queue sql_retry_exhausted claimed id=%s session=%s",
                item.get("id"),
                item.get("session_id"),
            )
            try:
                from observability.metrics import get_metrics

                get_metrics().observe_queue("sql_retry_replay", "sql")
            except Exception:  # noqa: BLE001
                pass
            return {
                "ok": True,
                "action": "recorded",
                "session_id": item.get("session_id"),
                "hint": "Re-run session or resume checkpoint to apply retry",
            }

        def _default(item: dict[str, Any]) -> dict[str, Any]:
            logger.info(
                "Claimed enterprise message topic=%s id=%s",
                item.get("topic"),
                item.get("id"),
            )
            return {"ok": True, "topic": item.get("topic")}

        return {
            "plugin_failure": _plugin_failure,
            "sql_retry_exhausted": _sql_retry,
            "*": _default,
        }

    def _thread_config(self, session_id: str | None = None) -> dict[str, Any]:
        """LangGraph invoke config with LangSmith tags/metadata + correlation id."""
        sid = session_id or self.session_id
        ctx = None
        try:
            from core.context import get_request_context

            ctx = get_request_context()
        except Exception:  # noqa: BLE001
            pass
        return {
            "configurable": {"thread_id": sid},
            "run_name": "sqlmind_orchestrator",
            "tags": langsmith_tags(
                provider=self.settings.llm_provider,
                dialect=self.schema.dialect,
            ),
            "metadata": run_metadata(
                session_id=sid,
                provider=self.settings.llm_provider,
                dialect=self.schema.dialect,
                database=self.schema.database_name,
                tenant_id=(ctx.tenant_id if ctx else self.settings.default_tenant_id),
                actor=(ctx.actor if ctx else self.settings.default_actor),
                extra={"request_id": ctx.request_id if ctx else ""},
            ),
        }

    def _authenticate(
        self,
        *,
        actor: str | None = None,
        tenant_id: str | None = None,
        api_key: str | None = None,
        password: str | None = None,
    ):
        """Establish IAM session — API key, password, or default local bootstrap."""
        raw_key = api_key or getattr(self.settings, "iam_api_key", "") or ""
        if raw_key:
            token = self.iam.authenticate_api_key(raw_key)
            if token is not None:
                return token

        username = actor or getattr(self.settings, "iam_username", "") or self.settings.default_actor
        pw = password or getattr(self.settings, "iam_password", "") or "local-dev"
        token = self.iam.authenticate(username, pw)
        if token is not None:
            if tenant_id and tenant_id != token.tenant_id:
                # Soft rebind workspace/tenant only for admin sessions
                if "admin" in token.roles:
                    token.tenant_id = tenant_id
            return token

        # Last resort: create session for default principal if auth failed but principal exists
        principal = self.iam.get_principal_by_username(username) or self.iam.get_principal_by_username(
            "local-user"
        )
        if principal is not None:
            return self.iam.create_session(principal)
        return None

    def _initial_state(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None,
        memory_summary: str = "",
        session_id: str | None = None,
        *,
        actor: str | None = None,
        tenant_id: str | None = None,
        api_key: str | None = None,
        password: str | None = None,
    ) -> GraphState:
        initial = empty_state(question, session_id or self.session_id)
        initial["conversation_history"] = conversation_history or []
        initial["memory_summary"] = memory_summary
        initial["max_retries"] = self.settings.max_sql_retries
        initial["max_runtime_replans"] = int(
            getattr(self.settings, "max_runtime_replans", 2) or 2
        )
        initial["runtime_replan_count"] = 0
        initial["schema_text"] = self.schema.to_prompt_text()
        initial["schema_dict"] = self.schema.to_dict()
        initial["database_name"] = self.schema.database_name
        initial["dialect"] = self.schema.dialect
        initial["episodic_context"] = self.episodic.format_for_prompt(
            question, database_name=self.schema.database_name
        )
        initial["capability_registry_version"] = int(self.registry.version)
        initial["workflow_memory_context"] = self.workflow_memory.format_for_prompt(
            question, database_name=self.schema.database_name
        )

        # Runtime IAM — every execution carries a security context
        iam_token = self._authenticate(
            actor=actor, tenant_id=tenant_id, api_key=api_key, password=password
        )
        if iam_token is not None:
            initial["iam_session"] = iam_token.model_dump(mode="json")
            initial["tenant_id"] = iam_token.tenant_id
            initial["workspace_id"] = iam_token.workspace_id
        else:
            initial["iam_session"] = {}
            initial["tenant_id"] = tenant_id or self.settings.default_tenant_id
            initial["workspace_id"] = "default"
            if getattr(self.settings, "enterprise_iam_enforcement", True):
                logger.warning("IAM authentication failed — graph will deny protected nodes")

        initial["plugin_catalog_version"] = int(self.registry.version)
        # Semantic memory preview for planner (also refreshed in memory_agent)
        if getattr(self.settings, "vector_memory_enabled", True):
            try:
                initial["vector_memory_context"] = self.vector_memory.format_for_prompt(
                    question,
                    tenant_id=initial["tenant_id"],
                    database_name=self.schema.database_name,
                )
                initial["learning_context"] = self.experience_store.format_for_planner(
                    question
                )
                self.metrics.observe_memory("retrieve", status="ok")
                safe_trace(
                    "memory_event",
                    kind="Memory Retrieval",
                    memories=(initial.get("vector_memory_context") or "")[:400],
                    detail={"source": "orchestrator_preview"},
                )
            except Exception:  # noqa: BLE001
                initial["vector_memory_context"] = ""
                initial["learning_context"] = ""
        return initial

    def _bind_request_context(
        self,
        question: str,
        session_id: str | None,
        *,
        actor: str | None = None,
        tenant_id: str | None = None,
    ) -> RequestContext:
        ensure_trace_id()
        ctx = RequestContext(
            session_id=session_id or self.session_id,
            actor=actor or self.settings.default_actor,
            tenant_id=tenant_id or self.settings.default_tenant_id,
            database=self.schema.database_name,
            question=question,
        )
        set_request_context(ctx)
        try:
            self.message_bus.heartbeat("orchestrator", session_id=ctx.session_id)
        except Exception:  # noqa: BLE001
            pass
        return ctx

    def _clear_run_context(self) -> None:
        clear_request_context()
        clear_trace_context()

    def _runtime_trace_enabled(self) -> bool:
        return bool(getattr(self.settings, "runtime_trace_enabled", True))

    def _runtime_trace_dir(self) -> str:
        return str(
            getattr(self.settings, "runtime_trace_dir", "")
            or f"{self.settings.data_dir}/runtime_traces"
        )

    def _seed_trace_platform_evidence(self) -> None:
        """Record platform capabilities active for this run (plugins, CB, observability)."""
        try:
            catalog = []
            if getattr(self, "plugins", None) is not None:
                catalog = self.plugins.catalog()
            safe_trace(
                "plugin_event",
                kind="Plugin Discovery",
                status="ok",
                detail={"count": len(catalog), "ids": [c.get("id") for c in catalog[:20]]},
            )
            for item in catalog[:10]:
                safe_trace(
                    "plugin_event",
                    kind="Plugin Loading",
                    plugin_id=str(item.get("id") or ""),
                    status="ok",
                    detail={"enabled": item.get("enabled")},
                )
            safe_trace(
                "reliability_event",
                kind="Circuit Breaker",
                status="active"
                if getattr(self.settings, "enterprise_circuit_breaker", True)
                else "disabled",
                detail={"enforcement": True},
            )
            safe_trace(
                "reliability_event",
                kind="Health Check",
                status="ok",
                detail={"orchestrator": True},
            )
        except Exception:  # noqa: BLE001
            pass

    def runtime_dashboard_stats(self) -> dict[str, Any]:
        """Live enterprise runtime statistics for ops dashboard /metrics consumers."""
        stats: dict[str, Any] = {
            "metrics_url": getattr(self, "metrics_url", ""),
            "agents_online": self.message_bus.discover(),
            "message_queue_depth": len(self.message_bus.queue_snapshot(limit=500)),
            "iam": {
                "enforcement": bool(
                    getattr(self.settings, "enterprise_iam_enforcement", True)
                ),
            },
            "circuit_breakers": {},
            "parallel": {},
            "plugins": {},
            "exports": {},
            "enterprise_queue": {},
            "distributed": {},
            "memory": {},
            "replanning": {
                "max_runtime_replans": int(
                    getattr(self.settings, "max_runtime_replans", 2) or 2
                ),
            },
            "recovery": {
                "controller_enabled": bool(
                    getattr(self.settings, "recovery_controller_enabled", True)
                ),
            },
        }
        try:
            for name, br in self.breakers._breakers.items():  # noqa: SLF001
                stats["circuit_breakers"][name] = {
                    "state": br.state,
                    "failures": br.failures,
                }
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "parallel_metrics", None) is not None:
                stats["parallel"] = self.parallel_metrics.snapshot()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "plugin_runtime", None) is not None:
                stats["plugins"] = self.plugin_runtime.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "export_queue", None) is not None:
                stats["exports"] = self.export_queue.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "enterprise_queue", None) is not None:
                stats["enterprise_queue"] = self.enterprise_queue.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            if getattr(self, "distributed", None) is not None:
                stats["distributed"] = self.distributed.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            stats["memory"] = {
                "vector_enabled": bool(
                    getattr(self.settings, "vector_memory_enabled", True)
                ),
                "workflow_db": getattr(self.workflow_memory, "db_path", ""),
            }
        except Exception:  # noqa: BLE001
            pass
        try:
            stats["gauges"] = dict(self.metrics.gauge_values)
            stats["timelines"] = self.metrics.all_timelines()
        except Exception:  # noqa: BLE001
            pass
        return stats

    def _finalize_runtime_trace(
        self,
        *,
        success: bool,
        final_state: dict[str, Any] | None,
    ) -> dict[str, str]:
        session = get_active_trace()
        if session is None or not session.enabled:
            return {}
        try:
            return finalize_and_export(
                session,
                output_dir=self._runtime_trace_dir(),
                success=success,
                final_state=final_state,
                print_summary=bool(
                    getattr(self.settings, "runtime_trace_print_summary", True)
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Runtime trace export failed: %s", exc)
            return {}

    def _persist_episode(self, question: str, result: dict[str, Any]) -> None:
        try:
            self.episodic.record(
                question=question,
                sql=result.get("sql") or "",
                outcome=(result.get("insights") or result.get("final_response") or "")[:500],
                row_count=int(result.get("row_count") or 0),
                success=bool(result.get("query_success")),
                session_id=result.get("session_id") or self.session_id,
                database_name=self.schema.database_name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Episodic persist failed: %s", exc)
        persist_workflow_outcome(self.workflow_memory, result)
        # Phase 3 — vector memory + learning + events
        try:
            tenant_id = result.get("tenant_id") or self.settings.default_tenant_id
            self.memory_fabric.remember_outcome(result, tenant_id=tenant_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector memory persist failed: %s", exc)
        try:
            self.experience_store.record_from_result(
                result, tenant_id=result.get("tenant_id") or self.settings.default_tenant_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Learning persist failed: %s", exc)
        try:
            self.events.emit(
                "goal",
                "run_complete",
                session_id=result.get("session_id") or self.session_id,
                tenant_id=result.get("tenant_id") or self.settings.default_tenant_id,
                status="ok" if not result.get("error") else "error",
                payload={
                    "query_success": result.get("query_success"),
                    "has_plan": bool(result.get("adaptive_plan")),
                },
            )
            self.health.beat("orchestrator", session_id=result.get("session_id") or "")
            self.metrics.observe_goal(
                "run_complete",
                success=bool(result.get("query_success")),
            )
            self.metrics.record_run_outcome(success=bool(result.get("query_success")))
            try:
                self.metrics.set_gauge(
                    "replan_count",
                    float(result.get("runtime_replan_count") or 0),
                )
                if getattr(self, "export_queue", None) is not None:
                    self.metrics.set_gauge(
                        "export_count",
                        float(self.export_queue.stats().get("completed", 0)),
                    )
                if getattr(self, "distributed", None) is not None:
                    dstats = self.distributed.stats()
                    self.metrics.set_gauge(
                        "worker_count", float(dstats.get("healthy_workers", 0))
                    )
                    self.metrics.set_gauge(
                        "queue_length",
                        float(sum((dstats.get("queue_length") or {}).values())),
                    )
                if getattr(self, "enterprise_queue", None) is not None:
                    self.metrics.set_gauge(
                        "recovery_count",
                        float(self.metrics.gauge_values.get("recovery_count", 0)),
                    )
            except Exception:  # noqa: BLE001
                pass
            # Export Prometheus text snapshot when enabled
            if getattr(self.settings, "prometheus_enabled", True):
                try:
                    from pathlib import Path

                    path = Path(
                        getattr(self.settings, "metrics_export_path", "")
                        or f"{self.settings.data_dir}/sqlmind_metrics.prom"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(self.metrics.render_prometheus(), encoding="utf-8")
                except Exception as mex:  # noqa: BLE001
                    logger.debug("Metrics export failed: %s", mex)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Enterprise event emit failed: %s", exc)

    def run(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        memory_summary: str = "",
        session_id: str | None = None,
        actor: str | None = None,
        tenant_id: str | None = None,
    ) -> GraphState:
        self._bind_request_context(
            question, session_id, actor=actor, tenant_id=tenant_id
        )
        try:
            with execution_trace_session(
                session_id=session_id or self.session_id,
                user_id=actor or self.settings.default_actor,
                tenant_id=tenant_id or self.settings.default_tenant_id,
                llm_provider=self.settings.llm_provider,
                model_name=self.settings.resolve_model(),
                enabled=self._runtime_trace_enabled(),
                question=question,
            ):
                self._seed_trace_platform_evidence()
                initial = self._initial_state(
                    question, conversation_history, memory_summary, session_id,
                    actor=actor, tenant_id=tenant_id,
                )
                # Bind IAM identity into the active trace after auth
                trace = get_active_trace()
                if trace is not None:
                    trace.bind_ids_from_state(initial)
                logger.info("Running SQLMind graph for: %s", question[:120])
                result: dict[str, Any] = {}
                try:
                    with start_span(
                        "sqlmind.orchestrator.invoke",
                        attributes={
                            "sqlmind.session_id": session_id or self.session_id,
                            "sqlmind.provider": self.settings.llm_provider,
                            "sqlmind.dialect": self.schema.dialect,
                            "sqlmind.database": self.schema.database_name,
                            "sqlmind.tenant_id": tenant_id or self.settings.default_tenant_id,
                            "sqlmind.actor": actor or self.settings.default_actor,
                        },
                    ):
                        result = self.app.invoke(
                            initial, config=self._thread_config(session_id)
                        )
                    self._cache_df(result)
                    self._persist_episode(question, result)
                finally:
                    success = bool(result.get("query_success")) and not result.get("error")
                    if result.get("final_response") and not result.get("error"):
                        success = True
                    self._finalize_runtime_trace(success=success, final_state=result or initial)
                return result  # type: ignore[return-value]
        finally:
            self._clear_run_context()

    def stream_events(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        *,
        memory_summary: str = "",
        session_id: str | None = None,
        actor: str | None = None,
        tenant_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        self._bind_request_context(
            question, session_id, actor=actor, tenant_id=tenant_id
        )
        try:
            with execution_trace_session(
                session_id=session_id or self.session_id,
                user_id=actor or self.settings.default_actor,
                tenant_id=tenant_id or self.settings.default_tenant_id,
                llm_provider=self.settings.llm_provider,
                model_name=self.settings.resolve_model(),
                enabled=self._runtime_trace_enabled(),
                question=question,
            ):
                self._seed_trace_platform_evidence()
                initial = self._initial_state(
                    question, conversation_history, memory_summary, session_id,
                    actor=actor, tenant_id=tenant_id,
                )
                trace = get_active_trace()
                if trace is not None:
                    trace.bind_ids_from_state(initial)
                final: dict[str, Any] = {}
                try:
                    with start_span(
                        "sqlmind.orchestrator.stream",
                        attributes={
                            "sqlmind.session_id": session_id or self.session_id,
                            "sqlmind.provider": self.settings.llm_provider,
                            "sqlmind.dialect": self.schema.dialect,
                            "sqlmind.database": self.schema.database_name,
                        },
                    ):
                        for event in self.app.stream(
                            initial,
                            config=self._thread_config(session_id),
                            stream_mode="updates",
                        ):
                            for node_name, update in event.items():
                                if isinstance(update, dict):
                                    final.update(update)
                                # Subgraph-level evidence (parent stream keys)
                                if node_name in SUBGRAPH_NAMES:
                                    safe_trace(
                                        "observe_stream_update",
                                        node_name=node_name,
                                        update=update if isinstance(update, dict) else {},
                                    )
                                elif get_active_trace() is not None:
                                    if isinstance(update, dict) and update.get("next_agent"):
                                        safe_trace(
                                            "conditional_edge",
                                            source=str(node_name),
                                            dest=str(update.get("next_agent")),
                                            reason="stream update next_agent",
                                        )
                            yield event
                    self._cache_df(final)
                    self._persist_episode(question, final)
                finally:
                    success = bool(final.get("query_success")) and not final.get("error")
                    if final.get("final_response") and not final.get("error"):
                        success = True
                    self._finalize_runtime_trace(
                        success=success, final_state=final or initial
                    )
        finally:
            self._clear_run_context()

    def stream_answer_tokens(
        self,
        final_state: dict[str, Any],
    ) -> Iterator[str]:
        """Real provider token stream for a short closing narrative when supported.

        If streaming fails (provider/structured-only path), yields nothing — callers
        should show a labeled non-fake fallback instead of a typewriter effect.
        """
        insights = (final_state.get("insights") or "").strip()
        if insights:
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Write one short closing sentence for an analytics answer. "
                        "No markdown. Be specific to the findings.",
                    ),
                    (
                        "human",
                        "Question: {question}\nFindings:\n{insights}\n",
                    ),
                ]
            )
            try:
                yield from self.llm.stream_text(
                    prompt,
                    {
                        "question": final_state.get("question") or "",
                        "insights": insights[:2500],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("Token streaming unavailable: %s", exc)
                return

    def resume(
        self,
        resume_value: Any,
        *,
        session_id: str | None = None,
    ) -> GraphState:
        """Resume after a HITL ``interrupt()`` (e.g. clarification answer)."""
        from langgraph.types import Command

        with execution_trace_session(
            session_id=session_id or self.session_id,
            user_id=self.settings.default_actor,
            tenant_id=self.settings.default_tenant_id,
            llm_provider=self.settings.llm_provider,
            model_name=self.settings.resolve_model(),
            enabled=self._runtime_trace_enabled(),
            question="(resume)",
        ):
            safe_trace("interrupt", node_name="resume", reason="HITL resume begin")
            safe_trace("resume", resume_value=resume_value)
            result = self.app.invoke(
                Command(resume=resume_value),
                config=self._thread_config(session_id),
            )
            self._cache_df(result)
            success = bool(result.get("query_success")) or bool(
                result.get("final_response")
            )
            if result.get("error"):
                success = False
            self._finalize_runtime_trace(success=success, final_state=result)
            return result  # type: ignore[return-value]

    def _cache_df(self, state: dict[str, Any]) -> None:
        import pandas as pd

        records = state.get("dataframe_records") or []
        columns = state.get("columns") or []
        if records:
            self._last_df = pd.DataFrame(records)
        elif columns:
            self._last_df = pd.DataFrame(columns=columns)
