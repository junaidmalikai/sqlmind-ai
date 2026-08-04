"""Phase 2 autonomous agents — Goal, Planner, Decompose, Replan, Memory.

Existing specialists (Supervisor, SQL, Viz, …) are unchanged; these nodes
sit *in front of* / *beside* them and communicate via structured messages.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from graph.state import GraphState
from memory.workflow_memory import WorkflowMemoryStore
from planner.messages import (
    feedback_message,
    goal_message,
    plan_message,
    task_message,
)
from planner.models import (
    AdaptivePlan,
    DecompositionOutput,
    GoalSpec,
    PlanStep,
    PlannerOutput,
    ReplanDecision,
    TaskGraph,
    TaskSpec,
)
from planner.registry import AgentRegistry
from planner.selection import bind_task_to_agent, expand_skills, select_agent_for_task
from prompts.templates import (
    goal_understanding_prompt,
    planner_prompt,
    replan_prompt,
    task_decomposition_prompt,
)
from services.llm_service import LLMService
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Confidence below this triggers HITL clarify automatically
DEFAULT_CLARIFY_THRESHOLD = 0.55


def _log(agent: str, message: str, status: str = "ok", detail: Any = None) -> dict:
    entry: dict[str, Any] = {"agent": agent, "message": message, "status": status}
    if detail is not None:
        entry["detail"] = detail
    return entry


def _question(state: GraphState) -> str:
    return state.get("rewritten_question") or state.get("question") or ""


def _safe_goal(state: GraphState) -> GoalSpec:
    raw = state.get("goal_spec")
    if isinstance(raw, dict) and raw:
        try:
            return GoalSpec.model_validate(raw)
        except Exception:  # noqa: BLE001
            pass
    return GoalSpec(
        goal=_question(state) or "Analyze database",
        confidence=0.5,
        simple=True,
        rewritten_question=_question(state),
    )


# ---------------------------------------------------------------------------
# 1. Goal Understanding Agent
# ---------------------------------------------------------------------------

def make_goal_understanding_agent(
    llm: LLMService,
    *,
    clarify_threshold: float = DEFAULT_CLARIFY_THRESHOLD,
):
    def goal_understanding_agent(state: GraphState) -> dict[str, Any]:
        try:
            result: GoalSpec = llm.invoke_structured(
                goal_understanding_prompt(),
                GoalSpec,
                {
                    "question": state.get("question") or "",
                    "memory_summary": state.get("memory_summary") or "(none)",
                    "episodic_context": state.get("episodic_context") or "(none)",
                    "workflow_memory": state.get("workflow_memory_context") or "(none)",
                    "database_name": state.get("database_name") or "",
                    "dialect": state.get("dialect") or "",
                    "schema_loaded": bool(state.get("schema_text")),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Goal understanding failed: %s", exc)
            # Degrade to simple supervisor fast-path — do NOT force HITL on LLM tooling gaps
            result = GoalSpec(
                goal=state.get("question") or "Analyze data",
                objectives=[state.get("question") or ""],
                confidence=0.65,
                simple=True,
                needs_clarification=False,
                rewritten_question=state.get("question") or "",
                ambiguity_flags=[],
                expected_output="analytics answer",
                intent_label="query",
            )

        rewritten = result.rewritten_question or state.get("question") or ""
        out: dict[str, Any] = {
            "goal_spec": result.to_state_dict(),
            "intent": result.intent_label or state.get("intent") or "query",
            "rewritten_question": rewritten,
            "agent_messages": [
                goal_message("goal_understanding", result.to_state_dict()).to_state_dict()
            ],
            "agent_logs": [
                _log(
                    "goal_understanding",
                    f"goal={result.goal[:80]} conf={result.confidence:.2f} simple={result.simple}",
                    detail=result.objectives,
                )
            ],
            "route_history": ["goal_understanding"],
        }

        # HITL: low confidence or explicit ambiguity → clarify
        needs = bool(result.needs_clarification) or (
            result.confidence < clarify_threshold and bool(result.ambiguity_flags)
        )
        if needs:
            out["needs_clarification"] = True
            out["clarification_question"] = (
                result.clarification_question
                or "Could you clarify your analysis goal and expected output?"
            )
            out["next_agent"] = "clarify"
            out["agent_logs"].append(
                _log("goal_understanding", "Low confidence → HITL clarify", "warn")
            )
        else:
            out["needs_clarification"] = False
            out["next_agent"] = "planner"
        return out

    return goal_understanding_agent


# ---------------------------------------------------------------------------
# 2. Planner Agent
# ---------------------------------------------------------------------------

def make_planner_agent(
    llm: LLMService,
    agent_registry: AgentRegistry,
    *,
    experience_store: Any | None = None,
):
    def planner_agent(state: GraphState) -> dict[str, Any]:
        goal = _safe_goal(state)
        catalog = agent_registry.catalog_text()
        # Dynamic plugin marketplace discovery (no hardcoded plugins)
        try:
            plugin_hint = ""
            for cap in agent_registry.capability_registry.list(enabled_only=True):
                if getattr(cap, "plugin_id", None):
                    plugin_hint += f"\n- plugin:{cap.id} ({cap.name})"
            if plugin_hint:
                catalog = f"{catalog}\n\n## Marketplace plugins{plugin_hint}"
        except Exception:  # noqa: BLE001
            pass
        workflow_mem = state.get("workflow_memory_context") or "(none)"
        vector_mem = state.get("vector_memory_context") or ""
        if vector_mem:
            workflow_mem = f"{workflow_mem}\n\n{vector_mem}"
        learning = state.get("learning_context") or ""
        if not learning and experience_store is not None:
            try:
                learning = experience_store.format_for_planner(_question(state)) or ""
            except Exception:  # noqa: BLE001
                learning = ""
        if learning:
            workflow_mem = f"{workflow_mem}\n\n{learning}"

        try:
            from observability.metrics import get_metrics

            get_metrics().observe_planner("plan_start")
        except Exception:  # noqa: BLE001
            pass

        # Simple fast path — AdaptivePlan with supervisor fallback
        if goal.simple and goal.confidence >= 0.7:
            plan = AdaptivePlan(
                plan_id=f"plan-{uuid4().hex[:8]}",
                goal_summary=goal.goal,
                strategy="Fast path: Supervisor-mediated specialist routing",
                execution_mode="sequential",
                estimated_cost=1.0,
                use_supervisor_fallback=True,
                status="ready",
                reasoning="Goal marked simple — preserve existing Supervisor loop",
                memory_hints_used=["learning"] if learning else [],
            )
            return {
                "adaptive_plan": plan.to_state_dict(),
                "next_agent": "task_decomposition",
                "agent_messages": [
                    plan_message("planner", plan.to_state_dict()).to_state_dict()
                ],
                "agent_logs": [_log("planner", "Simple goal → supervisor fallback plan")],
                "route_history": ["planner"],
            }

        try:
            raw: PlannerOutput = llm.invoke_structured(
                planner_prompt(),
                PlannerOutput,
                {
                    "goal_json": json.dumps(goal.to_state_dict(), indent=2),
                    "agent_catalog": catalog,
                    "workflow_memory": workflow_mem,
                    "episodic_context": state.get("episodic_context") or "(none)",
                    "schema_loaded": bool(state.get("schema_text")),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Planner LLM failed: %s", exc)
            raw = PlannerOutput(
                strategy="Fallback: Supervisor routing due to planner error",
                execution_mode="sequential",
                estimated_cost=1.0,
                use_supervisor_fallback=True,
                reasoning=str(exc),
                required_skills_sequence=[["nl2sql"]],
            )

        steps: list[PlanStep] = []
        for i, skill_group in enumerate(raw.required_skills_sequence or []):
            mode = "parallel" if len(skill_group) > 1 else "sequential"
            if raw.execution_mode == "parallel" and len(skill_group) > 1:
                mode = "parallel"
            steps.append(
                PlanStep(
                    step_id=f"s{i + 1}",
                    task_ids=[],  # filled after decomposition
                    mode=mode,  # type: ignore[arg-type]
                    rationale=f"skills={skill_group}",
                )
            )

        hints_used = []
        if workflow_mem != "(none)":
            hints_used.append("workflow_memory")
        if state.get("vector_memory_context"):
            hints_used.append("vector_memory")
        if learning:
            hints_used.append("planner_learning")

        plan = AdaptivePlan(
            plan_id=f"plan-{uuid4().hex[:8]}",
            goal_summary=goal.goal,
            strategy=raw.strategy,
            steps=steps,
            execution_mode=raw.execution_mode,
            estimated_cost=raw.estimated_cost,
            use_supervisor_fallback=raw.use_supervisor_fallback,
            status="draft",
            reasoning=raw.reasoning,
            memory_hints_used=hints_used,
        )
        return {
            "adaptive_plan": plan.to_state_dict(),
            "planner_output": raw.model_dump(mode="json"),
            "next_agent": "task_decomposition",
            "agent_messages": [
                plan_message("planner", plan.to_state_dict()).to_state_dict()
            ],
            "agent_logs": [
                _log(
                    "planner",
                    f"strategy={raw.strategy[:80]} mode={raw.execution_mode}",
                    detail=raw.reasoning,
                )
            ],
            "route_history": ["planner"],
        }

    return planner_agent


# ---------------------------------------------------------------------------
# 3. Task Decomposition Agent
# ---------------------------------------------------------------------------

def make_task_decomposition_agent(llm: LLMService, agent_registry: AgentRegistry):
    def task_decomposition_agent(state: GraphState) -> dict[str, Any]:
        goal = _safe_goal(state)
        plan = AdaptivePlan.from_state(state.get("adaptive_plan"))  # type: ignore[arg-type]
        catalog = agent_registry.catalog_text()

        if plan and plan.use_supervisor_fallback:
            # Minimal placeholder graph — coordinator will hand off to supervisor
            graph = TaskGraph(tasks=[], roots=[])
            if plan:
                plan.status = "ready"
            return {
                "task_graph": graph.to_state_dict(),
                "adaptive_plan": plan.to_state_dict() if plan else None,
                "next_agent": "execution_coordinator",
                "plan_active": False,
                "agent_logs": [
                    _log("task_decomposition", "Skipped — supervisor fallback")
                ],
                "route_history": ["task_decomposition"],
            }

        try:
            raw: DecompositionOutput = llm.invoke_structured(
                task_decomposition_prompt(),
                DecompositionOutput,
                {
                    "goal_json": json.dumps(goal.to_state_dict(), indent=2),
                    "plan_json": json.dumps(
                        (plan.to_state_dict() if plan else {}), indent=2
                    ),
                    "agent_catalog": catalog,
                    "planner_skills": json.dumps(
                        (state.get("planner_output") or {}).get(
                            "required_skills_sequence", []
                        )
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Decomposition failed: %s", exc)
            # Deterministic minimal task list from goal expected output
            raw = DecompositionOutput(
                tasks=_fallback_tasks(goal),
                reasoning=f"Fallback decomposition: {exc}",
            )

        # Bind each task to an agent via registry (dynamic selection)
        bound_tasks: list[TaskSpec] = []
        agent_ids: list[str] = []
        for task in raw.tasks:
            match, _decision = select_agent_for_task(agent_registry, task)
            updated = bind_task_to_agent(task, match)
            if (
                match is not None
                and match.descriptor.execution_strategy.value == "parallel_safe"
            ):
                updated = updated.model_copy(update={"parallel_safe": True})
            # Mark viz/insight parallel-safe by skill
            skills = expand_skills(updated.required_skills)
            if skills & {"visualize", "chart_recommend", "insights", "business_analysis"}:
                updated = updated.model_copy(update={"parallel_safe": True})
            bound_tasks.append(updated)
            if updated.responsible_agent_id:
                agent_ids.append(updated.responsible_agent_id)

        roots = [t.id for t in bound_tasks if not t.dependencies]
        graph = TaskGraph(tasks=bound_tasks, roots=roots)

        if plan:
            # Attach task ids to plan steps when possible
            if plan.steps and bound_tasks:
                # One step per task if steps empty of task_ids
                new_steps: list[PlanStep] = []
                for i, t in enumerate(bound_tasks):
                    mode = "parallel" if t.parallel_safe else "sequential"
                    new_steps.append(
                        PlanStep(
                            step_id=f"s{i + 1}",
                            task_ids=[t.id],
                            mode=mode,  # type: ignore[arg-type]
                            rationale=t.title,
                        )
                    )
                plan.steps = new_steps
            plan.agent_ids = agent_ids
            plan.status = "ready"
            plan.estimated_cost = sum(t.estimated_cost for t in bound_tasks)

        msgs = [
            task_message("task_decomposition", t.model_dump(mode="json")).to_state_dict()
            for t in bound_tasks
        ]
        return {
            "task_graph": graph.to_state_dict(),
            "adaptive_plan": plan.to_state_dict() if plan else None,
            "next_agent": "execution_coordinator",
            "plan_active": True,
            "agent_messages": msgs,
            "agent_logs": [
                _log(
                    "task_decomposition",
                    f"{len(bound_tasks)} tasks; agents={agent_ids}",
                    detail=raw.reasoning,
                )
            ],
            "route_history": ["task_decomposition"],
        }

    return task_decomposition_agent


def _fallback_tasks(goal: GoalSpec) -> list[TaskSpec]:
    """Deterministic task list when LLM decomposition fails."""
    expected = (goal.expected_output or "").lower()
    intent = (goal.intent_label or "").lower()
    tasks: list[TaskSpec] = [
        TaskSpec(
            id="t1",
            title="Ensure schema context",
            required_skills=["schema_discovery"],
            provides_any=["schema_text"],
            expected_output="schema_text",
            estimated_cost=0.5,
        ),
    ]
    if intent in {"summary"} or "summary" in expected:
        tasks.append(
            TaskSpec(
                id="t2",
                title="Database summary",
                required_skills=["database_summary"],
                provides_any=["database_summary"],
                dependencies=["t1"],
                expected_output="database_summary",
            )
        )
    elif intent in {"dashboard", "kpi"} or "dashboard" in expected:
        tasks.append(
            TaskSpec(
                id="t2",
                title="Dashboard blueprint",
                required_skills=["dashboard_design"],
                provides_any=["dashboard_spec"],
                dependencies=["t1"],
                expected_output="dashboard_spec",
            )
        )
    elif intent in {"schema"}:
        pass
    else:
        tasks.append(
            TaskSpec(
                id="t2",
                title="Author and run SQL",
                required_skills=["nl2sql"],
                provides_any=["sql"],
                dependencies=["t1"],
                expected_output="sql result",
                estimated_cost=2.0,
            )
        )
        tasks.append(
            TaskSpec(
                id="t3",
                title="Visualize results",
                required_skills=["visualize"],
                provides_any=["chart_spec"],
                dependencies=["t2"],
                parallel_safe=True,
                expected_output="chart",
            )
        )
        tasks.append(
            TaskSpec(
                id="t4",
                title="Generate insights",
                required_skills=["insights"],
                provides_any=["insights"],
                dependencies=["t2"],
                parallel_safe=True,
                expected_output="insights",
            )
        )
    if "export" in expected or intent == "export":
        last = tasks[-1].id if tasks else "t1"
        tasks.append(
            TaskSpec(
                id=f"t{len(tasks) + 1}",
                title="Export report",
                required_skills=["export"],
                provides_any=["export_paths"],
                dependencies=[last],
                expected_output="export files",
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# 7. Reflection & Replanning Agent (true replan — not bare retry)
# ---------------------------------------------------------------------------

def make_replan_agent(llm: LLMService, agent_registry: AgentRegistry):
    def replan_agent(state: GraphState) -> dict[str, Any]:
        goal = _safe_goal(state)
        plan = AdaptivePlan.from_state(state.get("adaptive_plan"))  # type: ignore[arg-type]
        graph = TaskGraph.from_state(state.get("task_graph"))  # type: ignore[arg-type]
        reason = state.get("replan_reason") or state.get("sql_error") or "execution failure"

        # Configurable max replans — avoid infinite Failure→Reflection→Planner loops
        runtime_count = int(state.get("runtime_replan_count") or 0)
        plan_count = int(plan.replan_count) if plan else 0
        replan_count = max(runtime_count, plan_count)
        max_replans = int(
            state.get("max_runtime_replans")
            or (plan.max_replans if plan else 2)
            or 2
        )
        if replan_count >= max_replans:
            try:
                from observability.metrics import get_metrics

                get_metrics().observe_replan("max_exceeded")
            except Exception:  # noqa: BLE001
                pass
            return {
                "next_agent": "fail",
                "status": "failed",
                "error": f"Max replans exceeded ({replan_count}/{max_replans})",
                "final_response": (
                    "I could not recover after multiple replanning attempts. "
                    f"Last failure: {reason}"
                ),
                "runtime_replan_count": replan_count,
                "replan_decision": {
                    "strategy": "abort",
                    "reasoning": "max_replans_exceeded",
                    "failure_analysis": reason,
                },
                "agent_logs": [
                    _log(
                        "replan_agent",
                        f"max replans exceeded ({replan_count}/{max_replans})",
                        "error",
                    )
                ],
                "route_history": ["replan_agent"],
            }

        # Merge previous execution context for the new plan cycle
        prior_context = {
            "sql": state.get("sql") or "",
            "sql_error": state.get("sql_error") or "",
            "retry_diagnosis": state.get("retry_diagnosis") or "",
            "fix_hint": state.get("fix_hint") or "",
            "reflection_notes": state.get("reflection_notes") or "",
            "reflection_verdict": state.get("reflection_verdict") or "",
            "route_history": list(state.get("route_history") or [])[-20:],
            "completed_tasks": [
                t.get("id")
                for t in ((state.get("task_graph") or {}).get("tasks") or [])
                if isinstance(t, dict) and t.get("status") == "completed"
            ],
            "failed_tasks": [
                t.get("id")
                for t in ((state.get("task_graph") or {}).get("tasks") or [])
                if isinstance(t, dict) and t.get("status") == "failed"
            ],
            "row_count": state.get("row_count"),
            "query_success": state.get("query_success"),
            "replan_count": replan_count,
        }

        try:
            decision: ReplanDecision = llm.invoke_structured(
                replan_prompt(),
                ReplanDecision,
                {
                    "goal_json": json.dumps(goal.to_state_dict(), indent=2),
                    "plan_json": json.dumps(
                        plan.to_state_dict() if plan else {}, indent=2
                    ),
                    "task_graph_json": json.dumps(
                        graph.to_state_dict() if graph else {}, indent=2
                    ),
                    "failure_reason": reason,
                    "sql_error": state.get("sql_error") or "",
                    "retry_diagnosis": state.get("retry_diagnosis") or "",
                    "route_history": str(list(state.get("route_history") or [])),
                    "agent_catalog": agent_registry.catalog_text(max_chars=2000),
                    "prior_execution_context": json.dumps(prior_context, default=str)[:4000],
                    "replan_count": replan_count,
                    "max_replans": max_replans,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Replan agent failed: %s", exc)
            decision = ReplanDecision(
                strategy="continue_with_supervisor",
                reasoning=str(exc),
                failure_analysis=reason,
                confidence=0.4,
            )

        new_count = replan_count + 1
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_replan(str(decision.strategy))
        except Exception:  # noqa: BLE001
            pass

        out: dict[str, Any] = {
            "replan_decision": decision.model_dump(mode="json"),
            "runtime_replan_count": new_count,
            "max_runtime_replans": max_replans,
            "prior_execution_context": prior_context,
            "agent_messages": [
                feedback_message(
                    "replan_agent",
                    {
                        **decision.model_dump(mode="json"),
                        "replan_count": new_count,
                        "prior_context_keys": list(prior_context.keys()),
                    },
                ).to_state_dict()
            ],
            "agent_logs": [
                _log(
                    "replan_agent",
                    f"strategy={decision.strategy} replan={new_count}/{max_replans}",
                    detail=decision.reasoning,
                )
            ],
            "route_history": ["replan_agent"],
        }

        if plan:
            plan.replan_count = new_count
            plan.max_replans = max_replans

        if decision.strategy == "ask_clarify":
            out["needs_clarification"] = True
            out["clarification_question"] = (
                decision.clarification_question
                or "The plan failed — could you clarify what you need?"
            )
            out["next_agent"] = "clarify"
            if plan:
                out["adaptive_plan"] = plan.to_state_dict()
            return out

        if decision.strategy == "abort":
            out["next_agent"] = "fail"
            out["error"] = decision.failure_analysis or decision.reasoning
            out["final_response"] = (
                decision.reasoning
                or "I could not complete the analysis after replanning."
            )
            out["status"] = "failed"
            if plan:
                out["adaptive_plan"] = plan.to_state_dict()
            return out

        if decision.strategy == "continue_with_supervisor":
            if plan:
                plan.use_supervisor_fallback = True
                plan.status = "ready"
                out["adaptive_plan"] = plan.to_state_dict()
            out["plan_active"] = False
            out["next_agent"] = "supervisor"
            return out

        # revise_tasks / swap_agent / narrow_scope → rebuild via planner+decompose
        updated_goal = goal.model_copy(
            update={
                "objectives": list(goal.objectives) + list(decision.add_objectives),
                "constraints": goal.constraints,
            }
        )
        if decision.updated_goal_notes:
            updated_goal = updated_goal.model_copy(
                update={
                    "goal": f"{goal.goal} | note: {decision.updated_goal_notes}",
                    "simple": False,
                }
            )
        if decision.preferred_skills:
            updated_goal = updated_goal.model_copy(
                update={
                    "objectives": updated_goal.objectives
                    + [f"prefer_skills:{','.join(decision.preferred_skills)}"],
                    "simple": False,
                }
            )

        if plan:
            plan.status = "replanning"
            if graph and decision.drop_task_ids:
                graph.tasks = [
                    t for t in graph.tasks if t.id not in set(decision.drop_task_ids)
                ]
                out["task_graph"] = graph.to_state_dict()
            # Preserve memory: keep completed task outcomes in execution_progress
            progress = dict(state.get("execution_progress") or {})
            timeline = list(progress.get("timeline") or [])
            timeline.append(
                {
                    "event": "replan_cycle",
                    "replan_count": new_count,
                    "strategy": decision.strategy,
                    "prior_context": {
                        k: prior_context[k]
                        for k in ("completed_tasks", "failed_tasks", "sql_error")
                        if k in prior_context
                    },
                }
            )
            progress["timeline"] = timeline
            out["execution_progress"] = progress
            out["adaptive_plan"] = plan.to_state_dict()

        out["goal_spec"] = updated_goal.to_state_dict()
        out["plan_active"] = True
        out["next_agent"] = "planner"  # full replan cycle — Planner re-entry
        out["fix_hint"] = decision.reasoning
        # Preserve hybrid memory fields already on state (no wipe)
        return out

    return replan_agent


# ---------------------------------------------------------------------------
# 8. Long-Term Memory Agent (read/write workflow memory)
# ---------------------------------------------------------------------------

def make_memory_agent(
    workflow_store: WorkflowMemoryStore | None,
    *,
    memory_fabric: Any | None = None,
    experience_store: Any | None = None,
):
    """Injects hybrid long-term memory into state for the Planner.

    When ``memory_fabric`` is provided, semantic vector retrieval is merged with
    workflow + episodic stores. Learning hints are appended when available.
    """

    def memory_agent(state: GraphState) -> dict[str, Any]:
        question = state.get("question") or ""
        database_name = state.get("database_name") or ""
        tenant_id = state.get("tenant_id") or "default"
        sections: list[str] = []
        vector_ctx = ""

        if memory_fabric is not None:
            try:
                from observability.otel_setup import start_span

                with start_span(
                    "sqlmind.memory.retrieve",
                    attributes={
                        "sqlmind.tenant_id": tenant_id,
                        "sqlmind.database": database_name,
                    },
                ):
                    hybrid = memory_fabric.retrieve_for_planner(
                        question, tenant_id=tenant_id, database_name=database_name
                    )
                if hybrid:
                    sections.append(hybrid)
                    vector_ctx = hybrid
            except Exception as exc:  # noqa: BLE001
                logger.warning("MemoryFabric retrieve failed: %s", exc)

        if not sections and workflow_store is not None:
            try:
                ctx = workflow_store.format_for_prompt(
                    question, database_name=database_name
                )
                if ctx:
                    sections.append(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Workflow memory retrieve failed: %s", exc)

        learning_ctx = ""
        if experience_store is not None:
            try:
                learning_ctx = experience_store.format_for_planner(question) or ""
                if learning_ctx:
                    sections.append(learning_ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Learning retrieve failed: %s", exc)

        # Persist preferences when available
        chart = state.get("chart_type") or ""
        if (
            workflow_store is not None
            and chart
            and chart != "none"
            and state.get("query_success")
        ):
            try:
                workflow_store.record_visualization_pref(
                    chart,
                    question=question,
                    session_id=state.get("session_id") or "",
                    database_name=database_name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Viz pref persist failed: %s", exc)

        combined = "\n\n".join(sections) if sections else "(no long-term memory)"
        try:
            from observability.runtime_trace import safe_trace

            safe_trace(
                "memory_event",
                kind="Hybrid Search" if vector_ctx else "Memory Retrieval",
                memories=combined[:500],
                detail={
                    "has_vector": bool(vector_ctx),
                    "has_learning": bool(learning_ctx),
                    "sections": len(sections),
                },
            )
            if vector_ctx:
                safe_trace(
                    "memory_event",
                    kind="Semantic Search",
                    memories=vector_ctx[:300],
                )
                safe_trace(
                    "memory_event",
                    kind="Vector Search",
                    memories=vector_ctx[:300],
                )
            if learning_ctx:
                safe_trace(
                    "memory_event",
                    kind="Planner Context",
                    memories=learning_ctx[:300],
                )
        except Exception:  # noqa: BLE001
            pass
        return {
            "workflow_memory_context": combined,
            "vector_memory_context": vector_ctx,
            "learning_context": learning_ctx,
            "next_agent": "goal_understanding",
            "agent_logs": [
                _log(
                    "memory_agent",
                    "Loaded hybrid memory"
                    + (" + learning" if learning_ctx else ""),
                )
            ],
            "route_history": ["memory_agent"],
        }

    return memory_agent


def persist_workflow_outcome(
    store: WorkflowMemoryStore | None,
    state: dict[str, Any],
) -> None:
    """Called by orchestrator after a run completes."""
    if store is None:
        return
    try:
        plan = state.get("adaptive_plan")
        success = bool(state.get("query_success") or state.get("final_response")) and not (
            state.get("status") == "failed"
        )
        store.record_plan_outcome(
            question=state.get("question") or "",
            goal_summary=(state.get("goal_spec") or {}).get("goal", "")
            if isinstance(state.get("goal_spec"), dict)
            else "",
            plan=plan if isinstance(plan, dict) else None,
            success=success,
            session_id=state.get("session_id") or "",
            database_name=state.get("database_name") or "",
        )
        chart = state.get("chart_type") or ""
        if success and chart and chart != "none":
            store.record_visualization_pref(
                chart,
                question=state.get("question") or "",
                session_id=state.get("session_id") or "",
                database_name=state.get("database_name") or "",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workflow memory persist failed: %s", exc)
