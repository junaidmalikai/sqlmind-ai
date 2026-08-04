"""LangGraph agent node implementations (bind_tools + structured output).

Security-critical validation/execution remain deterministic inside tools/guards.
Export and finalize are deterministic utilities (not AI agents).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.react_sql import make_sql_react_agent
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaSnapshot
from graph.state import GraphState
from models.structured import (
    ChartRecommendationModel,
    DashboardModel,
    DatabaseSummaryModel,
    InsightModel,
    OptimizationAdviceModel,
    ReflectionModel,
    RetryDecisionModel,
    SuggestedQuestionsModel,
)
from observability.otel_setup import start_span
from prompts.templates import (
    SUPERVISOR_SYSTEM,
    chart_prompt,
    dashboard_prompt,
    export_narrative_prompt,
    insight_prompt,
    optimize_prompt,
    reflection_prompt,
    retry_prompt,
    suggest_prompt,
    summary_prompt,
)
from services.llm_service import LLMService
from services.visualization import dataframe_profile
from kernel.exceptions import CapabilityNotFoundError, CapabilityNotRoutableError
from kernel.registry import CapabilityRegistry
from kernel.routing import resolve_next_node
from tools.routing_tools import ROUTE_TOOL_TO_NODE, build_routing_tools
from utils.logging_config import get_logger
from utils.security import SQLSecurityGuard

logger = get_logger(__name__)


def _log(agent: str, message: str, status: str = "ok", detail: Any = None) -> dict:
    entry: dict[str, Any] = {"agent": agent, "message": message, "status": status}
    if detail is not None:
        entry["detail"] = detail
    return entry


def _question(state: GraphState) -> str:
    return state.get("rewritten_question") or state.get("question") or ""


# ---------------------------------------------------------------------------
# Supervisor — bind_tools routing (LLM tool-call selects next node)
# ---------------------------------------------------------------------------

def make_supervisor_agent(
    llm: LLMService,
    registry: CapabilityRegistry | None = None,
):
    """Supervisor selects next capability via bind_tools against the registry catalog."""
    routing_tools = build_routing_tools(registry)
    bound = llm.bind_tools(routing_tools)

    def supervisor_agent(state: GraphState) -> dict[str, Any]:
        visits = int(state.get("supervisor_visit") or 0) + 1
        max_visits = int(state.get("max_supervisor_visits") or 12)
        if visits > max_visits:
            return {
                "supervisor_visit": visits,
                "next_agent": "finalize",
                "supervisor_reasoning": "Supervisor visit budget exhausted — finalizing",
                "agent_logs": [
                    _log("supervisor", "Max supervisor visits reached → finalize", "error")
                ],
                "route_history": ["finalize"],
            }

        history = llm.history_messages(state.get("conversation_history"))
        catalog_hint = ""
        if registry is not None:
            from kernel.routing import describe_routable_catalog

            catalog_hint = (
                f"Capability catalog v{registry.version}:\n"
                f"{describe_routable_catalog(registry)}\n"
            )
        goal = state.get("goal_spec") or {}
        plan = state.get("adaptive_plan") or {}
        plan_hint = ""
        if isinstance(goal, dict) and goal.get("goal"):
            plan_hint += f"Goal: {goal.get('goal')}\n"
            plan_hint += f"Expected output: {goal.get('expected_output') or '(n/a)'}\n"
        if isinstance(plan, dict) and plan.get("strategy"):
            plan_hint += (
                f"Plan strategy: {plan.get('strategy')}\n"
                f"Plan mode: {plan.get('execution_mode')}\n"
                f"Plan status: {plan.get('status')}\n"
            )
        human = HumanMessage(
            content=(
                f"Database: {state.get('database_name', '')} ({state.get('dialect', '')})\n"
                f"Memory summary: {state.get('memory_summary') or '(none)'}\n"
                f"Episodic memory:\n{state.get('episodic_context') or '(none)'}\n"
                f"Workflow memory:\n{state.get('workflow_memory_context') or '(none)'}\n"
                f"{plan_hint}"
                f"{catalog_hint}"
                f"Schema loaded: {bool(state.get('schema_text'))}\n"
                f"Has SQL: {bool(state.get('sql'))}\n"
                f"Query success: {bool(state.get('query_success'))}\n"
                f"Row count: {state.get('row_count', 0)}\n"
                f"Has insights: {bool(state.get('insights'))}\n"
                f"Has chart: {bool(state.get('chart_spec'))}\n"
                f"Reflection verdict: {state.get('reflection_verdict') or '(none)'}\n"
                f"Last error: {state.get('sql_error') or '(none)'}\n"
                f"Route history: {list(state.get('route_history') or [])}\n"
                f"User question: {state.get('question', '')}\n"
                "Choose the next step by calling exactly one routing tool."
            )
        )
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM), *history, human]

        try:
            response: AIMessage = bound.invoke(messages)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supervisor bind_tools invoke failed: %s", exc)
            # Real failure path — do NOT inject DEFAULT_PLANS
            return {
                "supervisor_visit": visits,
                "next_agent": "fail",
                "error": f"Supervisor routing failed: {exc}",
                "final_response": (
                    "I could not decide the next step (LLM routing error). "
                    f"Details: {exc}"
                ),
                "status": "failed",
                "agent_logs": [_log("supervisor", str(exc), "error")],
                "route_history": ["fail"],
            }

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # One corrective re-ask
            retry_msgs = messages + [
                response,
                HumanMessage(
                    content="You must call exactly one routing tool. Do not reply with plain text."
                ),
            ]
            try:
                response = bound.invoke(retry_msgs)  # type: ignore[assignment]
                tool_calls = getattr(response, "tool_calls", None) or []
            except Exception as exc:  # noqa: BLE001
                return {
                    "supervisor_visit": visits,
                    "next_agent": "fail",
                    "error": str(exc),
                    "agent_logs": [_log("supervisor", f"No tool call: {exc}", "error")],
                    "route_history": ["fail"],
                }

        if not tool_calls:
            return {
                "supervisor_visit": visits,
                "next_agent": "fail",
                "error": "Supervisor produced no tool call",
                "final_response": (
                    "Routing failed: the model did not select a next step via tool call."
                ),
                "status": "failed",
                "agent_logs": [_log("supervisor", "No tool_calls in response", "error")],
                "route_history": ["fail"],
            }

        tc = tool_calls[0]
        tool_name = tc.get("name") or ""
        args = tc.get("args") or {}
        try:
            if registry is not None:
                next_agent = resolve_next_node(registry, tool_name)
            else:
                next_agent = ROUTE_TOOL_TO_NODE.get(tool_name)
                if not next_agent:
                    raise CapabilityNotFoundError(f"Unknown routing tool: {tool_name}")
        except (CapabilityNotFoundError, CapabilityNotRoutableError, ValueError) as exc:
            return {
                "supervisor_visit": visits,
                "next_agent": "fail",
                "error": f"Unknown routing tool: {tool_name} ({exc})",
                "agent_logs": [_log("supervisor", f"Unknown tool {tool_name}", "error")],
                "route_history": ["fail"],
            }

        reasoning = str(args.get("reasoning") or response.content or tool_name)
        out: dict[str, Any] = {
            "supervisor_visit": visits,
            "next_agent": next_agent,
            "supervisor_reasoning": reasoning,
            "status": "routing",
            "agent_logs": [
                _log(
                    "supervisor",
                    f"tool={tool_name} → {next_agent}",
                    detail=reasoning,
                )
            ],
            "route_history": [next_agent],
            "messages": [response],
        }
        if next_agent == "clarify":
            out["needs_clarification"] = True
            out["clarification_question"] = str(
                args.get("question")
                or "Could you clarify what you'd like to analyze?"
            )
        return out

    return supervisor_agent


# ---------------------------------------------------------------------------
# Schema agent — LLM bind_tools selects schema_tool
# ---------------------------------------------------------------------------

def make_schema_agent(schema: SchemaSnapshot, llm: LLMService, tool_map: dict):
    schema_tool = tool_map.get("schema_tool")
    tools = [schema_tool] if schema_tool is not None else []

    def schema_agent(state: GraphState) -> dict[str, Any]:
        schema_text = schema.to_prompt_text()
        tool_logs: list[dict] = []

        if tools:
            bound = llm.bind_tools(tools)
            messages = [
                SystemMessage(
                    content=(
                        "You refresh database schema context. Call schema_tool once, then stop."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Database: {schema.database_name} ({schema.dialect}). "
                        "Call schema_tool now."
                    )
                ),
            ]
            try:
                response: AIMessage = bound.invoke(messages)  # type: ignore[assignment]
                if response.tool_calls:
                    from agents.react_sql import _execute_tool_calls

                    for tm in _execute_tool_calls(response, tools):
                        content = getattr(tm, "content", "") or ""
                        if content:
                            schema_text = content if isinstance(content, str) else str(content)
                        tool_logs.append(
                            _log("schema_agent", "tool_call:schema_tool", detail=str(content)[:200])
                        )
                else:
                    tool_logs.append(
                        _log("schema_agent", "No tool call — using snapshot text", "error")
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Schema bind_tools failed: %s", exc)
                tool_logs.append(_log("schema_agent", f"tool path failed: {exc}", "error"))

        suggestions: list[str] = []
        try:
            sug: SuggestedQuestionsModel = llm.invoke_structured(
                suggest_prompt(),
                SuggestedQuestionsModel,
                {
                    "database_name": schema.database_name,
                    "dialect": schema.dialect,
                    "schema_text": schema_text,
                },
            )
            suggestions = sug.questions
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI suggestions failed: %s", exc)
            # Real failure — empty list, not a fake static "AI" list
            suggestions = []

        return {
            "schema_text": schema_text,
            "schema_dict": schema.to_dict(),
            "suggested_questions": suggestions,
            "database_name": schema.database_name,
            "dialect": schema.dialect,
            "next_agent": "supervisor",
            "agent_logs": tool_logs
            + [
                _log(
                    "schema_agent",
                    f"Schema ready + {len(suggestions)} AI suggestions",
                )
            ],
        }

    return schema_agent


# ---------------------------------------------------------------------------
# SQL agent — ReAct with bind_tools + ToolNode
# ---------------------------------------------------------------------------

def make_sql_agent(llm: LLMService, tool_map: dict | None = None):
    """SQL specialist: ReAct tool loop when tools provided; else structured fallback."""
    if tool_map:
        react_tools = [
            t
            for name, t in tool_map.items()
            if name
            in {"schema_tool", "validate_sql_tool", "explain_tool", "statistics_tool"}
        ]
        return make_sql_react_agent(llm, react_tools)

    # Test/helper path without tools — still LLM structured SQL (no DEFAULT plan)
    from models.structured import SQLResponseModel
    from prompts.templates import sql_prompt

    def sql_agent(state: GraphState) -> dict[str, Any]:
        from services.column_semantics import ensure_schema_text_with_hints

        history = llm.history_messages(state.get("conversation_history"))
        try:
            result: SQLResponseModel = llm.invoke_structured(
                sql_prompt(),
                SQLResponseModel,
                {
                    "history": history,
                    "dialect": state.get("dialect", "sqlite"),
                    "schema_text": ensure_schema_text_with_hints(
                        state.get("schema_text") or "",
                        state.get("schema_dict"),
                    ),
                    "question": _question(state),
                    "memory_summary": state.get("memory_summary") or "(none)",
                    "prior_sql": state.get("sql") or "(none)",
                    "fix_hint": (
                        (state.get("fix_hint") or "")
                        + " | "
                        + (state.get("revised_approach") or "")
                        + " | "
                        + (state.get("retry_diagnosis") or "")
                    ),
                    "prior_error": state.get("sql_error")
                    or "; ".join(state.get("validation_errors") or [])
                    or "(none)",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("SQL generation failed")
            return {
                "sql": "",
                "sql_valid": False,
                "sql_error": str(exc),
                "agent_logs": [_log("sql_agent", str(exc), "error")],
            }
        return {
            "sql": result.sql.strip(),
            "sql_explanation": result.explanation,
            "sql_error": "",
            "react_messages": [],
            "agent_logs": [
                _log(
                    "sql_agent",
                    f"Generated SQL (confidence={result.confidence})",
                    detail=result.sql[:500],
                )
            ],
        }

    return sql_agent


# ---------------------------------------------------------------------------
# Validation — DETERMINISTIC (sqlglot). Never LLM.
# ---------------------------------------------------------------------------

def make_validation_node(security: SQLSecurityGuard, approval_policy: Any | None = None):
    """Deterministic sqlglot security gate — not an AI agent.

    Also evaluates approval needs so high-risk SQL routes to approval_gate.
    """

    def validation_node(state: GraphState) -> dict[str, Any]:
        sql = state.get("sql", "")
        with start_span(
            "sqlmind.validation",
            attributes={"sqlmind.sql_chars": len(sql or "")},
        ) as span:
            result = security.validate(sql)
            if span is not None:
                try:
                    span.set_attribute("sqlmind.sql_valid", result.is_safe)
                    if result.errors:
                        span.set_attribute(
                            "sqlmind.validation_errors",
                            "; ".join(result.errors)[:500],
                        )
                except Exception:  # noqa: BLE001
                    pass
        out: dict[str, Any] = {
            "sql": result.sql,
            "sql_valid": result.is_safe,
            "validation_errors": result.errors,
            "validation_warnings": result.warnings,
            "validation_meta": {
                "tables": result.tables,
                "columns": result.columns,
                "joins": result.joins,
                "statement_type": result.statement_type,
            },
            "sql_error": "; ".join(result.errors) if result.errors else "",
            "agent_logs": [
                _log(
                    "validation_node",
                    "Valid" if result.is_safe else "Rejected",
                    "ok" if result.is_safe else "error",
                    detail=result.errors or result.warnings,
                )
            ],
        }
        # Wire approval evaluation into the live security path
        if result.is_safe:
            try:
                from governance.approval import maybe_require_approval

                approval_patch = maybe_require_approval(state, approval_policy)
                if approval_patch:
                    out.update(approval_patch)
                    out["approval_resume_agent"] = "execution_node"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Approval evaluation skipped: %s", exc)
        try:
            from observability.metrics import get_metrics

            get_metrics().observe_sql(
                "validate", status="ok" if result.is_safe else "rejected"
            )
        except Exception:  # noqa: BLE001
            pass
        return out

    return validation_node


# ---------------------------------------------------------------------------
# Retry — LLM diagnosis; next_action can regenerate, replan, or give up
# ---------------------------------------------------------------------------

def make_retry_agent(llm: LLMService):
    def retry_agent(state: GraphState) -> dict[str, Any]:
        retries = state.get("retry_count") or 0
        max_retries = state.get("max_retries") or 3
        if retries >= max_retries:
            # Exhausted in-graph retries → enterprise retry queue + recovery/replan
            try:
                from reliability.enterprise_queue import get_enterprise_queue

                get_enterprise_queue().enqueue_retry(
                    topic="sql_retry_exhausted",
                    payload={
                        "session_id": state.get("session_id") or "",
                        "question": (state.get("question") or "")[:500],
                        "sql": (state.get("sql") or "")[:500],
                        "error": state.get("sql_error")
                        or "; ".join(state.get("validation_errors") or []),
                        "retry_count": retries,
                    },
                    error="sql_retry_exhausted",
                    session_id=str(state.get("session_id") or ""),
                    node="retry_agent",
                )
            except Exception:  # noqa: BLE001
                pass
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
            if replan_count < max_replans:
                return {
                    "should_retry": False,
                    "retry_next_action": "replan",
                    "replan_reason": "SQL retries exhausted — requesting replan",
                    "fix_hint": "",
                    "next_agent": "replan_agent",
                    "agent_logs": [
                        _log(
                            "retry_agent",
                            "Max SQL retries — escalating to replan + enterprise queue",
                            "warn",
                        )
                    ],
                }
            return {
                "should_retry": False,
                "retry_next_action": "give_up",
                "fix_hint": "",
                "next_agent": "fail",
                "agent_logs": [_log("retry_agent", "Max retries reached", "error")],
            }

        try:
            decision: RetryDecisionModel = llm.invoke_structured(
                retry_prompt(),
                RetryDecisionModel,
                {
                    "dialect": state.get("dialect", ""),
                    "schema_text": state.get("schema_text", ""),
                    "question": _question(state),
                    "sql": state.get("sql", ""),
                    "errors": state.get("sql_error")
                    or "; ".join(state.get("validation_errors") or []),
                    "retry_count": retries,
                    "max_retries": max_retries,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retry decision failed: %s", exc)
            # Real failure — do not silently retry the same thing
            return {
                "should_retry": False,
                "retry_next_action": "give_up",
                "retry_count": retries + 1,
                "final_response": f"Retry diagnosis failed: {exc}",
                "next_agent": "fail",
                "agent_logs": [_log("retry_agent", str(exc), "error")],
            }

        action = decision.next_action
        if not decision.should_retry:
            action = "give_up"

        # Python ceiling still applies
        can_retry = retries + 1 < max_retries
        if action == "regenerate_sql" and not can_retry:
            action = "give_up"

        next_agent = {
            "regenerate_sql": "sql_agent",
            "replan": "replan_agent",
            "give_up": "fail",
        }.get(action, "fail")
        if action == "replan":
            # Prefer replan under budget even without a prior plan_id
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
            if replan_count >= max_replans:
                next_agent = "supervisor"

        fix = decision.fix_hint
        if decision.revised_approach:
            fix = f"{fix}\nApproach change: {decision.revised_approach}".strip()

        out = {
            "should_retry": action in {"regenerate_sql", "replan"},
            "retry_next_action": action,
            "fix_hint": fix,
            "retry_diagnosis": decision.reasoning,
            "revised_approach": decision.revised_approach,
            "retry_count": retries + 1,
            "next_agent": next_agent,
            "final_response": (
                decision.give_up_message
                if action == "give_up"
                else state.get("final_response", "")
            ),
            "react_messages": [],  # reset ReAct buffer for fresh attempt
            "agent_logs": [
                _log(
                    "retry_agent",
                    f"action={action} class={decision.failure_class}",
                    detail=decision.reasoning,
                )
            ],
        }
        if action == "replan":
            out["replan_reason"] = decision.reasoning
        return out

    return retry_agent


# ---------------------------------------------------------------------------
# Execution — DETERMINISTIC utility (not an AI agent)
# ---------------------------------------------------------------------------

def make_execution_node(executor: QueryExecutor, tool_map: dict):
    """Deterministic query execution after validation — not an AI agent."""

    def execution_node(state: GraphState) -> dict[str, Any]:
        sql = state.get("sql", "")
        raw = tool_map["query_tool"].invoke({"sql": sql})
        payload = json.loads(raw) if isinstance(raw, str) else raw

        if not payload.get("success"):
            return {
                "query_success": False,
                "sql_error": payload.get("error") or "Execution failed",
                "dataframe_records": [],
                "columns": [],
                "row_count": 0,
                "execution_time": payload.get("execution_time") or 0.0,
                "agent_logs": [
                    _log("execution_node", payload.get("error") or "failed", "error")
                ],
            }

        full = executor.execute(sql)
        safe_records = json.loads(json.dumps(full.to_records(10_000), default=str))
        dtypes = {c: str(full.dataframe[c].dtype) for c in full.dataframe.columns}

        return {
            "query_success": True,
            "sql_error": "",
            "sql": full.sql or sql,
            "dataframe_records": safe_records,
            "columns": list(full.dataframe.columns),
            "dtypes": dtypes,
            "row_count": full.row_count,
            "execution_time": full.execution_time,
            "truncated": full.truncated,
            "validation_warnings": full.warnings,
            "agent_logs": [
                _log(
                    "execution_node",
                    f"Returned {full.row_count} rows in {full.execution_time:.3f}s",
                )
            ],
        }

    return execution_node


# ---------------------------------------------------------------------------
# Visualization / Insight / Summary / Dashboard
# ---------------------------------------------------------------------------

def make_visualization_agent(llm: LLMService):
    def visualization_agent(state: GraphState) -> dict[str, Any]:
        records = state.get("dataframe_records") or []
        columns = state.get("columns") or []
        df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
        profile = dataframe_profile(df)

        try:
            rec: ChartRecommendationModel = llm.invoke_structured(
                chart_prompt(),
                ChartRecommendationModel,
                {
                    "question": _question(state),
                    "columns": profile["columns"],
                    "dtypes": profile["dtypes"],
                    "statistics": json.dumps(profile["statistics"], default=str)[:3000],
                    "sample_rows": json.dumps(profile["sample"], default=str)[:4000],
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chart AI failed: %s", exc)
            rec = ChartRecommendationModel(
                chart_type="table" if not df.empty else "none",
                title="Query Results",
                rationale=str(exc),
            )

        return {
            "chart_type": rec.chart_type,
            "chart_spec": rec.model_dump(),
            "next_agent": "join_post_query" if state.get("parallel_job") else "supervisor",
            "agent_logs": [
                _log("visualization_agent", f"Recommended {rec.chart_type}", detail=rec.rationale)
            ],
        }

    return visualization_agent


def make_insight_agent(llm: LLMService):
    def insight_agent(state: GraphState) -> dict[str, Any]:
        history = llm.history_messages(state.get("conversation_history"), limit=6)
        records = (state.get("dataframe_records") or [])[:30]
        variables = {
            "history": history,
            "question": _question(state),
            "sql": state.get("sql", ""),
            "row_count": state.get("row_count", 0),
            "sample_rows": json.dumps(records, default=str)[:4000],
        }

        # Prefer structured; also support token streaming into state for UI
        stream_buf: list[str] = []
        try:
            result: InsightModel = llm.invoke_structured(
                insight_prompt(),
                InsightModel,
                variables,
            )
        except Exception as exc:  # noqa: BLE001
            # Fall back to real token streaming free-text if structured fails
            try:
                for chunk in llm.stream_text(insight_prompt(), variables):
                    stream_buf.append(chunk)
                text = "".join(stream_buf)
                result = InsightModel(summary=text or f"Unable to generate insights: {exc}", bullets=[])
            except Exception as exc2:  # noqa: BLE001
                result = InsightModel(summary=f"Unable to generate insights: {exc2}", bullets=[])

        bullets = "\n".join(f"- {b}" for b in result.bullets)
        trends = "\n".join(f"- {t}" for t in result.trends)
        parts = [result.summary]
        if bullets:
            parts.append(f"\n**Key points**\n{bullets}")
        if trends:
            parts.append(f"\n**Trends**\n{trends}")
        if result.recommendations:
            parts.append(
                "\n**Recommendations**\n"
                + "\n".join(f"- {r}" for r in result.recommendations)
            )
        insights = "\n".join(parts)

        return {
            "insights": insights,
            "insight_structured": result.model_dump(),
            "stream_tokens": insights if stream_buf else "",
            "next_agent": "join_post_query" if state.get("parallel_job") else "supervisor",
            "agent_logs": [_log("insight_agent", "Generated structured insights")],
        }

    return insight_agent


def make_summary_agent(llm: LLMService):
    def summary_agent(state: GraphState) -> dict[str, Any]:
        try:
            result: DatabaseSummaryModel = llm.invoke_structured(
                summary_prompt(),
                DatabaseSummaryModel,
                {
                    "database_name": state.get("database_name", ""),
                    "dialect": state.get("dialect", ""),
                    "schema_text": state.get("schema_text", ""),
                },
            )
        except Exception as exc:  # noqa: BLE001
            result = DatabaseSummaryModel(
                business_overview=str(exc),
                database_purpose="Unknown",
            )

        return {
            "database_summary": result.model_dump(),
            "suggested_questions": result.suggested_questions
            or state.get("suggested_questions")
            or [],
            "next_agent": "supervisor",
            "agent_logs": [_log("summary_agent", "Generated database summary")],
        }

    return summary_agent


def make_dashboard_agent(llm: LLMService):
    def dashboard_agent(state: GraphState) -> dict[str, Any]:
        try:
            result: DashboardModel = llm.invoke_structured(
                dashboard_prompt(),
                DashboardModel,
                {
                    "database_name": state.get("database_name", ""),
                    "dialect": state.get("dialect", ""),
                    "schema_text": state.get("schema_text", ""),
                },
            )
        except Exception as exc:  # noqa: BLE001
            result = DashboardModel(title="Dashboard", narrative=str(exc))

        return {
            "dashboard_spec": result.model_dump(),
            "next_agent": "supervisor",
            "agent_logs": [_log("dashboard_agent", "Designed KPI dashboard")],
        }

    return dashboard_agent


def make_optimization_agent(llm: LLMService, tool_map: dict):
    def optimization_agent(state: GraphState) -> dict[str, Any]:
        sql = state.get("sql") or ""
        explain_text = state.get("explain_plan") or ""
        explain_tool = tool_map.get("explain_tool")
        tool_logs: list[dict] = []

        if sql and explain_tool is not None:
            bound = llm.bind_tools([explain_tool])
            try:
                response: AIMessage = bound.invoke(  # type: ignore[assignment]
                    [
                        SystemMessage(
                            content=(
                                "Call explain_tool with the SQL to obtain an EXPLAIN plan, then stop."
                            )
                        ),
                        HumanMessage(content=f"SQL:\n{sql}"),
                    ]
                )
                if response.tool_calls:
                    from agents.react_sql import _execute_tool_calls

                    for tm in _execute_tool_calls(response, [explain_tool]):
                        explain_text = str(getattr(tm, "content", "") or explain_text)
                        tool_logs.append(_log("optimization_agent", "tool_call:explain_tool"))
            except Exception as exc:  # noqa: BLE001
                explain_text = explain_text or f"EXPLAIN unavailable: {exc}"
                tool_logs.append(_log("optimization_agent", str(exc), "error"))

        try:
            advice: OptimizationAdviceModel = llm.invoke_structured(
                optimize_prompt(),
                OptimizationAdviceModel,
                {
                    "dialect": state.get("dialect", ""),
                    "schema_text": state.get("schema_text", ""),
                    "sql": sql or "(no sql — advise generally from schema)",
                    "explain_plan": explain_text or "(not available)",
                },
            )
        except Exception as exc:  # noqa: BLE001
            advice = OptimizationAdviceModel(summary=str(exc))

        tips = (
            [advice.summary]
            + advice.index_suggestions
            + advice.rewrite_suggestions
            + advice.risk_notes
        )
        return {
            "explain_plan": explain_text,
            "optimization_tips": tips,
            "optimization_structured": advice.model_dump(),
            "next_agent": "supervisor",
            "agent_logs": tool_logs
            + [_log("optimization_agent", "Produced optimization advice")],
        }

    return optimization_agent


# ---------------------------------------------------------------------------
# Reflection — second LLM pass that can change the next path
# ---------------------------------------------------------------------------

def make_reflection_agent(llm: LLMService):
    def reflection_agent(state: GraphState) -> dict[str, Any]:
        count = int(state.get("reflection_count") or 0)
        max_r = int(state.get("max_reflections") or 2)
        if count >= max_r:
            return {
                "reflection_verdict": "accept",
                "reflection_notes": "Reflection budget exhausted — accepting",
                "next_agent": "finalize",
                "agent_logs": [_log("reflection_agent", "Budget exhausted → accept")],
            }

        from services.column_semantics import (
            assess_result_bi_quality,
            build_hints_from_schema_dict,
        )

        records = (state.get("dataframe_records") or [])[:15]
        columns = list(state.get("columns") or [])
        bi = assess_result_bi_quality(
            columns,
            state.get("rewritten_question") or state.get("question") or "",
            schema_dict=state.get("schema_dict"),
            row_count=int(state.get("row_count") or 0),
        )
        if bi.thin:
            bi_quality_note = (
                f"THIN RESULT — {bi.reason}. Suggested fix: {bi.improvement_hint}"
            )
        elif bi.scalar_request:
            bi_quality_note = "User requested scalar/aggregate-only answer — do not force enrichment."
        else:
            bi_quality_note = (
                f"Column count={bi.column_count}; appears adequately contextual "
                f"(has_aggregate={bi.has_aggregate})."
            )

        schema_hints = (build_hints_from_schema_dict(state.get("schema_dict")) or "")[:3500]
        if not schema_hints:
            schema_hints = (state.get("schema_text") or "")[:2500] or "(none)"

        try:
            result: ReflectionModel = llm.invoke_structured(
                reflection_prompt(),
                ReflectionModel,
                {
                    "question": _question(state),
                    "sql": state.get("sql", ""),
                    "query_success": state.get("query_success"),
                    "row_count": state.get("row_count", 0),
                    "columns": ", ".join(columns) if columns else "(none)",
                    "bi_quality_note": bi_quality_note,
                    "chart_type": state.get("chart_type", "none"),
                    "insights": (state.get("insights") or "")[:3000],
                    "sample_rows": json.dumps(records, default=str)[:3000],
                    "schema_hints": schema_hints,
                    "reflection_count": count,
                    "max_reflections": max_r,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "reflection_verdict": "accept",
                "reflection_notes": str(exc),
                "reflection_count": count + 1,
                "next_agent": "finalize",
                "agent_logs": [_log("reflection_agent", f"Failed: {exc}", "error")],
            }

        verdict = result.verdict
        improvement_hint = result.improvement_hint or ""

        # Deterministic BI gate: thin Name+COUNT style → force one SQL retry
        # when budget remains and the model accepted or gave a weak path.
        if (
            bi.thin
            and bi.improvement_hint
            and count + 1 < max_r
            and state.get("query_success")
            and verdict in {"accept", "improve_insights"}
        ):
            verdict = "retry_sql"
            improvement_hint = bi.improvement_hint
            result.reasoning = (
                f"{result.reasoning} | BI quality gate: {bi.reason}"
            ).strip(" |")

        # True replan when Phase 2 plan exists; else Supervisor soft recovery
        replan_target = (
            "replan_agent"
            if (state.get("adaptive_plan") or {}).get("plan_id")
            or state.get("plan_active")
            else "supervisor"
        )
        next_map = {
            "accept": "finalize",
            "retry_sql": "sql_agent",
            "improve_insights": "insight_agent",
            "replan": replan_target,
            "clarify": "clarify",
        }
        next_agent = next_map.get(verdict, "finalize")
        out: dict[str, Any] = {
            "reflection_verdict": verdict,
            "reflection_notes": result.reasoning,
            "reflection_count": count + 1,
            "next_agent": next_agent,
            "fix_hint": improvement_hint or state.get("fix_hint", ""),
            "agent_logs": [
                _log("reflection_agent", f"verdict={verdict}", detail=result.reasoning)
            ],
        }
        if verdict == "replan":
            out["replan_reason"] = result.reasoning
            out["plan_active"] = True
        if verdict == "clarify":
            out["needs_clarification"] = True
            out["clarification_question"] = (
                result.clarification_question
                or "Could you clarify the analysis goal?"
            )
        if verdict == "retry_sql":
            out["react_messages"] = []
            out["retry_diagnosis"] = result.reasoning
        return out

    return reflection_agent


# ---------------------------------------------------------------------------
# Export / Finalize — DETERMINISTIC utilities (not AI agents)
# ---------------------------------------------------------------------------

def make_export_node(export_dir: str, llm: LLMService | None = None, tool_map: dict | None = None):
    """Export node — optional LLM narrative, then ExportManager / async ExportQueue."""

    def export_node(state: GraphState) -> dict[str, Any]:
        records = state.get("dataframe_records") or []
        columns = state.get("columns") or []
        df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
        if df.empty:
            return {
                "export_paths": {},
                "next_agent": "supervisor",
                "agent_logs": [_log("export_node", "No data to export", "skip")],
            }

        insights = state.get("insights", "")
        recommendations = ""
        structured = state.get("insight_structured") or {}
        if isinstance(structured, dict) and structured.get("recommendations"):
            recommendations = "\n".join(f"- {r}" for r in structured["recommendations"][:8])
        if llm is not None and insights:
            try:
                narrative = llm.invoke_text(
                    export_narrative_prompt(),
                    {
                        "question": state.get("question", ""),
                        "sql": state.get("sql", ""),
                        "insights": insights[:3000],
                        "row_count": state.get("row_count", 0),
                    },
                )
                if narrative:
                    insights = f"{insights}\n\nReport narrative:\n{narrative}"
            except Exception as exc:  # noqa: BLE001
                logger.info("Export narrative skipped: %s", exc)

        meta = {
            "rows": state.get("row_count", len(df)),
            "time_s": state.get("execution_time"),
            "database": state.get("database_name", ""),
        }

        use_async = True
        try:
            from config.settings import get_settings

            use_async = bool(getattr(get_settings(), "export_async_enabled", True))
        except Exception:  # noqa: BLE001
            pass

        if use_async:
            try:
                from services.export_queue import get_export_queue

                job = get_export_queue(export_dir).enqueue(
                    df,
                    formats=["csv", "excel", "json", "markdown", "html", "pdf"],
                    label=state.get("question", "query") or "query",
                    question=state.get("question", ""),
                    sql=state.get("sql", ""),
                    insights=insights,
                    recommendations=recommendations,
                    meta=meta,
                    session_id=str(state.get("session_id") or ""),
                )
                # Non-blocking: return download metadata immediately; poll via Runtime Ops
                # Optional short wait only when dataset is tiny (instant completes)
                current = get_export_queue(export_dir).get(job.job_id) or job
                if current.status not in {"completed", "failed"} and len(df) <= 200:
                    import time as _time

                    for _ in range(10):
                        current = get_export_queue(export_dir).get(job.job_id) or current
                        if current.status in {"completed", "failed"}:
                            break
                        _time.sleep(0.05)
                download_meta = {
                    "job_id": current.job_id,
                    "status": current.status,
                    "progress": current.progress,
                    "formats": list(current.formats),
                    "paths": dict(current.paths),
                    "rows_total": current.rows_total,
                    "download_ready": current.status == "completed" and bool(current.paths),
                    "poll_hint": "Use Runtime Ops or export_progress state to poll job status",
                }
                try:
                    from observability.metrics import get_metrics

                    get_metrics().observe_export("async_enqueue", status=current.status)
                except Exception:  # noqa: BLE001
                    pass
                # Offer distributed worker follow-up when enabled (does not block graph)
                try:
                    from config.settings import get_settings

                    if getattr(get_settings(), "distributed_execution_enabled", True):
                        # Handled by orchestrator-registered export_job if submitted elsewhere
                        pass
                except Exception:  # noqa: BLE001
                    pass
                return {
                    "export_paths": dict(current.paths),
                    "export_job_id": current.job_id,
                    "export_progress": current.to_dict(),
                    "export_download_meta": download_meta,
                    "next_agent": "supervisor",
                    "agent_logs": [
                        _log(
                            "export_node",
                            f"Async export {current.status} · job={current.job_id} · "
                            f"{len(current.paths)} formats · progress={current.progress:.0%}",
                        )
                    ],
                    "route_history": ["export_node"],
                }
            except Exception as exc:  # noqa: BLE001
                logger.info("Async export fallback to sync: %s", exc)

        from services.exporters import ExportManager

        paths = ExportManager(export_dir).run(
            df,
            state.get("question", "query"),
            question=state.get("question", ""),
            sql=state.get("sql", ""),
            insights=insights,
            recommendations=recommendations,
            meta=meta,
            include_pdf=True,
        )
        try:
            from pathlib import Path

            from services.export_queue import export_html

            paths["html"] = export_html(
                df,
                Path(export_dir),
                state.get("question", "query") or "query",
                question=state.get("question", ""),
                insights=insights,
                meta=meta,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("HTML export skipped: %s", exc)
        return {
            "export_paths": paths,
            "next_agent": "supervisor",
            "agent_logs": [
                _log(
                    "export_node",
                    f"Excel/PDF/CSV/JSON/MD/HTML exporters completed · {len(paths)} formats",
                )
            ],
        }

    return export_node


def finalize_node(state: GraphState) -> dict[str, Any]:
    """Deterministic response assembly utility — not an AI agent."""
    if state.get("needs_clarification") and not state.get("insights"):
        q = state.get("clarification_question") or (
            "Could you clarify what you'd like to analyze in the database?"
        )
        return {
            "final_response": q,
            "status": "clarify",
            "agent_logs": [_log("finalize", "Clarification requested")],
        }

    intent = state.get("intent", "query")
    parts: list[str] = []

    if state.get("database_summary"):
        s = state["database_summary"]
        parts.append(f"### {s.get('database_purpose', 'Database Summary')}")
        parts.append(s.get("business_overview", ""))
        if s.get("important_tables"):
            parts.append(
                "**Important tables:** " + ", ".join(s["important_tables"])
            )
        if s.get("potential_kpis"):
            parts.append(
                "**Potential KPIs:**\n"
                + "\n".join(f"- {k}" for k in s["potential_kpis"])
            )

    if state.get("dashboard_spec"):
        d = state["dashboard_spec"]
        parts.append(f"### {d.get('title', 'Dashboard')}")
        parts.append(d.get("narrative", ""))
        for kpi in d.get("kpis") or []:
            parts.append(
                f"- **{kpi.get('label')}** ({kpi.get('priority')}): {kpi.get('description')}"
            )
        if d.get("alerts"):
            parts.append("**Alerts**\n" + "\n".join(f"- {a}" for a in d["alerts"]))

    if state.get("insights"):
        parts.append(state["insights"])

    if state.get("sql") and intent in {"query", "optimize"}:
        parts.append(f"\n**SQL**\n```sql\n{state['sql']}\n```")
        if state.get("sql_explanation"):
            parts.append(f"_{state['sql_explanation']}_")

    if state.get("query_success"):
        parts.append(
            f"\n_Rows: {state.get('row_count')} · "
            f"Time: {state.get('execution_time', 0):.3f}s · "
            f"Chart: {state.get('chart_type', 'none')}_"
        )

    if state.get("optimization_tips"):
        parts.append(
            "\n**Optimization**\n"
            + "\n".join(f"- {t}" for t in state["optimization_tips"][:8])
        )

    if state.get("reflection_notes"):
        parts.append(f"\n_Reflection: {state.get('reflection_verdict')} — {state['reflection_notes']}_")

    if state.get("schema_text") and intent == "schema" and not parts:
        parts.append(f"### Database Schema\n\n```\n{state['schema_text'][:6000]}\n```")

    if not parts:
        parts.append(
            state.get("final_response")
            or state.get("sql_error")
            or state.get("error")
            or "Done."
        )

    return {
        "final_response": "\n\n".join(p for p in parts if p),
        "status": "complete",
        "agent_logs": [_log("finalize", "Assembled final response (deterministic)")],
    }


# Aliases removed — use finalize_node / fail_node directly


def fail_node(state: GraphState) -> dict[str, Any]:
    """Deterministic failure terminal — not an AI agent."""
    msg = (
        state.get("final_response")
        or (
            "Unable to complete the request after retries.\n\n"
            f"Last error: {state.get('sql_error') or state.get('validation_errors')}\n"
            f"Diagnosis: {state.get('retry_diagnosis') or '(none)'}"
        )
    )
    return {
        "final_response": msg,
        "status": "failed",
        "agent_logs": [_log("fail", "Failed", "error")],
    }


def make_clarify_node():
    """Human-in-the-loop clarification via LangGraph interrupt()."""

    def clarify_node(state: GraphState) -> dict[str, Any]:
        from langgraph.types import interrupt

        question = state.get("clarification_question") or (
            "Could you clarify what you'd like to analyze?"
        )
        # Pause the graph; resume value is the user's answer
        user_reply = interrupt({"clarification_question": question})
        reply = user_reply if isinstance(user_reply, str) else str(user_reply)
        return {
            "question": reply,
            "rewritten_question": reply,
            "needs_clarification": False,
            "clarification_question": "",
            "next_agent": "supervisor",
            "agent_logs": [_log("clarify", f"User replied: {reply[:120]}")],
            "route_history": ["supervisor"],
        }

    return clarify_node
