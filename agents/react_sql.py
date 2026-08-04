"""SQL ReAct specialist — LLM selects tools via bind_tools + ToolNode.

Security: ``validate_sql_tool`` and ``query_tool`` remain deterministic sqlglot-gated.
The LLM chooses *when* to call them; it cannot bypass validation inside those tools.
Submitted SQL still flows through the outer ``validation_node`` → ``execution_node``
edges for defense-in-depth.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from graph.state import GraphState
from prompts.templates import SQL_REACT_SYSTEM
from services.llm_service import LLMService
from utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_REACT_STEPS = 8


def _execute_tool_calls(
    response: AIMessage, tools: list[StructuredTool]
) -> list[ToolMessage]:
    """Execute model-emitted tool_calls (ToolNode-equivalent without graph config)."""
    by_name = {t.name: t for t in tools}
    out: list[ToolMessage] = []
    for tc in response.tool_calls or []:
        name = tc.get("name") or ""
        tool = by_name.get(name)
        call_id = tc.get("id") or name
        if tool is None:
            out.append(
                ToolMessage(
                    content=json.dumps({"error": f"Unknown tool: {name}"}),
                    tool_call_id=call_id,
                    name=name or "unknown",
                )
            )
            continue
        try:
            result = tool.invoke(tc.get("args") or {})
            content = result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as exc:  # noqa: BLE001
            content = json.dumps({"error": str(exc)})
        out.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
    return out


class SubmitSQLArgs(BaseModel):
    sql: str = Field(description="Single read-only SQL statement, no markdown")
    explanation: str = Field(description="What the query computes in plain English")


def build_submit_sql_tool() -> StructuredTool:
    def submit_sql(sql: str, explanation: str = "") -> str:
        return json.dumps(
            {"submitted": True, "sql": sql.strip(), "explanation": explanation},
            default=str,
        )

    return StructuredTool.from_function(
        func=submit_sql,
        name="submit_sql",
        description=(
            "Submit the final read-only SQL for the outer validation/execution pipeline. "
            "Call this when you have a correct statement. Do not invent tables/columns."
        ),
        args_schema=SubmitSQLArgs,
    )


def _extract_submitted_sql(messages: list[BaseMessage]) -> tuple[str, str] | None:
    """Find the latest successful submit_sql tool result."""
    for msg in reversed(messages or []):
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", None) not in {None, "submit_sql"} and msg.name != "submit_sql":
            # ToolMessage.name may be set to tool name
            pass
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("submitted") and payload.get("sql"):
            return str(payload["sql"]), str(payload.get("explanation") or "")
    # Also check tool_call args if the model called submit_sql but tool not yet run
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "submit_sql":
                    args = tc.get("args") or {}
                    sql = (args.get("sql") or "").strip()
                    if sql:
                        return sql, str(args.get("explanation") or "")
    return None


def _tool_name(msg: ToolMessage) -> str:
    return getattr(msg, "name", None) or ""


def make_sql_react_agent(llm: LLMService, tools: list[StructuredTool]):
    """Build a node function that runs an inner ReAct loop (bind_tools + ToolNode)."""
    submit = build_submit_sql_tool()
    react_tools = list(tools) + [submit]
    bound = llm.bind_tools(react_tools)

    def _build_messages(state: GraphState) -> list[BaseMessage]:
        existing = list(state.get("react_messages") or [])
        if existing:
            return existing

        from services.column_semantics import (
            build_bi_enrichment_checklist,
            ensure_schema_text_with_hints,
        )

        schema_block = ensure_schema_text_with_hints(
            state.get("schema_text") or "",
            state.get("schema_dict"),
        )
        bi_checklist = build_bi_enrichment_checklist(state.get("schema_dict"))

        parts = [
            f"Dialect: {state.get('dialect', 'sqlite')}",
            f"Question: {state.get('rewritten_question') or state.get('question', '')}",
            f"Memory summary: {state.get('memory_summary') or '(none)'}",
            f"Episodic hints: {state.get('episodic_context') or '(none)'}",
            f"Schema (may refresh via schema_tool):\n{schema_block}",
            bi_checklist,
            f"Prior SQL: {state.get('sql') or '(none)'}",
            f"Fix hint: {state.get('fix_hint') or '(none)'}",
            f"Prior error: {state.get('sql_error') or '; '.join(state.get('validation_errors') or []) or '(none)'}",
            f"Retry diagnosis: {state.get('retry_diagnosis') or '(none)'}",
        ]
        return [
            SystemMessage(content=SQL_REACT_SYSTEM),
            HumanMessage(content="\n".join(parts)),
        ]

    def sql_react_agent(state: GraphState) -> dict[str, Any]:
        messages = _build_messages(state)
        logs: list[dict[str, Any]] = []
        steps = 0
        submitted: tuple[str, str] | None = None

        while steps < MAX_REACT_STEPS:
            steps += 1
            try:
                response: AIMessage = bound.invoke(messages)  # type: ignore[assignment]
            except Exception as exc:  # noqa: BLE001
                logger.exception("SQL ReAct LLM invoke failed")
                return {
                    "sql": state.get("sql") or "",
                    "sql_valid": False,
                    "sql_error": str(exc),
                    "react_messages": messages,
                    "agent_logs": [
                        {"agent": "sql_agent", "message": str(exc), "status": "error"}
                    ],
                }

            messages = messages + [response]
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                submitted = _extract_submitted_sql(messages)
                if submitted:
                    break
                logs.append(
                    {
                        "agent": "sql_agent",
                        "message": "ReAct ended without submit_sql",
                        "status": "error",
                    }
                )
                break

            # Execute LLM-selected tools (bind_tools → tool_calls → invoke)
            try:
                tool_msgs = _execute_tool_calls(response, react_tools)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool execution failed: %s", exc)
                return {
                    "sql": "",
                    "sql_error": str(exc),
                    "react_messages": messages,
                    "agent_logs": [
                        {"agent": "sql_agent", "message": f"Tool error: {exc}", "status": "error"}
                    ],
                }

            messages = messages + list(tool_msgs)
            for tm in tool_msgs:
                name = _tool_name(tm) if isinstance(tm, ToolMessage) else "tool"
                logs.append(
                    {
                        "agent": "sql_agent",
                        "message": f"tool_call:{name}",
                        "status": "ok",
                        "detail": (tm.content if isinstance(tm, ToolMessage) else "")[:300],
                    }
                )

            submitted = _extract_submitted_sql(messages)
            if submitted:
                break

        if not submitted:
            # Last resort: structured SQL generation still LLM-driven (no DEFAULT plan)
            from models.structured import SQLResponseModel
            from prompts.templates import sql_prompt

            try:
                from services.column_semantics import ensure_schema_text_with_hints

                result: SQLResponseModel = llm.invoke_structured(
                    sql_prompt(),
                    SQLResponseModel,
                    {
                        "history": llm.history_messages(state.get("conversation_history")),
                        "dialect": state.get("dialect", "sqlite"),
                        "schema_text": ensure_schema_text_with_hints(
                            state.get("schema_text") or "",
                            state.get("schema_dict"),
                        ),
                        "question": state.get("rewritten_question") or state.get("question") or "",
                        "memory_summary": state.get("memory_summary") or "(none)",
                        "prior_sql": state.get("sql") or "(none)",
                        "fix_hint": (state.get("fix_hint") or "")
                        + " | "
                        + (state.get("retry_diagnosis") or ""),
                        "prior_error": state.get("sql_error")
                        or "; ".join(state.get("validation_errors") or [])
                        or "(none)",
                    },
                )
                submitted = (result.sql.strip(), result.explanation)
                logs.append(
                    {
                        "agent": "sql_agent",
                        "message": "Fallback structured SQL after ReAct (still LLM)",
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "sql": "",
                    "sql_valid": False,
                    "sql_error": f"SQL ReAct failed to submit SQL: {exc}",
                    "react_messages": messages,
                    "agent_logs": logs
                    + [{"agent": "sql_agent", "message": str(exc), "status": "error"}],
                }

        sql, explanation = submitted
        return {
            "sql": sql,
            "sql_explanation": explanation,
            "sql_error": "",
            "react_messages": messages,
            "agent_logs": logs
            + [
                {
                    "agent": "sql_agent",
                    "message": f"Submitted SQL after {steps} ReAct step(s)",
                    "status": "ok",
                    "detail": sql[:500],
                }
            ],
        }

    return sql_react_agent


def route_react_continue(state: GraphState) -> Literal["tools", "done"]:
    """Helper for optional compiled subgraph (used in tests)."""
    msgs = state.get("react_messages") or []
    if not msgs:
        return "done"
    last = msgs[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "done"
