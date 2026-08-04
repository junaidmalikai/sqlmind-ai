"""Tests proving bind_tools / ToolNode SQL ReAct path is LLM-driven."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool

from agents.react_sql import _extract_submitted_sql, make_sql_react_agent
from graph.state import empty_state
from models.structured import SQLResponseModel


class ReActFakeLLM:
    def __init__(self, script: list[AIMessage]) -> None:
        self.script = list(script)
        self.bind_count = 0

    def history_messages(self, history, limit: int = 12):  # noqa: ANN001
        return []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        self.bind_count += 1
        script = self.script

        def _invoke(messages):  # noqa: ANN001
            if not script:
                return AIMessage(content="done")
            return script.pop(0)

        return RunnableLambda(_invoke)

    def invoke_structured(self, prompt, schema, variables):  # noqa: ANN001
        return SQLResponseModel(
            sql="SELECT 1 AS n",
            explanation="fallback",
            confidence=0.5,
        )


def test_extract_submitted_sql_from_tool_message() -> None:
    msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "1",
                    "name": "submit_sql",
                    "args": {"sql": "SELECT 1", "explanation": "x"},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"submitted": True, "sql": "SELECT 1", "explanation": "x"}),
            tool_call_id="1",
            name="submit_sql",
        ),
    ]
    got = _extract_submitted_sql(msgs)
    assert got is not None
    assert got[0] == "SELECT 1"


def test_sql_react_calls_bind_tools_and_submit() -> None:
    schema_tool = StructuredTool.from_function(
        func=lambda: "TABLE t(a INT)",
        name="schema_tool",
        description="schema",
    )
    validate_tool = StructuredTool.from_function(
        func=lambda sql: json.dumps({"is_safe": True, "sql": sql, "errors": []}),
        name="validate_sql_tool",
        description="validate",
    )

    llm = ReActFakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "schema_tool",
                        "args": {},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "submit_sql",
                        "args": {
                            "sql": "SELECT a FROM t LIMIT 5",
                            "explanation": "list a",
                        },
                    }
                ],
            ),
        ]
    )
    agent = make_sql_react_agent(llm, [schema_tool, validate_tool])  # type: ignore[arg-type]
    state = empty_state("list a")
    state["dialect"] = "sqlite"
    out = agent(state)
    assert llm.bind_count >= 1
    assert "SELECT a FROM t" in out["sql"]
    assert any("tool_call:schema_tool" in (log.get("message") or "") for log in out["agent_logs"])
    assert any("tool_call:submit_sql" in (log.get("message") or "") for log in out["agent_logs"])
