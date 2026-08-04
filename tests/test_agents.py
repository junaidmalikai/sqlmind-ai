"""Unit tests for agent nodes (mocked LLM — no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents.nodes import (
    fail_node,
    finalize_node,
    make_execution_node,
    make_retry_agent,
    make_schema_agent,
    make_sql_agent,
    make_validation_node,
    make_visualization_agent,
)
from graph.state import empty_state
from models.structured import (
    ChartRecommendationModel,
    RetryDecisionModel,
    SQLResponseModel,
    SuggestedQuestionsModel,
)
from utils.security import SQLSecurityGuard


class FakeLLM:
    """Minimal LLMService stand-in returning predetermined structured models / tool calls."""

    def __init__(
        self,
        responses: dict[type, Any] | None = None,
        *,
        tool_call_script: list[dict[str, Any]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[type] = []
        self.tool_call_script = list(tool_call_script or [])
        self.bind_calls = 0

    def history_messages(self, history, limit: int = 12):  # noqa: ANN001
        return []

    def invoke_structured(self, prompt, schema, variables):  # noqa: ANN001
        self.calls.append(schema)
        if schema in self.responses:
            resp = self.responses[schema]
            if callable(resp):
                return resp(variables)
            return resp
        raise AssertionError(f"No fake response for {schema}")

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        self.bind_calls += 1
        script = self.tool_call_script

        def _invoke(messages):  # noqa: ANN001
            if script:
                spec = script.pop(0)
                return AIMessage(
                    content=spec.get("content", ""),
                    tool_calls=[
                        {
                            "id": spec.get("id", "call_1"),
                            "name": spec["name"],
                            "args": spec.get("args") or {},
                        }
                    ],
                )
            # Default: call first tool if any
            name = tools[0].name if tools else "noop"
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": name, "args": {}}],
            )

        return RunnableLambda(_invoke)

    def stream_text(self, prompt, variables):  # noqa: ANN001
        yield "streamed "
        yield "token"


@pytest.fixture
def guard() -> SQLSecurityGuard:
    return SQLSecurityGuard(
        read_only=True,
        max_rows=50,
        dialect="sqlite",
        known_tables={"customers"},
        known_columns={"customers": {"customer_id", "name", "city"}},
    )


def test_sql_agent_generates_sql() -> None:
    llm = FakeLLM(
        {
            SQLResponseModel: SQLResponseModel(
                sql="SELECT name FROM customers LIMIT 10",
                explanation="List customer names",
                confidence=0.9,
            )
        }
    )
    state = empty_state("list customers")
    state["schema_text"] = "TABLE customers(customer_id, name, city)"
    state["dialect"] = "sqlite"
    out = make_sql_agent(llm)(state)  # type: ignore[arg-type]
    assert "SELECT" in out["sql"]
    assert out["sql_explanation"]


def test_validation_node_accepts_safe_sql(guard: SQLSecurityGuard) -> None:
    state = empty_state("q")
    state["sql"] = "SELECT name FROM customers LIMIT 5"
    out = make_validation_node(guard)(state)
    assert out["sql_valid"] is True
    assert out["validation_errors"] == []


def test_validation_node_rejects_delete(guard: SQLSecurityGuard) -> None:
    state = empty_state("q")
    state["sql"] = "DELETE FROM customers"
    out = make_validation_node(guard)(state)
    assert out["sql_valid"] is False
    assert out["validation_errors"]


def test_retry_agent_diagnoses_and_sets_approach() -> None:
    llm = FakeLLM(
        {
            RetryDecisionModel: RetryDecisionModel(
                should_retry=True,
                reasoning="typo in column names",
                failure_class="unknown_object",
                next_action="regenerate_sql",
                fix_hint="Use column name `name` not `names`",
                revised_approach="Select name, city instead of names",
            )
        }
    )
    state = empty_state("q")
    state["sql"] = "SELECT names FROM customers"
    state["sql_error"] = "no such column: names"
    state["retry_count"] = 0
    state["max_retries"] = 3
    out = make_retry_agent(llm)(state)  # type: ignore[arg-type]
    assert out["should_retry"] is True
    assert out["retry_next_action"] == "regenerate_sql"
    assert out["next_agent"] == "sql_agent"
    assert "name" in out["fix_hint"]
    assert "Approach change" in out["fix_hint"]
    assert out["retry_count"] == 1


def test_retry_different_errors_produce_different_hints() -> None:
    """Prove LLM diagnosis content differs by failure context (not blind replay)."""

    def factory(variables: dict[str, Any]) -> RetryDecisionModel:
        err = variables.get("errors") or ""
        if "DELETE" in err or "mutation" in err.lower() or "forbidden" in err.lower():
            return RetryDecisionModel(
                should_retry=True,
                reasoning="Mutation blocked by security gate",
                failure_class="permission_or_safety",
                next_action="regenerate_sql",
                fix_hint="Rewrite as SELECT only",
                revised_approach="Read-only projection of customers",
            )
        return RetryDecisionModel(
            should_retry=True,
            reasoning="Unknown column",
            failure_class="unknown_object",
            next_action="regenerate_sql",
            fix_hint="Replace `names` with `name`",
            revised_approach="Use schema column name",
        )

    llm = FakeLLM({RetryDecisionModel: factory})
    agent = make_retry_agent(llm)  # type: ignore[arg-type]

    s1 = empty_state("q")
    s1.update(
        {
            "sql": "DELETE FROM customers",
            "sql_error": "forbidden mutation DELETE",
            "retry_count": 0,
            "max_retries": 3,
        }
    )
    o1 = agent(s1)

    s2 = empty_state("q")
    s2.update(
        {
            "sql": "SELECT names FROM customers",
            "sql_error": "no such column: names",
            "retry_count": 0,
            "max_retries": 3,
        }
    )
    o2 = agent(s2)

    assert o1["fix_hint"] != o2["fix_hint"]
    assert o1["retry_diagnosis"] != o2["retry_diagnosis"]


def test_retry_agent_stops_at_max() -> None:
    llm = FakeLLM()
    state = empty_state("q")
    state["retry_count"] = 3
    state["max_retries"] = 3
    state["max_runtime_replans"] = 0  # force give_up (no replan budget)
    out = make_retry_agent(llm)(state)  # type: ignore[arg-type]
    assert out["should_retry"] is False
    assert out["next_agent"] in {"fail", "replan_agent"}


def test_retry_can_replan_to_supervisor() -> None:
    llm = FakeLLM(
        {
            RetryDecisionModel: RetryDecisionModel(
                should_retry=True,
                reasoning="Need schema refresh first",
                failure_class="unknown_object",
                next_action="replan",
                fix_hint="Supervisor should call schema_agent",
            )
        }
    )
    state = empty_state("q")
    state["retry_count"] = 0
    state["max_retries"] = 3
    state["sql_error"] = "no such table"
    state["max_runtime_replans"] = 2
    out = make_retry_agent(llm)(state)  # type: ignore[arg-type]
    # Prefer replan_agent under budget (was soft-routed to supervisor)
    assert out["next_agent"] == "replan_agent"
    assert out["retry_next_action"] == "replan"


def test_schema_agent_uses_bind_tools() -> None:
    schema = MagicMock()
    schema.to_prompt_text.return_value = "TABLE customers(...)"
    schema.to_dict.return_value = {"tables": []}
    schema.database_name = "retail"
    schema.dialect = "sqlite"

    tool = MagicMock()
    tool.name = "schema_tool"
    tool.invoke.return_value = "TABLE customers(id, name)"

    # ToolNode needs a real StructuredTool-like; use langchain tool
    from langchain_core.tools import StructuredTool

    real_tool = StructuredTool.from_function(
        func=lambda: "TABLE customers(id, name)",
        name="schema_tool",
        description="schema",
    )

    llm = FakeLLM(
        {
            SuggestedQuestionsModel: SuggestedQuestionsModel(
                questions=["Top cities?", "Revenue by month?", "Best products?"],
                themes=["geo"],
            )
        },
        tool_call_script=[
            {"name": "schema_tool", "args": {}},
        ],
    )
    out = make_schema_agent(schema, llm, {"schema_tool": real_tool})(empty_state("q"))  # type: ignore[arg-type]
    assert llm.bind_calls >= 1
    assert len(out["suggested_questions"]) >= 3
    assert out["database_name"] == "retail"


def test_visualization_agent_recommends_chart() -> None:
    llm = FakeLLM(
        {
            ChartRecommendationModel: ChartRecommendationModel(
                chart_type="bar",
                x_axis="city",
                y_axis="n",
                title="By city",
                rationale="categorical",
            )
        }
    )
    state = empty_state("by city")
    state["dataframe_records"] = [{"city": "Lahore", "n": 3}, {"city": "Karachi", "n": 2}]
    state["columns"] = ["city", "n"]
    out = make_visualization_agent(llm)(state)  # type: ignore[arg-type]
    assert out["chart_type"] == "bar"
    assert out["chart_spec"]["x_axis"] == "city"


def test_execution_node_uses_tool_and_executor(guard: SQLSecurityGuard, tmp_path) -> None:
    from pathlib import Path

    from database.connector import DatabaseConfig, DatabaseConnector
    from database.query_executor import QueryExecutor
    from sample_data.seed import create_sample_database
    from tools.sqlmind_tools import build_toolbelt, tools_by_name

    db = create_sample_database(Path(tmp_path) / "retail.db")
    connector = DatabaseConnector(
        DatabaseConfig(dialect="sqlite", sqlite_path=db, database="retail")
    )
    engine = connector.connect()
    from database.schema_inspector import SchemaInspector

    schema = SchemaInspector(engine, "sqlite", "retail").discover()
    security = SQLSecurityGuard(
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
        dialect="sqlite",
        max_rows=100,
    )
    executor = QueryExecutor(engine, security=security, database_label="retail")
    tools = tools_by_name(
        build_toolbelt(schema=schema, executor=executor, security=security, export_dir=str(tmp_path))
    )
    state = empty_state("q")
    state["sql"] = "SELECT name, city FROM customers LIMIT 5"
    out = make_execution_node(executor, tools)(state)
    assert out["query_success"] is True
    assert out["row_count"] >= 1
    connector.dispose()


def test_finalize_is_deterministic_utility() -> None:
    state = empty_state("hello")
    state["needs_clarification"] = True
    state["clarification_question"] = "Which city?"
    out = finalize_node(state)
    assert "city" in out["final_response"].lower()

    failed_state = empty_state("x")
    failed_state["sql_error"] = "boom"
    failed = fail_node(failed_state)
    assert "Unable to complete" in failed["final_response"] or "boom" in failed["final_response"]
