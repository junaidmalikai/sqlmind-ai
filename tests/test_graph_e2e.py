"""End-to-end LangGraph: bind_tools supervisor + SQL path + retry (mocked LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from database.connector import DatabaseConfig, DatabaseConnector
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaInspector
from graph.state import empty_state
from graph.workflow import (
    build_sqlmind_graph,
    route_after_execution,
    route_after_retry,
    route_after_validation,
)
from models.structured import (
    ChartRecommendationModel,
    InsightModel,
    ReflectionModel,
    RetryDecisionModel,
    SQLResponseModel,
    SuggestedQuestionsModel,
)
from planner.models import (
    DecompositionOutput,
    GoalSpec,
    PlannerOutput,
    ReplanDecision,
    TaskSpec,
)
from sample_data.seed import create_sample_database
from utils.security import SQLSecurityGuard


class ScriptedLLM:
    """Scripted bind_tools + structured outputs for graph e2e without network."""

    def __init__(self) -> None:
        self.sql_attempts = 0
        self.supervisor_calls = 0
        self._supervisor_queue: list[str] = [
            "route_to_sql_agent",
            "route_to_reflection_agent",
            "route_to_finalize",
        ]

    def history_messages(self, history, limit: int = 12):  # noqa: ANN001
        return []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        tool_names = {t.name for t in tools}

        def _invoke(messages):  # noqa: ANN001
            # Supervisor routing tools
            if "route_to_sql_agent" in tool_names:
                self.supervisor_calls += 1
                if self._supervisor_queue:
                    name = self._supervisor_queue.pop(0)
                else:
                    name = "route_to_finalize"
                args: dict[str, Any] = {"reasoning": f"scripted:{name}"}
                if name == "ask_user_clarification":
                    args = {"question": "Which metric?"}
                return AIMessage(
                    content="",
                    tool_calls=[{"id": f"s{self.supervisor_calls}", "name": name, "args": args}],
                )

            # Schema agent
            if "schema_tool" in tool_names and "submit_sql" not in tool_names and "explain_tool" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "schema1",
                            "name": "schema_tool",
                            "args": {},
                        }
                    ],
                )

            # Optimization explain
            if "explain_tool" in tool_names and "submit_sql" not in tool_names:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "ex1",
                            "name": "explain_tool",
                            "args": {"sql": "SELECT 1"},
                        }
                    ],
                )

            # SQL ReAct — submit immediately (optionally after validate)
            if "submit_sql" in tool_names:
                self.sql_attempts += 1
                sql = (
                    "SELECT c.city, COUNT(*) AS n "
                    "FROM customers c GROUP BY c.city ORDER BY n DESC LIMIT 20"
                )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": f"sql{self.sql_attempts}",
                            "name": "submit_sql",
                            "args": {"sql": sql, "explanation": "Customers by city"},
                        }
                    ],
                )

            return AIMessage(content="no tools matched")

        return RunnableLambda(_invoke)

    def invoke_structured(self, prompt, schema, variables):  # noqa: ANN001
        if schema is GoalSpec:
            q = str(variables.get("question") or "query")
            return GoalSpec(
                goal=q,
                objectives=[q],
                expected_output="table + insights",
                confidence=0.9,
                simple=True,
                rewritten_question=q,
                intent_label="query",
            )
        if schema is PlannerOutput:
            return PlannerOutput(
                strategy="Supervisor fast path",
                execution_mode="sequential",
                estimated_cost=1.0,
                use_supervisor_fallback=True,
                reasoning="scripted simple plan",
                required_skills_sequence=[["nl2sql"]],
            )
        if schema is DecompositionOutput:
            return DecompositionOutput(
                tasks=[
                    TaskSpec(
                        id="t1",
                        title="SQL",
                        required_skills=["nl2sql"],
                        provides_any=["sql"],
                    )
                ],
                reasoning="scripted",
            )
        if schema is ReplanDecision:
            return ReplanDecision(
                strategy="continue_with_supervisor",
                reasoning="scripted replan → supervisor",
                failure_analysis="scripted",
                confidence=0.7,
            )
        if schema is SuggestedQuestionsModel:
            return SuggestedQuestionsModel(
                questions=["Q1?", "Q2?", "Q3?"],
                themes=["sales"],
            )
        if schema is SQLResponseModel:
            self.sql_attempts += 1
            return SQLResponseModel(
                sql=(
                    "SELECT c.city, COUNT(*) AS n "
                    "FROM customers c GROUP BY c.city ORDER BY n DESC LIMIT 20"
                ),
                explanation="Customers by city",
                confidence=0.95,
            )
        if schema is ChartRecommendationModel:
            return ChartRecommendationModel(
                chart_type="bar",
                x_axis="city",
                y_axis="n",
                title="Customers by city",
                rationale="categorical count",
            )
        if schema is InsightModel:
            return InsightModel(
                summary="Lahore leads in customer count among sample cities.",
                bullets=["City distribution is uneven"],
                trends=[],
                anomalies=[],
                recommendations=["Drill into revenue by city"],
            )
        if schema is RetryDecisionModel:
            return RetryDecisionModel(
                should_retry=True,
                reasoning="fix column name",
                failure_class="unknown_object",
                next_action="regenerate_sql",
                fix_hint="Use city not cities",
                revised_approach="Group by city",
            )
        if schema is ReflectionModel:
            return ReflectionModel(
                verdict="accept",
                reasoning="Results answer the question",
                issues=[],
            )
        raise AssertionError(f"Unhandled schema {schema}")

    def stream_text(self, prompt, variables):  # noqa: ANN001
        yield "ok"

    def invoke_text(self, prompt, variables):  # noqa: ANN001
        return "narrative"


@pytest.fixture
def sample_env(tmp_path: Path):
    db = create_sample_database(tmp_path / "retail.db")
    config = DatabaseConfig(dialect="sqlite", sqlite_path=db, database="retail")
    connector = DatabaseConnector(config)
    engine = connector.connect()
    schema = SchemaInspector(engine, "sqlite", "retail").discover()
    security = SQLSecurityGuard(
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
        dialect="sqlite",
        max_rows=200,
    )
    executor = QueryExecutor(engine, security=security, database_label="retail")
    llm = ScriptedLLM()
    from langgraph.checkpoint.memory import MemorySaver

    app = build_sqlmind_graph(
        schema,
        executor,
        llm=llm,  # type: ignore[arg-type]
        security=security,
        checkpointer=MemorySaver(),
    )
    yield {
        "app": app,
        "schema": schema,
        "executor": executor,
        "llm": llm,
        "connector": connector,
        "security": security,
    }
    connector.dispose()


def test_routing_helpers() -> None:
    from graph.workflow import route_failure

    s = empty_state("q")
    s["sql_valid"] = True
    assert route_after_validation(s) == "execution_node"
    s["sql_valid"] = False
    # Enterprise default: fail → recovery_graph (legacy retry when forced)
    assert route_after_validation(s) == route_failure(s)
    s["force_retry_agent"] = True
    assert route_after_validation(s) == "retry_agent"
    del s["force_retry_agent"]

    s["query_success"] = False
    assert route_after_execution(s) == route_failure(s)
    s["query_success"] = True
    fan = route_after_execution(s)
    assert isinstance(fan, list) and len(fan) == 2

    s["should_retry"] = True
    s["retry_next_action"] = "regenerate_sql"
    s["next_agent"] = "sql_agent"
    assert route_after_retry(s) == "sql_agent"
    s["retry_next_action"] = "replan"
    s["next_agent"] = "supervisor"
    s["max_runtime_replans"] = 2
    s["runtime_replan_count"] = 0
    # Under replan budget, prefer replan_agent even from supervisor soft route
    assert route_after_retry(s) == "replan_agent"
    s["adaptive_plan"] = {"plan_id": "p1"}
    s["retry_next_action"] = "replan"
    s["next_agent"] = "replan_agent"
    assert route_after_retry(s) == "replan_agent"
    s["adaptive_plan"] = {}
    s["plan_active"] = False
    s["retry_next_action"] = "give_up"
    s["should_retry"] = False
    s["next_agent"] = "fail"
    s["final_response"] = "give up"
    assert route_after_retry(s) in {"fail", "finalize"}


def test_graph_happy_path(sample_env: dict[str, Any]) -> None:
    app = sample_env["app"]
    initial = empty_state("Show customers by city", session_id="test-happy")
    initial["schema_text"] = sample_env["schema"].to_prompt_text()
    initial["schema_dict"] = sample_env["schema"].to_dict()
    initial["database_name"] = "retail"
    initial["dialect"] = "sqlite"
    initial["max_retries"] = 3
    initial["max_supervisor_visits"] = 8

    result = app.invoke(initial, config={"configurable": {"thread_id": "test-happy"}})
    assert result.get("status") == "complete"
    assert result.get("query_success") is True
    assert result.get("row_count", 0) >= 1
    assert result.get("sql")
    assert result.get("chart_type") == "bar"
    assert result.get("insights")
    assert result.get("final_response")
    assert sample_env["llm"].supervisor_calls >= 1
    assert sample_env["llm"].sql_attempts >= 1


def test_graph_retry_after_bad_sql(sample_env: dict[str, Any]) -> None:
    """Force validation failure then retry with fixed SQL via scripted LLM."""

    class FlakySQL(ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self._supervisor_queue = [
                "route_to_sql_agent",
                "route_to_reflection_agent",
                "route_to_finalize",
            ]

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001
            tool_names = {t.name for t in tools}
            parent = super().bind_tools(tools, **kwargs)

            if "submit_sql" not in tool_names:
                return parent

            def _invoke(messages):  # noqa: ANN001
                self.sql_attempts += 1
                if self.sql_attempts == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "bad",
                                "name": "submit_sql",
                                "args": {
                                    "sql": "DELETE FROM customers",
                                    "explanation": "bad",
                                },
                            }
                        ],
                    )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "good",
                            "name": "submit_sql",
                            "args": {
                                "sql": "SELECT name, city FROM customers LIMIT 5",
                                "explanation": "fixed",
                            },
                        }
                    ],
                )

            return RunnableLambda(_invoke)

        def invoke_structured(self, prompt, schema, variables):  # noqa: ANN001
            if schema is RetryDecisionModel:
                return RetryDecisionModel(
                    should_retry=True,
                    reasoning="mutation blocked — regenerate read-only",
                    failure_class="permission_or_safety",
                    next_action="regenerate_sql",
                    fix_hint="Use SELECT only",
                    revised_approach="Projection of name, city",
                )
            if schema is ReflectionModel:
                return ReflectionModel(verdict="accept", reasoning="ok")
            if schema is ChartRecommendationModel:
                return ChartRecommendationModel(
                    chart_type="table", title="Customers", rationale="rows"
                )
            if schema is InsightModel:
                return InsightModel(summary="Customer list retrieved.", bullets=[])
            return super().invoke_structured(prompt, schema, variables)

    from langgraph.checkpoint.memory import MemorySaver

    schema = sample_env["schema"]
    executor = sample_env["executor"]
    security = sample_env["security"]
    llm = FlakySQL()
    app = build_sqlmind_graph(
        schema,
        executor,
        llm=llm,  # type: ignore[arg-type]
        security=security,
        checkpointer=MemorySaver(),
    )
    initial = empty_state("list customers", session_id="test-retry")
    initial["schema_text"] = schema.to_prompt_text()
    initial["database_name"] = "retail"
    initial["dialect"] = "sqlite"
    initial["max_retries"] = 3
    initial["max_supervisor_visits"] = 10

    result = app.invoke(initial, config={"configurable": {"thread_id": "test-retry"}})
    assert llm.sql_attempts >= 2
    assert result.get("query_success") is True
    assert "DELETE" not in (result.get("sql") or "").upper()
