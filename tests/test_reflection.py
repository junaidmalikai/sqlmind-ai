"""Reflection agent routes based on LLM verdict."""

from __future__ import annotations

from agents.nodes import make_reflection_agent
from database.schema_inspector import ColumnInfo, ForeignKeyInfo, SchemaSnapshot, TableInfo
from graph.state import empty_state
from models.structured import ReflectionModel
from tests.test_agents import FakeLLM


def _schema_dict() -> dict:
    return SchemaSnapshot(
        dialect="sqlite",
        database_name="shop",
        tables=[
            TableInfo(
                name="orders",
                columns=[
                    ColumnInfo("id", "INTEGER", primary_key=True),
                    ColumnInfo("customer_id", "INTEGER"),
                    ColumnInfo("total_amount", "REAL"),
                ],
                foreign_keys=[ForeignKeyInfo(["customer_id"], "customers", ["id"])],
            ),
            TableInfo(
                name="customers",
                columns=[
                    ColumnInfo("id", "INTEGER", primary_key=True),
                    ColumnInfo("full_name", "TEXT"),
                    ColumnInfo("email", "TEXT"),
                    ColumnInfo("phone", "TEXT"),
                ],
            ),
        ],
    ).to_dict()


def test_reflection_accept_goes_finalize() -> None:
    llm = FakeLLM(
        {
            ReflectionModel: ReflectionModel(
                verdict="accept",
                reasoning="Looks faithful",
            )
        }
    )
    state = empty_state("q")
    state["sql"] = "SELECT 1"
    state["query_success"] = True
    state["insights"] = "One row"
    state["columns"] = ["value"]
    state["dataframe_records"] = [{"value": 1}]
    out = make_reflection_agent(llm)(state)  # type: ignore[arg-type]
    assert out["next_agent"] == "finalize"
    assert out["reflection_verdict"] == "accept"


def test_reflection_retry_sql_changes_path() -> None:
    llm = FakeLLM(
        {
            ReflectionModel: ReflectionModel(
                verdict="retry_sql",
                reasoning="Wrong grain — need daily not monthly",
                improvement_hint="Aggregate by day",
            )
        }
    )
    state = empty_state("daily revenue")
    state["sql"] = "SELECT strftime('%m', d), SUM(x) FROM t GROUP BY 1"
    state["query_success"] = True
    out = make_reflection_agent(llm)(state)  # type: ignore[arg-type]
    assert out["next_agent"] == "sql_agent"
    assert "day" in out["fix_hint"].lower() or "Aggregate" in out["fix_hint"]


def test_reflection_bi_gate_forces_retry_on_thin_result() -> None:
    """Name + Total_Orders should not finalize when richer schema columns exist."""
    llm = FakeLLM(
        {
            ReflectionModel: ReflectionModel(
                verdict="accept",
                reasoning="Technically correct ranking",
            )
        }
    )
    state = empty_state("Who placed the most orders?")
    state["sql"] = (
        "SELECT c.full_name AS Customer_Name, COUNT(*) AS Total_Orders "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "GROUP BY c.full_name ORDER BY Total_Orders DESC LIMIT 1"
    )
    state["query_success"] = True
    state["columns"] = ["Customer_Name", "Total_Orders"]
    state["dataframe_records"] = [{"Customer_Name": "Sara Malik", "Total_Orders": 9}]
    state["row_count"] = 1
    state["schema_dict"] = _schema_dict()
    state["max_reflections"] = 2
    state["reflection_count"] = 0
    out = make_reflection_agent(llm)(state)  # type: ignore[arg-type]
    assert out["reflection_verdict"] == "retry_sql"
    assert out["next_agent"] == "sql_agent"
    assert out["fix_hint"]
    assert "JOIN" in out["fix_hint"] or "enrich" in out["fix_hint"].lower()


def test_reflection_bi_gate_respects_scalar_only_ask() -> None:
    llm = FakeLLM(
        {
            ReflectionModel: ReflectionModel(
                verdict="accept",
                reasoning="Scalar answer as requested",
            )
        }
    )
    state = empty_state("Only count the total orders")
    state["sql"] = "SELECT COUNT(*) AS Total_Orders FROM orders"
    state["query_success"] = True
    state["columns"] = ["Total_Orders"]
    state["dataframe_records"] = [{"Total_Orders": 42}]
    state["row_count"] = 1
    state["schema_dict"] = _schema_dict()
    state["max_reflections"] = 2
    out = make_reflection_agent(llm)(state)  # type: ignore[arg-type]
    assert out["reflection_verdict"] == "accept"
    assert out["next_agent"] == "finalize"
