"""Tests for sample DB, schema, execution, plan normalization, models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from database.connector import DatabaseConfig, DatabaseConnector
from database.query_executor import QueryExecutor
from database.schema_inspector import SchemaInspector
from graph.state import normalize_plan
from models.structured import ChartRecommendationModel, IntentModel, SQLResponseModel
from sample_data.seed import create_sample_database
from services.visualization import build_figure_from_recommendation
from utils.security import SQLSecurityGuard


@pytest.fixture
def sample_db(tmp_path: Path) -> str:
    return create_sample_database(tmp_path / "retail.db")


def test_sample_schema_discovery(sample_db: str) -> None:
    config = DatabaseConfig(dialect="sqlite", sqlite_path=sample_db, database="retail")
    connector = DatabaseConnector(config)
    engine = connector.connect()
    schema = SchemaInspector(engine, "sqlite", "retail").discover()
    names = set(schema.table_names())
    assert {"customers", "orders", "products", "order_items", "employees"} <= names
    assert len(schema.relationships) >= 3
    connector.dispose()


def test_safe_query_execution(sample_db: str) -> None:
    config = DatabaseConfig(dialect="sqlite", sqlite_path=sample_db, database="retail")
    connector = DatabaseConnector(config)
    engine = connector.connect()
    schema = SchemaInspector(engine, "sqlite", "retail").discover()
    guard = SQLSecurityGuard(
        known_tables=set(schema.table_names()),
        known_columns=schema.columns_map(),
        max_rows=50,
        dialect="sqlite",
    )
    executor = QueryExecutor(engine, security=guard, database_label="retail")
    result = executor.execute(
        """
        SELECT c.city, SUM(oi.quantity * oi.unit_price) AS revenue
        FROM customers c
        JOIN orders o ON o.customer_id = c.customer_id
        JOIN order_items oi ON oi.order_id = o.order_id
        GROUP BY c.city
        ORDER BY revenue DESC
        """
    )
    assert result.success
    assert result.row_count >= 1
    assert "revenue" in result.dataframe.columns
    connector.dispose()


def test_blocks_mutation(sample_db: str) -> None:
    config = DatabaseConfig(dialect="sqlite", sqlite_path=sample_db, database="retail")
    connector = DatabaseConnector(config)
    engine = connector.connect()
    executor = QueryExecutor(engine, database_label="retail")
    result = executor.execute("DELETE FROM customers")
    assert not result.success
    connector.dispose()


def test_chart_from_ai_recommendation() -> None:
    df = pd.DataFrame({"city": ["Lahore", "Karachi"], "revenue": [1000, 800]})
    rec = ChartRecommendationModel(
        chart_type="bar",
        x_axis="city",
        y_axis="revenue",
        title="Revenue by City",
        rationale="Categorical vs numeric",
    )
    fig = build_figure_from_recommendation(df, rec)
    assert fig is not None


def test_normalize_plan() -> None:
    plan = normalize_plan(["schema_agent", "sql_agent", "bogus"], "query")
    assert "bogus" not in plan
    assert "schema_agent" in plan
    assert "sql_agent" in plan
    # Empty input must NOT inject DEFAULT_PLANS
    assert normalize_plan([], "query") == []


def test_structured_models_roundtrip() -> None:
    intent = IntentModel(
        intent="query",
        reasoning="analytics",
    )
    assert intent.intent == "query"
    sql = SQLResponseModel(sql="SELECT 1", explanation="test", confidence=0.9)
    assert "SELECT" in sql.sql
