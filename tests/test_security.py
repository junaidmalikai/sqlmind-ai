"""Adversarial + depth tests for deterministic sqlglot SQL security."""

from __future__ import annotations

import pytest

from utils.security import SQLSecurityGuard, SQLSecurityError


@pytest.fixture
def guard() -> SQLSecurityGuard:
    return SQLSecurityGuard(
        read_only=True,
        max_rows=100,
        dialect="sqlite",
        known_tables={"customers", "orders", "products", "order_items"},
        known_columns={
            "customers": {"customer_id", "name", "city", "email"},
            "orders": {"order_id", "customer_id", "order_date", "status"},
            "products": {"product_id", "name", "unit_price", "category"},
            "order_items": {"order_item_id", "order_id", "product_id", "quantity", "unit_price"},
        },
    )


# --- Allow path ---


def test_allows_select(guard: SQLSecurityGuard) -> None:
    result = guard.validate("SELECT name, city FROM customers LIMIT 10")
    assert result.is_safe
    assert "customers" in result.tables


def test_allows_with_cte(guard: SQLSecurityGuard) -> None:
    sql = """
    WITH top_cities AS (
      SELECT city, COUNT(*) AS n FROM customers GROUP BY city
    )
    SELECT * FROM top_cities ORDER BY n DESC LIMIT 5
    """
    result = guard.validate(sql)
    assert result.is_safe


def test_allows_union(guard: SQLSecurityGuard) -> None:
    result = guard.validate(
        "SELECT name FROM customers LIMIT 5 UNION SELECT name FROM products LIMIT 5"
    )
    assert result.is_safe


# --- DDL / DML hard blocks ---


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DROP VIEW IF EXISTS v",
        "DELETE FROM customers",
        "DELETE FROM customers WHERE 1=1",
        "UPDATE customers SET name='x'",
        "UPDATE customers SET name='x' WHERE customer_id=1",
        "ALTER TABLE customers ADD COLUMN hack TEXT",
        "TRUNCATE TABLE customers",
        "INSERT INTO customers (name, city) VALUES ('a','b')",
        "CREATE TABLE evil (id INT)",
        "CREATE INDEX idx ON customers(name)",
        "REPLACE INTO customers (customer_id, name, city) VALUES (1,'a','b')",
        "GRANT ALL ON customers TO public",
        "REVOKE ALL ON customers FROM public",
        "VACUUM",
        "ATTACH DATABASE 'evil.db' AS evil",
    ],
)
def test_rejects_mutations(guard: SQLSecurityGuard, sql: str) -> None:
    result = guard.validate(sql)
    assert not result.is_safe, f"Should reject: {sql}"


def test_rejects_select_into(guard: SQLSecurityGuard) -> None:
    # May be caught by regex and/or AST depending on dialect parse
    result = guard.validate("SELECT * INTO new_table FROM customers")
    assert not result.is_safe


# --- Injection / multi-statement ---


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE customers",
        "SELECT name FROM customers; DELETE FROM orders",
        "SELECT 1; SELECT 2",
        "SELECT name FROM customers WHERE name = 'x'; DROP TABLE customers;--'",
    ],
)
def test_rejects_multi_statement_and_injection(guard: SQLSecurityGuard, sql: str) -> None:
    result = guard.validate(sql)
    assert not result.is_safe


def test_comment_stripped_mutation_blocked(guard: SQLSecurityGuard) -> None:
    result = guard.validate("DELETE /*x*/ FROM customers")
    assert not result.is_safe


# --- Schema allow-list ---


def test_unknown_table(guard: SQLSecurityGuard) -> None:
    result = guard.validate("SELECT * FROM secrets")
    assert not result.is_safe
    assert any("Unknown table" in e for e in result.errors)


def test_adds_limit(guard: SQLSecurityGuard) -> None:
    result = guard.validate("SELECT name FROM customers")
    assert result.is_safe
    assert "LIMIT" in result.sql.upper()


def test_raise_if_unsafe(guard: SQLSecurityGuard) -> None:
    result = guard.validate("UPDATE customers SET name='x'")
    with pytest.raises(SQLSecurityError):
        result.raise_if_unsafe()


def test_ast_join_extraction(guard: SQLSecurityGuard) -> None:
    result = guard.validate(
        "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.customer_id"
    )
    assert result.is_safe
    assert "customers" in result.tables
    assert "orders" in result.tables
    assert result.joins


def test_syntax_error(guard: SQLSecurityGuard) -> None:
    result = guard.validate("SELEC name FORM customers")
    assert not result.is_safe


def test_empty_sql(guard: SQLSecurityGuard) -> None:
    assert not guard.validate("").is_safe
    assert not guard.validate("   ").is_safe


def test_executor_gate_blocks_delete(tmp_path) -> None:
    from pathlib import Path

    from database.connector import DatabaseConfig, DatabaseConnector
    from database.query_executor import QueryExecutor
    from sample_data.seed import create_sample_database

    db = create_sample_database(Path(tmp_path) / "t.db")
    connector = DatabaseConnector(
        DatabaseConfig(dialect="sqlite", sqlite_path=db, database="retail")
    )
    engine = connector.connect()
    executor = QueryExecutor(engine, database_label="retail")
    result = executor.execute("DELETE FROM customers")
    assert not result.success
    assert result.error
    connector.dispose()
