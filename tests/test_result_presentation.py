"""Tests for presentation-layer result UX helpers."""

from __future__ import annotations

import pandas as pd

from services.result_presentation import (
    build_data_overview,
    build_query_summary,
    compose_executive_answer,
    detect_entities_from_columns,
    humanize_column_name,
    humanize_dataframe,
    is_weak_answer,
)


def test_humanize_common_columns():
    assert humanize_column_name("customer_name") == "Customer Name"
    assert humanize_column_name("email") == "Email"
    assert humanize_column_name("phone") == "Phone Number"
    assert humanize_column_name("mobile_no") == "Mobile Number"
    assert humanize_column_name("order_qty") == "Order Quantity"
    assert humanize_column_name("total_orders") == "Total Orders"
    assert humanize_column_name("product_name") == "Product"
    assert humanize_column_name("created_at") == "Created At"
    assert humanize_column_name("username") == "Username"
    assert humanize_column_name("student_name") == "Student Name"


def test_humanize_dataframe_display_only():
    df = pd.DataFrame([{"customer_name": "Sara", "total_orders": 9}])
    view = humanize_dataframe(df)
    assert list(view.columns) == ["Customer Name", "Total Orders"]
    assert list(df.columns) == ["customer_name", "total_orders"]


def test_weak_answer_detection():
    assert is_weak_answer("Done.")
    assert is_weak_answer("Analysis complete.")
    assert not is_weak_answer("Retrieved 183 customer orders with city context.")


def test_query_summary_success():
    state = {
        "query_success": True,
        "row_count": 183,
        "sql": "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id",
        "validation_meta": {"tables": ["orders", "customers"]},
        "columns": ["customer_name", "city", "order_date"],
    }
    df = pd.DataFrame(
        [{"customer_name": "A", "city": "Lahore", "order_date": "2026-01-01"}] * 3
    )
    text = build_query_summary(state, question="List orders with customer city", df=df)
    assert "## Query Summary" in text
    assert "183" in text
    assert "Success" in text
    assert "Done." not in text


def test_query_summary_empty():
    state = {
        "query_success": True,
        "row_count": 0,
        "sql": "SELECT * FROM orders WHERE 1=0",
        "validation_meta": {"tables": ["orders"]},
        "columns": [],
    }
    text = build_query_summary(state, question="Show orders from 1990", df=pd.DataFrame())
    assert "no matching records" in text.lower()
    assert "Suggested next questions" in text
    assert "0 rows" not in text.lower() or "0 records" in text.lower()


def test_data_overview_lists_friendly_fields():
    df = pd.DataFrame(
        [{"customer_name": "Sara", "product_name": "Phone", "order_qty": 2}]
    )
    overview = build_data_overview(df, row_count=1)
    assert "Data Overview" in overview
    assert "Customer Name" in overview
    assert "Product" in overview
    assert "Order Quantity" in overview


def test_compose_replaces_done():
    state = {
        "query_success": True,
        "row_count": 2,
        "sql": "SELECT customer_name, city FROM customers",
        "validation_meta": {"tables": ["customers"]},
        "columns": ["customer_name", "city"],
        "insights": "Done.",
        "final_response": "Done.",
    }
    df = pd.DataFrame([{"customer_name": "A", "city": "Karachi"}] * 2)
    out = compose_executive_answer(
        state,
        question="List customers and city",
        df=df,
        response_text="Done.",
        executive={"summary": "Done.", "bullets": []},
    )
    assert "Query Summary" in out["summary"]
    assert not is_weak_answer(out["summary"])
    assert out["result_cards"]
    assert out["data_overview"]


def test_detect_entities():
    ents = detect_entities_from_columns(
        ["Customer_Name", "Email", "Total_Orders", "Product_Name"]
    )
    assert "Customer" in ents
    assert "Order" in ents or "Product" in ents
