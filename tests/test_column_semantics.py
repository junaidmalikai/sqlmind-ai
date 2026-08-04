"""Tests for dynamic semantic column detection."""

from __future__ import annotations

from database.schema_inspector import ColumnInfo, ForeignKeyInfo, SchemaSnapshot, TableInfo
from services.column_semantics import (
    assess_result_bi_quality,
    build_bi_enrichment_checklist,
    build_business_column_hints,
    build_join_enrichment_hints,
    is_scalar_only_request,
    preferred_columns_for_table,
    score_column,
)


def _commerce_schema() -> SchemaSnapshot:
    return SchemaSnapshot(
        dialect="sqlite",
        database_name="shop",
        tables=[
            TableInfo(
                name="orders",
                columns=[
                    ColumnInfo("id", "INTEGER", primary_key=True),
                    ColumnInfo("order_number", "TEXT"),
                    ColumnInfo("total_amount", "REAL"),
                    ColumnInfo("order_date", "TEXT"),
                    ColumnInfo("status", "TEXT"),
                    ColumnInfo("customer_id", "INTEGER"),
                    ColumnInfo("product_id", "INTEGER"),
                    ColumnInfo("quantity", "INTEGER"),
                    ColumnInfo("unit_price", "REAL"),
                ],
                foreign_keys=[
                    ForeignKeyInfo(["customer_id"], "customers", ["id"]),
                    ForeignKeyInfo(["product_id"], "products", ["id"]),
                ],
            ),
            TableInfo(
                name="customers",
                columns=[
                    ColumnInfo("id", "INTEGER", primary_key=True),
                    ColumnInfo("full_name", "TEXT"),
                    ColumnInfo("email", "TEXT"),
                    ColumnInfo("phone", "TEXT"),
                    ColumnInfo("city", "TEXT"),
                    ColumnInfo("country", "TEXT"),
                ],
            ),
            TableInfo(
                name="products",
                columns=[
                    ColumnInfo("id", "INTEGER", primary_key=True),
                    ColumnInfo("product_name", "TEXT"),
                    ColumnInfo("category", "TEXT"),
                    ColumnInfo("brand", "TEXT"),
                    ColumnInfo("price", "REAL"),
                    ColumnInfo("stock", "INTEGER"),
                ],
            ),
        ],
    )


def test_prefers_name_over_id():
    hit = score_column("student_name")
    assert hit is not None
    assert hit.kind == "person"
    id_hit = score_column("student_id", primary_key=True)
    assert id_hit is not None
    assert id_hit.rank > hit.rank


def test_detects_seller_and_specialization():
    assert score_column("seller_name").kind == "person"
    assert score_column("specialization").kind == "title"
    assert score_column("payment_method").kind == "metric"
    assert score_column("email_address").kind == "contact"


def test_unknown_schema_adapts_without_hardcoded_table():
    table = TableInfo(
        name="learners",
        columns=[
            ColumnInfo("learner_id", "INTEGER", primary_key=True),
            ColumnInfo("display_name", "TEXT"),
            ColumnInfo("email", "TEXT"),
            ColumnInfo("cgpa", "REAL"),
            ColumnInfo("dept_code", "TEXT"),
        ],
        foreign_keys=[
            ForeignKeyInfo(["dept_code"], "departments", ["code"]),
        ],
    )
    prefs = preferred_columns_for_table(table)
    names = [c.name for c in prefs]
    assert "display_name" in names
    assert "email" in names
    assert names[0] != "learner_id"


def test_hints_include_only_existing_columns():
    schema = _commerce_schema()
    text = build_business_column_hints(schema)
    assert "order_number" in text
    assert "full_name" in text
    assert "customer_name" not in text  # must not invent
    assert "FK" in text
    assert "Join enrichment map" in text
    assert "customers.full_name" in text
    assert "products.product_name" in text


def test_join_enrichment_hints_are_schema_dynamic():
    hints = build_join_enrichment_hints(_commerce_schema())
    assert "orders.(customer_id) → customers" in hints
    assert "full_name" in hints
    assert "email" in hints
    assert "invented_column" not in hints


def test_bi_checklist_includes_join_map_from_dict():
    schema = _commerce_schema()
    checklist = build_bi_enrichment_checklist(schema.to_dict())
    assert "BI enrichment checklist" in checklist
    assert "Join enrichment map" in checklist
    assert "customers" in checklist


def test_to_prompt_text_appends_hints():
    schema = SchemaSnapshot(
        dialect="sqlite",
        database_name="demo",
        tables=[
            TableInfo(
                name="staff",
                columns=[
                    ColumnInfo("staff_id", "INTEGER", primary_key=True),
                    ColumnInfo("employee_name", "TEXT"),
                    ColumnInfo("salary", "REAL"),
                ],
            )
        ],
    )
    prompt = schema.to_prompt_text()
    assert "Business column preferences" in prompt
    assert "employee_name" in prompt


def test_thin_name_plus_count_detected():
    assessment = assess_result_bi_quality(
        ["Customer_Name", "Total_Orders"],
        "Who placed the most orders?",
        schema_dict=_commerce_schema().to_dict(),
        row_count=1,
    )
    assert assessment.thin is True
    assert assessment.has_aggregate is True
    assert "JOIN" in assessment.improvement_hint or "join" in assessment.improvement_hint.lower()


def test_scalar_only_request_not_forced():
    assert is_scalar_only_request("Only total orders please")
    assessment = assess_result_bi_quality(
        ["Total_Orders"],
        "Only count the orders",
        schema_dict=_commerce_schema().to_dict(),
        row_count=1,
    )
    assert assessment.thin is False
    assert assessment.scalar_request is True


def test_rich_result_not_thin():
    assessment = assess_result_bi_quality(
        [
            "Customer Name",
            "Email",
            "Phone",
            "Total Orders",
            "Last Order Date",
            "City",
        ],
        "Who placed the most orders?",
        schema_dict=_commerce_schema().to_dict(),
        row_count=5,
    )
    assert assessment.thin is False
