"""Business KPI helpers (structural metrics remain deterministic)."""

from __future__ import annotations

from typing import Any

from database.schema_inspector import SchemaSnapshot


def compute_table_kpis(schema: SchemaSnapshot) -> list[dict[str, Any]]:
    """Lightweight structural KPIs from schema metadata (deterministic)."""
    cards: list[dict[str, Any]] = [
        {"label": "Tables", "value": len(schema.tables), "help": "Discovered user tables"},
        {
            "label": "Relationships",
            "value": len(schema.relationships),
            "help": "Foreign-key links",
        },
        {
            "label": "Columns",
            "value": sum(len(t.columns) for t in schema.tables),
            "help": "Total columns across tables",
        },
    ]
    total_rows = sum(t.row_estimate or 0 for t in schema.tables)
    cards.append(
        {
            "label": "Est. Rows",
            "value": f"{total_rows:,}",
            "help": "Sum of table row estimates",
        }
    )
    return cards
