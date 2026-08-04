"""Automatic schema discovery for PostgreSQL, MySQL, and SQLite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    default: Any = None


@dataclass
class ForeignKeyInfo:
    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]


@dataclass
class IndexInfo:
    name: str
    columns: list[str]
    unique: bool = False


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    indexes: list[IndexInfo] = field(default_factory=list)
    row_estimate: int | None = None


@dataclass
class SchemaSnapshot:
    """Full discovered schema for a connected database."""

    dialect: str
    database_name: str
    tables: list[TableInfo] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def columns_map(self) -> dict[str, set[str]]:
        return {t.name: {c.name for c in t.columns} for t in self.tables}

    def to_prompt_text(self, max_tables: int = 50) -> str:
        """Compact schema text for LLM prompts."""
        lines: list[str] = [
            f"Dialect: {self.dialect}",
            f"Database: {self.database_name}",
            f"Tables ({len(self.tables)}):",
        ]
        for table in self.tables[:max_tables]:
            cols = ", ".join(
                f"{c.name}:{c.data_type}"
                + (" PK" if c.primary_key else "")
                + ("" if c.nullable else " NOT NULL")
                for c in table.columns
            )
            lines.append(f"- {table.name}({cols})")
            for fk in table.foreign_keys:
                lines.append(
                    f"  FK {table.name}.({', '.join(fk.constrained_columns)})"
                    f" -> {fk.referred_table}.({', '.join(fk.referred_columns)})"
                )
        if self.relationships:
            lines.append("Relationships:")
            for rel in self.relationships[:100]:
                lines.append(
                    f"- {rel['from_table']}.{rel['from_column']}"
                    f" -> {rel['to_table']}.{rel['to_column']}"
                )
        try:
            from services.column_semantics import build_business_column_hints

            hints = build_business_column_hints(self, max_tables=max_tables)
            if hints:
                lines.append("")
                lines.append(hints)
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SchemaInspector:
    """Inspects database metadata via SQLAlchemy Inspector."""

    def __init__(self, engine: Engine, dialect: str, database_name: str) -> None:
        self.engine = engine
        self.dialect = dialect
        self.database_name = database_name

    def discover(self) -> SchemaSnapshot:
        """Discover tables, columns, keys, indexes, and relationships."""
        inspector = inspect(self.engine)
        tables: list[TableInfo] = []
        relationships: list[dict[str, Any]] = []

        for table_name in inspector.get_table_names():
            columns_raw = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name) or {}
            pk_cols = list(pk.get("constrained_columns") or [])

            columns = [
                ColumnInfo(
                    name=col["name"],
                    data_type=str(col.get("type", "UNKNOWN")),
                    nullable=bool(col.get("nullable", True)),
                    primary_key=col["name"] in pk_cols,
                    default=col.get("default"),
                )
                for col in columns_raw
            ]

            fks: list[ForeignKeyInfo] = []
            for fk in inspector.get_foreign_keys(table_name):
                info = ForeignKeyInfo(
                    constrained_columns=list(fk.get("constrained_columns") or []),
                    referred_table=fk.get("referred_table") or "",
                    referred_columns=list(fk.get("referred_columns") or []),
                )
                fks.append(info)
                for src, dst in zip(info.constrained_columns, info.referred_columns):
                    relationships.append(
                        {
                            "from_table": table_name,
                            "from_column": src,
                            "to_table": info.referred_table,
                            "to_column": dst,
                        }
                    )

            indexes = [
                IndexInfo(
                    name=idx.get("name") or "",
                    columns=list(idx.get("column_names") or []),
                    unique=bool(idx.get("unique")),
                )
                for idx in inspector.get_indexes(table_name)
            ]

            row_estimate = self._estimate_rows(table_name)

            tables.append(
                TableInfo(
                    name=table_name,
                    columns=columns,
                    primary_keys=pk_cols,
                    foreign_keys=fks,
                    indexes=indexes,
                    row_estimate=row_estimate,
                )
            )

        logger.info(
            "Discovered %d tables, %d relationships in %s",
            len(tables),
            len(relationships),
            self.database_name,
        )
        return SchemaSnapshot(
            dialect=self.dialect,
            database_name=self.database_name,
            tables=tables,
            relationships=relationships,
        )

    def _estimate_rows(self, table_name: str) -> int | None:
        """Best-effort row count (may be slow on huge tables — capped)."""
        try:
            # Quote identifiers safely via SQLAlchemy text with bound table is hard;
            # use inspector-validated name only.
            if not table_name.replace("_", "").isalnum():
                return None
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                return int(result.scalar() or 0)
        except Exception:  # noqa: BLE001
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
                    return int(result.scalar() or 0)
            except Exception:  # noqa: BLE001
                return None
