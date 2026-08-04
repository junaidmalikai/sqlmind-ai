"""AST-based SQL validation with sqlglot + regex safety net.

Security decisions are DETERMINISTIC — never delegated to an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from core.exceptions import SQLSecurityError as _BaseSQLSecurityError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SQLSecurityError(_BaseSQLSecurityError):
    """Raised when a SQL statement violates security policy."""


# Secondary safety net only — primary validation is sqlglot AST.
FORBIDDEN_STATEMENT_RE = re.compile(
    r"""(?ix)
    \b(
        DROP\s+(TABLE|VIEW|INDEX|DATABASE|SCHEMA|TRIGGER|PROCEDURE|FUNCTION)|
        DELETE\s+FROM|
        UPDATE\s+\w+|
        ALTER\s+(TABLE|VIEW|DATABASE|SCHEMA)|
        TRUNCATE\s+(TABLE)?|
        INSERT\s+INTO|
        CREATE\s+(TABLE|VIEW|INDEX|DATABASE|SCHEMA|TRIGGER|PROCEDURE|FUNCTION)|
        REPLACE\s+INTO|
        MERGE\s+INTO|
        GRANT\s+|
        REVOKE\s+|
        EXEC(?:UTE)?\s+|
        CALL\s+\w+|
        SELECT\s+.+\s+INTO\s+|
        ATTACH\s+|
        DETACH\s+|
        VACUUM\b|
        REINDEX\b|
        LOAD\s+DATA|
        COPY\s+\w+
    )
    """,
)

_COMMENT_RE = re.compile(r"(--.*?$)|(/\*.*?\*/)", re.MULTILINE | re.DOTALL)
_MULTI_STMT_RE = re.compile(r";\s*\S")

ALLOWED_ROOT_TYPES = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.With,
    exp.Subquery,
)

FORBIDDEN_NODE_TYPES = (
    exp.Delete,
    exp.Update,
    exp.Insert,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Grant,
    exp.Set,
)


DIALECT_MAP = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
}


@dataclass
class ValidationResult:
    """Outcome of SQL security / structural validation."""

    is_safe: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    statement_type: str = ""

    def raise_if_unsafe(self) -> None:
        if not self.is_safe:
            raise SQLSecurityError("; ".join(self.errors) or "Unsafe SQL")


class SQLSecurityGuard:
    """Deterministic SQL validator using sqlglot AST + regex safety net."""

    def __init__(
        self,
        *,
        read_only: bool = True,
        max_rows: int = 1000,
        dialect: str = "sqlite",
        known_tables: set[str] | None = None,
        known_columns: dict[str, set[str]] | None = None,
    ) -> None:
        self.read_only = read_only
        self.max_rows = max_rows
        self.dialect = DIALECT_MAP.get(dialect.lower(), dialect.lower())
        self.known_tables = {t.lower() for t in (known_tables or set())}
        self.known_columns = {
            t.lower(): {c.lower() for c in cols}
            for t, cols in (known_columns or {}).items()
        }

    def strip_comments(self, sql: str) -> str:
        return _COMMENT_RE.sub(" ", sql).strip()

    def normalize(self, sql: str) -> str:
        cleaned = self.strip_comments(sql)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()
        return cleaned

    def validate(self, sql: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        tables: list[str] = []
        columns: list[str] = []
        joins: list[str] = []
        statement_type = ""

        if not sql or not sql.strip():
            return ValidationResult(False, sql or "", ["Empty SQL statement"])

        cleaned = self.normalize(sql)

        # Injection: multi-statement
        if _MULTI_STMT_RE.search(cleaned + " "):
            errors.append("Multiple SQL statements are not allowed")

        # Regex safety net (defense in depth)
        if self.read_only:
            match = FORBIDDEN_STATEMENT_RE.search(cleaned)
            if match:
                errors.append(
                    f"Forbidden statement detected: {match.group(0).strip()[:48]}"
                )

        # Primary: sqlglot AST
        try:
            expressions = sqlglot.parse(cleaned, read=self.dialect)
        except ParseError as exc:
            return ValidationResult(
                False,
                cleaned,
                errors + [f"SQL syntax error: {exc}"],
                warnings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlglot parse failed, regex-only fallback: %s", exc)
            expressions = []

        if not expressions:
            if not errors:
                errors.append("Unable to parse SQL")
            return ValidationResult(False, cleaned, errors, warnings)

        if len(expressions) > 1:
            errors.append("Multiple SQL statements are not allowed")

        tree = expressions[0]
        if tree is None:
            errors.append("Empty parsed SQL")
            return ValidationResult(False, cleaned, errors, warnings)

        statement_type = type(tree).__name__
        root = tree
        if isinstance(tree, exp.With):
            root = tree.this
        if isinstance(root, exp.Subquery):
            root = root.this

        if self.read_only:
            if not isinstance(root, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
                # Allow SHOW/DESCRIBE-like commands that parse as Command in some dialects
                if isinstance(tree, exp.Command):
                    cmd = (tree.name or "").upper()
                    if cmd not in {"SHOW", "DESCRIBE", "DESC", "EXPLAIN"}:
                        errors.append(f"Only read queries are allowed (got {statement_type})")
                elif not isinstance(tree, exp.Select):
                    # EXPLAIN wraps differently across dialects — check string start
                    first = cleaned.split(None, 1)[0].upper()
                    if first not in {"SELECT", "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "DESC"}:
                        errors.append(
                            f"Only read queries are allowed (got {statement_type})"
                        )

            for node in tree.walk():
                if isinstance(node, FORBIDDEN_NODE_TYPES):
                    errors.append(
                        f"Forbidden AST node: {type(node).__name__}"
                    )

        # Extract tables / columns / joins from AST
        for table in tree.find_all(exp.Table):
            name = (table.name or "").lower()
            if name:
                tables.append(name)
        for col in tree.find_all(exp.Column):
            name = (col.name or "").lower()
            if name and name != "*":
                columns.append(name)
        for join in tree.find_all(exp.Join):
            joins.append(join.sql(dialect=self.dialect))

        tables_u = list(dict.fromkeys(tables))
        columns_u = list(dict.fromkeys(columns))

        if self.known_tables and tables_u:
            unknown = set(tables_u) - self.known_tables
            # Ignore CTE names
            cte_names = {
                (c.alias_or_name or "").lower()
                for c in tree.find_all(exp.CTE)
            }
            unknown -= cte_names
            if unknown:
                errors.append(f"Unknown table(s): {', '.join(sorted(unknown))}")

        if self.known_columns and columns_u:
            # Soft check: column must exist in at least one known table (or be *)
            all_cols = {c for cols in self.known_columns.values() for c in cols}
            # Common aliases / expressions may not match — warn only for obvious misses
            suspicious = [
                c for c in columns_u
                if c not in all_cols and not c.isdigit()
            ]
            if suspicious and len(suspicious) == len(columns_u):
                warnings.append(
                    f"Columns not found in schema catalog: {', '.join(suspicious[:8])}"
                )

        # Warnings from AST
        if tree.find(exp.Star):
            warnings.append("SELECT * may be inefficient; prefer explicit columns")

        has_limit = tree.find(exp.Limit) is not None
        if not has_limit and isinstance(root, exp.Select):
            warnings.append(f"No LIMIT clause; max_rows={self.max_rows} will be applied")
            cleaned = self.enforce_limit(cleaned)

        return ValidationResult(
            is_safe=len(errors) == 0,
            sql=cleaned,
            errors=list(dict.fromkeys(errors)),
            warnings=list(dict.fromkeys(warnings)),
            tables=tables_u,
            columns=columns_u,
            joins=joins[:20],
            statement_type=statement_type,
        )

    def enforce_limit(self, sql: str) -> str:
        """Append LIMIT via sqlglot when possible; string fallback otherwise."""
        try:
            tree = sqlglot.parse_one(sql, read=self.dialect)
            if tree.find(exp.Limit) is None:
                tree = tree.limit(self.max_rows)
                return tree.sql(dialect=self.dialect)
            return sql
        except Exception:  # noqa: BLE001
            if "LIMIT" in sql.upper():
                return sql
            return f"{sql.rstrip()} LIMIT {self.max_rows}"
