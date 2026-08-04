"""Dynamic semantic column detection for business-readable SQL.

Scores columns that already exist on each table — never invents names,
never assumes a fixed schema (customer_name / product_name may or may not exist).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from database.schema_inspector import SchemaSnapshot, TableInfo

# Pattern vocabulary only — matched against whatever columns the DB actually has.
# Priority: lower rank = preferred for SELECT / JOIN display.
_SEMANTIC_PATTERNS: list[tuple[str, int, tuple[str, ...]]] = [
    # Person / entity labels
    (
        "person",
        10,
        (
            "full_name",
            "displayname",
            "display_name",
            "username",
            "user_name",
            "first_name",
            "lastname",
            "last_name",
            "firstname",
            "student_name",
            "employee_name",
            "staff_name",
            "teacher_name",
            "doctor_name",
            "vendor_name",
            "supplier_name",
            "customer_name",
            "client_name",
            "owner_name",
            "patient_name",
            "member_name",
            "contact_name",
            "person_name",
            "seller_name",
            "manager_name",
            "agent_name",
            "salesperson",
            "sales_person",
            "name",
        ),
    ),
    # Titles / product labels
    (
        "title",
        20,
        (
            "product_name",
            "item_name",
            "course_name",
            "book_title",
            "job_title",
            "position",
            "designation",
            "specialization",
            "speciality",
            "specialty",
            "title",
            "model",
            "sku",
            "brand",
            "category",
            "subject",
            "label",
            "store_name",
            "store",
        ),
    ),
    # Organization / location
    (
        "org",
        30,
        (
            "company_name",
            "company",
            "organization",
            "organisation",
            "department",
            "dept_name",
            "branch",
            "school_name",
            "college_name",
            "institution",
            "team_name",
            "office",
            "city",
            "country",
            "region",
            "state",
            "province",
            "address",
            "semester",
            "section",
            "class",
            "grade_level",
            "campus",
        ),
    ),
    # Identity contact
    (
        "contact",
        40,
        (
            "email",
            "email_address",
            "e_mail",
            "mail",
            "phone",
            "phone_number",
            "mobile",
            "mobile_number",
            "contact",
            "contact_number",
            "telephone",
            "cell",
            "whatsapp",
        ),
    ),
    # Business / document identifiers (prefer over opaque surrogate keys)
    (
        "business_id",
        50,
        (
            "order_number",
            "order_no",
            "invoice_number",
            "invoice_no",
            "registration_no",
            "registration_number",
            "roll_number",
            "roll_no",
            "employee_code",
            "employee_id",
            "emp_code",
            "emp_id",
            "student_code",
            "student_id",
            "sku",
            "code",
            "ref_no",
            "reference_no",
        ),
    ),
    # Dates
    (
        "date",
        60,
        (
            "created_at",
            "updated_at",
            "order_date",
            "purchase_date",
            "sale_date",
            "appointment_date",
            "enrollment_date",
            "joining_date",
            "hire_date",
            "start_date",
            "end_date",
            "payment_date",
            "shipped_date",
            "delivery_date",
            "dob",
            "birth_date",
            "date",
        ),
    ),
    # Metrics / business attributes
    (
        "metric",
        70,
        (
            "quantity",
            "qty",
            "ordered_quantity",
            "units",
            "unit_price",
            "price",
            "list_price",
            "sale_price",
            "salary",
            "revenue",
            "total",
            "total_price",
            "total_amount",
            "subtotal",
            "grand_total",
            "amount",
            "spend",
            "stock",
            "stock_quantity",
            "stock_qty",
            "quantity_in_stock",
            "inventory",
            "units_in_stock",
            "score",
            "marks",
            "cgpa",
            "gpa",
            "rating",
            "status",
            "order_status",
            "payment_status",
            "payment_method",
            "payment_type",
            "grade",
            "count",
            "discount",
        ),
    ),
    # Internal IDs — last resort
    ("internal_id", 90, ("uuid", "guid")),
]

# Aggregate / metric-looking result headers (normalized substrings)
_AGGREGATE_HINTS = frozenset(
    {
        "count",
        "total",
        "sum",
        "avg",
        "average",
        "mean",
        "max",
        "min",
        "orders",
        "total_orders",
        "order_count",
        "num_orders",
        "quantity",
        "qty",
        "revenue",
        "amount",
        "sales",
        "visits",
        "courses",
        "units_sold",
        "row_count",
        "n",
        "cnt",
    }
)

# User asked for a scalar / aggregate-only answer — do not force enrichment
_SCALAR_ONLY_RE = re.compile(
    r"\b("
    r"only\s+(total|count|average|avg|sum|revenue|number|ids?|id)|"
    r"just\s+(the\s+)?(total|count|average|avg|sum|number)|"
    r"how\s+many\b|"
    r"what\s+is\s+the\s+(total|count|average|avg|sum|number)\b|"
    r"give\s+me\s+(only\s+)?(the\s+)?(total|count|average|avg|sum)\b"
    r")\b",
    re.I,
)

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RankedColumn:
    name: str
    kind: str
    rank: int
    score: int


@dataclass(frozen=True)
class BIQualityAssessment:
    """Whether a result set looks thin for enterprise BI presentation."""

    thin: bool
    reason: str
    improvement_hint: str
    column_count: int
    has_aggregate: bool
    scalar_request: bool


def _normalize(name: str) -> str:
    return _TOKEN_RE.sub("_", (name or "").strip().lower()).strip("_")


def _tokens(name: str) -> set[str]:
    n = _normalize(name)
    parts = {p for p in n.split("_") if p}
    parts.add(n)
    return parts


def _is_likely_surrogate_id(col: str, *, primary_key: bool = False) -> bool:
    n = _normalize(col)
    if n in {"id", "pk"}:
        return True
    if n.endswith("_id") or n.endswith("_pk") or n.endswith("_fk"):
        return True
    if primary_key and n.endswith("id"):
        return True
    return False


def _looks_like_aggregate_header(col: str) -> bool:
    n = _normalize(col)
    toks = _tokens(col)
    if n in _AGGREGATE_HINTS or toks & _AGGREGATE_HINTS:
        return True
    for prefix in ("total_", "sum_", "avg_", "count_", "num_", "n_"):
        if n.startswith(prefix):
            return True
    return False


def score_column(name: str, *, primary_key: bool = False) -> RankedColumn | None:
    """Return best semantic rank for an existing column, or None if uninteresting."""
    n = _normalize(name)
    toks = _tokens(name)

    if _is_likely_surrogate_id(name, primary_key=primary_key):
        # Keep bare IDs only as lowest priority when nothing else exists
        # Exception: business-style *employee_id* / *student_id* may still match
        # business_id patterns below when not treated as pure surrogate — but
        # _is_likely_surrogate_id catches *_id. Prefer descriptive labels first.
        biz_override = None
        for kind, rank, patterns in _SEMANTIC_PATTERNS:
            if kind != "business_id":
                continue
            for pat in patterns:
                p = _normalize(pat)
                if n == p:
                    biz_override = RankedColumn(
                        name=name, kind=kind, rank=rank, score=100 - rank + min(len(p), 24)
                    )
                    break
            if biz_override:
                break
        if biz_override:
            return biz_override
        return RankedColumn(name=name, kind="internal_id", rank=95, score=1)

    best: RankedColumn | None = None
    for kind, rank, patterns in _SEMANTIC_PATTERNS:
        for pat in patterns:
            p = _normalize(pat)
            hit = False
            # Exact or suffix / contains — still only if the column exists on the table
            if n == p or n.endswith("_" + p) or p in toks:
                hit = True
            elif len(p) >= 4 and (p in n or n in p):
                hit = True
            if not hit:
                continue
            # Prefer exact / longer matches so email_address beats token "address"
            score = 100 - rank + min(len(p), 24)
            if n == p:
                score += 50
            elif n.endswith("_" + p):
                score += 20
            elif p in toks and len(p) >= 6:
                score += 10
            elif p in toks:
                score += 2
            cand = RankedColumn(name=name, kind=kind, rank=rank, score=score)
            if best is None or cand.score > best.score or (
                cand.score == best.score and cand.rank < best.rank
            ):
                best = cand
    return best


def rank_table_columns(
    columns: Iterable[tuple[str, bool]],
    *,
    limit: int = 8,
) -> list[RankedColumn]:
    """columns: iterable of (name, is_primary_key)."""
    ranked: list[RankedColumn] = []
    seen: set[str] = set()
    for name, is_pk in columns:
        hit = score_column(name, primary_key=is_pk)
        if hit is None or hit.name.lower() in seen:
            continue
        seen.add(hit.name.lower())
        ranked.append(hit)

    # Prefer non-ID semantics first; drop surrogate IDs if we already have labels
    descriptive = [c for c in ranked if c.kind != "internal_id"]
    ids = [c for c in ranked if c.kind == "internal_id"]
    descriptive.sort(key=lambda c: (c.rank, -c.score, c.name.lower()))
    chosen = descriptive[:limit]
    if not chosen:
        ids.sort(key=lambda c: c.name.lower())
        chosen = ids[: min(2, limit)]
    return chosen


def preferred_columns_for_table(table: "TableInfo", *, limit: int = 8) -> list[RankedColumn]:
    cols = [(c.name, bool(c.primary_key)) for c in (table.columns or [])]
    return rank_table_columns(cols, limit=limit)


def _label_columns_for_table(table: "TableInfo", *, limit: int = 4) -> list[RankedColumn]:
    """Best display columns when joining TO this table (name/contact/org first)."""
    prefs = preferred_columns_for_table(table, limit=limit + 2)
    preferred_kinds = {"person", "title", "contact", "org", "business_id", "date", "metric"}
    chosen = [c for c in prefs if c.kind in preferred_kinds][:limit]
    return chosen or prefs[:limit]


def build_join_enrichment_hints(
    schema: "SchemaSnapshot | None",
    *,
    max_edges: int = 24,
) -> str:
    """Concrete FK → label SELECT suggestions from THIS schema only."""
    if schema is None or not getattr(schema, "tables", None):
        return ""

    by_name = {t.name: t for t in schema.tables if t.name}
    lines: list[str] = [
        "Join enrichment map (use only columns listed — never invent):",
        "When SELECTing an FK, JOIN the referred table and project its preferred labels.",
    ]
    edges = 0
    for table in schema.tables:
        for fk in table.foreign_keys or []:
            if edges >= max_edges:
                break
            ref_name = fk.referred_table
            ref = by_name.get(ref_name)
            if ref is None:
                continue
            labels = _label_columns_for_table(ref, limit=4)
            if not labels:
                continue
            fk_cols = ", ".join(fk.constrained_columns)
            label_txt = ", ".join(f"{ref_name}.{c.name} [{c.kind}]" for c in labels)
            lines.append(
                f"- {table.name}.({fk_cols}) → {ref_name}: prefer SELECT {label_txt}"
            )
            edges += 1
        if edges >= max_edges:
            break

    if edges == 0:
        return ""
    return "\n".join(lines)


def build_bi_enrichment_checklist(schema_dict: dict | None = None) -> str:
    """Short checklist injected into the SQL ReAct human message."""
    base = [
        "BI enrichment checklist (schema-aware — only use real columns/FKs):",
        "1. Target 4–8 high-value columns when the schema allows (not just Name + COUNT).",
        "2. For entities (customer/student/employee/user/supplier/doctor/…): include name + "
        "identifier and/or contact (email/phone) + one business attribute when present.",
        "3. For orders/sales: join related entity + product/item tables via FKs; include "
        "qty/price/total/date/status when those columns exist.",
        "4. For products: prefer name, category/brand, price, stock, supplier label over Count alone.",
        "5. Aggregate rankings (top customers / most orders): keep the metric AND enrich with "
        "contact, last activity date, spend/revenue, category — if available via joins.",
        "6. Respect explicit scalar asks (only total/count/avg/sum/ids) — do not over-enrich those.",
        "7. Avoid duplicate rows and unnecessary joins; use DISTINCT or proper GROUP BY.",
        "8. Before submit_sql: would a business user / recruiter find this table useful? If not, rewrite.",
    ]
    join_block = ""
    if schema_dict:
        snap = _schema_from_dict(schema_dict)
        if snap is not None:
            join_block = build_join_enrichment_hints(snap)
    if join_block:
        return "\n".join(base) + "\n\n" + join_block
    return "\n".join(base)


def build_business_column_hints(
    schema: "SchemaSnapshot | None",
    *,
    max_tables: int = 40,
) -> str:
    """Compact hint block for SQL agent prompts (empty if no schema)."""
    if schema is None or not getattr(schema, "tables", None):
        return ""

    lines: list[str] = [
        "Business column preferences (detected from THIS schema only — do not invent columns):",
        "Priority: human-readable name → title/username → email → phone → business id → "
        "org/location → date → metric → internal id (last resort).",
        "When JOINing on FKs, SELECT descriptive columns from the referred table instead of bare *_id.",
        "Prefer 4–8 meaningful columns for business users — never return only Name + COUNT/SUM "
        "unless the user explicitly asked for only a number.",
        "Use clear AS aliases (e.g. Customer Name, Total Orders) for report-ready headers.",
        "Aggregates must include entity context (contact, category, dates, spend) when present.",
    ]
    for table in list(schema.tables)[:max_tables]:
        prefs = preferred_columns_for_table(table, limit=7)
        if not prefs:
            continue
        parts = [f"{c.name} [{c.kind}]" for c in prefs]
        fk_notes: list[str] = []
        for fk in table.foreign_keys or []:
            ref = fk.referred_table
            cols = ", ".join(fk.constrained_columns)
            fk_notes.append(f"FK ({cols})→{ref}")
        extra = f" · {'; '.join(fk_notes)}" if fk_notes else ""
        lines.append(f"- {table.name}: {', '.join(parts)}{extra}")

    join_hints = build_join_enrichment_hints(schema)
    if join_hints:
        lines.append("")
        lines.append(join_hints)
    return "\n".join(lines)


def is_scalar_only_request(question: str) -> bool:
    """True when the user explicitly wants a single number / ids only."""
    q = (question or "").strip()
    if not q:
        return False
    return bool(_SCALAR_ONLY_RE.search(q))


def assess_result_bi_quality(
    columns: list[str] | None,
    question: str = "",
    *,
    schema_dict: dict | None = None,
    row_count: int | None = None,
) -> BIQualityAssessment:
    """Detect thin Name+COUNT style results that should trigger SQL regeneration.

    Never invents columns — only signals that richer schema context likely exists.
    """
    cols = [str(c) for c in (columns or []) if str(c).strip()]
    n = len(cols)
    scalar = is_scalar_only_request(question)
    if scalar:
        return BIQualityAssessment(
            thin=False,
            reason="User asked for a scalar/aggregate-only answer",
            improvement_hint="",
            column_count=n,
            has_aggregate=any(_looks_like_aggregate_header(c) for c in cols),
            scalar_request=True,
        )

    has_agg = any(_looks_like_aggregate_header(c) for c in cols)
    id_heavy = sum(1 for c in cols if _is_likely_surrogate_id(c)) >= max(1, n - 1) and n <= 3

    # Schema suggests descriptive columns exist somewhere
    schema_has_labels = False
    if schema_dict:
        snap = _schema_from_dict(schema_dict)
        if snap is not None:
            for table in snap.tables:
                prefs = preferred_columns_for_table(table, limit=3)
                if any(c.kind in {"person", "title", "contact", "org"} for c in prefs):
                    schema_has_labels = True
                    break

    thin = False
    reason = ""
    hint = ""

    if n == 0 and (row_count or 0) == 0:
        thin = False
        reason = "Empty result — may be correct"
    elif n <= 2 and has_agg and schema_has_labels:
        thin = True
        reason = (
            f"Result has only {n} column(s) including an aggregate while the schema "
            "has richer entity/contact/product attributes"
        )
        hint = (
            "Regenerate SQL as a business-ready report: keep the metric, JOIN related "
            "tables via FKs, and SELECT available name/email/phone/category/date/status "
            "columns (4–8 high-value columns). Do not invent columns."
        )
    elif n <= 2 and schema_has_labels and not has_agg:
        # Possible ranking without context, or ID-only
        thin = True
        reason = (
            f"Result has only {n} column(s); schema has richer descriptive columns available"
        )
        hint = (
            "Enrich the SELECT with identifying and contextual columns from related "
            "tables (name, contact, business attributes, dates) using existing FKs."
        )
    elif id_heavy and schema_has_labels:
        thin = True
        reason = "Result is dominated by surrogate keys while descriptive columns exist"
        hint = (
            "Replace bare *_id columns with JOINed human-readable labels "
            "(name/title/email) from referred tables."
        )

    return BIQualityAssessment(
        thin=thin,
        reason=reason,
        improvement_hint=hint,
        column_count=n,
        has_aggregate=has_agg,
        scalar_request=False,
    )


def resolve_schema_hints(
    schema: "SchemaSnapshot | None" = None,
    schema_text: str = "",
) -> str:
    """Prefer live SchemaSnapshot; otherwise leave empty (LLM still has schema_text)."""
    if schema is not None:
        return build_business_column_hints(schema)
    return ""


def _schema_from_dict(schema_dict: dict | None) -> Any:
    if not schema_dict or not isinstance(schema_dict, dict):
        return None
    try:
        from database.schema_inspector import (
            ColumnInfo,
            ForeignKeyInfo,
            SchemaSnapshot,
            TableInfo,
        )
    except Exception:  # noqa: BLE001
        return None

    tables: list = []
    for t in schema_dict.get("tables") or []:
        if not isinstance(t, dict):
            continue
        tables.append(
            TableInfo(
                name=str(t.get("name") or ""),
                columns=[
                    ColumnInfo(
                        name=str(c.get("name") or ""),
                        data_type=str(c.get("data_type") or ""),
                        primary_key=bool(c.get("primary_key")),
                    )
                    for c in (t.get("columns") or [])
                    if isinstance(c, dict)
                ],
                foreign_keys=[
                    ForeignKeyInfo(
                        constrained_columns=list(fk.get("constrained_columns") or []),
                        referred_table=str(fk.get("referred_table") or ""),
                        referred_columns=list(fk.get("referred_columns") or []),
                    )
                    for fk in (t.get("foreign_keys") or [])
                    if isinstance(fk, dict)
                ],
            )
        )
    if not tables:
        return None
    return SchemaSnapshot(
        dialect=str(schema_dict.get("dialect") or "sqlite"),
        database_name=str(schema_dict.get("database_name") or ""),
        tables=tables,
    )


def build_hints_from_schema_dict(schema_dict: dict | None) -> str:
    """Rebuild hints from GraphState.schema_dict (asdict of SchemaSnapshot)."""
    snap = _schema_from_dict(schema_dict)
    if snap is None:
        return ""
    return build_business_column_hints(snap)


def ensure_schema_text_with_hints(schema_text: str, schema_dict: dict | None = None) -> str:
    """Append dynamic business-column hints when missing from cached schema_text."""
    block = schema_text or ""
    # Refresh if older cache lacks join enrichment map
    has_prefs = "Business column preferences" in block
    has_joins = "Join enrichment map" in block
    if has_prefs and has_joins:
        return block
    hints = build_hints_from_schema_dict(schema_dict)
    if not hints:
        return block or "(call schema_tool)"
    if not block or block.strip() in {"(call schema_tool)", "(none)"}:
        return hints
    if has_prefs and not has_joins:
        join_only = ""
        snap = _schema_from_dict(schema_dict)
        if snap is not None:
            join_only = build_join_enrichment_hints(snap)
        if join_only:
            return f"{block}\n\n{join_only}"
        return block
    return f"{block}\n\n{hints}"
