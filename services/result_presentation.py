"""Presentation-layer helpers for enterprise analytics UX.

Formats final answers, friendly column labels, data overviews, and empty-state
copy. Does not change LangGraph, agents, or SQL generation.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Weak finalize / fallback strings that should never be shown alone
_WEAK_ANSWERS = frozenset(
    {
        "done.",
        "done",
        "analysis complete.",
        "analysis complete",
        "ok",
        "success",
        "",
    }
)

_ACRONYMS = frozenset(
    {
        "id",
        "sku",
        "url",
        "api",
        "gpa",
        "cgpa",
        "qty",
        "avg",
        "sum",
        "min",
        "max",
        "pdf",
        "csv",
        "sql",
        "kpi",
        "usd",
        "eur",
        "pkr",
    }
)

_SPECIAL_LABELS: dict[str, str] = {
    "email": "Email",
    "e_mail": "Email",
    "email_address": "Email",
    "mail": "Email",
    "phone": "Phone Number",
    "phone_number": "Phone Number",
    "mobile": "Mobile Number",
    "mobile_no": "Mobile Number",
    "mobile_number": "Mobile Number",
    "telephone": "Phone Number",
    "cell": "Mobile Number",
    "username": "Username",
    "user_name": "Username",
    "qty": "Quantity",
    "quantity": "Quantity",
    "order_qty": "Order Quantity",
    "ordered_quantity": "Order Quantity",
    "product_name": "Product",
    "item_name": "Product",
    "city_name": "City",
    "country_name": "Country",
    "created_at": "Created At",
    "updated_at": "Updated At",
    "order_date": "Order Date",
    "total_orders": "Total Orders",
    "invoice_total": "Invoice Total",
    "sales_amount": "Sales Amount",
    "unit_price": "Unit Price",
    "total_amount": "Total Amount",
    "total_price": "Total Price",
    "payment_method": "Payment Method",
    "roll_no": "Roll Number",
    "roll_number": "Roll Number",
    "emp_id": "Employee ID",
    "employee_id": "Employee ID",
    "student_id": "Student ID",
}

_ENTITY_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("Customer", ("customer", "client", "buyer")),
    ("Employee", ("employee", "staff", "worker")),
    ("Student", ("student", "learner", "pupil")),
    ("User", ("user", "username", "account")),
    ("Supplier", ("supplier",)),
    ("Vendor", ("vendor",)),
    ("Doctor", ("doctor", "physician")),
    ("Patient", ("patient",)),
    ("Teacher", ("teacher", "instructor")),
    ("Seller", ("seller", "salesperson", "sales_person")),
    ("Manager", ("manager",)),
    ("Agent", ("agent",)),
    ("Product", ("product", "item", "sku")),
    ("Order", ("order",)),
    ("Invoice", ("invoice",)),
    ("Course", ("course",)),
    ("Company", ("company", "organization", "organisation")),
    ("Restaurant", ("restaurant",)),
]

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def humanize_column_name(name: str) -> str:
    """Convert raw DB / SQL alias names into report-ready headers."""
    raw = str(name or "").strip()
    if not raw:
        return "Value"

    # Already Title Case with spaces — keep lightly cleaned
    if " " in raw and raw == raw.title() and "_" not in raw:
        return raw

    key = _TOKEN_RE.sub("_", raw).strip("_").lower()
    if key in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[key]

    # Split snake / kebab / camel
    spaced = _TOKEN_RE.sub(" ", raw).strip()
    spaced = _CAMEL_RE.sub(" ", spaced)
    parts = [p for p in spaced.replace("-", " ").split() if p]
    if not parts:
        return raw

    words: list[str] = []
    for part in parts:
        low = part.lower()
        if low in _ACRONYMS:
            words.append(low.upper())
        elif low in {"no", "num", "number"} and words:
            # "Roll No" → "Roll Number" when trailing
            if low in {"no", "num"}:
                words.append("Number")
            else:
                words.append("Number")
        else:
            words.append(low.capitalize())

    label = " ".join(words)
    # Soften common trailing Id
    label = re.sub(r"\bId\b", "ID", label)
    return label


def humanize_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return a display copy with friendly column headers (raw DF unchanged)."""
    if df is None:
        return None
    if df.empty and len(df.columns) == 0:
        return df.copy()
    renamed = {c: humanize_column_name(str(c)) for c in df.columns}
    # Avoid duplicate display headers
    seen: dict[str, int] = {}
    unique: dict[Any, str] = {}
    for src, label in renamed.items():
        n = seen.get(label, 0)
        seen[label] = n + 1
        unique[src] = label if n == 0 else f"{label} ({n + 1})"
    out = df.copy()
    out.columns = [unique[c] for c in df.columns]
    return out


def detect_entities_from_columns(columns: list[str] | None) -> list[str]:
    """Infer primary business entities from result column names."""
    cols = [str(c).lower() for c in (columns or [])]
    blob = " ".join(cols)
    found: list[str] = []
    for label, hints in _ENTITY_HINTS:
        if any(h in blob for h in hints):
            found.append(label)
    return found[:5]


def extract_tables_used(state: dict[str, Any] | None) -> list[str]:
    """Pull table names from validation metadata or SQL text (presentation only)."""
    state = state or {}
    meta = state.get("validation_meta") or {}
    tables = meta.get("tables") or []
    names: list[str] = []
    for t in tables:
        if isinstance(t, str) and t.strip():
            names.append(t.strip())
        elif isinstance(t, (list, tuple)) and t:
            names.append(str(t[0]))
    if names:
        return _unique_preserve(names)

    sql = state.get("sql") or ""
    if sql:
        # Lightweight FROM/JOIN scrape — display only, not a parser replacement
        for m in re.finditer(
            r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)",
            sql,
            flags=re.I,
        ):
            token = m.group(1).split(".")[-1]
            if token.lower() not in {"select", "where", "on", "as"}:
                names.append(token)
    return _unique_preserve(names)


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _title_case_table(name: str) -> str:
    return humanize_column_name(name)


def is_weak_answer(text: str | None) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t.lower() in _WEAK_ANSWERS:
        return True
    # Single-word stubs
    if len(t.split()) <= 2 and t.lower().rstrip(".") in {"done", "ok", "complete", "success"}:
        return True
    return False


def build_suggested_followups(
    *,
    question: str = "",
    columns: list[str] | None = None,
    entities: list[str] | None = None,
    empty: bool = False,
    suggested: list[str] | None = None,
) -> list[str]:
    """Business-language follow-up prompts (never SQL jargon)."""
    if suggested:
        return [str(s) for s in suggested[:5] if str(s).strip()]

    ents = entities or detect_entities_from_columns(columns)
    primary = ents[0] if ents else "records"
    followups: list[str] = []

    if empty:
        followups = [
            f"Show all {primary.lower()} without filters",
            f"What {primary.lower()} fields are available in this database?",
            "Summarize the main datasets I can explore",
        ]
    else:
        followups = [
            f"Break this down by date or time period",
            f"Show the top {primary.lower()} by key metrics",
            f"Compare these results with related categories",
        ]
        if "Customer" in ents or "Order" in ents:
            followups[1] = "Who are the top customers by order volume?"
        if "Product" in ents:
            followups.append("Which products contribute most to totals?")
        if "Student" in ents:
            followups[1] = "Show students by department or semester"

    # De-dupe and keep short
    return _unique_preserve(followups)[:4]


def build_query_summary(
    state: dict[str, Any] | None,
    *,
    question: str = "",
    df: pd.DataFrame | None = None,
) -> str:
    """Executive Query Summary in business language (markdown)."""
    state = state or {}
    row_count = int(state.get("row_count") or (0 if df is None else len(df)))
    success = bool(state.get("query_success"))
    error = (state.get("sql_error") or state.get("error") or "").strip()
    tables = extract_tables_used(state)
    table_txt = ", ".join(_title_case_table(t) for t in tables[:6]) if tables else "connected datasets"
    columns = list(df.columns) if df is not None and len(df.columns) else list(state.get("columns") or [])
    entities = detect_entities_from_columns([str(c) for c in columns])
    entity_txt = ", ".join(entities) if entities else "business records"
    friendly_cols = [humanize_column_name(str(c)) for c in columns[:8]]

    if error and not success:
        return (
            "## Query Summary\n\n"
            "The request could not be completed successfully.\n\n"
            f"**What happened:** {error[:400]}\n\n"
            "You can rephrase the question or try one of the suggestions below."
        )

    if success and row_count == 0:
        reasons = [
            "No records satisfy the current filters",
            "The selected date range may have no activity",
            "The requested relationship may not exist in the data",
            "The database currently has no matching entries",
        ]
        followups = build_suggested_followups(
            question=question,
            columns=[str(c) for c in columns],
            entities=entities,
            empty=True,
            suggested=list(state.get("suggested_questions") or [])[:3] or None,
        )
        lines = [
            "## Query Summary\n",
            "Execution completed successfully.",
            "",
            "However, **no matching records were found**.",
            "",
            "Possible reasons:",
            *[f"• {r}" for r in reasons],
            "",
            f"• Datasets checked: {table_txt}",
            f"• Status: Success (0 records)",
            "",
            "Suggested next questions:",
            *[f"• {s}" for s in followups],
        ]
        return "\n".join(lines)

    status = "Success" if success else ("Completed" if row_count else "Partial")
    intro_bits = [
        "The request was successfully understood and executed."
        if success
        else "The analysis finished with partial results."
    ]
    if tables:
        if len(tables) == 1:
            intro_bits.append(
                f"The system searched the **{_title_case_table(tables[0])}** dataset."
            )
        else:
            joined = ", ".join(_title_case_table(t) for t in tables[:4])
            intro_bits.append(
                f"The system searched and combined: **{joined}**."
            )
    if entities:
        intro_bits.append(f"Primary entities in focus: **{', '.join(entities)}**.")

    lines = [
        "## Query Summary\n",
        " ".join(intro_bits),
        "",
        "Execution completed successfully." if success else "Execution finished.",
        "",
        "Result:",
        f"• Records found: **{row_count:,}**",
        f"• Datasets used: **{table_txt}**",
        f"• Status: **{status}**",
    ]
    if friendly_cols:
        lines.append(f"• Fields returned: {', '.join(friendly_cols)}")
    if row_count > 0:
        lines.extend(["", "The complete dataset is shown below."])
    return "\n".join(lines)


def build_data_overview(
    df: pd.DataFrame | None,
    *,
    row_count: int | None = None,
    entities: list[str] | None = None,
) -> str:
    """Short overview shown above the results table."""
    if df is None:
        return ""
    n = int(row_count if row_count is not None else len(df))
    cols = [humanize_column_name(str(c)) for c in df.columns]
    ents = entities or detect_entities_from_columns([str(c) for c in df.columns])

    if n == 0:
        return (
            "### Data Overview\n\n"
            "No matching records were found for this request.\n\n"
            "Execution succeeded, but nothing in the database matched your criteria."
        )

    subject = ents[0].lower() + " records" if len(ents) == 1 else (
        " and ".join(e.lower() for e in ents[:2]) + " records" if ents else "records"
    )
    # Prefer "customer orders" style when Order + Customer
    if "Order" in (ents or []) and "Customer" in (ents or []):
        subject = "customer orders"
    elif "Product" in (ents or []) and n:
        subject = "product records"

    lines = [
        "### Data Overview\n",
        f"Retrieved **{n:,}** {subject}.",
        "",
        "The dataset includes:",
        *[f"• {c}" for c in cols[:10]],
    ]
    if len(cols) > 10:
        lines.append(f"• …and {len(cols) - 10} more fields")
    return "\n".join(lines)


def build_result_cards(state: dict[str, Any] | None, df: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """Compact metric cards for the answer header."""
    state = state or {}
    rows = int(state.get("row_count") or (0 if df is None else len(df)))
    tables = extract_tables_used(state)
    duration = float(state.get("execution_time") or state.get("_wall_time") or 0)
    success = bool(state.get("query_success"))
    has_sql = bool(state.get("sql"))
    chart = (state.get("chart_type") or "").strip().lower()
    insights = bool((state.get("insights") or "").strip()) or bool(state.get("insight_structured"))

    from utils.helpers import format_duration

    status = "Success" if success else ("Failed" if state.get("error") or state.get("sql_error") else "—")
    chart_label = "Yes" if chart and chart not in {"none", "table", ""} else "No"
    if chart and chart not in {"none", "table", ""}:
        chart_label = chart.replace("_", " ").title()

    cards: list[tuple[str, str]] = [
        ("Records Found", f"{rows:,}"),
        ("Datasets Used", str(len(tables)) if tables else ("1" if has_sql else "—")),
        ("Response Time", format_duration(duration) if duration else "—"),
        ("Status", status),
        ("Analysis Query", "Ready" if has_sql else "—"),
        ("Visual", chart_label),
        ("Insights", "Ready" if insights else "—"),
    ]
    return cards


def compose_executive_answer(
    state: dict[str, Any] | None,
    *,
    question: str = "",
    df: pd.DataFrame | None = None,
    response_text: str = "",
    executive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge LLM insights with a guaranteed Query Summary for portfolio-quality UX."""
    state = state or {}
    executive = dict(executive or {})
    summary = (executive.get("summary") or "").strip()
    response = (response_text or "").strip()

    query_summary = build_query_summary(state, question=question, df=df)
    row_count = int(state.get("row_count") or (0 if df is None else len(df)))

    # Prefer rich insight summary when present; always prepend Query Summary if weak/missing
    if is_weak_answer(summary) and is_weak_answer(response):
        hero = query_summary
        bullets = list(executive.get("bullets") or [])
    elif is_weak_answer(summary):
        # Keep insight body from response if useful
        hero = query_summary
        if response and not is_weak_answer(response) and "## Query Summary" not in response:
            executive["analysis_extra"] = "\n\n".join(
                p for p in [executive.get("analysis_extra") or "", response] if p
            ).strip()
        bullets = list(executive.get("bullets") or [])
    else:
        # Strong insight — lead with Query Summary, then insight
        if "## Query Summary" not in summary:
            hero = f"{query_summary}\n\n### Key Insight\n\n{summary}"
        else:
            hero = summary
        bullets = list(executive.get("bullets") or [])

    columns = list(df.columns) if df is not None else list(state.get("columns") or [])
    entities = detect_entities_from_columns([str(c) for c in columns])
    overview = build_data_overview(df, row_count=row_count, entities=entities) if df is not None else ""

    followups = build_suggested_followups(
        question=question,
        columns=[str(c) for c in columns],
        entities=entities,
        empty=row_count == 0 and bool(state.get("query_success")),
        suggested=list(state.get("suggested_questions") or [])[:3] or None,
    )

    # Soften SQL jargon in analysis extras
    extra = executive.get("analysis_extra") or ""
    extra = extra.replace("**SQL optimization**", "**Performance notes**")
    extra = re.sub(r"\bprimary key\b", "unique identifier", extra, flags=re.I)
    extra = re.sub(r"\bforeign key\b", "related reference", extra, flags=re.I)
    extra = re.sub(r"\btuples?\b", "records", extra, flags=re.I)
    extra = re.sub(r"\bcursor\b", "result set", extra, flags=re.I)

    return {
        "summary": hero,
        "bullets": bullets[:8],
        "trends": list(executive.get("trends") or []),
        "anomalies": list(executive.get("anomalies") or []),
        "recommendations": list(executive.get("recommendations") or []),
        "analysis_extra": extra,
        "data_overview": overview,
        "followups": followups,
        "entities": entities,
        "query_summary": query_summary,
        "result_cards": build_result_cards(state, df),
        "empty_result": bool(state.get("query_success")) and row_count == 0,
    }
