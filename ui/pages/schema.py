"""Schema Explorer — enterprise database documentation (presentation only)."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from config.settings import Settings
from models.structured import DatabaseSummaryModel
from prompts.templates import summary_prompt
from services.llm_service import LLMService
from ui.components import (
    empty_state,
    health_pills,
    kpi_grid,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.ui_helpers import esc, inject_html


def _schema_stats(schema) -> dict[str, int]:
    tables = list(schema.tables or [])
    columns = sum(len(t.columns) for t in tables)
    pks = sum(len(t.primary_keys or []) for t in tables)
    fks = sum(len(t.foreign_keys or []) for t in tables)
    indexes = sum(len(t.indexes or []) for t in tables)
    relationships = len(schema.relationships or [])
    if relationships == 0:
        relationships = fks
    return {
        "tables": len(tables),
        "columns": columns,
        "relationships": relationships,
        "indexes": indexes,
        "primary_keys": pks,
        "foreign_keys": fks,
        "views": 0,
        "triggers": 0,
        "stored_objects": 0,
    }


def render_schema(settings: Settings) -> None:
    page_header(
        "Schema",
        "Tables, keys, relationships, and AI documentation",
        settings=settings,
    )

    if not st.session_state.get("connected"):
        empty_state(
            "Connect a database",
            "Schema documentation and statistics appear after you connect.",
            icon="SC",
        )
        page_footer(settings.app_version)
        return

    schema = st.session_state.schema
    stats = _schema_stats(schema)
    est_rows = sum((t.row_estimate or 0) for t in schema.tables)

    health_pills(
        [
            ("Database", schema.database_name, "ok"),
            ("Dialect", str(schema.dialect).upper(), "accent"),
            ("Tables", str(stats["tables"]), "ok"),
            ("Relationships", str(stats["relationships"]), "info"),
        ]
    )

    tab_overview, tab_catalog, tab_docs, tab_raw = st.tabs(
        [
            "Overview",
            f"Catalog · {stats['tables']}",
            "Documentation",
            "Schema text",
        ]
    )

    with tab_overview:
        section_title("Database", "Connected source")
        inject_html(
            f"""
            <div class="sq-card sq-card--accent">
              <div class="sq-card__title">Connected database</div>
              <div class="sq-card__heading">{esc(schema.database_name)}</div>
              <div class="sq-kv-grid">
                <div><span class="sq-kv-label">Dialect</span><strong>{esc(schema.dialect)}</strong></div>
                <div><span class="sq-kv-label">Tables</span><strong>{esc(stats["tables"])}</strong></div>
                <div><span class="sq-kv-label">Columns</span><strong>{esc(stats["columns"])}</strong></div>
                <div><span class="sq-kv-label">Est. rows</span><strong>{esc(f"{est_rows:,}")}</strong></div>
              </div>
              <div class="sq-badge-row sq-badge-row--top">
                {status_badge("Read-only", kind="default")}
                {status_badge(f"{stats['primary_keys']} PK", kind="ok")}
                {status_badge(f"{stats['foreign_keys']} FK", kind="info")}
                {status_badge(f"{stats['indexes']} indexes", kind="accent")}
              </div>
            </div>
            """
        )

        section_title("Statistics", "Schema footprint")
        kpi_grid(
            [
                ("Tables", str(stats["tables"])),
                ("Columns", str(stats["columns"])),
                ("Relationships", str(stats["relationships"])),
                ("Indexes", str(stats["indexes"])),
            ]
        )
        kpi_grid(
            [
                ("Primary keys", str(stats["primary_keys"])),
                ("Foreign keys", str(stats["foreign_keys"])),
                ("Est. rows", f"{est_rows:,}"),
                ("Views", str(stats["views"])),
            ]
        )

        section_title("Relationships", "Foreign-key graph")
        rels = _collect_relationships(schema)
        if rels:
            cards_html = "".join(
                f"""
                <div class="sq-pin-card">
                  <div class="sq-pin-card__top">{status_badge("FK", kind="info")}</div>
                  <div class="sq-pin-card__title">{esc(e.get("from_table", ""))} → {esc(e.get("to_table", ""))}</div>
                  <p class="sq-pin-card__body">{esc(f"{e.get('from_column', '')} → {e.get('to_column', '')}")}</p>
                </div>
                """
                for e in rels[:40]
            )
            inject_html(f'<div class="sq-pin-list">{cards_html}</div>')
            if len(rels) > 40:
                st.caption(f"Showing 40 of {len(rels)} relationships.")
        else:
            empty_state(
                "No relationships found",
                "No foreign-key relationships were discovered for this database.",
                icon="FK",
            )

        section_title("Snapshot", "Session metadata")
        inject_html(
            f"""
            <div class="sq-card">
              <div class="sq-card__title">Loaded</div>
              <p class="sq-card__body">
                Schema for <strong>{esc(schema.database_name)}</strong> ·
                {esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}
              </p>
            </div>
            """
        )

    with tab_catalog:
        _render_catalog(schema)

    with tab_docs:
        _render_ai_summary(settings, schema)

    with tab_raw:
        section_title("Prompt schema", "Text fed to AI agents")
        inject_html(
            f"""
            <div class="sq-card">
              <div class="sq-card__title">Schema text</div>
              <div class="sq-badge-row sq-badge-row--flush">
                {status_badge(schema.dialect, kind="accent")}
                {status_badge(schema.database_name, kind="info")}
                {status_badge(f"{stats['tables']} tables", kind="ok")}
              </div>
            </div>
            """
        )
        st.code(schema.to_prompt_text(), language="text")

    page_footer(settings.app_version)


def _collect_relationships(schema) -> list[dict]:
    rels = list(schema.relationships or [])
    if rels:
        # Normalize object/dict forms
        out: list[dict] = []
        for e in rels:
            if isinstance(e, dict):
                out.append(e)
            else:
                out.append(
                    {
                        "from_table": getattr(e, "from_table", ""),
                        "from_column": getattr(e, "from_column", ""),
                        "to_table": getattr(e, "to_table", ""),
                        "to_column": getattr(e, "to_column", ""),
                    }
                )
        return out

    out = []
    for table in schema.tables:
        for fk in table.foreign_keys or []:
            for src, dst in zip(fk.constrained_columns, fk.referred_columns):
                out.append(
                    {
                        "from_table": table.name,
                        "from_column": src,
                        "to_table": fk.referred_table,
                        "to_column": dst,
                    }
                )
    return out


def _render_catalog(schema) -> None:
    section_title("Catalog", "Tables · columns · keys · indexes")
    c1, c2, c3 = st.columns([3, 1, 1])
    search = c1.text_input(
        "Search",
        placeholder="Filter tables or columns…",
        key="schema_search",
        label_visibility="collapsed",
    )
    filter_mode = c2.selectbox(
        "Filter",
        ["All tables", "With PK", "With FK", "With indexes"],
        key="schema_filter",
    )
    sort_mode = c3.selectbox("Sort", ["Name", "Columns", "Rows"], key="schema_sort")
    q = (search or "").strip().lower()

    tables = list(schema.tables)
    if filter_mode == "With PK":
        tables = [t for t in tables if t.primary_keys]
    elif filter_mode == "With FK":
        tables = [t for t in tables if t.foreign_keys]
    elif filter_mode == "With indexes":
        tables = [t for t in tables if t.indexes]
    if sort_mode == "Columns":
        tables = sorted(tables, key=lambda t: len(t.columns), reverse=True)
    elif sort_mode == "Rows":
        tables = sorted(tables, key=lambda t: t.row_estimate or 0, reverse=True)
    else:
        tables = sorted(tables, key=lambda t: t.name.lower())

    shown = 0
    for table in tables:
        hay = f"{table.name} " + " ".join(c.name for c in table.columns)
        if q and q not in hay.lower():
            continue
        shown += 1
        pk_n = len(table.primary_keys or [])
        fk_n = len(table.foreign_keys or [])
        idx_n = len(table.indexes or [])
        rows_est = table.row_estimate if table.row_estimate is not None else "?"

        inject_html(
            f"""
            <div class="sq-pin-card">
              <div class="sq-pin-card__top">
                {status_badge("Table", kind="accent")}
                {status_badge(f"{len(table.columns)} cols", kind="info")}
                {status_badge(f"PK {pk_n}", kind="ok")}
                {status_badge(f"FK {fk_n}", kind="info")}
                {status_badge(f"Idx {idx_n}", kind="default")}
                {status_badge(f"~{rows_est} rows", kind="default")}
              </div>
              <div class="sq-pin-card__title">{esc(table.name)}</div>
            </div>
            """
        )

        with st.expander(f"Columns · {table.name}", expanded=False):
            for c in table.columns:
                badges = []
                if c.primary_key or c.name in (table.primary_keys or []):
                    badges.append('<span class="sq-badge sq-badge-ok">PK</span>')
                if any(c.name in fk.constrained_columns for fk in (table.foreign_keys or [])):
                    badges.append('<span class="sq-badge sq-badge-info">FK</span>')
                badges.append(
                    '<span class="sq-badge">NULL</span>'
                    if c.nullable
                    else '<span class="sq-badge sq-badge-accent">NOT NULL</span>'
                )
                badges.append(f'<span class="sq-badge-type">{esc(c.data_type)}</span>')
                inject_html(
                    f'<div class="sq-col-row"><span>{esc(c.name)}</span>'
                    f'<span>{" ".join(badges)}</span></div>'
                )
            if table.foreign_keys:
                st.markdown("**Foreign keys**")
                for fk in table.foreign_keys:
                    st.write(
                        f"- ({', '.join(fk.constrained_columns)}) → "
                        f"{fk.referred_table}({', '.join(fk.referred_columns)})"
                    )
            if table.indexes:
                st.markdown("**Indexes**")
                for idx in table.indexes:
                    uniq = " unique" if idx.unique else ""
                    st.write(f"- {idx.name}{uniq}: {', '.join(idx.columns)}")

    if shown == 0:
        empty_state("No matching tables", "Try a different search or filter.", icon="--")
    else:
        st.caption(f"Showing {shown} table{'s' if shown != 1 else ''}")

    section_title("Views & stored objects")
    inject_html(
        """
        <div class="sq-card-grid">
          <div class="sq-card sq-card--flush">
            <div class="sq-card__title">Views</div>
            <p class="sq-card__body">
              View discovery is dialect-dependent. None were exposed by the current inspector.
            </p>
          </div>
          <div class="sq-card sq-card--flush">
            <div class="sq-card__title">Triggers &amp; procedures</div>
            <p class="sq-card__body">
              Tracked when the dialect exposes them. Catalog above covers tables, keys, and indexes.
            </p>
          </div>
        </div>
        """
    )


def _render_ai_summary(settings: Settings, schema) -> None:
    from ui.gate import is_llm_ready

    section_title("AI documentation", "Summary Agent report")
    llm_ok, llm_reason = is_llm_ready(settings)

    inject_html(
        f"""
        <div class="sq-card">
          <div class="sq-card__title">Documentation</div>
          <div class="sq-card__heading">Generate a business-oriented schema brief</div>
          <p class="sq-card__body">
            The Summary Agent reads live schema metadata and proposes purpose, key tables,
            relationships, KPIs, and follow-up questions.
          </p>
          <div class="sq-badge-row sq-badge-row--top">
            {status_badge("LLM ready" if llm_ok else "LLM setup required", kind="ok" if llm_ok else "warn")}
            {status_badge(schema.database_name, kind="info")}
          </div>
        </div>
        """
    )

    if st.button("Generate AI summary", type="primary", disabled=not llm_ok, use_container_width=True):
        llm = st.session_state.get("llm_service") or LLMService(settings)
        with st.spinner("Analyzing schema…"):
            result = llm.invoke_structured(
                summary_prompt(),
                DatabaseSummaryModel,
                {
                    "database_name": schema.database_name,
                    "dialect": schema.dialect,
                    "schema_text": schema.to_prompt_text(),
                },
            )
            st.session_state.ai_summary = result.model_dump()
            if result.suggested_questions:
                st.session_state.suggested_questions = result.suggested_questions
            st.rerun()
    if not llm_ok:
        st.caption(llm_reason)

    summary = st.session_state.get("ai_summary")
    if isinstance(summary, str) and summary.strip():
        summary = {"business_overview": summary, "database_purpose": "Overview"}
    if not isinstance(summary, dict) or not summary:
        empty_state(
            "No documentation yet",
            "Generate an AI summary to see executive findings and recommendations.",
            icon="DOC",
        )
        return

    purpose = summary.get("database_purpose") or "Database overview"
    overview = summary.get("business_overview") or ""
    important = summary.get("important_tables") or []
    relationships = summary.get("relationships_summary") or ""
    kpis = summary.get("potential_kpis") or []
    metrics = summary.get("business_metrics") or []
    questions = summary.get("suggested_questions") or []

    findings = []
    if important:
        findings.append("Core tables: " + ", ".join(important[:8]))
    if purpose:
        findings.append(f"Purpose: {purpose}")
    if relationships:
        findings.append(relationships[:280] + ("…" if len(relationships) > 280 else ""))

    recommendations = questions[:5] if questions else [
        "Explore high-value joins between core entities",
        "Track KPIs listed in Important Metrics",
    ]
    opportunities = questions[5:10] if len(questions) > 5 else (
        [f"Analyze trends on {t}" for t in important[:3]]
        if important
        else ["Identify growth drivers from fact tables"]
    )
    risks = [
        "Validate foreign-key coverage before multi-table joins",
        "Prefer aggregations with explicit filters to avoid oversized scans",
        "Treat estimated row counts as approximate until queried",
    ]

    cards = [
        ("EX", "ok", "Executive Summary", overview or purpose),
        ("KF", "accent", "Key Findings", findings),
        ("BI", "info", "Business Insights", relationships or "See Relationship Summary."),
        ("RC", "ok", "Recommendations", recommendations),
        ("RK", "warn", "Risks", risks),
        ("OP", "accent", "Opportunities", opportunities),
        ("IM", "info", "Important Metrics", metrics or kpis or ["Refresh the summary to generate metrics."]),
    ]
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, card in zip(cols, cards[i : i + 2]):
            with col:
                _report_card(*card)


def _report_card(icon: str, tone: str, title: str, body) -> None:
    tone_cls = {
        "ok": "sq-report sq-report--ok",
        "warn": "sq-report sq-report--warn",
        "accent": "sq-report sq-report--accent",
        "info": "sq-report sq-report--info",
    }.get(tone, "sq-report")
    if isinstance(body, list):
        items = "".join(f"<li>{esc(str(x))}</li>" for x in body if str(x).strip())
        body_html = (
            f"<ul class='sq-report__list'>{items}</ul>"
            if items
            else "<p class='sq-report__body'>—</p>"
        )
    else:
        body_html = f"<p class='sq-report__body'>{esc(str(body))}</p>"
    inject_html(
        f"""
        <div class="{tone_cls}">
          <div class="sq-report__head">
            <span class="sq-report__icon">{esc(icon)}</span>
            <span class="sq-report__status"></span>
            <span class="sq-report__title">{esc(title)}</span>
          </div>
          {body_html}
        </div>
        """
    )
