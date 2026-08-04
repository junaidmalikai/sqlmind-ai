"""Exports — polished multi-format download center (presentation only)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import Settings
from services.export_service import ExportService
from ui.components import (
    empty_state,
    health_pills,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.ui_helpers import esc, inject_html
from utils.helpers import format_duration

_FORMAT_META: dict[str, dict[str, str]] = {
    "csv": {
        "ext": "CSV",
        "title": "CSV",
        "body": "Raw tabular data for spreadsheets and pipelines.",
    },
    "json": {
        "ext": "JSON",
        "title": "JSON",
        "body": "Structured records for APIs and notebooks.",
    },
    "markdown": {
        "ext": "MD",
        "title": "Markdown",
        "body": "Readable table report for docs and wikis.",
    },
    "excel": {
        "ext": "XLSX",
        "title": "Excel report",
        "body": "Workbook with question, SQL, and results.",
    },
    "pdf": {
        "ext": "PDF",
        "title": "PDF report",
        "body": "Shareable analysis summary for stakeholders.",
    },
    "sql": {
        "ext": "SQL",
        "title": "SQL file",
        "body": "The validated query used for this analysis.",
    },
}


def render_exports(settings: Settings) -> None:
    page_header(
        "Exports",
        "Download analysis results in multiple formats",
        settings=settings,
    )

    result = st.session_state.get("last_result")
    df = st.session_state.get("last_df")
    if result is None or df is None or getattr(df, "empty", True):
        empty_state(
            "Nothing to export yet",
            "Run a successful Chat query first, then return here for multi-format downloads.",
            icon="XP",
        )
        section_title("Supported formats")
        inject_html(_format_preview_grid(["csv", "json", "markdown", "excel", "pdf", "sql"]))
        page_footer(settings.app_version)
        return

    question = (
        result.get("question")
        or result.get("rewritten_question")
        or "Analytics query"
    )
    if str(question).strip().lower() in {"sqlmind report", "report", "analytics report"}:
        question = "Analytics query"
    sql = result.get("sql") or ""
    insights = result.get("insights") or ""
    rows_n = len(df)
    duration = format_duration(float(result.get("execution_time") or 0))
    chart = result.get("chart_type") or "—"
    db_name = result.get("database_name") or (
        st.session_state.get("db_config").display_name
        if st.session_state.get("db_config")
        else "—"
    )

    health_pills(
        [
            ("Status", "Ready", "ok"),
            ("Rows", f"{rows_n:,}", "ok"),
            ("Runtime", duration, "info"),
            ("Database", str(db_name), "accent"),
        ]
    )

    q_short = (question[:96] + "…") if len(question) > 96 else question
    inject_html(
        f"""
        <div class="sq-card sq-card--accent">
          <div class="sq-card__title">Latest analysis</div>
          <div class="sq-card__heading">{esc(q_short)}</div>
          <div class="sq-kv-grid">
            <div><span class="sq-kv-label">Rows</span><strong>{esc(f"{rows_n:,}")}</strong></div>
            <div><span class="sq-kv-label">Duration</span><strong>{esc(duration)}</strong></div>
            <div><span class="sq-kv-label">Chart</span><strong>{esc(chart)}</strong></div>
            <div><span class="sq-kv-label">Database</span><strong>{esc(db_name)}</strong></div>
          </div>
          <div class="sq-badge-row sq-badge-row--flush sq-badge-row--top">
            {status_badge("Ready", kind="ok")}
            {status_badge(f"{rows_n:,} rows", kind="info")}
            {status_badge(duration, kind="default")}
          </div>
        </div>
        """
    )

    # Build payloads
    payload: dict[str, bytes] = {
        "csv": df.to_csv(index=False).encode("utf-8"),
        "json": df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
    }
    try:
        md = df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        md = df.to_string(index=False)
    payload["markdown"] = f"# {question}\n\n{md}\n".encode("utf-8")

    service = ExportService(settings.export_dir)
    try:
        excel_path = service.to_excel_report(
            df,
            label=question[:48],
            question=question,
            sql=sql,
            insights=insights,
            meta={
                "rows": result.get("row_count"),
                "time_s": round(float(result.get("execution_time") or 0), 3),
                "database": db_name,
                "chart_type": chart,
                "question": question,
            },
        )
        payload["excel"] = Path(excel_path).read_bytes()
    except Exception:  # noqa: BLE001
        pass
    try:
        pdf_path = service.build_pdf_report(
            title="SQLMind Report",
            question=question,
            sql=sql,
            insights=insights,
            df=df,
            meta={
                "rows": result.get("row_count"),
                "time_s": round(float(result.get("execution_time") or 0), 3),
                "database": db_name,
            },
        )
        payload["pdf"] = Path(pdf_path).read_bytes()
    except Exception:  # noqa: BLE001
        pass
    if sql:
        payload["sql"] = sql.encode("utf-8")

    mime = {
        "csv": "text/csv",
        "json": "application/json",
        "markdown": "text/markdown",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
        "sql": "text/plain",
    }
    ext = {
        "csv": "csv",
        "json": "json",
        "markdown": "md",
        "excel": "xlsx",
        "pdf": "pdf",
        "sql": "sql",
    }

    section_title("Download", "Choose a format")
    # Format cards then download buttons aligned underneath
    order = [k for k in ("csv", "excel", "pdf", "json", "markdown", "sql") if k in payload]
    inject_html(_format_preview_grid(order, sizes={k: _fmt_size(payload[k]) for k in order}))

    cols = st.columns(3)
    for i, name in enumerate(order):
        meta = _FORMAT_META.get(name, {"title": name.upper()})
        with cols[i % 3]:
            st.download_button(
                f"Download {meta['title']}",
                data=payload[name],
                file_name=f"sqlmind_export.{ext.get(name, name)}",
                mime=mime.get(name, "application/octet-stream"),
                key=f"export_center_{name}",
                use_container_width=True,
                type="primary" if name in {"excel", "pdf"} else "secondary",
            )

    tab_preview, tab_sql, tab_insight = st.tabs(["Data preview", "SQL", "Insights"])

    with tab_preview:
        section_title("Preview", "First 25 rows")
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    with tab_sql:
        if sql:
            section_title("Validated SQL", "Query used for this analysis")
            st.code(sql, language="sql")
            st.download_button(
                "Download SQL",
                data=sql.encode("utf-8"),
                file_name="sqlmind_query.sql",
                mime="text/plain",
                key="export_sql_dl_tab",
                use_container_width=True,
            )
        else:
            empty_state("No SQL", "This result has no SQL payload.", icon="QL")

    with tab_insight:
        if insights:
            section_title("Insights", "From the analysis run")
            st.markdown(insights)
        else:
            empty_state("No insights", "Insights appear after a successful Chat analysis.", icon="IN")

    page_footer(settings.app_version)


def _fmt_size(data: bytes) -> str:
    n = len(data)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _format_preview_grid(keys: list[str], sizes: dict[str, str] | None = None) -> str:
    sizes = sizes or {}
    cells: list[str] = []
    for key in keys:
        meta = _FORMAT_META.get(key, {"ext": key.upper(), "title": key.upper(), "body": ""})
        size = sizes.get(key, "")
        size_html = f'<div class="sq-export-card__meta">{esc(size)}</div>' if size else ""
        cells.append(
            f"""
            <div class="sq-export-card">
              <div class="sq-export-card__ext">{esc(meta["ext"])}</div>
              <div class="sq-export-card__title">{esc(meta["title"])}</div>
              <p class="sq-export-card__body">{esc(meta.get("body", ""))}</p>
              {size_html}
            </div>
            """
        )
    return f'<div class="sq-export-grid">{"".join(cells)}</div>'
