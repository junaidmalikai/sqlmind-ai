"""Agent execution timeline + chat result helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.ui_helpers import esc, inject_html
from utils.helpers import format_duration

NODE_CATALOG: dict[str, dict[str, str]] = {
    "memory_agent": {
        "icon": "MM",
        "title": "Memory Agent",
        "done": "Loaded long-term memory",
        "active": "Retrieving memory…",
    },
    "goal_understanding": {
        "icon": "GL",
        "title": "Goal Understanding",
        "done": "Structured the analysis goal",
        "active": "Understanding goal…",
    },
    "planner": {
        "icon": "PL",
        "title": "Planner",
        "done": "Created execution plan",
        "active": "Planning…",
    },
    "task_decomposition": {
        "icon": "TD",
        "title": "Task Decomposition",
        "done": "Built task graph",
        "active": "Decomposing tasks…",
    },
    "goal_tracking": {
        "icon": "GT",
        "title": "Goal Tracking",
        "done": "Updated goal progress",
        "active": "Tracking goal…",
    },
    "execution_coordinator": {
        "icon": "CO",
        "title": "Execution Coordinator",
        "done": "Scheduled specialist work",
        "active": "Coordinating…",
    },
    "supervisor": {
        "icon": "SV",
        "title": "Supervisor",
        "done": "Routed next AI Agent",
        "active": "Routing…",
    },
    "schema_agent": {
        "icon": "SC",
        "title": "Schema Agent",
        "done": "Loaded schema context",
        "active": "Discovering schema…",
    },
    "sql_agent": {
        "icon": "QL",
        "title": "SQL Agent",
        "done": "Authored read-only SQL",
        "active": "Generating SQL…",
    },
    "validation_node": {
        "icon": "VL",
        "title": "Validation",
        "done": "Security validation passed",
        "active": "Validating SQL…",
    },
    "approval_gate": {
        "icon": "AP",
        "title": "Approval",
        "done": "Approval recorded",
        "active": "Awaiting approval…",
    },
    "execution_node": {
        "icon": "EX",
        "title": "Execution",
        "done": "Retrieved query results",
        "active": "Executing…",
    },
    "insight_agent": {
        "icon": "IN",
        "title": "Insight",
        "done": "Generated business insights",
        "active": "Generating insights…",
    },
    "visualization_agent": {
        "icon": "VZ",
        "title": "Visualization",
        "done": "Recommended charts",
        "active": "Building visualization…",
    },
    "join_post_query": {
        "icon": "JN",
        "title": "Join",
        "done": "Merged parallel analytics",
        "active": "Joining branches…",
    },
    "optimization_agent": {
        "icon": "OP",
        "title": "Optimization",
        "done": "Reviewed query plan",
        "active": "Optimizing…",
    },
    "dashboard_agent": {
        "icon": "DB",
        "title": "Dashboard",
        "done": "Designed KPI dashboard",
        "active": "Designing dashboard…",
    },
    "summary_agent": {
        "icon": "SM",
        "title": "Summary",
        "done": "Summarized database",
        "active": "Summarizing…",
    },
    "reflection_agent": {
        "icon": "RF",
        "title": "Reflection",
        "done": "Quality review complete",
        "active": "Reflecting…",
    },
    "retry_agent": {
        "icon": "RT",
        "title": "Retry",
        "done": "Diagnosed and retried",
        "active": "Diagnosing failure…",
    },
    "replan_agent": {
        "icon": "RP",
        "title": "Replan",
        "done": "Revised execution strategy",
        "active": "Replanning…",
    },
    "recovery_controller": {
        "icon": "RC",
        "title": "Recovery",
        "done": "Recovery policy applied",
        "active": "Recovering…",
    },
    "plugin_runtime_agent": {
        "icon": "PK",
        "title": "Plugin Runtime",
        "done": "Plugin skill completed",
        "active": "Invoking plugin…",
    },
    "export_node": {
        "icon": "XP",
        "title": "Export",
        "done": "Exports prepared",
        "active": "Exporting…",
    },
    "export_graph": {
        "icon": "XG",
        "title": "Export Graph",
        "done": "Export subgraph finished",
        "active": "Running export graph…",
    },
    "finalize": {
        "icon": "OK",
        "title": "Finalize",
        "done": "Response assembled",
        "active": "Finalizing…",
    },
    "clarify": {
        "icon": "CL",
        "title": "Clarify",
        "done": "Clarification received",
        "active": "Waiting for clarification…",
    },
    "fail": {
        "icon": "FL",
        "title": "Failed",
        "done": "Could not complete",
        "active": "Handling failure…",
    },
}

_SKIP = {"join_post_query", "__start__", "__end__"}


def node_meta(node_name: str) -> dict[str, str]:
    if node_name in NODE_CATALOG:
        return NODE_CATALOG[node_name]
    clean = node_name.replace("_agent", "").replace("_node", "").replace("_", " ").title()
    return {"icon": "•", "title": clean, "done": "Completed", "active": "Working…"}


def should_show_node(node_name: str) -> bool:
    return bool(node_name) and node_name not in _SKIP and not node_name.startswith("after_")


def step_message(node_name: str, update: dict[str, Any] | None = None) -> str:
    meta = node_meta(node_name)
    update = update or {}
    if node_name == "execution_node" and update.get("row_count") is not None:
        return f"Retrieved {update.get('row_count', 0)} records"
    if node_name == "validation_node" and update.get("sql_valid") is False:
        return "Validation issues detected"
    return meta["done"]


def render_timeline_html(
    steps: list[dict[str, Any]],
    *,
    running: bool = False,
    header: str = "Analyzing your request…",
) -> str:
    has_error = any(s.get("state") == "error" for s in steps)
    if running:
        status, status_cls = "Running", "is-running"
    elif has_error:
        status, status_cls = "Failed", "is-failed"
    else:
        status, status_cls = "Completed", "is-done"

    rows: list[str] = []
    for step in steps:
        state = step.get("state", "done")
        if state == "error":
            icon_cls, check = "is-error", ""
        elif state == "active":
            icon_cls, check = "is-active", ""
        else:
            icon_cls, check = "is-done", "✓ "
        rows.append(
            f"""
            <div class="sq-timeline__step">
              <div class="sq-timeline__icon {icon_cls}">{esc(step.get("icon", "•"))}</div>
              <div>
                <div class="sq-timeline__step-title">{esc(step.get("title", "Step"))}</div>
                <div class="sq-timeline__step-msg">{check}{esc(step.get("message", ""))}</div>
              </div>
            </div>"""
        )

    done_n = len([s for s in steps if s.get("state") == "done"])
    if running:
        pct = min(92, 12 + done_n * 11)
    elif steps:
        pct = 100
    else:
        pct = 8

    body = (
        f'<div class="sq-timeline__rail">{"".join(rows)}</div>'
        if rows
        else '<div class="sq-skeleton"></div><div class="sq-skeleton"></div>'
    )
    bar_cls = "is-running" if running else ("is-failed" if has_error else "")
    return f"""
    <div class="sq-timeline">
      <div class="sq-timeline__header">
        <div>
          <div class="sq-timeline__eyebrow">Agent workflow</div>
          <div class="sq-timeline__title">{esc(header)}</div>
        </div>
        <span class="sq-timeline__status {status_cls}">{status}</span>
      </div>
      <div>{body}</div>
      <div class="sq-timeline__bar">
        <div class="sq-timeline__bar-fill {bar_cls}" style="--sq-progress:{pct}%;"></div>
      </div>
    </div>
    """


def render_timeline(
    steps: list[dict[str, Any]],
    *,
    running: bool = False,
    header: str = "Analyzing your request…",
    slot: Any | None = None,
) -> None:
    markup = render_timeline_html(steps, running=running, header=header)
    target = slot if slot is not None else st
    if hasattr(target, "html"):
        target.html(markup)
    else:
        target.markdown(markup, unsafe_allow_html=True)


def extract_confidence(state: dict[str, Any]) -> float | None:
    for log in reversed(state.get("agent_logs") or []):
        msg = str((log or {}).get("message") or "")
        match = re.search(r"confidence\s*=\s*([0-9.]+)", msg, flags=re.I)
        if match:
            try:
                val = float(match.group(1))
                return val if val <= 1 else val / 100.0
            except ValueError:
                pass
    return None


def parse_executive_content(state: dict[str, Any], response_text: str) -> dict[str, Any]:
    structured = state.get("insight_structured") or {}
    summary = structured.get("summary") or ""
    bullets = list(structured.get("bullets") or [])
    trends = list(structured.get("trends") or [])
    anomalies = list(structured.get("anomalies") or [])
    recommendations = list(structured.get("recommendations") or [])
    if not summary:
        source = state.get("insights") or response_text or ""
        summary = source.strip().split("\n\n")[0][:800]
    if not bullets:
        for line in (state.get("insights") or response_text or "").splitlines():
            stripped = line.strip().lstrip("-•* ").strip()
            if line.strip().startswith(("-", "•", "*")) and stripped:
                bullets.append(stripped)
    parts: list[str] = []
    if trends:
        parts.append("**Growth & trends**\n" + "\n".join(f"- {t}" for t in trends))
    if anomalies:
        parts.append("**Outliers**\n" + "\n".join(f"- {a}" for a in anomalies))
    if recommendations:
        parts.append("**Recommendations**\n" + "\n".join(f"- {r}" for r in recommendations))
    tips = state.get("optimization_tips") or []
    if tips:
        parts.append("**Performance notes**\n" + "\n".join(f"- {t}" for t in tips[:6]))
    return {
        "summary": summary or response_text[:500] or "",
        "bullets": bullets[:8],
        "trends": trends,
        "anomalies": anomalies,
        "recommendations": recommendations,
        "analysis_extra": "\n\n".join(parts),
    }


def build_auto_charts(df: pd.DataFrame, primary: go.Figure | None = None) -> list[tuple[str, go.Figure]]:
    from services.chart_builder import build_plotly_figures

    charts: list[tuple[str, go.Figure]] = []
    if primary is not None:
        charts.append(("Primary", primary))
    charts.extend(build_plotly_figures(df))
    seen: set[str] = set()
    unique: list[tuple[str, go.Figure]] = []
    for label, fig in charts:
        if label not in seen:
            seen.add(label)
            unique.append((label, fig))
    return unique[:5]


def render_data_table(
    df: pd.DataFrame,
    key_prefix: str,
    *,
    show_overview: bool = True,
) -> None:
    if df is None or df.empty:
        from ui.components import empty_state

        empty_state(
            "No matching records",
            "Execution succeeded, but the database has no entries for this request. "
            "Try a broader question or remove filters.",
            icon="--",
        )
        return

    from services.result_presentation import humanize_column_name, humanize_dataframe

    # Display-only friendly headers (caller may already have humanized)
    view_source = df
    raw_cols = [str(c) for c in df.columns]
    if any("_" in c or c != humanize_column_name(c) for c in raw_cols):
        # Only re-humanize if headers still look raw
        if any(("_" in c) or (c == c.lower()) for c in raw_cols):
            view_source = humanize_dataframe(df) or df

    if show_overview:
        n = len(view_source)
        cols = [str(c) for c in view_source.columns][:8]
        inject_html(
            '<div class="sq-data-overview">'
            f"<strong>Data Overview</strong> — {n:,} records · "
            f"fields: {esc(', '.join(cols))}"
            + (f" (+{len(view_source.columns) - 8} more)" if len(view_source.columns) > 8 else "")
            + "</div>"
        )

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    query = c1.text_input(
        "Search table",
        key=f"{key_prefix}_search",
        placeholder="Search records…",
        label_visibility="collapsed",
    )
    sort_col = c2.selectbox(
        "Sort",
        ["—"] + [str(c) for c in view_source.columns],
        key=f"{key_prefix}_sort",
        label_visibility="collapsed",
    )
    page_size = c3.selectbox(
        "Page size",
        [10, 25, 50, 100],
        index=1,
        key=f"{key_prefix}_psize",
        label_visibility="collapsed",
    )
    view = view_source.copy()
    if query and query.strip():
        q = query.strip().lower()
        mask = pd.Series(False, index=view_source.index)
        for col in view_source.columns:
            mask = mask | view_source[col].astype(str).str.lower().str.contains(q, na=False)
        view = view_source[mask]
    if sort_col and sort_col != "—":
        try:
            view = view.sort_values(sort_col, ascending=True)
        except Exception:  # noqa: BLE001
            pass
    total = len(view)
    pages = max(1, (total + int(page_size) - 1) // int(page_size))
    page = c4.number_input(
        "Page",
        min_value=1,
        max_value=pages,
        value=1,
        key=f"{key_prefix}_page",
        label_visibility="collapsed",
    )
    start = (int(page) - 1) * int(page_size)
    end = start + int(page_size)
    inject_html(
        f'<div class="sq-table-toolbar">Showing {min(total, start + 1)}–{min(total, end)} of {total:,} records · {len(view_source.columns)} fields</div>'
    )
    st.dataframe(view.iloc[start:end], use_container_width=True, hide_index=True, height=360)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export filtered CSV",
        data=csv_bytes,
        file_name="sqlmind_table.csv",
        mime="text/csv",
        key=f"{key_prefix}_tbl_csv",
    )


def render_sql_accordion(
    sql: str,
    key_prefix: str,
    *,
    duration_s: float | None = None,
    rows: int | None = None,
    dialect: str | None = None,
    status: str = "Completed",
) -> None:
    if not sql:
        return
    cfg = st.session_state.get("db_config")
    engine = dialect or (cfg.dialect if cfg else "SQL")
    badges: list[str] = [f'<span class="sq-badge sq-badge-accent">{esc(str(engine).upper())}</span>']
    if status:
        kind = "ok" if status.lower() in {"completed", "success", "ok"} else "warn"
        cls = "sq-badge sq-badge-ok" if kind == "ok" else "sq-badge sq-badge-warn"
        badges.append(f'<span class="{cls}">{esc(status)}</span>')
    if duration_s is not None:
        badges.append(f'<span class="sq-badge">{esc(format_duration(float(duration_s)))}</span>')
    if rows is not None:
        badges.append(f'<span class="sq-badge">{esc(f"{int(rows):,} records")}</span>')

    with st.expander("Analysis query used", expanded=False):
        inject_html(
            '<div class="sq-sql-panel"><div class="sq-sql-meta">'
            + "".join(badges)
            + "</div></div>"
        )
        st.code(sql, language="sql")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download query",
                data=sql.encode("utf-8"),
                file_name="sqlmind_query.sql",
                mime="text/plain",
                key=f"{key_prefix}_sql_dl",
                use_container_width=True,
            )
        with c2:
            if st.button("Copy query", key=f"{key_prefix}_sql_copy", use_container_width=True):
                st.session_state[f"{key_prefix}_sql_copied"] = sql
                st.toast("Query ready — copy from the code block")


def render_kpi_strip(
    *,
    duration_s: float | None = None,
    rows: int | None = None,
    confidence: float | None = None,
    chart_type: str | None = None,
    tokens: str | int | None = None,
) -> None:
    """Compact run metadata as badges (not bulky Streamlit metrics)."""
    from ui.components import status_badge

    parts: list[str] = []
    if duration_s is not None:
        parts.append(status_badge(format_duration(float(duration_s)), kind="default"))
    if rows is not None:
        parts.append(status_badge(f"{int(rows):,} records", kind="info"))
    if confidence is not None:
        parts.append(status_badge(f"{float(confidence) * 100:.0f}% confidence", kind="accent"))
    if tokens not in (None, "", 0, "—"):
        parts.append(status_badge(f"{tokens} tokens", kind="default"))
    if chart_type and chart_type not in {"none", "table", ""}:
        parts.append(status_badge(str(chart_type).title(), kind="ok"))
    if not parts:
        return
    inject_html(f'<div class="sq-chat-toolbar">{"".join(parts)}</div>')
