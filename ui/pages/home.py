"""Home — polished product workspace (presentation only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import Settings
from models.structured import DashboardModel
from prompts.templates import dashboard_prompt
from services.chart_builder import build_plotly_figures
from services.kpi_service import compute_table_kpis
from services.llm_service import LLMService
from services.visualization import build_figure
from ui.components import (
    empty_state,
    feature_grid,
    health_pills,
    kpi_grid,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.components.charts import render_plotly_chart_tabs
from ui.gate import is_llm_ready, require_llm
from ui.ui_helpers import esc, inject_html
from utils.helpers import format_duration


def render_home(settings: Settings) -> None:
    page_header(
        "Home",
        "Enterprise natural-language SQL analytics",
        settings=settings,
    )

    llm_ok, llm_reason = is_llm_ready(settings)
    connected = bool(st.session_state.get("connected"))
    cfg = st.session_state.get("db_config")
    schema = st.session_state.get("schema")
    can_ask = llm_ok and connected

    history: list = []
    try:
        history = st.session_state.history_store.list_history(limit=100) or []
    except Exception:  # noqa: BLE001
        history = []

    charts_n = sum(
        1
        for h in history
        if h.get("chart_type") and h.get("chart_type") not in {"", "none", "table"}
    )
    total_time = sum(float(h.get("execution_time") or 0) for h in history)
    success_n = sum(1 for h in history if h.get("success"))

    # —— Product hero ——
    inject_html(
        f"""
        <div class="sq-about-hero sq-about-hero--ink">
          <div class="sq-about-hero__mark">SQ</div>
          <div>
            <div class="sq-eyebrow">SQLMind AI</div>
            <h1>Ask questions. Get trusted SQL answers.</h1>
            <p>
              Multi-agent planning, sqlglot validation, read-only execution,
              then charts and insights — ready for Streamlit Community Cloud.
            </p>
            <div class="sq-about-meta">
              {status_badge(f"v{settings.app_version}", kind="accent")}
              {status_badge("LangGraph", kind="info")}
              {status_badge("Read-only", kind="ok")}
              {status_badge(cfg.display_name if cfg and connected else "No database", kind="ok" if connected else "off")}
            </div>
          </div>
        </div>
        """
    )

    if not llm_ok:
        st.warning(llm_reason)

    health_pills(
        [
            ("Database", "Connected" if connected else "Offline", "ok" if connected else "off"),
            ("LLM", "Ready" if llm_ok else "Setup required", "ok" if llm_ok else "warn"),
            ("Queries", str(len(history)), "ok" if history else "off"),
            ("Success", str(success_n), "ok" if success_n else "off"),
        ]
    )

    if connected and schema:
        cards = compute_table_kpis(schema)
        kpi_grid([(c["label"], str(c["value"])) for c in cards[:4]])
    else:
        kpi_grid(
            [
                ("Queries", str(len(history))),
                ("Tables", "—"),
                ("Charts", str(charts_n)),
                ("Runtime", format_duration(total_time) if history else "—"),
            ]
        )

    tab_work, tab_product, tab_insights = st.tabs(
        ["Workspace", "Product", "Insights"]
    )

    with tab_work:
        section_title("Quick actions", "Jump to a workspace area")
        a1, a2, a3, a4 = st.columns(4)
        if a1.button("Open Chat", type="primary", use_container_width=True, disabled=not can_ask):
            st.session_state.nav_page = "Chat"
            st.rerun()
        if a2.button("Schema", use_container_width=True, disabled=not connected):
            st.session_state.nav_page = "Schema"
            st.rerun()
        if a3.button("Runtime", use_container_width=True):
            st.session_state.nav_page = "Runtime"
            st.rerun()
        if a4.button("Settings", use_container_width=True):
            st.session_state.nav_page = "Settings"
            st.rerun()

        if history:
            section_title("Recent activity", "Re-run or remove a past question")
            for row in history[:5]:
                q = row.get("question") or ""
                ok = bool(row.get("success"))
                hid = row.get("id")
                meta = (
                    f"{format_duration(row.get('execution_time') or 0)} · "
                    f"{row.get('row_count') or 0} rows"
                )
                cols = st.columns([5, 0.9, 0.9])
                with cols[0]:
                    inject_html(
                        f"""
                        <div class="sq-pin-card">
                          <div class="sq-pin-card__top">
                            {status_badge("Success" if ok else "Failed", kind="ok" if ok else "error")}
                            {status_badge(meta, kind="default")}
                          </div>
                          <div class="sq-pin-card__title">{esc(q[:96])}{"…" if len(q) > 96 else ""}</div>
                        </div>
                        """
                    )
                if cols[1].button(
                    "Run",
                    key=f"home_hist_run_{hid}",
                    use_container_width=True,
                    type="primary",
                    disabled=not can_ask,
                ):
                    st.session_state["_pending_question"] = q
                    st.session_state.nav_page = "Chat"
                    st.rerun()
                if cols[2].button(
                    "Delete",
                    key=f"home_hist_del_{hid}",
                    use_container_width=True,
                ):
                    try:
                        st.session_state.history_store.delete_history(int(hid))
                    except Exception:  # noqa: BLE001
                        pass
                    st.rerun()
        else:
            section_title("Recent activity")
            empty_state(
                "No activity yet",
                "Run a Chat question to populate recent analyses here.",
                icon="HX",
            )

    with tab_product:
        section_title("Capabilities", "What SQLMind does")
        feature_grid(
            [
                (
                    "Natural language → SQL",
                    "Ask in plain English. Agents author dialect-aware, read-only SQL.",
                ),
                (
                    "Deterministic security",
                    "Every statement passes a sqlglot AST gate before database I/O.",
                ),
                (
                    "Charts & insights",
                    "Results become Plotly charts and executive summaries automatically.",
                ),
                (
                    "Multi-agent workflow",
                    "Planner, Supervisor, and specialists coordinate on LangGraph.",
                ),
                (
                    "Enterprise runtime",
                    "Queues, circuit breakers, IAM hooks, and live observability.",
                ),
                (
                    "Exports",
                    "Download CSV, Excel, PDF, JSON, or Markdown from any result.",
                ),
            ]
        )

        section_title("Architecture", "Control plane at a glance")
        inject_html(
            """
            <div class="sq-layer-grid">
              <div class="sq-layer sq-layer--control">
                <div class="sq-layer__label">Control</div>
                <div class="sq-layer__title">Plan &amp; coordinate</div>
                <p class="sq-layer__body">Goal · Plan · Coordinate · Supervise</p>
              </div>
              <div class="sq-layer sq-layer--data">
                <div class="sq-layer__label">Data</div>
                <div class="sq-layer__title">Author &amp; execute</div>
                <p class="sq-layer__body">Schema · SQL · Validate · Execute</p>
              </div>
              <div class="sq-layer sq-layer--analytics">
                <div class="sq-layer__label">Analytics</div>
                <div class="sq-layer__title">Explain &amp; deliver</div>
                <p class="sq-layer__body">Visualize · Insight · Reflect · Export</p>
              </div>
              <div class="sq-layer sq-layer--platform">
                <div class="sq-layer__label">Platform</div>
                <div class="sq-layer__title">Operate &amp; extend</div>
                <p class="sq-layer__body">Memory · IAM · Plugins · Metrics</p>
              </div>
            </div>
            """
        )

        inject_html(
            f"""
            <div class="sq-flow">
              <div class="sq-flow__title">Execution flow</div>
              <div class="sq-flow__steps">
                <span class="sq-flow__step">Question</span>
                <span class="sq-flow__arrow">→</span>
                <span class="sq-flow__step">Plan</span>
                <span class="sq-flow__arrow">→</span>
                <span class="sq-flow__step">SQL</span>
                <span class="sq-flow__arrow">→</span>
                <span class="sq-flow__step is-gate">Validate</span>
                <span class="sq-flow__arrow">→</span>
                <span class="sq-flow__step">Execute</span>
                <span class="sq-flow__arrow">→</span>
                <span class="sq-flow__step is-parallel">Charts ∥ Insights</span>
              </div>
            </div>
            """
        )

    with tab_insights:
        section_title("Dashboard", "Optional AI KPI blueprint")
        if not connected:
            empty_state(
                "Connect a database",
                "Schema KPIs and the AI dashboard appear after you connect.",
                icon="DB",
            )
        elif not require_llm(settings, action="generate an AI dashboard"):
            pass
        else:
            _render_dashboard_block(settings)

    page_footer(settings.app_version)


def _render_dashboard_block(settings: Settings) -> None:
    schema = st.session_state.schema
    spec = st.session_state.get("dashboard_spec") or {}

    inject_html(
        f"""
        <div class="sq-card">
          <div class="sq-card__title">Dashboard Agent</div>
          <div class="sq-card__heading">Design a KPI blueprint from schema metadata</div>
          <p class="sq-card__body">
            Proposes KPIs, chart ideas, and alerts for
            <strong>{esc(getattr(schema, "database_name", "database"))}</strong>.
          </p>
        </div>
        """
    )
    if st.button(
        "Generate dashboard",
        type="primary",
        use_container_width=True,
        key="home_gen_dash",
    ):
        llm = st.session_state.get("llm_service") or LLMService(settings)
        with st.spinner("Designing KPI blueprint…"):
            result = llm.invoke_structured(
                dashboard_prompt(),
                DashboardModel,
                {
                    "database_name": schema.database_name,
                    "dialect": schema.dialect,
                    "schema_text": schema.to_prompt_text(),
                },
            )
            st.session_state.dashboard_spec = result.model_dump()
            st.rerun()

    if spec:
        st.markdown(f"**{spec.get('title') or 'Dashboard'}**")
        narrative = spec.get("narrative") or ""
        if narrative:
            st.markdown(narrative)
        kpis = spec.get("kpis") or []
        if kpis:
            items = []
            for kpi in kpis[:4]:
                label = kpi.get("label", "KPI")
                value = kpi.get("description") or kpi.get("metric") or kpi.get("priority") or "—"
                if isinstance(value, str) and len(value) > 40:
                    value = value[:37] + "…"
                items.append((label, str(value)))
            if items:
                kpi_grid(items)
        if spec.get("alerts"):
            for a in spec["alerts"]:
                st.warning(a)
    else:
        st.caption("Generate a dashboard to see KPI suggestions here.")

    section_title("Latest charts", "From your most recent query")
    df: pd.DataFrame | None = st.session_state.get("last_df")
    result = st.session_state.get("last_result")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        empty_state(
            "No charts yet",
            "Run a successful Chat query to populate charts and a preview table here.",
            icon="CH",
        )
        return

    fig = build_figure(df, (result or {}).get("chart_spec"))
    charts = build_plotly_figures(df)
    if fig is not None:
        charts = [("Primary", fig)] + charts
    if charts:
        db_name = ""
        schema = st.session_state.get("schema")
        if schema is not None:
            db_name = getattr(schema, "database_name", "") or ""
        chart_type = (result or {}).get("chart_type") or "auto"
        key_prefix = f"home_{db_name}_{chart_type}"
        render_plotly_chart_tabs(
            charts,
            key_prefix=key_prefix,
            config={"displayModeBar": False},
        )
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)
    if result and result.get("insights"):
        with st.expander("Insights", expanded=False):
            st.markdown(result["insights"])
