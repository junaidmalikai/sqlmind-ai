"""Chat — multi-agent natural-language SQL analytics."""

from __future__ import annotations

import io
import re
import time
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import Settings, get_settings
from memory.conversation import QueryHistoryItem
from services.chart_builder import build_plotly_figures
from services.result_presentation import (
    compose_executive_answer,
    humanize_dataframe,
    is_weak_answer,
)
from services.visualization import build_figure
from ui.components import health_pills, kpi_grid, page_footer, page_header, status_badge
from ui.components.charts import render_plotly_chart_tabs
from ui.components.timeline import (
    extract_confidence,
    node_meta,
    parse_executive_content,
    render_data_table,
    render_kpi_strip,
    render_sql_accordion,
    render_timeline,
    should_show_node,
    step_message,
)
from ui.ui_helpers import esc, inject_html
from utils.helpers import format_duration, utc_now_iso


def render_chat(settings: Settings) -> None:
    page_header(
        "Chat",
        "Ask in plain English. Get summaries, results, visuals, and insights.",
        settings=settings,
    )

    from ui.gate import require_db_and_llm, is_llm_ready

    if not require_db_and_llm(settings, action="chat with your data"):
        page_footer(settings.app_version)
        return

    connected = bool(st.session_state.get("connected"))
    cfg = st.session_state.get("db_config")
    llm_ok, _ = is_llm_ready(settings)
    msgs = st.session_state.get("chat_messages") or []
    health_pills(
        [
            ("Database", cfg.display_name if cfg and connected else "—", "ok" if connected else "off"),
            ("LLM", "Ready" if llm_ok else "Setup", "ok" if llm_ok else "warn"),
            ("Messages", str(len(msgs)), "accent" if msgs else "off"),
            ("Mode", "Read-only", "ok"),
        ]
    )

    if st.session_state.get("memory_summary"):
        with st.expander("Conversation memory", expanded=False):
            st.write(st.session_state.memory_summary)

    if not msgs:
        _render_chat_empty()
    else:
        _render_message_history()

    prompt = st.chat_input("Ask about your data…")
    if prompt:
        _handle_user_question(prompt, settings)

    page_footer(settings.app_version)


def _render_chat_empty() -> None:
    suggestions = st.session_state.get("suggested_questions") or [
        "Show top 5 best-selling products by quantity",
        "What are monthly revenue trends?",
        "Which customers placed the most orders?",
    ]
    chips = "".join(
        f'<span class="sq-chat-chip">{esc(str(s)[:64])}</span>' for s in suggestions[:4]
    )
    inject_html(
        f"""
        <div class="sq-chat-empty">
          <h3>Start an analysis</h3>
          <p>Ask a question about your connected database. You will get a clear summary, results, visuals, and next-step suggestions.</p>
          <div class="sq-chat-suggestions">{chips}</div>
        </div>
        """
    )


def _render_message_history() -> None:
    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                inject_html(f'<div class="sq-chat-user">{esc(msg["content"])}</div>')
                continue
            _render_assistant_message(msg)


def _render_assistant_message(msg: dict[str, Any]) -> None:
    meta = msg.get("meta_dict") or {}
    executive = msg.get("executive") or {}

    # —— Result cards (enterprise run metadata) ——
    cards = executive.get("result_cards") or msg.get("result_cards") or []
    if cards:
        kpi_grid([(str(k), str(v)) for k, v in cards[:7]])
    else:
        render_kpi_strip(
            duration_s=meta.get("duration_s"),
            rows=meta.get("rows"),
            confidence=meta.get("confidence"),
            chart_type=meta.get("chart_type"),
            tokens=meta.get("tokens"),
        )

    summary = executive.get("summary") or msg.get("content") or ""

    # —— Answer (hero) ——
    inject_html(
        '<div class="sq-chat-answer"><div class="sq-chat-answer__label">Query Summary</div></div>'
    )
    st.markdown(summary)
    bullets = executive.get("bullets") or []
    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    analysis = executive.get("analysis_extra") or ""
    if analysis:
        st.markdown(analysis)
    elif msg.get("content") and msg.get("content") != summary:
        cleaned = re.sub(r"\n?\*\*SQL\*\*.*$", "", msg["content"], flags=re.S).strip()
        if cleaned and cleaned != summary.strip() and not is_weak_answer(cleaned):
            st.markdown(cleaned)

    followups = executive.get("followups") or []
    if followups and executive.get("empty_result"):
        inject_html('<div class="sq-chat-answer__label" style="margin-top:0.75rem">Try next</div>')
        for s in followups[:4]:
            st.markdown(f"- {s}")

    # —— Detail tabs ——
    has_sql = bool(msg.get("sql"))
    df = msg.get("df")
    has_df = df is not None
    empty_ok = bool(executive.get("empty_result") or msg.get("empty_result"))
    charts = msg.get("charts") or []
    if not charts and msg.get("fig") is not None:
        charts = [("Primary", msg["fig"])]
    if not charts and has_df and not getattr(df, "empty", True):
        charts = build_plotly_figures(df)
    has_charts = bool(charts)
    has_exports = bool(msg.get("exports"))

    tab_labels: list[str] = []
    if has_df or empty_ok:
        tab_labels.append("Results")
    if has_charts:
        tab_labels.append("Charts")
    if has_sql:
        tab_labels.append("Analysis Query")
    if has_exports:
        tab_labels.append("Export")
    tab_labels.append("Activity")

    tabs = st.tabs(tab_labels)
    tab_map = {label: tabs[i] for i, label in enumerate(tab_labels)}

    if "Results" in tab_map:
        with tab_map["Results"]:
            overview = executive.get("data_overview") or ""
            if overview:
                st.markdown(overview)
            if empty_ok or (has_df and getattr(df, "empty", False)):
                inject_html(
                    '<div class="sq-empty sq-empty--inline">'
                    "<h3>No matching records</h3>"
                    "<p>Execution succeeded, but nothing matched this request. "
                    "Try broadening filters or asking a different question.</p>"
                    "</div>"
                )
                if followups and not executive.get("empty_result"):
                    st.markdown("**Suggested next questions**")
                    for s in followups[:4]:
                        st.markdown(f"- {s}")
            elif has_df:
                render_data_table(
                    humanize_dataframe(df),
                    key_prefix=msg.get("id", "hist"),
                    show_overview=False,
                )

    if "Charts" in tab_map:
        with tab_map["Charts"]:
            msg_id = msg.get("id") or "hist"
            db_name = ""
            schema = st.session_state.get("schema")
            if schema is not None:
                db_name = getattr(schema, "database_name", "") or ""
            chart_type = (msg.get("meta_dict") or {}).get("chart_type") or ""
            key_prefix = f"chat_{msg_id}_{db_name}_{chart_type}"
            render_plotly_chart_tabs(charts, key_prefix=key_prefix)

    if "Analysis Query" in tab_map:
        with tab_map["Analysis Query"]:
            render_sql_accordion(
                msg["sql"],
                key_prefix=msg.get("id", "hist"),
                duration_s=meta.get("duration_s"),
                rows=meta.get("rows"),
                status="Completed",
            )

    if "Export" in tab_map:
        with tab_map["Export"]:
            _download_buttons(msg["exports"], key_prefix=msg.get("id", "hist"))

    with tab_map["Activity"]:
        if msg.get("timeline"):
            render_timeline(msg["timeline"], running=False, header="Workflow complete")
        else:
            st.caption("No activity timeline stored for this message.")
        if msg.get("logs"):
            st.markdown("**Activity log**")
            for log in msg["logs"]:
                st.write(
                    f"**{log.get('agent')}** · {log.get('status', 'ok')} — {log.get('message')}"
                )

    # —— Footer actions ——
    mid = msg.get("id", "hist")
    ts = meta.get("timestamp") or msg.get("timestamp") or ""
    if ts:
        inject_html(
            f'<div class="sq-badge-row sq-badge-row--flush">'
            f'{status_badge("Completed", kind="ok")}'
            f'{status_badge(str(ts)[:19].replace("T", " "), kind="default")}'
            f"</div>"
        )
    a1, a2, a3 = st.columns(3)
    if a1.button("Copy answer", key=f"copyans_{mid}", use_container_width=True):
        st.session_state[f"copy_buf_{mid}"] = summary
        st.toast("Answer copied to session buffer")
    if a2.button("Regenerate", key=f"regen_{mid}", use_container_width=True, type="primary"):
        q = msg.get("question") or ""
        if q:
            st.session_state["_pending_question"] = q
            st.rerun()
    if a3.button("Exports page", key=f"goto_exp_{mid}", use_container_width=True):
        st.session_state.nav_page = "Exports"
        st.rerun()


def _handle_user_question(question: str, settings: Settings) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    memory = st.session_state.memory
    memory.add("user", question)

    orchestrator = st.session_state.orchestrator
    session_id = st.session_state.get("session_id") or "default"

    if hasattr(orchestrator, "summarizer"):
        summary = orchestrator.summarizer.maybe_summarize(
            memory.as_dicts(),
            st.session_state.get("memory_summary", ""),
        )
        st.session_state.memory_summary = summary

    timeline_slot = st.empty()
    steps: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    t0 = time.perf_counter()

    render_timeline(
        [],
        running=True,
        header="Analyzing your request…",
        slot=timeline_slot,
    )

    try:
        for event in orchestrator.stream_events(
            question,
            memory.as_dicts()[:-1],
            memory_summary=st.session_state.get("memory_summary", ""),
            session_id=session_id,
            actor=settings.default_actor,
            tenant_id=settings.default_tenant_id,
        ):
            for node_name, update in event.items():
                if not should_show_node(node_name):
                    if isinstance(update, dict):
                        final_state.update(update)
                        if update.get("agent_logs"):
                            st.session_state.agent_logs.extend(update["agent_logs"])
                    continue

                # Mark previous active step as done
                for step in steps:
                    if step.get("state") == "active":
                        step["state"] = "done"

                meta = node_meta(node_name)
                upd = update if isinstance(update, dict) else {}
                status = "error" if (upd.get("status") == "failed" or node_name == "fail") else "active"
                message = step_message(node_name, upd)
                if status == "error":
                    message = upd.get("error") or upd.get("sql_error") or message

                # Avoid duplicate consecutive identical nodes unless retry
                if steps and steps[-1].get("node") == node_name and node_name != "retry_agent":
                    steps[-1]["message"] = message
                    steps[-1]["state"] = "done" if status != "error" else "error"
                else:
                    steps.append(
                        {
                            "node": node_name,
                            "icon": meta["icon"],
                            "title": meta["title"],
                            "message": message if status != "active" else meta["active"],
                            "state": status,
                        }
                    )

                if isinstance(update, dict):
                    final_state.update(update)
                    if update.get("agent_logs"):
                        st.session_state.agent_logs.extend(update["agent_logs"])
                    # Enrich execution step with row count once known
                    if node_name == "execution_node" and update.get("row_count") is not None:
                        steps[-1]["message"] = f"Retrieved {update.get('row_count', 0)} records"

                render_timeline(
                    steps,
                    running=True,
                    header="Analyzing your request…",
                    slot=timeline_slot,
                )

        if not final_state.get("final_response") and not final_state.get("dataframe_records"):
            final_state = orchestrator.run(
                question,
                memory.as_dicts()[:-1],
                memory_summary=st.session_state.get("memory_summary", ""),
                session_id=session_id,
                actor=settings.default_actor,
                tenant_id=settings.default_tenant_id,
            )

        for step in steps:
            if step.get("state") == "active":
                step["state"] = "done"
        if not any(s.get("node") == "finalize" for s in steps):
            fin = node_meta("finalize")
            steps.append(
                {
                    "node": "finalize",
                    "icon": fin["icon"],
                    "title": fin["title"],
                    "message": fin["done"],
                    "state": "done",
                }
            )

        elapsed = time.perf_counter() - t0
        render_timeline(
            steps,
            running=False,
            header="Analysis complete",
            slot=timeline_slot,
        )
        render_kpi_strip(
            duration_s=float(final_state.get("execution_time") or elapsed),
            rows=final_state.get("row_count"),
            confidence=extract_confidence(final_state),
            chart_type=final_state.get("chart_type"),
        )
    except Exception as exc:  # noqa: BLE001
        for step in steps:
            if step.get("state") == "active":
                step["state"] = "error"
        fail = node_meta("fail")
        steps.append(
            {
                "node": "fail",
                "icon": fail["icon"],
                "title": fail["title"],
                "message": str(exc)[:160],
                "state": "error",
            }
        )
        render_timeline(
            steps,
            running=False,
            header="Workflow failed",
            slot=timeline_slot,
        )
        err = f"Workflow error: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": err})
        st.error(err)
        st.rerun()
        return

    response_text = final_state.get("final_response") or final_state.get("insights") or ""
    if is_weak_answer(response_text):
        # Presentation-layer guarantee — never show bare "Done."
        from services.result_presentation import build_query_summary

        records = final_state.get("dataframe_records") or []
        columns = final_state.get("columns") or []
        preview_df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns or None)
        response_text = build_query_summary(
            final_state, question=question, df=preview_df
        )

    with st.chat_message("assistant"):
        # Real provider token stream for a closing narrative when available.
        # Never fake a typewriter over a finished string.
        streamed_extra = ""
        token_iter = None
        if hasattr(orchestrator, "stream_answer_tokens"):
            try:
                token_iter = orchestrator.stream_answer_tokens(final_state)
            except Exception:  # noqa: BLE001
                token_iter = None

        # Show a skeleton while tokens arrive; content is streamed honestly
        st.markdown(response_text)
        if token_iter is not None:
            try:
                streamed_extra = st.write_stream(token_iter) or ""
            except Exception:  # noqa: BLE001
                st.caption("Live narrative unavailable for this provider — showing the completed answer.")
        if streamed_extra and not is_weak_answer(str(streamed_extra)):
            response_text = f"{response_text}\n\n{streamed_extra}"

    final_state["final_response"] = response_text
    final_state["_timeline"] = steps
    final_state["_wall_time"] = time.perf_counter() - t0
    _present_result(question, final_state, settings)
    st.rerun()


def _present_result(question: str, state: dict[str, Any], settings: Settings) -> None:
    records = state.get("dataframe_records") or []
    columns = state.get("columns") or []
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=columns or None)
    primary_fig = build_figure(df, state.get("chart_spec"))
    charts = build_plotly_figures(df) if not df.empty else []
    if primary_fig is not None:
        charts = [("Primary", primary_fig)] + [c for c in charts if c[0] != "Primary"]
    response = state.get("final_response") or state.get("insights") or ""
    sql = state.get("sql") or ""
    base_executive = parse_executive_content(state, response)
    executive = compose_executive_answer(
        state,
        question=question,
        df=df,
        response_text=response,
        executive=base_executive,
    )
    # Prefer composed summary as the stored answer content
    response = executive.get("summary") or response
    if is_weak_answer(response):
        response = executive.get("query_summary") or "Query Summary ready."

    rec_text = "\n".join(f"- {r}" for r in (executive.get("recommendations") or [])[:8])
    display_df = humanize_dataframe(df) if not df.empty else df
    download_payload = _build_download_payload(
        display_df if display_df is not None else df,
        question,
        state,
        primary_fig,
        recommendations=rec_text,
    )

    duration = float(state.get("execution_time") or state.get("_wall_time") or 0)
    confidence = extract_confidence(state)
    row_count = int(state.get("row_count") or (0 if df is None else len(df)))
    empty_result = bool(state.get("query_success")) and row_count == 0
    meta_dict = {
        "duration_s": duration,
        "rows": row_count,
        "confidence": confidence,
        "chart_type": state.get("chart_type") or "",
        "tokens": "—",
        "timestamp": utc_now_iso(),
    }
    meta = ""
    if state.get("query_success"):
        conf_txt = f"{confidence * 100:.0f}%" if confidence is not None else "—"
        meta = (
            f"Records: {meta_dict['rows']} · "
            f"Time: {format_duration(duration)} · "
            f"Confidence: {conf_txt} · "
            f"Visual: {state.get('chart_type', 'none')}"
        )

    msg_id = f"m{len(st.session_state.chat_messages)}"
    # Keep empty DF for Results tab empty-state UX (not silent omission)
    stored_df = df if (not df.empty or empty_result or bool(columns)) else None
    st.session_state.chat_messages.append(
        {
            "id": msg_id,
            "role": "assistant",
            "question": question,
            "content": response,
            "executive": executive,
            "result_cards": executive.get("result_cards") or [],
            "empty_result": empty_result,
            "sql": sql,
            "plan": state.get("plan") or [],
            "route_history": state.get("route_history") or [],
            "df": stored_df,
            "fig": primary_fig,
            "charts": charts,
            "meta": meta,
            "meta_dict": meta_dict,
            "exports": download_payload,
            "logs": state.get("agent_logs") or [],
            "timeline": state.get("_timeline") or [],
            "insight_structured": state.get("insight_structured") or {},
        }
    )

    memory = st.session_state.memory
    memory.add("assistant", response, sql=sql)

    # Persist question for Export Center / PDF / Excel
    state["question"] = question
    cfg = st.session_state.get("db_config")
    if cfg and not state.get("database_name"):
        state["database_name"] = cfg.display_name
    st.session_state.last_result = state
    st.session_state.last_df = df

    if state.get("suggested_questions"):
        st.session_state.suggested_questions = state["suggested_questions"]
    if state.get("database_summary"):
        st.session_state.ai_summary = state["database_summary"]
    if state.get("dashboard_spec"):
        st.session_state.dashboard_spec = state["dashboard_spec"]

    item = QueryHistoryItem(
        question=question,
        sql=sql,
        execution_time=float(state.get("execution_time") or 0),
        timestamp=utc_now_iso(),
        database=cfg.display_name if cfg else "",
        row_count=int(state.get("row_count") or 0),
        success=bool(state.get("query_success")),
        exports=state.get("export_paths") or {},
        insights=state.get("insights") or "",
        chart_type=state.get("chart_type") or "",
    )
    st.session_state.history_store.add_history(item)


def _build_download_payload(
    df: pd.DataFrame,
    label: str,
    state: dict[str, Any] | None = None,
    fig: Any | None = None,
    recommendations: str = "",
) -> dict[str, bytes]:
    if df is None or df.empty:
        return {}

    payload: dict[str, bytes] = {
        "csv": df.to_csv(index=False).encode("utf-8"),
        "json": df.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"),
    }
    try:
        md = df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        md = df.to_string(index=False)
    payload["markdown"] = f"# {label}\n\n{md}\n".encode("utf-8")

    db_name = (state or {}).get("database_name") or (
        st.session_state.get("db_config").display_name
        if st.session_state.get("db_config")
        else "—"
    )
    insights = (state or {}).get("insights", "") if state else ""
    sql = (state or {}).get("sql", "") if state else ""
    meta = {
        "rows": (state or {}).get("row_count"),
        "time_s": round(float((state or {}).get("execution_time") or 0), 3),
        "database": db_name,
        "chart_type": (state or {}).get("chart_type") or "—",
        "question": label,
    }

    try:
        from pathlib import Path

        from services.export_service import ExportService

        service = ExportService(get_settings().export_dir)
        excel_path = service.to_excel_report(
            df,
            label=label[:48] or "sqlmind_report",
            question=label,
            sql=sql,
            insights=insights,
            meta=meta,
            recommendations=recommendations,
        )
        payload["excel"] = Path(excel_path).read_bytes()
    except Exception:  # noqa: BLE001
        try:
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            payload["excel"] = buf.getvalue()
        except Exception:  # noqa: BLE001
            pass

    try:
        from pathlib import Path

        from services.export_service import ExportService

        path = ExportService(get_settings().export_dir).build_pdf_report(
            title=label,
            question=label,
            sql=sql,
            insights=insights,
            df=df,
            meta=meta,
            recommendations=recommendations,
        )
        payload["pdf"] = Path(path).read_bytes()
    except Exception:  # noqa: BLE001
        pass
    return payload


def _download_buttons(payload: dict[str, bytes], key_prefix: str) -> None:
    if not payload:
        return
    cols = st.columns(len(payload))
    mime = {
        "csv": "text/csv",
        "json": "application/json",
        "markdown": "text/markdown",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pdf": "application/pdf",
    }
    ext = {"csv": "csv", "json": "json", "markdown": "md", "excel": "xlsx", "pdf": "pdf"}
    labels = {
        "csv": "CSV",
        "json": "JSON",
        "markdown": "Markdown",
        "excel": "Excel Report",
        "pdf": "PDF Report",
    }
    for i, (name, data) in enumerate(payload.items()):
        cols[i].download_button(
            label=labels.get(name, name.upper()),
            data=data,
            file_name=f"sqlmind_export.{ext.get(name, name)}",
            mime=mime.get(name, "application/octet-stream"),
            key=f"{key_prefix}_{name}",
            use_container_width=True,
        )
