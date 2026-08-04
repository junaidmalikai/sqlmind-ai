"""History — polished activity timeline (presentation only)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from config.settings import Settings
from ui.components import (
    empty_state,
    health_pills,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.gate import is_llm_ready
from ui.ui_helpers import esc, inject_html
from utils.helpers import format_duration


def render_history(settings: Settings) -> None:
    page_header(
        "History",
        "Search, re-run, or bookmark past analyses",
        settings=settings,
    )
    store = st.session_state.history_store
    rows = store.list_history(limit=100) or []
    llm_ok, _ = is_llm_ready(settings)

    if not rows:
        empty_state(
            "No queries yet",
            "Ask a question in Chat to build a searchable history of analyses.",
            icon="HX",
        )
        page_footer(settings.app_version)
        return

    success_n = sum(1 for r in rows if r.get("success"))
    failed_n = len(rows) - success_n
    health_pills(
        [
            ("Total", str(len(rows)), "accent"),
            ("Success", str(success_n), "ok"),
            ("Failed", str(failed_n), "error" if failed_n else "off"),
            ("Re-run", "Ready" if llm_ok else "LLM setup", "ok" if llm_ok else "warn"),
        ]
    )

    f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
    search = f1.text_input(
        "Search",
        placeholder="Search questions or SQL…",
        key="hist_search",
        label_visibility="collapsed",
    )
    status = f2.selectbox("Status", ["All", "Success", "Failed"], key="hist_status")
    db_opts = ["All"] + sorted({str(r.get("database_name") or "—") for r in rows})
    db_filter = f3.selectbox("Database", db_opts, key="hist_db")
    sort = f4.selectbox("Sort", ["Newest", "Oldest"], key="hist_sort")

    q = (search or "").strip().lower()
    filtered: list[dict] = []
    for row in rows:
        if status == "Success" and not row.get("success"):
            continue
        if status == "Failed" and row.get("success"):
            continue
        if db_filter != "All" and str(row.get("database_name") or "—") != db_filter:
            continue
        hay = f"{row.get('question', '')} {row.get('sql', '')} {row.get('insights', '')}".lower()
        if q and q not in hay:
            continue
        filtered.append(row)

    if sort == "Oldest":
        filtered = list(reversed(filtered))

    if not filtered:
        empty_state("No matches", "Try a different search or clear filters.", icon="--")
        page_footer(settings.app_version)
        return

    inject_html(
        f'<div class="sq-badge-row sq-badge-row--flush">'
        f'{status_badge(f"{len(filtered)} of {len(rows)} queries", kind="info")}'
        f'{status_badge(status, kind="default")}'
        f'{status_badge(db_filter if db_filter != "All" else "All databases", kind="default")}'
        f"</div>"
    )

    if not llm_ok:
        st.caption("Apply LLM settings to re-run queries from history.")

    groups = _group_rows(filtered)
    for label, items in groups.items():
        if not items:
            continue
        section_title(label, f"{len(items)} quer{'ies' if len(items) != 1 else 'y'}")
        for row in items:
            _render_history_item(row, store=store, llm_ok=llm_ok)

    page_footer(settings.app_version)


def _render_history_item(row: dict, *, store, llm_ok: bool) -> None:
    title = (row.get("question") or "Untitled query").strip()
    title_short = title[:90] + ("…" if len(title) > 90 else "")
    ok = bool(row.get("success"))
    ts = str(row.get("timestamp") or "")[:19].replace("T", " ")
    duration = format_duration(row.get("execution_time") or 0)
    rows_n = row.get("row_count") or 0
    db = row.get("database_name") or "—"
    chart = row.get("chart_type") or ""
    rid = row["id"]

    status_html = status_badge("Success" if ok else "Failed", kind="ok" if ok else "error")
    meta_badges = status_html + status_badge(duration, kind="default")
    meta_badges += status_badge(f"{rows_n} rows", kind="info")
    if chart and chart not in {"", "none", "table"}:
        meta_badges += status_badge(str(chart).title(), kind="accent")

    c1, c2 = st.columns([5, 1.2])
    with c1:
        inject_html(
            f"""
            <div class="sq-pin-card">
              <div class="sq-pin-card__top">{meta_badges}</div>
              <div class="sq-pin-card__title">{esc(title_short)}</div>
              <div class="sq-pin-card__meta">
                <span class="sq-pin-card__date">{esc(ts)}</span>
                <span class="sq-pin-card__date">·</span>
                <span class="sq-pin-card__date">{esc(db)}</span>
              </div>
            </div>
            """
        )
    with c2:
        if st.button(
            "Re-run",
            key=f"rerun_{rid}",
            use_container_width=True,
            type="primary",
            disabled=not llm_ok or not title,
        ):
            st.session_state["_pending_question"] = row["question"]
            st.session_state.nav_page = "Chat"
            st.rerun()
        if st.button("Bookmark", key=f"bm_{rid}", use_container_width=True):
            store.add_bookmark(title_short, row["question"], row.get("sql") or "")
            st.toast("Bookmarked")
            st.rerun()

    with st.expander(f"Details · {ts or rid}", expanded=False):
        if row.get("sql"):
            st.code(row["sql"], language="sql")
        else:
            st.caption("No SQL stored for this run.")
        if row.get("insights"):
            section_title("Insights")
            st.markdown(row["insights"])
        a1, a2 = st.columns(2)
        if a1.button("Save query", key=f"sq_{rid}", use_container_width=True):
            store.save_query(title_short, row["question"], row.get("sql") or "")
            st.toast("Saved to library")
            st.rerun()
        if a2.button("Open in Chat", key=f"open_{rid}", use_container_width=True, disabled=not llm_ok):
            st.session_state["_pending_question"] = row["question"]
            st.session_state.nav_page = "Chat"
            st.rerun()


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    now = datetime.now(timezone.utc)
    today = now.date()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)
    groups: dict[str, list[dict]] = {
        "Today": [],
        "Yesterday": [],
        "Last Week": [],
        "Older": [],
    }
    for row in rows:
        ts = row.get("timestamp") or ""
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            d = dt.date()
        except Exception:  # noqa: BLE001
            groups["Older"].append(row)
            continue
        if d == today:
            groups["Today"].append(row)
        elif d == yesterday:
            groups["Yesterday"].append(row)
        elif d >= week_ago:
            groups["Last Week"].append(row)
        else:
            groups["Older"].append(row)
    return groups
