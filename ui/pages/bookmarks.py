"""Bookmarks & saved queries — polished library UI (presentation only)."""

from __future__ import annotations

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


def render_bookmarks(settings: Settings) -> None:
    page_header(
        "Bookmarks",
        "Pinned questions and saved queries for quick reuse",
        settings=settings,
    )

    store = st.session_state.history_store
    bookmarks = store.list_bookmarks() or []
    saved = store.list_saved_queries() or []
    llm_ok, _ = is_llm_ready(settings)

    health_pills(
        [
            ("Bookmarks", str(len(bookmarks)), "ok" if bookmarks else "off"),
            ("Saved queries", str(len(saved)), "ok" if saved else "off"),
            ("Ask", "Ready" if llm_ok else "LLM setup", "ok" if llm_ok else "warn"),
            ("Library", str(len(bookmarks) + len(saved)), "accent"),
        ]
    )

    s1, s2 = st.columns([3, 1])
    with s1:
        search = st.text_input(
            "Search",
            placeholder="Filter by title, question, or SQL…",
            key="bm_search",
            label_visibility="collapsed",
        )
    with s2:
        sort = st.selectbox(
            "Sort",
            ["Newest", "Oldest", "A–Z"],
            key="bm_sort",
            label_visibility="collapsed",
        )
    q = (search or "").strip().lower()

    if not llm_ok:
        st.caption("Apply LLM settings to enable Ask from bookmarks.")

    tab_bm, tab_saved = st.tabs(
        [f"Bookmarks · {len(bookmarks)}", f"Saved queries · {len(saved)}"]
    )

    with tab_bm:
        shown = _filter_items(
            bookmarks,
            q,
            title_key="title",
            question_key="question",
            sql_key="sql",
        )
        shown = _sort_items(shown, sort, title_key="title")
        if not shown:
            empty_state(
                "No bookmarks yet" if not q else "No matches",
                "Star a query from History to pin it here."
                if not q
                else "Try a different search.",
                icon="BM",
            )
        else:
            section_title("Pinned", f"{len(shown)} item{'s' if len(shown) != 1 else ''}")
            for bm in shown:
                _render_pin_item(
                    kind="Bookmark",
                    title=bm.get("title") or "Untitled",
                    question=bm.get("question") or "",
                    sql=bm.get("sql") or "",
                    created=bm.get("created_at") or "",
                    item_id=bm["id"],
                    ask_key=f"ask_bm_{bm['id']}",
                    del_key=f"del_bm_{bm['id']}",
                    llm_ok=llm_ok,
                    on_delete=lambda i=bm["id"]: store.delete_bookmark(i),
                )

    with tab_saved:
        shown = _filter_items(
            saved,
            q,
            title_key="name",
            question_key="question",
            sql_key="sql",
        )
        shown = _sort_items(shown, sort, title_key="name")
        if not shown:
            empty_state(
                "No saved queries yet" if not q else "No matches",
                "Save a query from History to reuse it here."
                if not q
                else "Try a different search.",
                icon="SQ",
            )
        else:
            section_title("Library", f"{len(shown)} item{'s' if len(shown) != 1 else ''}")
            for sq in shown:
                _render_pin_item(
                    kind="Saved",
                    title=sq.get("name") or "Untitled",
                    question=sq.get("question") or "",
                    sql=sq.get("sql") or "",
                    created=sq.get("created_at") or "",
                    item_id=sq["id"],
                    ask_key=f"ask_sq_{sq['id']}",
                    del_key=f"del_sq_{sq['id']}",
                    llm_ok=llm_ok,
                    on_delete=lambda i=sq["id"]: store.delete_saved_query(i),
                    show_sql=True,
                )

    page_footer(settings.app_version)


def _render_pin_item(
    *,
    kind: str,
    title: str,
    question: str,
    sql: str,
    created: str,
    item_id: int,  # noqa: ARG001 — reserved for future deep-links
    ask_key: str,
    del_key: str,
    llm_ok: bool,
    on_delete,
    show_sql: bool = False,
) -> None:
    # Avoid duplicate title/question when History saved them identical
    body = ""
    if question and question.strip().lower() != title.strip().lower():
        body = question.strip()

    date_label = str(created)[:19].replace("T", " ") if created else ""
    badges = status_badge(kind, kind="accent" if kind == "Bookmark" else "info")
    if sql:
        badges += status_badge("Has SQL", kind="ok")
    date_html = (
        f'<span class="sq-pin-card__date">{esc(date_label)}</span>' if date_label else ""
    )
    body_html = (
        f'<p class="sq-pin-card__body">{esc(body[:180])}{"…" if len(body) > 180 else ""}</p>'
        if body
        else ""
    )

    c1, c2 = st.columns([5, 1.2])
    with c1:
        inject_html(
            f"""
            <div class="sq-pin-card">
              <div class="sq-pin-card__top">
                {badges}
              </div>
              <div class="sq-pin-card__title">{esc(title)}</div>
              {body_html}
              <div class="sq-pin-card__meta">{date_html}</div>
            </div>
            """
        )
        if show_sql and sql:
            with st.expander(f"SQL · {title[:40]}", expanded=False):
                st.code(sql, language="sql")
    with c2:
        if st.button(
            "Ask",
            key=ask_key,
            use_container_width=True,
            type="primary",
            disabled=not llm_ok or not question,
        ):
            st.session_state["_pending_question"] = question
            st.session_state.nav_page = "Chat"
            st.rerun()
        if st.button("Delete", key=del_key, use_container_width=True):
            on_delete()
            st.rerun()


def _filter_items(
    items: list[dict],
    q: str,
    *,
    title_key: str,
    question_key: str,
    sql_key: str,
) -> list[dict]:
    if not q:
        return list(items)
    out: list[dict] = []
    for item in items:
        hay = " ".join(
            [
                str(item.get(title_key) or ""),
                str(item.get(question_key) or ""),
                str(item.get(sql_key) or ""),
            ]
        ).lower()
        if q in hay:
            out.append(item)
    return out


def _sort_items(items: list[dict], sort: str, *, title_key: str) -> list[dict]:
    rows = list(items)
    if sort == "Oldest":
        rows.reverse()
    elif sort == "A–Z":
        rows.sort(key=lambda r: str(r.get(title_key) or "").lower())
    return rows
