"""Reusable visual components — unified design system."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.ui_helpers import esc, inject_html


def status_badge(label: str, *, kind: str = "default") -> str:
    cls = {
        "ok": "sq-badge sq-badge-ok",
        "warn": "sq-badge sq-badge-warn",
        "error": "sq-badge sq-badge-error",
        "off": "sq-badge sq-badge-off",
        "accent": "sq-badge sq-badge-accent",
        "info": "sq-badge sq-badge-info",
    }.get(kind, "sq-badge")
    return f'<span class="{cls}">{esc(label)}</span>'


def page_status_badges(settings: Any | None = None) -> str:
    from ui.gate import is_llm_ready

    connected = bool(st.session_state.get("connected"))
    cfg = st.session_state.get("db_config")
    llm_ok = False
    if settings is not None:
        llm_ok, _ = is_llm_ready(settings)

    db_label = cfg.display_name if cfg else "No database"
    badges = [
        status_badge(
            f"{'Connected' if connected else 'Disconnected'} · {db_label}",
            kind="ok" if connected else "off",
        ),
        status_badge(
            "LLM ready" if llm_ok else "LLM setup required",
            kind="accent" if llm_ok else "warn",
        ),
        status_badge("Read-only", kind="default"),
    ]
    return '<div class="sq-badge-row">' + "".join(badges) + "</div>"


def page_header(title: str, subtitle: str = "", settings: Any | None = None, right_html: str = "") -> None:
    badges = page_status_badges(settings) if settings is not None else ""
    inject_html(
        f"""
        <div class="sq-page-head">
          <div>
            <div class="sq-eyebrow">SQLMind AI</div>
            <h1>{esc(title)}</h1>
            <p>{esc(subtitle)}</p>
            {badges}
          </div>
          <div>{right_html}</div>
        </div>
        """
    )


def hero(title: str, subtitle: str, eyebrow: str = "SQLMind AI") -> None:
    inject_html(
        f"""
        <div class="sq-hero">
          <div class="sq-hero__eyebrow">{esc(eyebrow)}</div>
          <h1>{esc(title)}</h1>
          <p>{esc(subtitle)}</p>
        </div>
        """
    )


def card(
    *,
    title: str = "",
    heading: str = "",
    body: str = "",
    meta: str = "",
    kind: str = "",
    flush: bool = False,
    extra_html: str = "",
) -> None:
    classes = ["sq-card"]
    if kind in {"accent", "ok", "warn", "error"}:
        classes.append(f"sq-card--{kind}")
    if flush:
        classes.append("sq-card--flush")
    parts = [f'<div class="{" ".join(classes)}">']
    if title:
        parts.append(f'<div class="sq-card__title">{esc(title)}</div>')
    if heading:
        parts.append(f'<div class="sq-card__heading">{esc(heading)}</div>')
    if body:
        parts.append(f'<p class="sq-card__body">{esc(body)}</p>')
    if meta:
        parts.append(f'<div class="sq-card__meta">{meta}</div>')
    if extra_html:
        parts.append(extra_html)
    parts.append("</div>")
    inject_html("".join(parts))


def kv_card(title: str, items: list[tuple[str, Any]], *, columns: int = 2) -> None:
    """Key-value card for Runtime / Settings."""
    grid = "sq-kv-grid sq-kv-grid--3" if columns == 3 else "sq-kv-grid"
    cells = "".join(
        f'<div><span class="sq-kv-label">{esc(k)}</span><strong>{esc(v if v is not None else "—")}</strong></div>'
        for k, v in items
    )
    inject_html(
        f'<div class="sq-card"><div class="sq-card__title">{esc(title)}</div>'
        f'<div class="{grid}">{cells}</div></div>'
    )


def health_pills(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, kind) where kind is ok|warn|off|error."""
    parts: list[str] = []
    for label, value, kind in items:
        dot = {"ok": "", "warn": " warn", "error": " error", "off": " off"}.get(kind, " off")
        parts.append(
            f'<span class="sq-health-pill"><span class="sq-status-dot{dot}"></span>'
            f'{esc(label)} <strong>{esc(value)}</strong></span>'
        )
    inject_html(f'<div class="sq-health-row">{"".join(parts)}</div>')


def kpi_grid(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    cells = "".join(
        f'<div class="sq-kpi"><div class="l">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in items
    )
    inject_html(f'<div class="sq-kpi-grid">{cells}</div>')


def feature_grid(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    cells = "".join(
        f'<div class="sq-feature"><div class="sq-feature__title">{esc(t)}</div>'
        f'<p class="sq-feature__body">{esc(b)}</p></div>'
        for t, b in items
    )
    inject_html(f'<div class="sq-feature-grid">{cells}</div>')


def empty_state(title: str, body: str, icon: str = "SQ") -> None:
    mark = (icon or "SQ")[:2]
    inject_html(
        f"""
        <div class="sq-empty">
          <div class="sq-empty__mark">{esc(mark)}</div>
          <h3>{esc(title)}</h3>
          <p>{esc(body)}</p>
        </div>
        """
    )


def section_title(title: str, subtitle: str = "", *, eyebrow: str = "") -> None:
    eye = f'<div class="sq-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="sq-section__sub">{esc(subtitle)}</p>' if subtitle else ""
    inject_html(
        f"""
        <div class="sq-section">
          {eye}
          <div class="sq-section__title">{esc(title)}</div>
          {sub}
        </div>
        """
    )


def connection_status_card(
    *,
    connected: bool,
    name: str = "",
    engine: str = "",
    tables: int | str = "—",
    rows: int | str = "—",
    size: str = "—",
) -> None:
    del rows, size  # compact sidebar — unused
    dot = "sq-status-dot" if connected else "sq-status-dot off"
    status = "Connected" if connected else "Not connected"
    inject_html(
        f"""
        <div class="sq-conn">
          <div class="sq-conn__head">
            <span class="{dot}"></span>
            <span class="sq-conn__status">{esc(status)}</span>
          </div>
          <div class="sq-conn__name">{esc(name or "No database")}</div>
          <div class="sq-conn__grid">
            <div>Engine<br/><strong>{esc(engine or "—")}</strong></div>
            <div>Tables<br/><strong>{esc(tables)}</strong></div>
          </div>
        </div>
        """
    )


def brand_block(app_name: str = "SQLMind AI", tagline: str = "Enterprise SQL analytics") -> None:
    inject_html(
        f"""
        <div class="sq-brand">
          <div class="sq-brand__mark">SQ</div>
          <div>
            <div class="sq-brand__name">{esc(app_name)}</div>
            <div class="sq-brand__tag">{esc(tagline)}</div>
          </div>
        </div>
        """
    )


def page_footer(version: str = "") -> None:
    ver = f"v{esc(version)}" if version else "SQLMind"
    inject_html(
        f"""
        <div class="sq-footer">
          <div><strong>{ver}</strong> · Read-only SQL analytics</div>
          <div class="sq-footer__contact">
            <div class="sq-footer__name">Muhammad Junaid</div>
            <div>Phone: 0304-1659295</div>
            <div>Email: <a href="mailto:junaidfazal08@gmail.com">junaidfazal08@gmail.com</a></div>
          </div>
        </div>
        """
    )


def metric_row(items: list[tuple[str, Any]]) -> None:
    """Streamlit metric strip — values coerced to safe display types."""
    if not items:
        return
    cols = st.columns(min(4, len(items)))
    for i, (label, value) in enumerate(items):
        cols[i % len(cols)].metric(label, _metric_display(value))


def _metric_display(value: Any) -> str | int | float:
    """st.metric only accepts scalar-like values — never lists/dicts."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value if value else "—"
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        if "count" in value and isinstance(value["count"], (int, float)):
            return value["count"]
        if "total" in value and isinstance(value["total"], (int, float)):
            return value["total"]
        return len(value)
    return str(value)[:48]
