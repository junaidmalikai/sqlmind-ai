"""Enterprise Runtime — operations console (presentation only)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import Settings
from ui.components import (
    card,
    empty_state,
    health_pills,
    kpi_grid,
    kv_card,
    metric_row,
    page_footer,
    page_header,
    section_title,
    status_badge,
)
from ui.session import get_orchestrator
from ui.ui_helpers import esc, inject_html


def render_runtime_ops(settings: Settings) -> None:
    page_header(
        "Runtime",
        "System health, agents, queues, plugins, and reliability",
        settings=settings,
    )

    orch = get_orchestrator()
    if orch is None:
        empty_state(
            "Runtime idle",
            "Connect a database to start the orchestrator and activate live monitoring.",
            icon="RT",
        )
        section_title("Telemetry", "Configured endpoints")
        kv_card(
            "Observability",
            [
                ("Prometheus", f"http://{settings.metrics_http_host}:{settings.metrics_http_port}/metrics"),
                ("Runtime API", f"http://{settings.metrics_http_host}:{settings.metrics_http_port}/runtime"),
                ("Metrics file", settings.metrics_export_path),
            ],
        )
        _render_primer()
        page_footer(settings.app_version)
        return

    stats = orch.runtime_dashboard_stats()
    metrics_url = stats.get("metrics_url") or (
        f"http://{settings.metrics_http_host}:{settings.metrics_http_port}"
    )
    parallel = _as_dict(stats.get("parallel"))
    exports = _as_dict(stats.get("exports"))
    dist = _as_dict(stats.get("distributed"))
    eq = _as_dict(stats.get("enterprise_queue"))
    plugins = _as_dict(stats.get("plugins"))
    breakers = _as_dict(stats.get("circuit_breakers"))
    iam = _as_dict(stats.get("iam"))
    agents = stats.get("agents_online") or []
    if not isinstance(agents, list):
        agents = []

    workers_n = _countish(dist.get("healthy_workers"), fallback=_countish(dist.get("workers")))
    exports_n = _countish(exports.get("completed"))
    waves_n = _countish(parallel.get("total_parallel_executions"))
    max_par = _countish(parallel.get("max_parallelism"))
    bus_depth = _countish(stats.get("message_queue_depth"))

    # —— System health ——
    section_title("System health", "Live session overview")
    health_pills(
        [
            ("Orchestrator", "Online", "ok"),
            ("Agents", str(len(agents)), "ok" if agents else "off"),
            ("Workers", str(workers_n), "ok" if workers_n else "off"),
            ("Exports", str(exports_n), "ok"),
            ("Breakers", _breaker_summary(breakers), _breaker_kind(breakers)),
        ]
    )
    kpi_grid(
        [
            ("Parallel waves", str(waves_n)),
            ("Max parallelism", str(max_par)),
            ("Healthy workers", str(workers_n)),
            ("Exports done", str(exports_n)),
        ]
    )

    inject_html(
        f"""
        <div class="sq-card">
          <div class="sq-card__title">Endpoints</div>
          <div class="sq-kv-grid sq-kv-grid--3">
            <div><span class="sq-kv-label">Prometheus</span><strong>{esc(metrics_url)}/metrics</strong></div>
            <div><span class="sq-kv-label">Runtime API</span><strong>{esc(metrics_url)}/runtime</strong></div>
            <div><span class="sq-kv-label">Message bus</span><strong>{esc(bus_depth)}</strong></div>
          </div>
        </div>
        """
    )

    tab_overview, tab_agents, tab_parallel, tab_queues, tab_plugins, tab_reliab, tab_obs = st.tabs(
        [
            "Overview",
            "Agents",
            "Parallel",
            "Queues",
            "Plugins",
            "Reliability",
            "Observability",
        ]
    )

    with tab_overview:
        _render_primer()
        section_title("Memory & planner", "Session fabric")
        recovery = _as_dict(stats.get("recovery"))
        replanning = _as_dict(stats.get("replanning"))
        gauges = stats.get("gauges") or {}
        gauge_n = len(gauges) if isinstance(gauges, dict) else _countish(gauges)
        kpi_grid(
            [
                ("Replans", str(_countish(_dig(replanning, "count", "total", default=0)))),
                ("Recovery", str(_dig(recovery, "state", "status", default="idle"))),
                ("Active gauges", str(gauge_n)),
                ("Bus depth", str(bus_depth)),
            ]
        )

    with tab_agents:
        section_title("Agent registry", "Discoverable capabilities")
        inject_html(
            f'<div class="sq-badge-row sq-badge-row--flush">'
            f'{status_badge(f"{len(agents)} registered", kind="accent")}'
            f'{status_badge(f"Bus depth {bus_depth}", kind="info")}'
            f"</div>"
        )
        if agents:
            # Normalize to tabular rows when possible
            if agents and isinstance(agents[0], dict):
                st.dataframe(agents, use_container_width=True, hide_index=True)
            else:
                rows = [{"agent": str(a)} for a in agents]
                st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            empty_state(
                "No agents online",
                "Agents appear here once the orchestrator registers capabilities.",
                icon="AG",
            )

    with tab_parallel:
        section_title("Parallel execution", "Waves and workers")
        kpi_grid(
            [
                ("Total waves", str(waves_n)),
                ("Max parallelism", str(max_par)),
                (
                    "Active workers",
                    str(
                        _countish(
                            parallel.get("active_workers"),
                            fallback=workers_n,
                        )
                    ),
                ),
                ("Timeline events", str(_countish(parallel.get("worker_timeline")))),
            ]
        )
        timeline = parallel.get("worker_timeline") or []
        if isinstance(timeline, list) and timeline:
            section_title("Worker timeline")
            st.dataframe(timeline, use_container_width=True, hide_index=True)
        _raw_expander("Raw parallel metrics", parallel)

    with tab_queues:
        section_title("Distributed workers", "Local in-process pool")
        total_w = _countish(dist.get("total_workers"), fallback=_countish(dist.get("workers")))
        pending = _countish(dist.get("pending"), fallback=_countish(dist.get("queued")))
        kpi_grid(
            [
                ("Healthy", str(workers_n)),
                ("Total", str(total_w)),
                ("Pending", str(pending)),
                ("Pool", "Local" if workers_n or total_w else "Idle"),
            ]
        )
        worker_list = dist.get("workers")
        if isinstance(worker_list, list) and worker_list and isinstance(worker_list[0], dict):
            section_title("Worker roster")
            st.dataframe(worker_list, use_container_width=True, hide_index=True)
        _raw_expander("Distributed details", dist)

        section_title("Enterprise queue", "Retry · delayed · DLQ · poison")
        kpi_grid(
            [
                ("Retry", str(_countish(_dig(eq, "retry", "retry_count", default=0)))),
                ("Delayed", str(_countish(_dig(eq, "delayed", "delayed_count", default=0)))),
                ("Dead letters", str(_countish(_dig(eq, "dlq", "dead_letter", default=0)))),
                ("Poison", str(_countish(_dig(eq, "poison", "poison_count", default=0)))),
            ]
        )
        _raw_expander("Queue details", eq)

        if hasattr(orch, "enterprise_queue"):
            try:
                dead = orch.enterprise_queue.list_dlq(limit=20)
            except Exception:  # noqa: BLE001
                dead = []
            if dead:
                section_title("Dead / poison letters")
                st.dataframe(dead, use_container_width=True, hide_index=True)
                replay_id = st.text_input("Dead-letter id to replay", key="runtime_dlq_replay")
                if st.button("Replay to retry queue", use_container_width=True) and replay_id:
                    try:
                        result = orch.enterprise_queue.replay_dlq(replay_id)
                        st.success(f"Replayed · {result}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

    with tab_plugins:
        section_title("Plugin runtime", "Marketplace skills")
        if plugins:
            items = [(str(k), _display_val(v)) for k, v in list(plugins.items())[:8]]
            kv_card("Plugin status", items)
        else:
            card(title="Plugins", body="No plugin telemetry yet for this session.")
        caps: list[Any] = []
        try:
            caps = orch.plugin_runtime.list_capabilities()
        except Exception:  # noqa: BLE001
            pass
        if caps:
            st.dataframe(caps, use_container_width=True, hide_index=True)
        if st.button("Run echo skill health check", use_container_width=True):
            try:
                result = orch.plugin_runtime.invoke("skill.echo", {"probe": True})
                st.success(str(result))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        _raw_expander("Raw plugin metrics", plugins)

    with tab_reliab:
        section_title("Circuit breakers", "Failure isolation")
        if breakers:
            rows = []
            for name, state in breakers.items():
                if isinstance(state, dict):
                    rows.append(
                        {
                            "breaker": name,
                            "state": state.get("state", state.get("status", "—")),
                            "failures": state.get("failures", state.get("failure_count", "—")),
                        }
                    )
                else:
                    rows.append({"breaker": name, "state": str(state), "failures": "—"})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            card(title="Circuit breakers", body="No breaker events recorded yet.")
        _raw_expander("Raw breaker state", breakers)

        section_title("IAM", "Identity and access")
        if iam:
            kv_card("IAM", [(str(k), _display_val(v)) for k, v in list(iam.items())[:6]])
        else:
            card(title="IAM", body="IAM enforcement is idle or not enabled for this session.")
        _raw_expander("Raw IAM state", iam)

        section_title("Recovery", "Replanning and gauges")
        kpi_grid(
            [
                (
                    "Replans",
                    str(_countish(_dig(_as_dict(stats.get("replanning")), "count", "total", default=0))),
                ),
                (
                    "Recovery",
                    str(_dig(_as_dict(stats.get("recovery")), "state", "status", default="idle")),
                ),
                (
                    "Gauges",
                    str(len(stats.get("gauges") or {}) if isinstance(stats.get("gauges"), dict) else 0),
                ),
                ("Breakers", _breaker_summary(breakers)),
            ]
        )
        _raw_expander(
            "Raw recovery state",
            {
                "replanning": stats.get("replanning"),
                "recovery": stats.get("recovery"),
                "gauges": stats.get("gauges"),
            },
        )

    with tab_obs:
        section_title("Timelines", "Recent enterprise events")
        timelines = stats.get("timelines") or {}
        if not timelines:
            empty_state(
                "No timeline events",
                "Events appear after workflow and export activity.",
                icon="TL",
            )
        else:
            for kind, rows in timelines.items():
                label = str(kind).replace("_", " ").title()
                recent = rows[-8:] if isinstance(rows, list) else rows
                with st.expander(label, expanded=False):
                    if isinstance(recent, list) and recent and isinstance(recent[0], dict):
                        st.dataframe(recent, use_container_width=True, hide_index=True)
                    else:
                        st.write(recent)

        section_title("Export jobs")
        if exports:
            kv_card(
                "Export metrics",
                [(str(k), _display_val(v)) for k, v in list(exports.items())[:8]],
            )
        else:
            card(title="Exports", body="No export jobs recorded yet.")
        _raw_expander("Raw export metrics", exports)

    page_footer(settings.app_version)


def _raw_expander(label: str, data: Any) -> None:
    with st.expander(label, expanded=False):
        st.json(data if data is not None else {})


def _as_dict(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else {}


def _countish(value: Any, *, fallback: Any = 0) -> int | str:
    """Coerce nested/list/dict telemetry into a scalar for display."""
    if value is None:
        if fallback is None:
            return 0
        return _countish(fallback) if fallback != 0 else 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return value if value else 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, dict):
        for k in ("count", "total", "size", "depth", "completed", "pending"):
            if k in value and isinstance(value[k], (int, float)):
                return int(value[k])
        return len(value)
    return fallback if isinstance(fallback, (int, str)) else 0


def _display_val(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple, set)):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    return str(value)


def _dig(obj: Any, *keys: str, default: Any = "—") -> Any:
    if not isinstance(obj, dict):
        return default
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _breaker_summary(breakers: Any) -> str:
    if not isinstance(breakers, dict) or not breakers:
        return "Closed"
    open_n = 0
    for v in breakers.values():
        state = (v.get("state") or v.get("status") or "") if isinstance(v, dict) else str(v)
        if str(state).lower() in {"open", "half_open", "half-open"}:
            open_n += 1
    return f"{open_n} open" if open_n else "Closed"


def _breaker_kind(breakers: Any) -> str:
    summary = _breaker_summary(breakers)
    if summary.endswith("open") and not summary.startswith("0"):
        return "warn"
    return "ok"


def _render_primer() -> None:
    section_title("How the runtime works", "Control plane at a glance")
    inject_html(
        """
        <div class="sq-layer-grid">
          <div class="sq-layer sq-layer--control">
            <div class="sq-layer__label">Execution</div>
            <div class="sq-layer__title">Plan → specialists</div>
            <p class="sq-layer__body">Question → validate → execute → analytics</p>
          </div>
          <div class="sq-layer sq-layer--data">
            <div class="sq-layer__label">Memory</div>
            <div class="sq-layer__title">Session fabric</div>
            <p class="sq-layer__body">Conversation · episodic · workflow · vector</p>
          </div>
          <div class="sq-layer sq-layer--analytics">
            <div class="sq-layer__label">Parallel</div>
            <div class="sq-layer__title">Viz ∥ Insight</div>
            <p class="sq-layer__body">Fan-out after query · join · finalize</p>
          </div>
          <div class="sq-layer sq-layer--platform">
            <div class="sq-layer__label">Platform</div>
            <div class="sq-layer__title">Operate &amp; recover</div>
            <p class="sq-layer__body">Queues · breakers · IAM · plugins · metrics</p>
          </div>
        </div>
        """
    )
