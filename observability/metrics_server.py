"""Production metrics HTTP endpoint — Prometheus scrape target.

Serves ``GET /metrics`` (and ``GET /health``, ``GET /runtime``) without
replacing Streamlit. Start via settings ``METRICS_HTTP_ENABLED=true``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from utils.logging_config import get_logger

logger = get_logger(__name__)

_SERVER: ThreadingHTTPServer | None = None
_THREAD: threading.Thread | None = None
_LOCK = threading.Lock()


def _runtime_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    try:
        from observability.metrics import get_metrics

        m = get_metrics()
        snap["gauges"] = dict(m.gauge_values)
        snap["timelines"] = {k: v[-10:] for k, v in m.all_timelines().items()}
    except Exception as exc:  # noqa: BLE001
        snap["metrics_error"] = str(exc)
    try:
        from observability.parallel_metrics import get_parallel_metrics

        snap["parallel"] = get_parallel_metrics().snapshot()
    except Exception as exc:  # noqa: BLE001
        snap["parallel_error"] = str(exc)
    try:
        from reliability.enterprise_queue import get_enterprise_queue

        snap["enterprise_queue"] = get_enterprise_queue().stats()
    except Exception as exc:  # noqa: BLE001
        snap["queue_error"] = str(exc)
    try:
        from distributed import get_distributed_executor

        snap["distributed"] = get_distributed_executor(autostart=False).stats()
    except Exception as exc:  # noqa: BLE001
        snap["distributed_error"] = str(exc)
    try:
        from services.export_queue import get_export_queue

        snap["exports"] = get_export_queue().stats()
    except Exception as exc:  # noqa: BLE001
        snap["exports_error"] = str(exc)
    try:
        from planner.messages import get_message_bus

        bus = get_message_bus()
        snap["message_bus"] = {
            "agents": bus.discover(),
            "queue_depth": len(bus.queue_snapshot(limit=500)),
            "history": len(bus.history(limit=500)),
        }
    except Exception as exc:  # noqa: BLE001
        snap["bus_error"] = str(exc)
    return snap


class _MetricsHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logger.debug("metrics_http: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/metrics", "/metrics/"}:
            try:
                from observability.metrics import get_metrics

                body = get_metrics().render_prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
            return
        if path in {"/health", "/healthz"}:
            self._json(200, {"status": "ok", "service": "SQLMind-AI"})
            return
        if path in {"/runtime", "/runtime/stats"}:
            self._json(200, _runtime_snapshot())
            return
        self._json(404, {"error": "not_found", "paths": ["/metrics", "/health", "/runtime"]})

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_metrics_server(host: str = "127.0.0.1", port: int = 9108) -> str:
    """Start background Prometheus HTTP server. Returns bind URL."""
    global _SERVER, _THREAD
    with _LOCK:
        if _SERVER is not None:
            return f"http://{host}:{_SERVER.server_address[1]}"
        server = ThreadingHTTPServer((host, port), _MetricsHandler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="sqlmind-metrics-http",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        _THREAD = thread
        url = f"http://{host}:{port}"
        logger.info("Prometheus metrics endpoint listening at %s/metrics", url)
        return url


def stop_metrics_server() -> None:
    global _SERVER, _THREAD
    with _LOCK:
        if _SERVER is not None:
            _SERVER.shutdown()
            _SERVER = None
            _THREAD = None
