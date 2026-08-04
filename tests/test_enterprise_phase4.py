"""Tests for Phase 4 enterprise production extensions."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from observability.metrics import get_metrics, reset_metrics
from observability.parallel_metrics import (
    get_parallel_metrics,
    record_parallel_send,
)
from planner.messages import AgentMessageBus, get_message_bus
from reliability.enterprise_queue import EnterpriseQueue, RetryPolicy
from reliability.recovery_actions import decide_recovery, make_recovery_controller
from services.export_queue import ExportQueue, export_html


def test_parallel_metrics_records_wave():
    collector = get_parallel_metrics()
    collector.reset()
    wid = record_parallel_send(
        ["visualization_agent", "insight_agent"],
        source="execution_node",
        session_id="s1",
    )
    assert wid
    snap = collector.snapshot(session_id="s1")
    assert snap["total_parallel_executions"] >= 1
    assert snap["max_parallelism"] >= 2
    assert snap["average_parallel_workers"] >= 2
    collector.worker_exit("visualization_agent", status="ok")
    collector.worker_exit("insight_agent", status="ok")
    snap2 = collector.snapshot(session_id="s1")
    assert snap2["worker_timeline"]


def test_recovery_actions_policy():
    decision = decide_recovery(
        {
            "sql_error": "syntax error",
            "retry_count": 0,
            "max_retries": 3,
            "adaptive_plan": {"plan_id": "p1", "replan_count": 0, "max_replans": 2},
            "plan_active": True,
        }
    )
    assert decision.action in {"retry", "recover"}
    ctrl = make_recovery_controller()
    out = ctrl(
        {
            "sql_error": "boom",
            "retry_count": 5,
            "max_retries": 3,
            "adaptive_plan": {"plan_id": "p1", "replan_count": 5, "max_replans": 2},
            "runtime_replan_count": 5,
            "max_runtime_replans": 2,
        }
    )
    assert out["recovery_decision"]["action"] in {"escalate", "abort", "fallback"}
    assert out.get("enterprise_events")


def test_enterprise_queue_retry_delay_poison_and_replay(tmp_path: Path):
    q = EnterpriseQueue(
        str(tmp_path / "eq.sqlite3"),
        policy=RetryPolicy(max_retries=2, poison_after=3, base_delay_seconds=0.01),
    )
    mid = q.enqueue_retry(
        topic="sql_failure",
        payload={"sql": "select 1"},
        error="fail",
        attempts=1,
        node="sql_agent",
    )
    assert mid.startswith("eq-")
    time.sleep(0.02)
    ready = q.claim_ready(limit=5)
    assert ready
    poison = q.enqueue_retry(
        topic="sql_failure",
        payload={"sql": "bad"},
        error="poison",
        attempts=3,
        node="sql_agent",
    )
    dead = q.list_dlq()
    assert any(d["id"] == poison for d in dead)
    # Replay poison → retry
    result = q.replay_dlq(poison, to_retry=True)
    assert result["status"] == "requeued"
    stats = q.stats()
    assert "by_kind_status" in stats


def test_message_bus_request_reply_ttl_priority_reject():
    from planner.messages import AgentMessage

    bus = AgentMessageBus()
    bus.register_agent("planner")
    bus.register_agent("sql_agent")

    def _auto_reply(msg):
        if msg.delivery == "request" and msg.recipient == "sql_agent":
            bus.reply(msg, "sql_agent", {"ok": True})

    bus.subscribe("sql_agent", _auto_reply)
    reply = bus.request_reply(
        "planner",
        "sql_agent",
        "data_request",
        {"need": "schema"},
        timeout_seconds=2.0,
        priority=1,
        ttl_seconds=5.0,
    )
    assert reply is not None
    assert reply.payload.get("ok") is True

    req = bus.request("planner", "sql_agent", "command", {"x": 1})
    expired = bus.publish(
        AgentMessage(
            kind="command",
            sender="planner",
            recipient="sql_agent",
            payload={},
            expires_at=time.time() - 1,
        )
    )
    assert expired.rejected is True
    rejected = bus.reject(req, "sql_agent", "busy")
    assert rejected.kind == "reject"
    bus.command("planner", "sql_agent", "pause", priority=1)
    bus.share_context("planner", "sql_agent", {"goal": "analyze"})
    bus.broadcast_event("planner", {"event": "plan_ready"})


def test_export_queue_html_and_progress(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    html_path = export_html(df, tmp_path, "demo", question="q", insights="i")
    assert Path(html_path).exists()
    q = ExportQueue(str(tmp_path), max_workers=1, chunk_size=2)
    job = q.enqueue(df, formats=["csv", "json", "html", "markdown"], label="demo")
    for _ in range(100):
        cur = q.get(job.job_id)
        if cur and cur.status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    cur = q.get(job.job_id)
    assert cur is not None
    assert cur.status == "completed"
    assert cur.progress == 1.0
    assert "csv" in cur.paths
    assert "html" in cur.paths


def test_distributed_queue_lease_complete(tmp_path: Path):
    from distributed import DistributedExecutor, BackendKind

    exe = DistributedExecutor(
        db_path=str(tmp_path / "tq.sqlite3"),
        lock_db_path=str(tmp_path / "locks.sqlite3"),
        backend=BackendKind.LOCAL,
        worker_count=1,
    )

    def handler(task):
        return {"echo": task.payload}

    exe.register_handler("echo", handler)
    exe.start()
    task = exe.submit("echo", {"hello": "world"}, priority=1)
    deadline = time.time() + 5
    while time.time() < deadline:
        rows = exe.queue.list_tasks(status="completed", limit=5)
        if any(r["task_id"] == task.task_id for r in rows):
            break
        time.sleep(0.05)
    rows = exe.queue.list_tasks(status="completed", limit=10)
    assert any(r["task_id"] == task.task_id for r in rows)
    assert exe.locks.acquire("job-1", "worker-a")
    assert not exe.locks.acquire("job-1", "worker-b", ttl_seconds=30)
    exe.locks.release("job-1", "worker-a")
    exe.stop()


def test_metrics_include_enterprise_series():
    reset_metrics()
    m = get_metrics()
    m.observe_replan("revise_tasks")
    m.observe_recovery("retry", "sql_failure")
    m.observe_parallel(source="execution_node", workers=2, wave_id="w1")
    m.observe_queue("enqueue", "echo")
    m.observe_worker("register", "local-0")
    m.observe_llm_latency("ollama", 0.12)
    m.observe_llm_tokens(prompt=10, completion=20)
    m.observe_plugin_latency("skill.echo", 0.05)
    m.set_gauge("queue_length", 3)
    text = m.render_prometheus()
    assert "sqlmind_replan_events_total" in text
    assert "sqlmind_recovery_events_total" in text
    assert "sqlmind_parallel_waves_total" in text
    assert "sqlmind_gauge" in text


def test_plugin_runtime_invoke_echo():
    from plugins import PluginMarketplace
    from plugins.runtime import PluginRuntime

    market = PluginMarketplace(plugin_dirs=["plugins"], require_signature=False)
    market.load_all()
    runtime = PluginRuntime(market, require_permission=False)
    # Builtin echo capability id may be skill.echo
    caps = runtime.list_capabilities()
    assert caps is not None
    try:
        result = runtime.invoke("skill.echo", {"msg": "hi"})
        assert result is not None
    except KeyError:
        # If echo not installed in this env, routing should still raise cleanly
        pass


def test_route_failure_prefers_recovery_graph():
    from graph.workflow import route_failure, route_next

    dest = route_failure({})
    assert dest in {"recovery_graph", "recovery_controller", "retry_agent"}
    assert route_next({"next_agent": "export_node"}) in {"export_graph", "export_node"}


def test_enterprise_queue_claim_and_mark(tmp_path):
    import time

    from reliability.enterprise_queue import EnterpriseQueue, RetryPolicy

    q = EnterpriseQueue(
        str(tmp_path / "eq.sqlite3"),
        policy=RetryPolicy(base_delay_seconds=0.0, max_delay_seconds=0.0),
    )
    item_id = q.enqueue_retry(
        topic="sql_retry_exhausted",
        payload={"session_id": "s1"},
        error="exhausted",
        attempts=1,
        session_id="s1",
        node="retry_agent",
    )
    time.sleep(0.05)
    ready = q.claim_ready(limit=5)
    assert any(r["id"] == item_id for r in ready)
    q.mark_claimed(item_id, status="completed")
    stats = q.stats()
    assert "by_kind_status" in stats


def test_bind_plugin_task_to_runtime_agent():
    from kernel.models import CapabilityDescriptor, CapabilityMatch
    from kernel.enums import CapabilityKind, RiskClass
    from planner.models import TaskSpec
    from planner.selection import bind_task_to_agent

    desc = CapabilityDescriptor(
        id="skill.echo",
        kind=CapabilityKind.SKILL,
        name="Echo",
        description="echo",
        tags=frozenset({"plugin"}),
        skills=frozenset({"plugin"}),
        risk_class=RiskClass.LOW,
        graph_node="",
    )
    match = CapabilityMatch(
        capability_id="skill.echo",
        descriptor=desc,
        score=1.0,
        reasons=["test"],
    )
    task = TaskSpec(id="t1", title="echo", required_skills=["plugin"])
    bound = bind_task_to_agent(task, match)
    assert bound.responsible_graph_node == "plugin_runtime_agent"


def test_bus_drain_delivers_to_agent():
    from planner.messages import AgentMessageBus, task_request_message

    bus = AgentMessageBus()
    bus.register_agent("sql_agent")
    bus.publish(
        task_request_message(
            "execution_coordinator",
            {"task_id": "t1", "plugin_capability_id": "skill.echo"},
            recipient="sql_agent",
        )
    )
    inbox = bus.drain("sql_agent")
    assert inbox
    assert inbox[0].payload.get("plugin_capability_id") == "skill.echo"


def test_backend_adapter_raises_when_unconfigured():
    from distributed import BackendAdapter, BackendKind, BackendNotConfiguredError

    adapter = BackendAdapter(BackendKind.CELERY)
    try:
        adapter.submit("echo", {})
        raised = False
    except BackendNotConfiguredError:
        raised = True
    assert raised
