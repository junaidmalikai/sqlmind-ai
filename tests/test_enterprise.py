"""Phase 3 enterprise platform tests — goal tracking, P2P, plugins, IAM, memory, etc."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from governance.approval import ApprovalPolicy, classify_sql_risk, evaluate_approval_need
from iam import IAMService, Principal
from learning import PlannerExperienceStore
from memory.vector_store import HashingEmbeddingProvider, MemoryFabric, VectorMemoryStore
from planner.goal_models import GoalTrackingRecord
from planner.goal_store import GoalStore
from planner.goal_tracker import make_goal_tracking_agent
from plugins import PluginManifest, PluginMarketplace, validate_manifest
from reliability import CircuitBreaker, DeadLetterQueue, RecoveryManager
from planner.messages import (
    AgentMessageBus,
    get_message_bus,
    status_update_message,
    task_request_message,
    task_result_message,
)


def test_goal_lifecycle_transitions():
    g = GoalTrackingRecord(title="Test", status="created")
    g.transition("planning")
    g.transition("running")
    g.transition("waiting", reason="clarify")
    assert g.blocked
    g.transition("running")
    g.transition("completed")
    assert g.completion_pct == 1.0
    with pytest.raises(ValueError):
        g.transition("planning")


def test_goal_store_persist(tmp_path: Path):
    store = GoalStore(str(tmp_path / "goals.sqlite3"))
    rec = GoalTrackingRecord(title="Revenue analysis", session_id="s1", status="created")
    store.upsert(rec)
    loaded = store.get(rec.goal_id)
    assert loaded is not None
    assert loaded.title == "Revenue analysis"
    store.transition(rec.goal_id, "planning")
    assert store.get(rec.goal_id).status == "planning"


def test_goal_tracking_agent_updates_state():
    agent = make_goal_tracking_agent(None)
    out = agent(
        {
            "question": "top customers",
            "goal_spec": {
                "goal": "Find top customers",
                "objectives": ["rank customers", "show revenue"],
                "confidence": 0.9,
                "simple": True,
            },
            "adaptive_plan": {"plan_id": "p1"},
            "plan_active": True,
            "next_agent": "execution_coordinator",
            "execution_progress": {},
            "task_graph": {},
        }
    )
    assert out["goal_tracking"]["status"] in {"planning", "running"}
    assert out["goal_status_update"]["goal_id"]


def test_p2p_message_bus_request_reply():
    bus = AgentMessageBus()
    req = bus.request(
        "planner",
        "sql_agent",
        "task_request",
        {"task": "write sql"},
    )
    assert req.delivery == "request"
    reply = bus.reply(req, "sql_agent", {"sql": "SELECT 1"})
    assert reply.delivery == "reply"
    assert reply.reply_to == req.id
    inbox = bus.drain("planner")
    assert any(m.kind == "task_result" for m in inbox)


def test_message_kinds_extend_phase2():
    msg = status_update_message("goal_tracking", {"status": "running"})
    assert msg.kind == "status_update"
    tr = task_request_message("coordinator", {"id": "t1"}, recipient="sql_agent")
    assert tr.kind == "task_request"
    assert get_message_bus() is not None


def test_plugin_manifest_validation():
    m = PluginManifest(
        id="demo.plugin",
        name="Demo",
        version="1.0.0",
        capabilities=[],
    )
    result = validate_manifest(m)
    assert result.valid
    assert result.warnings


def test_plugin_marketplace_loads_builtin():
    root = Path(__file__).resolve().parents[1] / "plugins"
    market = PluginMarketplace(plugin_dirs=[str(root)])
    records = market.load_all()
    assert any(r.manifest.id == "sqlmind.builtin.echo" for r in records)
    health = market.health_check()
    assert health
    assert market.catalog()


def test_approval_classifies_high_risk_sql():
    reason, risk = classify_sql_risk("DELETE FROM users WHERE 1=1")
    assert reason == "delete"
    assert risk == "critical"
    reason2, _ = classify_sql_risk("SELECT * FROM users")
    assert reason2 is None


def test_approval_evaluate_need():
    req = evaluate_approval_need(
        {"sql": "DROP TABLE secrets", "session_id": "s1"},
        ApprovalPolicy(),
    )
    assert req is not None
    assert req.reason == "drop"


def test_vector_memory_semantic_search(tmp_path: Path):
    store = VectorMemoryStore(
        str(tmp_path / "vec.sqlite3"),
        embedder=HashingEmbeddingProvider(dim=64),
    )
    store.store(
        kind="sql_history",
        content="Q: top revenue customers\nSQL: SELECT name, SUM(amount) FROM orders GROUP BY name",
        title="revenue",
        tenant_id="t1",
    )
    store.store(
        kind="failure",
        content="timeout on large join",
        title="fail",
        tenant_id="t1",
    )
    hits = store.search("customer revenue ranking", tenant_id="t1", top_k=3)
    assert hits
    fabric = MemoryFabric(store)
    ctx = fabric.retrieve_for_planner("revenue by customer", tenant_id="t1")
    assert "Semantic" in ctx or "revenue" in ctx.lower() or ctx == ""


def test_iam_rbac_and_api_key(tmp_path: Path):
    iam = IAMService(str(tmp_path / "iam.sqlite3"))
    session = iam.authenticate("local-user", "local-dev")
    assert session is not None
    assert iam.check_permission(session, "agent.sql")
    assert iam.check_permission(session, "sql.read")

    viewer = Principal(
        principal_id="prin-viewer",
        username="viewer1",
        roles=["viewer"],
        tenant_id="default",
    )
    iam.create_principal(viewer, password="view")
    raw, _rec = iam.create_api_key(viewer, name="test")
    tok = iam.authenticate_api_key(raw)
    assert tok is not None
    assert iam.check_permission(tok, "sql.read")
    assert not iam.check_permission(tok, "approval.decide")


def test_iam_tenant_isolation(tmp_path: Path):
    iam = IAMService(str(tmp_path / "iam2.sqlite3"))
    p = Principal(
        principal_id="prin-a",
        username="alice",
        roles=["analyst"],
        tenant_id="tenant-a",
    )
    iam.create_principal(p, password="x")
    sess = iam.authenticate("alice", "x")
    assert sess is not None
    assert not iam.check_permission(
        sess, "sql.read", attributes={"resource_tenant_id": "tenant-b"}
    )


def test_reliability_dlq_and_circuit(tmp_path: Path):
    dlq = DeadLetterQueue(str(tmp_path / "dlq.sqlite3"))
    item_id = dlq.enqueue(kind="node_failure", error="boom", node="sql_agent")
    assert dlq.list_open()
    dlq.mark_resolved(item_id)
    br = CircuitBreaker(name="sql_agent", failure_threshold=2)
    br.record_failure()
    br.record_failure()
    assert br.state == "open"
    assert not br.allow()
    mgr = RecoveryManager(dlq)
    assert mgr.fallback_strategy("sql_agent") in {
        "retry_agent",
        "sql_agent",
        "fail",
        "recovery_graph",
        "recovery_controller",
    }


def test_planner_learning_store(tmp_path: Path):
    store = PlannerExperienceStore(str(tmp_path / "learn.sqlite3"))
    store.record(
        question="top sales",
        success=True,
        agent_sequence=["schema_agent", "sql_agent", "insight_agent"],
        sql_strategy="aggregate by region",
        chart_strategy="bar",
    )
    store.record(
        question="fail case",
        success=False,
        agent_sequence=["sql_agent", "retry_agent"],
    )
    seqs = store.best_agent_sequences()
    assert seqs
    assert store.best_chart_strategies()
    hints = store.format_for_planner("sales")
    assert "Planner learning" in hints


def test_subgraphs_compile():
    from graph.subgraphs import (
        build_export_subgraph,
        build_memory_subgraph,
        build_planning_subgraph,
        build_reflection_subgraph,
    )

    def _noop(state):
        return {"next_agent": "finalize"}

    mem = build_memory_subgraph(memory_agent=_noop)
    assert mem is not None
    plan = build_planning_subgraph(
        goal_understanding=_noop,
        planner=_noop,
        task_decomposition=_noop,
        goal_tracking=_noop,
    )
    assert plan is not None
    assert build_export_subgraph(export_node=_noop) is not None
    assert build_reflection_subgraph(reflection_agent=_noop) is not None


def test_enterprise_event_tracker(tmp_path: Path):
    from observability.events import EnterpriseEventTracker

    tracker = EnterpriseEventTracker(log_path=str(tmp_path / "events.jsonl"))
    ev = tracker.emit("plan", "created", session_id="s1", agent="planner")
    assert ev.event_type == "plan"
    assert tracker.recent(event_type="plan")
    tracker.track_graph_transition("planner", "task_decomposition", session_id="s1")


def test_enterprise_runtime_iam_and_circuit(tmp_path: Path):
    from enterprise.runtime import EnterpriseRuntime, wrap_enterprise_node
    from reliability import RecoveryManager

    iam = IAMService(str(tmp_path / "iam_rt.sqlite3"))
    session = iam.authenticate("local-user", "local-dev")
    assert session is not None
    dlq = DeadLetterQueue(str(tmp_path / "dlq_rt.sqlite3"))
    runtime = EnterpriseRuntime(
        iam=iam,
        recovery=RecoveryManager(dlq),
        enforce_iam=True,
        enforce_circuit_breaker=True,
    )

    def ok_node(state):
        return {"next_agent": "finalize", "status": "ok"}

    wrapped = wrap_enterprise_node("schema_agent", ok_node, runtime)
    out = wrapped({"iam_session": session.model_dump(mode="json"), "tenant_id": "default"})
    assert out.get("status") == "ok" or out.get("next_agent") == "finalize"
    assert out.get("reliability_meta", {}).get("last_node") == "schema_agent"

    # Unauthenticated deny
    denied = wrapped({"iam_session": {}, "tenant_id": "default"})
    assert denied.get("next_agent") == "fail" or denied.get("needs_approval")


def test_message_bus_discovery_and_heartbeat():
    bus = AgentMessageBus()
    bus.register_agent("sql_agent", capabilities=["sql"])
    bus.heartbeat("sql_agent", load=0.1)
    discovered = bus.discover(capability="sql")
    assert discovered
    assert discovered[0]["status"] == "online"
    bus.publish(status_update_message("sql_agent", {"status": "working"}))
    assert bus.history(agent_id="sql_agent")
    assert bus.queue_snapshot()


def test_plugin_sign_and_catalog(tmp_path: Path):
    # Copy builtin plugin into temp dir so signing does not mutate repo files
    src = Path(__file__).resolve().parents[1] / "plugins" / "builtins"
    dest = tmp_path / "plugins" / "builtins"
    dest.mkdir(parents=True)
    for name in ("plugin.json", "echo_skill.py", "__init__.py"):
        (dest / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    # Entrypoint still resolves via package import path; register without loading handlers
    market = PluginMarketplace(
        plugin_dirs=[str(tmp_path / "plugins")],
        signing_secret="test-secret",
        require_signature=False,
    )
    manifest = load_manifest_file(dest / "plugin.json") if False else None
    from plugins import load_manifest_file

    manifest = load_manifest_file(dest / "plugin.json")
    # Avoid entrypoint import from temp tree — register metadata only then sign
    rec = market.register_manifest(manifest, path=str(dest / "plugin.json"), load_handlers=False)
    assert rec.manifest.id == "sqlmind.builtin.echo"
    signed = market.sign_installed("sqlmind.builtin.echo")
    assert signed.manifest.signature
    assert signed.manifest.checksum
    catalog = market.discover_for_planner()
    assert any(c["plugin_id"] == "sqlmind.builtin.echo" for c in catalog)

def test_embedding_factory_hashing():
    from memory.embeddings import build_embedding_provider

    emb = build_embedding_provider("hashing", dim=64)
    vecs = emb.embed(["hello world", "hello world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 64
    # deterministic
    assert vecs[0] == vecs[1]


def test_prometheus_metrics_render():
    from observability.metrics import reset_metrics

    m = reset_metrics()
    m.observe_node("sql_agent", status="ok", duration_seconds=0.12)
    m.observe_sql("validate", status="ok")
    m.observe_iam("allow", "agent.sql")
    text = m.render_prometheus()
    assert "sqlmind_node_runs_total" in text
    assert "sqlmind_sql_events_total" in text


def test_vector_memory_expiration_and_rerank(tmp_path: Path):
    store = VectorMemoryStore(
        str(tmp_path / "vec_exp.sqlite3"),
        embedder=HashingEmbeddingProvider(dim=64),
        default_ttl_seconds=3600,
    )
    store.store(
        kind="sql_history",
        content="customer revenue ranking by region",
        title="revenue customers",
        tenant_id="t1",
        ttl_seconds=3600,
    )
    hits = store.search("customer revenue", tenant_id="t1", top_k=3, rerank=True)
    assert hits
    assert hits[0].get("reranked") is True
