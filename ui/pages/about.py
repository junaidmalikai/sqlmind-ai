"""About — product story, architecture, and AI agent catalog."""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from ui.components import card, feature_grid, page_footer, page_header, section_title, status_badge
from ui.ui_helpers import esc, inject_html


# Agent catalog for presentation only — mirrors kernel/bootstrap registrations.
# Descriptions match implemented behavior; optional nodes are marked clearly.
_AI_AGENTS: list[dict[str, str]] = [
    {
        "name": "Memory Agent",
        "purpose": "Load long-term context before planning.",
        "responsibility": "Retrieve workflow memory, vector memory (when enabled), and planner learning hints.",
        "input": "User question, database name, tenant id",
        "output": "workflow_memory_context, vector_memory_context, learning_context",
        "role": "Upstream context provider",
        "position": "First step when autonomy planning is enabled",
        "enterprise": "Feeds the Memory Fabric into Goal Understanding and the Planner",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Goal Understanding",
        "purpose": "Turn a natural-language question into a structured goal.",
        "responsibility": "Extract objectives, constraints, confidence, ambiguity flags, and rewrite the question.",
        "input": "Question plus memory summaries",
        "output": "GoalSpec (goal, objectives, confidence, intent)",
        "role": "Intent extraction AI Agent",
        "position": "After Memory Agent; may route to Clarify on low confidence",
        "enterprise": "Defines the measurable objective the rest of the workflow must satisfy",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Planner",
        "purpose": "Create an adaptive execution plan from the goal.",
        "responsibility": "Select strategy and skill sequence using the Agent Registry catalog and past experience.",
        "input": "GoalSpec, agent catalog, memory / learning context",
        "output": "AdaptivePlan",
        "role": "Strategy AI Agent",
        "position": "After Goal Understanding; before Task Decomposition",
        "enterprise": "Plans work against discoverable capabilities instead of hardcoded routes",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Task Decomposition",
        "purpose": "Break the plan into a task graph.",
        "responsibility": "Create tasks with skills, dependencies, and expected outputs; bind each task to an AI Agent via the registry.",
        "input": "GoalSpec, AdaptivePlan, agent catalog",
        "output": "TaskGraph with agent bindings",
        "role": "Work breakdown AI Agent",
        "position": "After Planner; before Execution Coordinator",
        "enterprise": "Enables dependency-aware and parallel-safe scheduling",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Goal Tracking",
        "purpose": "Monitor goal lifecycle during execution.",
        "responsibility": "Track progress, blocked or failed goals, and notify the Planner.",
        "input": "Goal and execution progress",
        "output": "Updated goal status / progress signals",
        "role": "Progress monitor",
        "position": "Between decomposition and coordination when enterprise mode is on",
        "enterprise": "Persists goal lifecycle for observability and recovery",
        "optional": "Requires autonomy planning and enterprise mode",
    },
    {
        "name": "Execution Coordinator",
        "purpose": "Schedule and dispatch the task graph.",
        "responsibility": "Respect dependencies, run parallel-safe tasks, track progress, and hand failures to replanning.",
        "input": "TaskGraph, AdaptivePlan",
        "output": "Specialist invocations and execution progress",
        "role": "Deterministic scheduler",
        "position": "Control plane after planning; may fall back to Supervisor",
        "enterprise": "Coordinates multi-agent work without skipping security gates",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Supervisor",
        "purpose": "Route the next capability via tool calling.",
        "responsibility": "Select the next AI Agent or manager using LLM bind_tools against the live capability registry.",
        "input": "Current GraphState and registry catalog",
        "output": "next_agent / route decision",
        "role": "Outer orchestration AI Agent",
        "position": "Primary router when planning is off or as coordinator fallback",
        "enterprise": "Registry-driven routing — new capabilities become selectable without code changes",
        "optional": "",
    },
    {
        "name": "Schema Agent",
        "purpose": "Load and interpret database schema context.",
        "responsibility": "Refresh schema text and suggest analytical questions.",
        "input": "Connected database / schema snapshot",
        "output": "schema_text, suggested questions",
        "role": "Schema specialist AI Agent",
        "position": "Early specialist step before SQL authoring",
        "enterprise": "Grounds SQL generation in live metadata",
        "optional": "",
    },
    {
        "name": "SQL Agent",
        "purpose": "Author read-only SQL with a ReAct tool loop.",
        "responsibility": "Use schema, validate, explain, and submit tools; emit SQL for the security gate.",
        "input": "Question, schema context, dialect",
        "output": "Candidate SQL for validation",
        "role": "NL→SQL specialist AI Agent",
        "position": "Before Validation; never executes against the database itself",
        "enterprise": "Separates generation from enforcement — LLM cannot bypass validation",
        "optional": "",
    },
    {
        "name": "Validation",
        "purpose": "Enforce read-only SQL security.",
        "responsibility": "Deterministic sqlglot AST checks; flag high-risk statements for approval when configured.",
        "input": "Candidate SQL",
        "output": "sql_valid / errors; optional approval routing",
        "role": "Deterministic security node (not an AI Agent)",
        "position": "Hard-wired after SQL Agent; before Execution or Approval",
        "enterprise": "Security invariant — every statement is gated before DB I/O",
        "optional": "",
    },
    {
        "name": "Approval",
        "purpose": "Human approval for high-risk actions.",
        "responsibility": "Pause via LangGraph interrupt for high-risk SQL, sensitive exports, or long tasks.",
        "input": "Pending high-risk action",
        "output": "Approved or rejected resume signal",
        "role": "Governance HITL gate",
        "position": "Between Validation and Execution when approval gates are enabled",
        "enterprise": "Human-in-the-loop governance for elevated risk",
        "optional": "Requires enterprise mode and approval gates",
    },
    {
        "name": "Execution",
        "purpose": "Run validated read-only SQL.",
        "responsibility": "Execute with row caps and timeouts; return tabular results.",
        "input": "Validated SQL",
        "output": "Rows, columns, timing, success flags",
        "role": "Deterministic execution node (not an AI Agent)",
        "position": "After Validation (and Approval when required)",
        "enterprise": "Server-side limits and audit-friendly I/O",
        "optional": "",
    },
    {
        "name": "Visualization",
        "purpose": "Recommend charts for the result set.",
        "responsibility": "Produce a chart type and Plotly-ready specification.",
        "input": "Query result dataframe metadata",
        "output": "chart_spec / chart_type",
        "role": "Visualization AI Agent",
        "position": "Parallel with Insight after successful execution (legacy path) or via plan",
        "enterprise": "Turns tabular output into visual analytics",
        "optional": "",
    },
    {
        "name": "Insight",
        "purpose": "Generate business insights from results.",
        "responsibility": "Summarize findings, trends, anomalies, and recommendations.",
        "input": "Query results and question context",
        "output": "insights / structured insight fields",
        "role": "Analytics AI Agent",
        "position": "Parallel with Visualization after successful execution",
        "enterprise": "Converts raw rows into decision-oriented narrative",
        "optional": "",
    },
    {
        "name": "Join",
        "purpose": "Synchronize parallel post-query work.",
        "responsibility": "Barrier after Visualization and Insight fan-out; return to Coordinator or Supervisor.",
        "input": "Parallel branch outputs",
        "output": "Merged state for next routing step",
        "role": "Deterministic barrier",
        "position": "After parallel Visualization + Insight",
        "enterprise": "Safe join point for parallel analytics",
        "optional": "",
    },
    {
        "name": "Dashboard",
        "purpose": "Design a KPI dashboard blueprint.",
        "responsibility": "Propose KPIs, charts, and alerts for the connected database.",
        "input": "Schema context",
        "output": "dashboard_spec",
        "role": "Dashboard design AI Agent",
        "position": "Specialist invoked by Supervisor or plan when dashboard intent is detected",
        "enterprise": "Blueprints executive views from metadata",
        "optional": "",
    },
    {
        "name": "Summary",
        "purpose": "Produce an executive database summary.",
        "responsibility": "Describe purpose, important tables, metrics, and suggested questions.",
        "input": "Schema context",
        "output": "database_summary structured model",
        "role": "Documentation AI Agent",
        "position": "Specialist for schema/documentation intents",
        "enterprise": "Accelerates onboarding to unfamiliar databases",
        "optional": "",
    },
    {
        "name": "Optimization",
        "purpose": "Advise on SQL performance.",
        "responsibility": "Use EXPLAIN (after validation) and suggest indexes or rewrites.",
        "input": "SQL and dialect",
        "output": "optimization tips",
        "role": "Performance AI Agent",
        "position": "Specialist after SQL is available",
        "enterprise": "Improves query cost without changing the security gate",
        "optional": "",
    },
    {
        "name": "Reflection",
        "purpose": "Critique answer quality before finish.",
        "responsibility": "Review SQL, results, and insights; accept or request another pass / replan.",
        "input": "Near-final GraphState",
        "output": "reflection verdict and notes",
        "role": "Quality AI Agent",
        "position": "Late workflow, before Finalize when routed",
        "enterprise": "Self-critique loop for higher-confidence answers",
        "optional": "",
    },
    {
        "name": "Retry",
        "purpose": "Diagnose SQL failures and choose recovery.",
        "responsibility": "Decide regenerate_sql, replan, or give_up after validation/execution failure.",
        "input": "sql_error, validation feedback, attempt count",
        "output": "retry diagnosis and next route",
        "role": "Failure-diagnosis AI Agent",
        "position": "On validation or execution failure edges",
        "enterprise": "Bounded recovery before escalation",
        "optional": "",
    },
    {
        "name": "Replan",
        "purpose": "Build a new strategy after failure.",
        "responsibility": "True replanning (revise tasks, swap agents, narrow scope, clarify, or abort) — not a bare retry.",
        "input": "Goal, plan, task graph, failure context",
        "output": "ReplanDecision; may re-enter Planner",
        "role": "Strategy revision AI Agent",
        "position": "After Retry or Reflection when replan is chosen",
        "enterprise": "Closed-loop recovery with max-replan limits",
        "optional": "Requires autonomy planning",
    },
    {
        "name": "Recovery",
        "purpose": "Apply runtime recovery policy.",
        "responsibility": "Decide Recover / Retry / Fallback / Abort / Escalate after runtime failures.",
        "input": "Failure context from enterprise runtime",
        "output": "Recovery action and next route",
        "role": "Reliability controller",
        "position": "Enterprise recovery path (optional recovery graph)",
        "enterprise": "Policy engine for tool, SQL, plugin, and timeout failures",
        "optional": "Requires enterprise mode and recovery controller",
    },
    {
        "name": "Clarify",
        "purpose": "Ask the user for missing detail.",
        "responsibility": "Pause the workflow with LangGraph interrupt until the user answers.",
        "input": "clarification_question",
        "output": "User clarification; resumed state",
        "role": "Human-in-the-loop gate",
        "position": "From Goal Understanding or Replan when ambiguity is high",
        "enterprise": "Prevents unsafe guessing on unclear goals",
        "optional": "",
    },
    {
        "name": "Export",
        "purpose": "Package results for download.",
        "responsibility": "Orchestrate CSV, Excel, JSON, Markdown, and PDF exporters (async queue optional).",
        "input": "Result set, question, insights",
        "output": "export_paths / download payloads",
        "role": "Export manager (deterministic; optional narrative)",
        "position": "Late specialist or Export Center",
        "enterprise": "Consistent multi-format delivery",
        "optional": "",
    },
    {
        "name": "Plugin Runtime",
        "purpose": "Invoke marketplace plugin skills.",
        "responsibility": "Call a registered plugin capability selected from state or the Agent Registry.",
        "input": "plugin_capability_id / skill payload",
        "output": "Plugin result merged into state",
        "role": "Extensibility runtime",
        "position": "When a plan or Supervisor routes to a plugin skill",
        "enterprise": "Hot-reloadable Plugin Marketplace without rewriting the graph",
        "optional": "Requires enterprise mode and plugin runtime",
    },
    {
        "name": "Finalize",
        "purpose": "Assemble the final answer.",
        "responsibility": "Compose the user-facing response from current GraphState and end the run.",
        "input": "Completed specialist outputs",
        "output": "final_response",
        "role": "Deterministic response assembly",
        "position": "Terminal success path",
        "enterprise": "Stable close-out for chat, history, and exports",
        "optional": "",
    },
]


def render_about(settings: Settings) -> None:
    page_header(
        "About",
        "Architecture, agents, and platform capabilities",
        settings=settings,
    )

    # —— Product hero ——
    inject_html(
        f"""
        <div class="sq-about-hero">
          <div class="sq-about-hero__mark">SQ</div>
          <div>
            <div class="sq-eyebrow">Product</div>
            <h1>{esc(settings.app_name)}</h1>
            <p>
              Natural-language questions become validated, read-only SQL analytics.
              A multi-agent LangGraph workflow plans work, routes specialists, enforces
              deterministic security, and returns charts, insights, and exports.
            </p>
            <div class="sq-about-meta">
              {status_badge(f"v{settings.app_version}", kind="accent")}
              {status_badge("LangGraph", kind="info")}
              {status_badge("Read-only SQL", kind="ok")}
              {status_badge("Streamlit Cloud", kind="default")}
            </div>
          </div>
        </div>
        """
    )

    # —— Overview ——
    section_title("Overview", "What this product is")
    feature_grid(
        [
            (
                "What it is",
                "AI-native SQL analytics for trustworthy natural-language access to relational data.",
            ),
            (
                "Who it is for",
                "Analysts, engineers, and teams evaluating multi-agent system design on PostgreSQL, MySQL, or SQLite.",
            ),
            (
                "Core objective",
                "Answer business questions accurately with a hard split between LLM generation and SQL security.",
            ),
            (
                "Enterprise philosophy",
                "Discoverable capabilities, explicit plans, human-in-the-loop gates, and observable execution.",
            ),
        ]
    )

    # —— Design pillars ——
    section_title("Design", "Why this architecture")
    feature_grid(
        [
            (
                "LangGraph",
                "Stateful graph execution, checkpoints, interrupts, and controlled parallel fan-out.",
            ),
            (
                "Multi-agent",
                "Specialists own schema, SQL, visualization, insight, planning, and recovery.",
            ),
            (
                "SQL analytics",
                "Generate → validate with sqlglot → execute under limits → visualize results.",
            ),
            (
                "Local + cloud LLMs",
                "Ollama, OpenAI, Gemini, Claude, Groq — swap providers; the security gate stays the same.",
            ),
            (
                "Memory system",
                "Chat memory, episodic retrieval, workflow memory, optional vector fabric, checkpoints.",
            ),
            (
                "Security & observability",
                "AST read-only gate, audit logging, LangSmith / OTel, Prometheus, Runtime console.",
            ),
        ]
    )

    # —— Architecture layers ——
    section_title("Architecture", "Control plane and data plane")
    inject_html(
        """
        <div class="sq-layer-grid">
          <div class="sq-layer sq-layer--control">
            <div class="sq-layer__label">Control plane</div>
            <div class="sq-layer__title">Plan &amp; coordinate</div>
            <p class="sq-layer__body">Goal · Plan · Coordinate · Supervise</p>
          </div>
          <div class="sq-layer sq-layer--data">
            <div class="sq-layer__label">Data plane</div>
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

    # —— Workflow flow ——
    inject_html(
        """
        <div class="sq-flow">
          <div class="sq-flow__title">Execution flow</div>
          <div class="sq-flow__steps">
            <span class="sq-flow__step">Question</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Memory</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Goal</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Planner</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Coordinator</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Specialists</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step is-gate">Validate</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Execute</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step is-parallel">Viz ∥ Insight</span>
            <span class="sq-flow__arrow">→</span>
            <span class="sq-flow__step">Finalize</span>
          </div>
        </div>
        """
    )
    card(
        title="Runtime notes",
        body=(
            "GraphState carries question, schema, SQL, results, plan, and messages across nodes. "
            "An Agent Message Bus supports request/reply between planner components. "
            "Enterprise Runtime adds optional IAM, circuit breakers, queues, workers, plugins, "
            "and recovery — enabled via configuration, not required for core analytics."
        ),
    )

    # —— Capabilities ——
    section_title("Capabilities", "What ships in this build")
    inject_html(
        """
        <div class="sq-cap-grid">
          <div class="sq-cap">
            <div class="sq-cap__label">Databases</div>
            <div class="sq-cap__value">PostgreSQL · MySQL · SQLite</div>
          </div>
          <div class="sq-cap">
            <div class="sq-cap__label">Security</div>
            <div class="sq-cap__value">sqlglot read-only gate</div>
          </div>
          <div class="sq-cap">
            <div class="sq-cap__label">Charts</div>
            <div class="sq-cap__value">AI recommendations · Plotly</div>
          </div>
          <div class="sq-cap">
            <div class="sq-cap__label">Exports</div>
            <div class="sq-cap__value">CSV · Excel · PDF · JSON · MD</div>
          </div>
        </div>
        """
    )

    # —— Tech stack ——
    section_title("Tech stack", "Core dependencies")
    inject_html(
        f"""
        <div class="sq-stack-row">
          {status_badge("Python", kind="default")}
          {status_badge("LangGraph", kind="accent")}
          {status_badge("LangChain", kind="info")}
          {status_badge("sqlglot", kind="ok")}
          {status_badge("Streamlit", kind="default")}
          {status_badge("Plotly", kind="info")}
          {status_badge("SQLAlchemy", kind="default")}
          {status_badge("Prometheus", kind="warn")}
        </div>
        """
    )

    # —— Agents by group ——
    section_title("AI Agents", "Registered workflow participants")
    st.caption(
        "Every entry maps to an implemented graph node. "
        "Optional items appear only when matching configuration flags are enabled."
    )

    groups = _agent_groups()
    for group_name, agents in groups:
        section_title(group_name)
        # Preview cards (2-col)
        preview = "".join(
            f'<div class="sq-agent-card">'
            f'<div class="sq-agent-card__name">{esc(a["name"])}</div>'
            f'<p class="sq-agent-card__purpose">{esc(a["purpose"])}</p>'
            f'</div>'
            for a in agents
        )
        inject_html(f'<div class="sq-agent-grid">{preview}</div>')
        for agent in agents:
            opt = f" · Optional: {agent['optional']}" if agent.get("optional") else ""
            with st.expander(f"{agent['name']} — details", expanded=False):
                st.markdown(
                    f"""
**Purpose.** {agent['purpose']}

**Responsibility.** {agent['responsibility']}

**Input.** {agent['input']}

**Output.** {agent['output']}

**Communication role.** {agent['role']}

**Position in workflow.** {agent['position']}

**Enterprise responsibility.** {agent['enterprise']}{opt}
"""
                )

    # —— Extensibility ——
    section_title("Extensibility", "Plugins and deployment")
    c1, c2 = st.columns(2)
    with c1:
        card(
            title="Plugins",
            heading="Marketplace skills",
            body=(
                "Plugins register into the capability catalog. "
                "The Plugin Runtime invokes them when selected by the plan or Supervisor. "
                "A built-in echo skill is included for health checks."
            ),
            kind="accent",
        )
    with c2:
        card(
            title="Deployment",
            heading="Streamlit Community Cloud",
            body=(
                "Configure LLM secrets, connect a database "
                "(or use the built-in sample SQLite), and run analytics in the browser."
            ),
        )

    b1, b2 = st.columns(2)
    if b1.button("Open Settings", type="primary", use_container_width=True, key="about_to_settings"):
        st.session_state.nav_page = "Settings"
        st.rerun()
    if b2.button("Open Runtime", use_container_width=True, key="about_to_runtime"):
        st.session_state.nav_page = "Runtime"
        st.rerun()

    page_footer(settings.app_version)


def _agent_groups() -> list[tuple[str, list[dict[str, str]]]]:
    """Group agents for cleaner About presentation (same catalog, better layout)."""
    by_name = {a["name"]: a for a in _AI_AGENTS}
    order = [
        (
            "Planning & control",
            [
                "Memory Agent",
                "Goal Understanding",
                "Planner",
                "Task Decomposition",
                "Goal Tracking",
                "Execution Coordinator",
                "Supervisor",
            ],
        ),
        (
            "Data plane",
            ["Schema Agent", "SQL Agent", "Validation", "Approval", "Execution"],
        ),
        (
            "Analytics & quality",
            [
                "Visualization",
                "Insight",
                "Join",
                "Dashboard",
                "Summary",
                "Optimization",
                "Reflection",
            ],
        ),
        (
            "Reliability & delivery",
            [
                "Retry",
                "Replan",
                "Recovery",
                "Clarify",
                "Export",
                "Plugin Runtime",
                "Finalize",
            ],
        ),
    ]
    groups: list[tuple[str, list[dict[str, str]]]] = []
    for title, names in order:
        agents = [by_name[n] for n in names if n in by_name]
        if agents:
            groups.append((title, agents))
    return groups
