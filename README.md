# SQLMind AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Natural-language SQL analytics with a multi-agent LangGraph runtime.**

SQLMind AI turns business questions into validated, read-only SQL, executes them safely against PostgreSQL, MySQL, or SQLite, and returns charts, insights, and exportable reports.

Built for AI engineers, data teams, technical interviews, and client demos — deployable on [Streamlit Community Cloud](https://share.streamlit.io).

---

## Project Overview

Most “chat with your database” demos stop at a single LLM prompt that emits SQL. SQLMind AI is a full **agentic control plane**:

- Specialized AI Agents plan, discover schema, author SQL (ReAct tool loop), visualize, and critique results
- Deterministic security nodes (sqlglot AST) block unsafe SQL before any database I/O
- Optional enterprise runtime: IAM, approval gates, queues, plugins, vector memory, and observability

**Who it is for**

| Audience | Value |
|----------|--------|
| Recruiters / hiring managers | Demonstrates LangGraph, tool calling, security design, and production UX |
| Clients / stakeholders | Browser demo with sample retail data — no external DB required |
| Engineers | Clone, configure an LLM, and extend agents / tools / plugins |

---

## Features

| Area | Capability |
|------|------------|
| Databases | PostgreSQL, MySQL, SQLite (SQLAlchemy) with multi-DB switcher |
| NL → SQL | SQL Agent with ReAct `bind_tools` loop and structured submit |
| Schema | Auto-discovery via Schema Agent when routed |
| Security | sqlglot AST read-only gate + regex safety net (deterministic — not LLM) |
| AI Agents | Memory, Goal, Planner, Supervisor, Schema, SQL, Visualization, Insight, Dashboard, Summary, Optimization, Reflection, Retry, Replan, and more |
| Deterministic nodes | Validation, Execution, Export, Join, Finalize, Clarify / Approval HITL |
| Exporters | CSV, Excel, JSON, Markdown, PDF |
| Charts | AI chart recommendation → Plotly |
| Memory | Multi-turn chat, compression, episodic store, workflow memory, optional vector Memory Fabric, LangGraph checkpoints |
| Observability | LangSmith, OpenTelemetry, Prometheus metrics, runtime traces with `trace_id` |
| HITL | LangGraph `interrupt()` Clarify + Approval; orchestrator `resume()` |
| LLM providers | Ollama, OpenAI, Gemini, Claude, Groq |

Validation, Execution, Export, Join, and Finalize are **not** AI Agents — they are deterministic infrastructure nodes by design.

---

## AI Architecture

```
User Question
  → Memory → Goal Understanding → Planner → Task Decomposition
  → Execution Coordinator (or Supervisor bind_tools fallback)
  → Schema / SQL specialists as routed
  → Validation (sqlglot) → Approval? → Execution
  → parallel Visualization ∥ Insight → Join
  → Reflection / Export / Finalize
```

On validation or execution failure → Retry Agent (regenerate / replan / give up), then optional Replan Agent.

Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Multi-Agent Workflow

| Plane | Responsibility |
|-------|----------------|
| **Control** | Memory, Goal Understanding, Planner, Task Decomposition, Coordinator / Supervisor |
| **Data** | Schema Agent, SQL Agent, Validation, Approval, Execution |
| **Analytics** | Visualization, Insight, Dashboard, Summary, Optimization, Reflection, Export |
| **Platform** | IAM, Plugin Runtime, queues, circuit breakers, Memory Fabric, metrics |

The Supervisor binds live routing tools from the capability registry. SQL never reaches the database without passing `SQLSecurityGuard`.

---

## Tech Stack

- **UI:** Streamlit
- **Orchestration:** LangGraph + LangChain
- **LLMs:** Ollama / OpenAI / Gemini / Claude / Groq
- **SQL:** SQLAlchemy, sqlglot
- **Data:** PostgreSQL, MySQL, SQLite
- **Viz / export:** Plotly, pandas, openpyxl, reportlab
- **Observability:** LangSmith, OpenTelemetry, Prometheus
- **Packaging:** `requirements.txt` + `pyproject.toml` (uv-compatible)

---

## Installation

**Requirements:** Python 3.11+ (3.12 recommended), and a reachable LLM (Ollama locally or a cloud API key).

```bash
git clone https://github.com/<your-username>/sqlmind-ai.git
cd sqlmind-ai

# Recommended
uv sync

# Or
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Set `LLM_PROVIDER` and the matching API key (or keep Ollama defaults for local use).

---

## Run Locally

```bash
streamlit run app.py
```

1. Open the app in the browser  
2. Enable **Use built-in sample database** → **Connect SQLite**  
3. Ask a question in Chat  

Optional observability:

- LangSmith: set `LANGCHAIN_API_KEY`
- OpenTelemetry: set `OTEL_ENABLED=true`

---

## Streamlit Deployment

Main file path on Streamlit Community Cloud: **`app.py`**

1. Push this repository to GitHub  
2. Create an app at [share.streamlit.io](https://share.streamlit.io)  
3. Set Python 3.11+ (3.12 recommended)  
4. Add LLM secrets (see below)  
5. Deploy — use the built-in sample SQLite database for demos  

**Secrets example** (Streamlit Cloud → Settings → Secrets):

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
MAX_ROWS = 500
QUERY_TIMEOUT_SECONDS = 30
```

Full notes: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## Screenshots

Add product screenshots under `docs/screenshots/` after your first local or Cloud run:

| File | Suggested content |
|------|-------------------|
| `docs/screenshots/home.png` | Home / product overview |
| `docs/screenshots/chat.png` | Multi-agent chat with SQL + chart |
| `docs/screenshots/schema.png` | Schema explorer |
| `docs/screenshots/runtime.png` | Runtime / agent timeline |

Then embed them here, for example:

```markdown
![Chat](docs/screenshots/chat.png)
```

---

## Project Structure

```
sqlmind-ai/
├── app.py                 # Streamlit entry (Cloud main file)
├── requirements.txt       # Cloud / pip install
├── pyproject.toml         # Project metadata + uv deps
├── .streamlit/            # Theme and server config
├── .env.example           # Documented configuration
├── agents/                # AI Agent nodes + SQL ReAct
├── graph/                 # LangGraph state, workflow, orchestrator
├── planner/               # Goal, plan, coordinator, registry
├── models/                # Pydantic structured outputs
├── prompts/               # ChatPromptTemplates
├── tools/                 # StructuredTools + routing tools
├── database/              # Connector, schema, executor
├── services/              # LLM, visualization, export, KPI
├── memory/                # Conversation, episodic, checkpoints, Memory Fabric
├── observability/         # LangSmith, OTel, metrics, runtime trace
├── enterprise/            # Enterprise Runtime wiring
├── kernel/                # Autonomy Kernel / capability registry
├── iam/ governance/ …     # Optional enterprise packages
├── ui/                    # Streamlit pages, layouts, styles
├── sample_data/           # Demo SQLite seed (generated at runtime)
├── tests/
└── docs/
```

---

## Example Questions

Against the built-in sample retail database:

- Show total revenue by city  
- Top 10 customers by spend  
- Highest selling products  
- What percentage of orders are cancelled?  
- Which customers have never placed an order?  
- Compare revenue between Electronics and Furniture  

---

## Enterprise Features

Optional and configuration-gated (`ENTERPRISE_*` in `.env`):

- IAM (RBAC / ABAC), tenants, API keys  
- Approval gates for high-risk SQL / exports  
- Plugin marketplace with hot-reload  
- Circuit breakers, DLQ, recovery controller  
- Local distributed worker pool (SQLite-backed)  
- Vector Memory Fabric and planner learning store  

Core analytics works without enabling the enterprise layer.

---

## Security

| Control | Enforced by |
|---------|-------------|
| Read-only SQL | sqlglot AST + forbidden nodes |
| No multi-statement / injection chains | Parser + regex net |
| Unknown tables | Schema allow-list |
| Row cap / timeout | Executor (server-side) |
| Audit trail | JSONL with request context |

The LLM cannot bypass validation — graph edge + executor double-check. Tool calling does not weaken this gate.

Do not commit `.env` or `.streamlit/secrets.toml`. Prefer SELECT-only database users in production.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Muhammad Junaid**

- Phone: 0304-1659295  
- Email: [junaidfazal08@gmail.com](mailto:junaidfazal08@gmail.com)

---

## Docs

- [Architecture](docs/ARCHITECTURE.md)  
- [Deployment](docs/DEPLOYMENT.md)
