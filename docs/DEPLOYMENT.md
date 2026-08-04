# Deployment Guide

## Local

```bash
cp .env.example .env
uv sync   # or: pip install -r requirements.txt
streamlit run app.py
```

Ensure the selected LLM is reachable (Ollama daemon or cloud API key). Apply LLM settings in the sidebar or Settings page before using Chat.

The sample retail database is created on demand from `sample_data/seed.py` — nothing under `data/` needs to be committed.

## Streamlit Community Cloud

Designed for Cloud constraints:

- Native Streamlit widgets
- Minimal CSS (no layout-breaking hacks)
- No custom JavaScript
- Sample SQLite database generated in-app (no external database required for demos)
- Download buttons use in-memory bytes (ephemeral filesystem friendly)

### Required repository files

| File | Role |
|------|------|
| `app.py` | Main file path on Streamlit Cloud |
| `requirements.txt` | Dependency install |
| `.streamlit/config.toml` | Theme / server defaults |
| `sample_data/` | Demo DB seed |

`packages.txt` is not required for the current dependency set.

### Steps

1. Push the repository to GitHub (do not commit `.env` or secrets)
2. Create an app on [share.streamlit.io](https://share.streamlit.io)
3. Main file: `app.py`
4. Python version: 3.11+ (3.12 recommended)
5. Configure secrets for your LLM provider
6. Deploy and connect the built-in sample database

### Secrets example

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
MAX_ROWS = 500
QUERY_TIMEOUT_SECONDS = 30
```

See `.env.example` for the full configuration surface (enterprise, observability, memory).

## Production notes

- Keep `READ_ONLY_MODE=true`
- Point `HISTORY_DB_PATH` / `AUDIT_LOG_PATH` at persistent storage when available
- Prefer least-privilege database users (SELECT-only grants at the server)
- Do not commit `.env` or `.streamlit/secrets.toml`
- Enterprise Runtime features (IAM, distributed workers, plugins, approval gates) are optional and configuration-gated — core analytics works without them
