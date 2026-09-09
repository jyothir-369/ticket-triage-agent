# 🎫 Support-Ticket Triage Agent

An AI agent that classifies incoming support tickets, retrieves related past tickets and documentation, drafts response suggestions, and escalates ambiguous cases to human reviewers — with a full audit trail for every decision.

---

## ✨ Features

- **Intelligent Classification** — Automatically categorizes tickets by type (bug, feature request, billing, etc.) and urgency (P0–P3).
- **RAG-Powered Retrieval** — Searches historical tickets and docs to find relevant context before drafting a response.
- **Draft Response Generation** — Produces a suggested reply with citations to retrieved sources.
- **Confidence-Based Escalation** — Routes low-confidence tickets to human reviewers with tunable thresholds.
- **Loop Detection** — Detects and breaks out of agent loops (e.g., repeated failed tool calls).
- **Full Audit Trail** — Every decision (classification → retrieval → draft → routing) is traced with timestamps.
- **Observability** — OpenTelemetry traces exported to any OTLP-compatible backend.
- **Streamlit Dashboard** — Real-time triage metrics, escalation rates, and agent performance.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| Backend API | FastAPI |
| LLM Providers | OpenAI / Anthropic (swappable) |
| Vector DB | Qdrant |
| Relational DB | PostgreSQL |
| Background Jobs | Inngest + Redis |
| Observability | OpenTelemetry |
| Dashboard | Streamlit + Plotly |
| Testing | Pytest |
| Formatting | Black + Ruff + MyPy |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- An OpenAI or Anthropic API key

### 1. Clone and install

```bash
git clone <repo-url>
cd support-ticket-triage-agent
cp .env.example .env          # fill in your API keys
make install                  # install dependencies
```

### 2. Start infrastructure

```bash
make db-up                    # Postgres + Qdrant + Redis
make db-migrate               # run migrations
make db-seed                  # seed eval tickets
```

### 3. Run the application

```bash
make run                      # FastAPI on http://localhost:8000
make run-dashboard            # Streamlit on http://localhost:8501
```

### 4. Triage a ticket

```bash
curl -X POST http://localhost:8000/tickets/1/triage
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/tickets/{id}/triage` | Run the full triage pipeline on a ticket |
| `GET` | `/tickets/{id}/trace` | Retrieve the step-by-step decision trace |
| `GET` | `/dashboard/triage-metrics` | Aggregate metrics (accuracy, escalation rate) |
| `GET` | `/health` | Health check endpoint |

---

## 🧪 Development

```bash
# Run tests
make test

# Lint
make lint

# Format
make format

# Run evaluation harness
make eval
```

---

## 📁 Project Structure

```
support-ticket-triage-agent/
├── src/
│   ├── agent/              # LangGraph agent definition
│   ├── api/                # FastAPI routes
│   ├── models/             # SQLAlchemy + Pydantic models
│   ├── services/           # LLM, retrieval, classification services
│   ├── tools/              # Agent tool functions
│   ├── evaluation/         # Eval harness + labeled test set
│   └── scripts/            # DB seeding, one-off scripts
├── dashboard/              # Streamlit UI
├── alembic/                # Database migrations
├── tests/                  # Test suite
├── data/                   # Eval ticket fixtures
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## 📄 License

MIT
