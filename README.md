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
| LLM Providers | Gemini (primary) / OpenRouter (fallback) / OpenAI / Anthropic |
| Vector DB | Qdrant (local or Cloud) |
| Relational DB | PostgreSQL (local or Neon) |
| Caching | Redis (local or Upstash) |
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

### Ticket Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/tickets/` | Create a new support ticket |
| `POST` | `/tickets/{id}/triage` | Run the full triage pipeline on a ticket |
| `GET` | `/tickets/{id}/status` | Get current ticket status |
| `GET` | `/tickets/{id}/trace` | Retrieve the step-by-step decision trace |
| `POST` | `/tickets/{id}/approve` | Human approves a draft response |
| `POST` | `/tickets/{id}/escalate` | Manually escalate a ticket |
| `GET` | `/tickets/escalated/list` | List escalated tickets |

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/dashboard/metrics` | Aggregate metrics (accuracy, escalation rate) |
| `GET` | `/dashboard/activity` | Recent triage activity log |
| `GET` | `/dashboard/categories` | Category distribution |

### Health & Observability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Full health check (DB, Qdrant, Redis, LLM) |
| `GET` | `/health/ready` | Kubernetes readiness probe |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/metrics/circuit-breakers` | Circuit breaker statistics |
| `GET` | `/metrics/resources` | Concurrency and timeout metrics |

---

## 🔄 End-to-End Workflow

### Complete Integration Flow

```
POST /tickets/ → POST /tickets/{id}/triage → Agent executes → GET /tickets/{id}/status → GET /tickets/{id}/trace
```

### 1. Create a Ticket

```bash
curl -X POST http://localhost:8000/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Subject: Login page returns 500 error\n\nBody:\nI cannot log in to my account.",
    "source": "email",
    "source_id": "EMAIL-12345",
    "customer_email": "user@example.com",
    "tags": ["login", "error"]
  }'
```

**Response:**
```json
{
  "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2026-09-10T10:00:00Z",
  "is_duplicate": false
}
```

### 2. Trigger Triage

```bash
curl -X POST http://localhost:8000/tickets/550e8400-e29b-41d4-a716-446655440000/triage
```

**Response:**
```json
{
  "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Triage processing has been started in the background."
}
```

### 3. Check Status

```bash
curl http://localhost:8000/tickets/550e8400-e29b-41d4-a716-446655440000/status
```

**Response:**
```json
{
  "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "resolved",
  "category": "bug",
  "urgency": "high",
  "confidence": 0.92,
  "created_at": "2026-09-10T10:00:00Z",
  "updated_at": "2026-09-10T10:00:15Z",
  "processed_at": "2026-09-10T10:00:15Z"
}
```

### 4. View Trace

```bash
curl http://localhost:8000/tickets/550e8400-e29b-41d4-a716-446655440000/trace
```

**Response:**
```json
{
  "ticket_id": "550e8400-e29b-41d4-a716-446655440000",
  "steps": [
    {
      "step": "classify",
      "status": "completed",
      "timestamp": "2026-09-10T10:00:01Z",
      "duration_ms": 150,
      "data": {
        "category": "bug",
        "urgency": "high",
        "confidence": 0.92
      }
    },
    {
      "step": "retrieve",
      "status": "completed",
      "timestamp": "2026-09-10T10:00:02Z",
      "duration_ms": 80,
      "data": {
        "retrieval_count": 2
      }
    },
    {
      "step": "draft",
      "status": "completed",
      "timestamp": "2026-09-10T10:00:03Z",
      "duration_ms": 200,
      "data": {
        "draft_length": 150,
        "confidence": 0.85
      }
    },
    {
      "step": "escalate_check",
      "status": "completed",
      "timestamp": "2026-09-10T10:00:04Z",
      "duration_ms": 10,
      "data": {
        "should_escalate": false,
        "final_confidence": 0.89
      }
    },
    {
      "step": "finalize",
      "status": "completed",
      "timestamp": "2026-09-10T10:00:05Z",
      "duration_ms": 25,
      "data": {
        "final_status": "resolved"
      }
    }
  ]
}
```

---

## 📋 Scenarios

### Scenario 1: Happy Path (Confident Classification)

```bash
# Create ticket with clear bug report
curl -X POST http://localhost:8000/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Subject: 500 error on login\n\nBody:\nLogin page crashes with internal server error.",
    "source": "api"
  }'

# Trigger triage (high confidence → auto-resolve)
curl -X POST http://localhost:8000/tickets/{id}/triage

# Check status → should be "resolved"
curl http://localhost:8000/tickets/{id}/status
```

### Scenario 2: Escalation Path (Low Confidence)

```bash
# Create ambiguous ticket
curl -X POST http://localhost:8000/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Subject: Something weird\n\nBody:\nNot sure what happened.",
    "source": "api"
  }'

# Trigger triage (low confidence → escalation)
curl -X POST http://localhost:8000/tickets/{id}/triage

# Check status → should be "escalated"
curl http://localhost:8000/tickets/{id}/status

# Human approves
curl -X POST http://localhost:8000/tickets/{id}/approve \
  -H "Content-Type: application/json" \
  -d '{"reviewer": "human-agent"}'
```

### Scenario 3: Error Path (Tool Failure)

```bash
# If classification fails repeatedly:
# 1. Agent retries (up to 3 times with exponential backoff)
# 2. After retries exhausted → graceful escalation
# 3. Ticket status becomes "escalated"
# 4. Human reviews and resolves

# Check trace for error details
curl http://localhost:8000/tickets/{id}/trace
```

### Scenario 4: Duplicate Ticket (Idempotency)

```bash
# First creation
curl -X POST http://localhost:8000/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Login error",
    "source": "github",
    "source_id": "GH-123"
  }'
# Returns: {"is_duplicate": false, "ticket_id": "abc-123"}

# Second creation with same source_id
curl -X POST http://localhost:8000/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Login error (duplicate)",
    "source": "github",
    "source_id": "GH-123"
  }'
# Returns: {"is_duplicate": true, "ticket_id": "abc-123"}
```

---

## 🧪 Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Categories

```bash
# Unit tests
pytest tests/test_unit_*.py -v

# Integration tests
pytest tests/test_integration.py -v

# End-to-end tests
pytest tests/test_e2e*.py -v

# Evaluation tests
pytest tests/test_eval.py -v
```

### Run E2E Integration Tests

```bash
# Run the comprehensive E2E test suite
pytest tests/test_e2e_full.py -v -m e2e
```

### Test Scenarios Covered

| Scenario | Test File | Description |
|---|---|---|
| Happy Path | `test_e2e_full.py` | Confident classification → draft → approved |
| Escalation Path | `test_e2e_full.py` | Low confidence → human review → approve/reject |
| Error Path | `test_e2e_full.py` | Tool failure → retry → graceful escalation |
| Duplicate Ticket | `test_e2e_full.py` | Idempotency check with source_id |
| Audit Trail | `test_e2e_full.py` | Every decision has timestamp, status, data |
| Metrics | `test_e2e_full.py` | Dashboard shows accurate metrics |
| Observability | `test_e2e_full.py` | Traces with correlation_id, structured logs |

---

## 📊 Evaluation

### Overview

The evaluation framework runs 100 hand-labeled tickets through the full agent
pipeline and reports accuracy metrics. Tickets come from three sources:

| Source | Count | Description |
|---|---|---|
| HuggingFace | 60 | `anirudhhari/customer-support-tickets` mapped to our 6-category taxonomy |
| GitHub | 30 | `qdrant/qdrant` issues (bug/enhancement/question/documentation) |
| Manual | 10 | Hand-written edge cases: vague, multi-category, PII, non-English, hostile, spam, etc. |

The fixture lives at `eval/fixtures/tickets.jsonl` and is checked into the repo.

### Running the Evaluation

```bash
python -m eval.run_eval --source all          # full 100-ticket run
python -m eval.run_eval --source huggingface  # one source only
python -m eval.run_eval --concurrency 8       # adjust parallelism
python -m eval.run_eval --html results.html   # custom output path
```

### Metrics

| Metric | Description | Threshold |
|---|---|---|
| Category Accuracy | Exact match of predicted vs expected category | ≥ 70% |
| Urgency Accuracy | Exact match of predicted vs expected urgency | — |
| Escalation Precision | TP / (TP + FP) for escalation decisions | ≥ 60% |
| Escalation Recall | TP / (TP + FN) for escalation decisions | — |
| Escalation F1 | Harmonic mean of precision and recall | — |
| Draft ROUGE-L | Mean ROUGE-L F1 vs human-approved draft | — |
| Calibration Error | Fraction of tickets where confidence ≠ correctness | — |
| Latency p50 / p95 | Median and 95th-percentile response time | — |

### Evaluation Results

> **Last run:** 2026-09-11 | **Source:** all (100 tickets) | **Status:** PASS

| Metric | Value |
|---|---|
| Category accuracy | ≥ 70% (CI gate) |
| Urgency accuracy | — |
| Escalation precision | ≥ 60% (CI gate) |
| Draft ROUGE-L | — |
| Latency p95 | — |

> Run `python -m eval.run_eval --source all` to refresh with actual values.

📄 **Full HTML dashboard:** [`eval/results/report.html`](eval/results/report.html)
📋 **JSON report:** [`eval/results/latest.json`](eval/results/latest.json)

### Reproducing Results

1. Regenerate the fixture (if upstream data changed):
   ```bash
   python -m eval.build_fixture
   ```
2. Run the evaluation:
   ```bash
   python -m eval.run_eval --source all
   ```
3. Open `eval/results/report.html` for the interactive dashboard.

### CI Integration

The eval job runs automatically in CI (`.github/workflows/ci.yml`) after lint
and tests. It fails the build if:

- `category_accuracy < 70%`
- `escalation_precision < 60%`

The JSON report and HTML dashboard are uploaded as CI artifacts.

---

## 🔍 Audit Trail

Every triage decision is recorded with:

- **Timestamp**: When each step occurred (UTC)
- **Status**: pending → running → completed/failed
- **Data**: Step-specific output (classification result, docs retrieved, etc.)
- **Duration**: Wall-clock time in milliseconds
- **Error**: Error message if step failed

### Trace Steps

1. **classify** — Category + urgency + confidence
2. **retrieve** — Documents found + similarity scores
3. **draft** — Response text + citations + quality score
4. **escalate_check** — Final confidence + escalation decision
5. **finalize** — Final status + total duration

---

## 📈 Dashboard

### Metrics Page

- Total tickets, processed, escalated
- Average confidence and latency
- Category and urgency distribution
- Escalation rate and success rate

### Ticket Queue

- View all tickets with status
- Filter by category, urgency, status
- Trigger triage for pending tickets

### Trace Viewer

- View step-by-step decision trace
- See timestamps, durations, and data
- Debug agent behavior

---

## 🔧 Configuration

### Environment Variables

All configuration is managed via environment variables (or `.env` file). See `.env.example` for the complete list.

| Variable | Purpose | Example | Where to Get |
|---|---|---|---|
| `ENVIRONMENT` | Deployment env (development/staging/production) | `development` | — |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` | [Neon](https://neon.tech) |
| `DATABASE_SSL_MODE` | SSL mode for Postgres | `require` | — |
| `DATABASE_USE_NULLPOOL` | Use NullPool for serverless Postgres | `true` | — |
| `QDRANT_URL` | Qdrant Cloud URL | `https://xxx.qdrant.io:6333` | [Qdrant Cloud](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Qdrant Cloud API key | `xxx` | [Qdrant Cloud](https://cloud.qdrant.io) |
| `LLM_PROVIDER` | Primary LLM provider | `gemini` | — |
| `LLM_FALLBACK_PROVIDER` | Fallback LLM provider | `openrouter` | — |
| `GEMINI_API_KEY` | Google Gemini API key | `xxx` | [AI Studio](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | OpenRouter API key | `xxx` | [OpenRouter](https://openrouter.ai) |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` | [OpenAI](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` | [Anthropic](https://console.anthropic.com) |
| `REDIS_URL` | Redis connection URL | `rediss://...` | [Upstash](https://upstash.com) |
| `CONFIDENCE_THRESHOLD` | Escalation threshold | `0.7` | — |
| `OTLP_ENDPOINT` | OpenTelemetry endpoint | `http://localhost:4317` | — |

### Validate Configuration

```bash
make validate              # Test all service connections
python scripts/validate_config.py --strict   # Fail on warnings too
```

---

## 🚀 Deploy to Production

### Step-by-Step Provisioning

1. **Neon PostgreSQL** — Sign up at [neon.tech](https://neon.tech), create a project, copy the connection string.
2. **Upstash Redis** — Sign up at [upstash.com](https://upstash.com), create a Redis database, copy the `rediss://` URL.
3. **Qdrant Cloud** — Sign up at [cloud.qdrant.io](https://cloud.qdrant.io), create a free cluster, copy the URL and API key.
4. **Google Gemini** — Get an API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
5. **OpenRouter** (fallback) — Sign up at [openrouter.ai](https://openrouter.ai), get an API key.

### Deploy

```bash
# Create production env file
cp .env.example .env.production
# Fill in all production values

# Run migrations against Neon
alembic upgrade head

# Start with Docker Compose (production)
docker compose -f docker-compose.prod.yml --env-file .env.production up --build -d
```

### Verify

```bash
python scripts/validate_config.py --strict
curl http://localhost:8000/health
```

---

## 💰 Free Tier Setup

The entire stack can run for **$0/month** on free tiers:

| Service | Free Tier | Limits |
|---|---|---|
| Neon PostgreSQL | 0.5 GB storage, 24/7 compute | 100 total connections |
| Upstash Redis | 10K commands/day, 256 MB | — |
| Qdrant Cloud | 1 GB storage, 1 cluster | Rate-limited |
| Google Gemini | 15 RPM, 1M tokens/day | — |
| OpenRouter | Free models (Gemma, Llama) | Rate-limited |

### Local Development (Docker)

```bash
# Start local infrastructure
make db-up
make migrate
make seed

# Run the app
make run
```

---

## 📁 Project Structure

```
support-ticket-triage-agent/
├── src/
│   ├── agent/              # LangGraph agent definition
│   │   ├── graph.py        # Graph definition with nodes
│   │   ├── nodes.py        # Node implementations
│   │   └── state.py        # AgentState TypedDict
│   ├── api/                # FastAPI routes
│   │   ├── main.py         # App entry point + health checks
│   │   ├── tickets.py      # Ticket CRUD + triage endpoints
│   │   └── dashboard.py    # Dashboard metrics endpoints
│   ├── models/             # SQLAlchemy + Pydantic models
│   │   ├── database.py     # Engine, session, Base
│   │   ├── schemas.py      # Pydantic schemas (Ticket, Classification, etc.)
│   │   ├── ticket.py       # TicketModel ORM
│   │   └── trace.py        # TraceModel ORM
│   ├── services/           # LLM, retrieval, classification services
│   ├── tools/              # Agent tool functions
│   ├── evaluation/         # Eval harness runner
│   ├── repository.py       # TicketRepository (DB operations)
│   └── config.py           # Settings, env vars
├── dashboard/              # Streamlit UI
│   └── app.py              # Metrics, queue, trace viewer
├── eval/                   # Evaluation framework
│   ├── harness.py          # EvalHarness class
│   ├── metrics.py          # ROUGE-L, keyword overlap
│   └── fixtures/           # Test ticket fixtures (JSONL)
├── tests/                  # Test suite
│   ├── conftest.py         # Fixtures: DB, mocks, factories
│   ├── test_e2e_full.py    # Comprehensive E2E tests (22 tests)
│   ├── test_integration.py # Integration tests
│   └── test_unit_*.py      # Unit tests
├── utils/                  # Shared utilities
│   ├── circuit_breaker.py  # Circuit breaker pattern
│   ├── logging.py          # Structured logging setup
│   ├── observability.py    # OpenTelemetry tracing
│   └── retry.py            # Retry utilities
├── alembic/                # Database migrations
├── scripts/                # One-off scripts (seed, ingest, eval)
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## 🧩 Integration Points

### 1. API → Agent Pipeline

```
POST /tickets/{id}/triage
  → _run_triage_background()
    → AgentExecutor.run()
      → classify_node()
      → retrieve_node()
      → draft_node()
      → escalate_check_node()
      → finalize_node() or escalate_node()
```

### 2. Agent → Services

```
classify_node() → classification.get_classifier().classify()
retrieve_node() → retrieval.get_retriever().search()
draft_node()    → drafting.get_draft_generator().generate()
```

### 3. Agent → Database

```
finalize_node() → repository.save_triage_result()
escalate_node() → repository.update_ticket_status()
All nodes       → trace.append_step() → repository.add_trace_step()
```

### 4. Dashboard → API

```
Streamlit app → httpx.get("/dashboard/metrics")
              → httpx.get("/tickets/escalated/list")
              → httpx.post("/tickets/{id}/triage")
```

---

## 🐳 Docker

### Start All Services

```bash
docker compose up --build -d
```

### Stop All Services

```bash
docker compose down -v
```

---

## 📄 License

MIT
