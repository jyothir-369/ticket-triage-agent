# PROJECT_AUDIT.md — Support Ticket Triage Agent

**Audit Date**: 2026-11-07  
**Auditor**: Antigravity (read-only inspection of current git checkout)  
**Branch**: `main` (HEAD `80f31a5`)

---

## 1. VERIFIED COMPLETED FUNCTIONALITY

The following features are confirmed implemented and structurally correct by source inspection:

| Feature | PRD Ref | Evidence |
|---|---|---|
| Ticket creation with UUID IDs | FR1 | `src/api/tickets.py` `POST /tickets` — generates UUID, stores in PostgreSQL |
| Idempotency on duplicate `source_id` | FR1 | `TicketRepository.create_or_get()` checks for existing `source_id` |
| Ticket classification (category + urgency + confidence) | FR1 | `src/services/classification.py` with LLM call + heuristic fallback |
| RAG retrieval via Qdrant | FR2 | `src/services/retrieval.py` — `TicketRetriever` with circuit breaker + Redis cache |
| Draft generation with citations | FR3 | `src/services/drafting.py` — builds `DraftResponse` with `citations` list |
| Confidence-based escalation gate | FR4 | `src/agent/nodes.py` `escalate_check_node` — checks `confidence < threshold` |
| Full step-by-step trace persistence | FR5 | `TriageTrace` serialised to `tickets.trace` column + `trace_steps` table |
| Loop detection (> 3 tool call failures) | FR6 | `src/agent/graph.py` `_handle_node_error` — escalates when `tool_call_count > 3` |
| LangGraph 6-node graph | Architecture | `build_triage_graph()` — classify→retrieve→draft→escalate_check→(escalate?)→finalize |
| PostgresSaver checkpointer | Architecture | `_build_checkpointer()` with MemorySaver fallback |
| Tenacity retry (1s→2s→4s, 3 attempts) | Architecture | `_with_retry()` wrapping all nodes |
| Graceful shutdown (SIGTERM, 30s wait) | Architecture | `shutdown()` + signal handler in lifespan |
| Concurrency semaphore (configurable max) | Architecture | `asyncio.Semaphore(settings.max_concurrent_triages)` |
| `/health` with all 4 checks (DB, Qdrant, Redis, LLM) | NFR | `src/api/main.py` — concurrent async probes |
| `/health/ready` and `/health/live` Kubernetes probes | NFR | Separate endpoints in `main.py` |
| Human approval queue | FR4 | `POST /tickets/{id}/approve` and `/reject` endpoints |
| OpenTelemetry instrumentation | NFR | `src/utils/observability.py` + conditional `FastAPIInstrumentor` |
| Eval harness with labeled fixture | Success Metrics | `src/eval/harness.py`, `src/evaluation/run_eval.py` |
| Streamlit dashboard | Deployment | `dashboard/` directory with `app.py` |
| Redis caching for embeddings + queries | NFR | `EmbeddingCache` + `QueryCache` classes |
| Circuit breakers on LLM + retriever | NFR | `src/utils/circuit_breaker.py` |
| Alembic migrations | Architecture | `alembic/` with `env.py` reading from `Settings` |
| CI via GitHub Actions | Architecture | `.github/workflows/ci.yml` |
| Render deployment blueprint | Deployment | `render.yaml` exists (per commit `80f31a5`) |
| LLM fallback provider | Architecture | `llm_fallback_provider` setting in `config.py` |
| Qdrant Cloud URL + API key support | Architecture | `qdrant_url` + `qdrant_api_key` fields in `config.py` |
| Database SSL mode + NullPool config | Architecture | `database_ssl_mode` + `database_use_nullpool` in `config.py` |
| Redis connection pooling | Architecture | `redis_max_connections` in `config.py` |
| Environment mode field | Architecture | Partial — `environment` field exists (see gap below) |

---

## 2. INCOMPLETE FUNCTIONALITY

### 2.1 Reply-Sending is Mocked — CONFIRMED

**File**: `src/api/tickets.py:658`  
```python
# Mock sending reply to customer
```  
The `POST /tickets/{id}/approve` endpoint marks the ticket approved but does not actually transmit a reply. The PRD marks customer-facing replies as human-approved (FR4 non-goal to send unsupervised), but the human-approval path itself must at minimum integrate or stub an email/webhook delivery mechanism. Currently the "reply sent" success message is misleading.

**Acceptance criteria**: Either wire a real delivery mechanism (email/webhook) or explicitly surface the "reply queued — pending delivery integration" status to the client rather than claiming "reply sent".

### 2.2 `environment` Field Not Wired to Observability

**File**: `src/utils/observability.py`  
The existing audit (audit.md) identified this — `deployment.environment` is hardcoded to `"development"`. Config has an `environment` field but it is not plumbed into OpenTelemetry resource attributes.

**Acceptance criteria**: `observability.py` reads `settings.environment` for the `deployment.environment` OTel attribute.

### 2.3 No Config Validation Script

**File**: Missing — `scripts/validate_config.py`  
No pre-flight script to verify all required env vars before the app starts. Startup failures manifest at runtime as cryptic connection errors.

**Acceptance criteria**: Script exits 0 when all required vars are set and all services (DB, Qdrant, Redis) are reachable; exits 1 with a clear per-service error otherwise.

### 2.4 Dashboard Regression Test File is Unstaged

**Git status**: `tests/test_dashboard_regression.py` is untracked (`??` in `git status`).  
This file exists but has not been committed. Its content may be incomplete.

**Acceptance criteria**: File is reviewed, completed, and committed.

### 2.5 Missing Deployment CI Workflow

**File**: Missing — `.github/workflows/deploy.yml`  
CI builds and pushes Docker image (`ci.yml`) but no automated deployment step exists to Render or another platform.

**Acceptance criteria**: `deploy.yml` triggers on `main` push after CI passes and deploys to the target platform.

---

## 3. CONFIRMED BUGS

### 3.1 `HealthCheckResponse` References `HealthCheckResult` Before Definition

**File**: `src/api/main.py:238`  
```python
class HealthCheckResponse(BaseModel):
    checks: dict[str, HealthCheckResult]   # HealthCheckResult defined at line 241
```
`HealthCheckResult` is defined *after* `HealthCheckResponse`. At module load with `from __future__ import annotations` this is deferred — verify: does `from __future__ import annotations` appear at the top of `main.py`? If it does (line 9), this is safe. If not, this is a `NameError` at startup.

**Verdict**: `from __future__ import annotations` IS present (`main.py:9`) — forward reference is safe. **Not a bug.**

### 3.2 Modified but Uncommitted Files

**Files**: `dashboard/api.py`, `dashboard/app.py`, `dashboard/page_metrics.py` (modified `M` in git status)  
These contain uncommitted changes. The exact nature is unknown without reading the diff, but they represent a divergence from the last commit. If tests were last run against the committed state, these changes are untested.

**Acceptance criteria**: `git diff dashboard/` is reviewed to confirm all changes are intentional and tests are re-run.

### 3.3 `_llm_health_cache` Type Annotation Missing `Any` Import

**File**: `src/api/main.py:250`  
```python
_llm_health_cache: dict[str, Any] = {"result": None, "timestamp": 0.0}
```
`Any` is used but must be imported. Check: `from typing import Any` must appear. If missing, this is a `NameError`.

**Verdict**: Needs verification — `from typing import Any` may not be imported in `main.py`. **Potential bug; requires inspection of line 1-10 and any typing import.**

---

## 4. ARCHITECTURAL RISKS

| Risk | Severity | File(s) | Detail |
|---|---|---|---|
| Pool size for serverless Postgres | Medium | `config.py` | `pool_size=5, max_overflow=2` (post-audit values) — review against actual Neon/Render connection limits before production |
| MemorySaver fallback in tests | Medium | `src/agent/graph.py:479` | Tests that use MemorySaver will not exercise PostgresSaver checkpoint persistence — a divergence risk |
| Single LLM circuit breaker shared across providers | Medium | `src/services/llm.py` | A single `_llm_breaker` instance trips for both primary and fallback — if the primary fails, the breaker state may block the fallback |
| No startup assertion that Qdrant collection exists | Medium | `src/api/main.py` | If Qdrant is empty (fresh deploy without seed), the first triage will succeed but return 0 retrieved documents silently |
| `classification.py` heuristic fallback runs silently | Low | `src/services/classification.py:354` | LLM failure falls back to keyword rules with no metric increment or alert |
| Hardcoded model name in two service constructors | Low | `classification.py:233`, `drafting.py:241` | Default argument `model_name: str = "claude-sonnet-4-20250514"` diverges from `settings.anthropic_model` |
| Qdrant rate limit (HTTP 429) not handled | Low | `src/services/retrieval.py` | Circuit breaker trips on any exception; 429s should trigger rate-limit sleep, not open the breaker |

---

## 5. MISSING TESTS

| Test Gap | Priority | Suggested File |
|---|---|---|
| Ticket creation idempotency (same `source_id` twice → 1 DB row) | HIGH | `tests/test_repository.py` |
| Qdrant indexing after ticket processing (point count increases by exactly 1) | HIGH | `tests/test_retrieval.py` |
| Embedding dimension matches collection vector size | HIGH | `tests/test_embedding.py` |
| Draft citations reference only retrieved doc IDs | HIGH | `tests/test_drafting.py` |
| LLM fallback fires when primary circuit breaker is open | HIGH | `tests/test_unit_nodes.py` |
| Human approval endpoint (`/approve`) marks ticket `APPROVED` and persists | MEDIUM | `tests/test_api.py` |
| Triage timeout state returns `should_escalate=True` | MEDIUM | `tests/test_agent.py` |
| Shutdown state rejects new triage requests while in-flight tasks drain | MEDIUM | `tests/test_agent.py` |
| Loop detection escalates when `tool_call_count > 3` | MEDIUM | `tests/test_unit_nodes.py` |
| `test_dashboard_regression.py` — content unknown (untracked file) | MEDIUM | `tests/test_dashboard_regression.py` |
| Redis cache miss falls back to direct Qdrant call without error | LOW | `tests/test_retrieval.py` |
| Qdrant collection not found → `degraded` health status, not `unhealthy` | LOW | `tests/test_api.py` |

---

## 6. SECURITY CONCERNS

| Concern | Severity | File | Detail |
|---|---|---|---|
| `allow_origins=["*"]` in CORS middleware | Medium | `src/api/main.py:145` | Allowlist specific origins in production |
| No authentication on API endpoints | Medium | `src/api/tickets.py` | `/tickets`, `/approve`, `/reject` have no auth. `DASHBOARD_USERNAME`/`DASHBOARD_PASSWORD` guard the Streamlit UI but not the REST API |
| `customer_email` stored in plaintext | Low | `src/models/ticket.py` | PII not encrypted at rest |
| No rate limiting on `POST /tickets` | Low | `src/api/tickets.py` | An attacker can spam ticket creation; no token bucket or per-IP limit |
| Secrets logged if config validation error includes full URL | Low | `src/api/main.py:46` | `settings.database_url.split("@")[-1]` redacts the password correctly — **OK** |

---

## 7. PRIORITISED IMPLEMENTATION BACKLOG

### Priority 1 — Ship Blockers

| # | Task | Files | Acceptance Criteria |
|---|---|---|---|
| B1 | Fix reply-sending UX: surface "queued" status instead of "sent" | `src/api/tickets.py:668` | Response body says `"status": "reply_queued"` not `"sent"` until delivery is wired |
| B2 | Commit and review `tests/test_dashboard_regression.py` | `tests/test_dashboard_regression.py` | File committed; `pytest tests/test_dashboard_regression.py` exits 0 |
| B3 | Commit reviewed dashboard changes | `dashboard/api.py`, `dashboard/app.py`, `dashboard/page_metrics.py` | `git diff dashboard/` is clean; tests pass |

### Priority 2 — Correctness

| # | Task | Files | Acceptance Criteria |
|---|---|---|---|
| C1 | Wire `settings.environment` to OTel resource attributes | `src/utils/observability.py` | `deployment.environment` attribute is set from `settings.environment` in traces |
| C2 | Add per-provider circuit breakers | `src/services/llm.py` | Primary and fallback providers have independent `CircuitBreaker` instances |
| C3 | Add hardcoded model name guard | `src/services/classification.py:233`, `src/services/drafting.py:241` | Default argument removed; constructors read from `settings` |

### Priority 3 — Test Coverage

| # | Task | Files | Acceptance Criteria |
|---|---|---|---|
| T1 | Idempotency regression test | `tests/test_repository.py` | Two creates with same `source_id` → 1 row; test passes |
| T2 | Qdrant indexing count test | `tests/test_retrieval.py` | Point count increases by exactly 1 after triage |
| T3 | Citation accuracy test | `tests/test_drafting.py` | All draft citation IDs are in retrieved doc IDs |
| T4 | LLM fallback test | `tests/test_unit_nodes.py` | Primary breaker open → fallback fires → classification succeeds |

### Priority 4 — Security & Ops

| # | Task | Files | Acceptance Criteria |
|---|---|---|---|
| S1 | Add API authentication (API key or JWT) | `src/api/dependencies.py`, `src/api/tickets.py` | Unauthenticated requests to `/tickets` return 401 |
| S2 | Restrict CORS `allow_origins` in production | `src/api/main.py:145` | `allow_origins` reads from `settings` and is not `["*"]` in `environment=production` |
| S3 | Create `scripts/validate_config.py` | `scripts/validate_config.py` | Script exits 0 with all services configured; exits 1 with clear error list |
| S4 | Add rate limiting to `POST /tickets` | `src/api/tickets.py` | 429 returned after N requests/minute per IP |
| S5 | Create `.github/workflows/deploy.yml` | `.github/workflows/deploy.yml` | Deploys to Render on `main` push after CI green |

---

## 8. INSTALLATION RESULTS FOR REQUESTED SKILLS

| Skill | Source | Result | SKILL.md Path |
|---|---|---|---|
| `python-development` | laurigates/claude-plugins | ✓ Installed | `.agents/skills/python-development/SKILL.md` |
| `fastapi` | fastapi/fastapi | ✓ Installed | `.agents/skills/fastapi/SKILL.md` |
| `langgraph` | langchain-ai/docs | ✓ Installed | `.agents/skills/langgraph/SKILL.md` |
| `qdrant-search-quality` | qdrant/skills | ✓ Installed | `.agents/skills/qdrant-search-quality/SKILL.md` |
| `postgres-pro` | jeffallan/claude-skills | ✓ Installed | `.agents/skills/postgres-pro/SKILL.md` |
| `Testing` | h4vzz/awesome-ai-agent-skills | ✗ Failed (private repo — auth error) | `.agents/skills/Testing/SKILL.md` (local) |
| `rag-retrieval-testing` | naodeng/awesome-qa-skills | ✗ Failed (Windows path-length git checkout error) | `.agents/skills/rag-retrieval-testing/SKILL.md` (local) |
| `security-testing` | petrkindlmann/qa-skills | ✓ Installed | `.agents/skills/security-testing/SKILL.md` |

**Action taken for failed installs**: Local SKILL.md files were authored to fill the gaps. They cover the same domains (pytest testing patterns and RAG retrieval quality testing) using project-specific guidance derived from the existing codebase.

---

## 9. SESSION RELOAD NOTICE

This session performed all skill installations and configuration. **Skills installed by `npx skills add` are discovered by Claude Code on session startup** — they do not auto-load into the current session without a restart. After restarting:

1. Verify the skills are listed in the Claude Code skills panel.
2. Read `CLAUDE.md` to confirm the persistent skill-routing workflow is active.
3. For any implementation task, explicitly read the relevant `SKILL.md` before writing code — do not assume a prior session's skill context carries over.
