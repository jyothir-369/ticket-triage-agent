You are a senior backend engineer with deep experience in production Python systems, LangGraph agents, and cloud infrastructure. You've shipped similar systems at scale and you know exactly where the sharp edges are: SSL modes that silently fail, connection pools that exhaust under load, circuit breakers that never recover, fallback logic that masks real errors, and health checks that lie.

Your job now is to take a Support-Ticket Triage Agent that currently runs only against local Docker services and wire it end-to-end to five live external services: Neon PostgreSQL, Upstash Redis, Qdrant Cloud, Google Gemini (primary LLM), and OpenRouter (fallback LLM).

Do the work carefully. Think through each change before writing it. Consider failure modes. Consider what happens when a service is slow, down, or returns unexpected data. Write code that a reviewer would trust in production.

Apply every change below in the exact order listed. Do not skip any item. After all changes, run the full verification suite and report results honestly — including any failures you couldn't resolve.

---

PART 1: DEPENDENCIES

Update pyproject.toml:
- Add langchain-google-genai (required for Gemini)
- Add opentelemetry-instrumentation-fastapi (currently imported but missing from deps)
- Remove inngest and pandas (unused per audit)

Run pip install -e ".[dev]" and confirm zero import errors. If any dependency conflicts arise, resolve them by pinning compatible versions rather than removing functionality.

---

PART 2: CONFIGURATION LAYER

Rewrite src/config.py to add all fields required for live services. Think about each one:

PostgreSQL (Neon is serverless):
- database_ssl_mode: str = "require" — Neon requires SSL; without this, connections fail silently in some asyncpg versions
- database_use_nullpool: bool = False — set True for serverless Postgres to avoid holding open connections
- database_connect_timeout: int = 30
- database_pool_size: 5 (reduce from 20 — Neon free tier caps at ~100 total connections, and 4 uvicorn workers × 20 = 80 connections per worker pool alone)
- database_max_overflow: 2 (reduce from 10)

Redis (Upstash uses TLS):
- redis_max_connections: int = 10
- redis_socket_timeout: int = 5

Qdrant Cloud (URL + API key auth):
- qdrant_url: Optional[str] = None
- qdrant_api_key: str = ""

LLM Providers:
- gemini_api_key: str = ""
- gemini_model: str = "gemini-2.5-flash"
- openrouter_api_key: str = ""
- openrouter_base_url: str = "https://openrouter.ai/api/v1"
- openrouter_model: str = "google/gemma-4-31b-it:free"
- llm_fallback_provider: Optional[Literal["openai", "anthropic", "gemini", "openrouter"]] = None
- llm_timeout_seconds: int = 30
- llm_max_retries: int = 3
- openai_base_url: Optional[str] = None
- anthropic_base_url: Optional[str] = None

Environment:
- environment: Literal["development", "staging", "production"] = "development"

Add a validate() method that runs on every Settings instantiation via model_post_init. It must:
- Raise ValueError if environment == "production" and no LLM provider has a valid API key (check gemini, openai, anthropic, openrouter in order)
- Raise ValueError if environment == "production" and database_url contains "localhost"
- Log a warning (not fail) if qdrant_api_key is empty and environment != "development"
- Log a warning if redis_url starts with "redis://" and environment == "production" (Upstash requires rediss://)
- Log which providers are configured — never print the actual keys, only whether they're set

Be defensive: assume the user might set environment=production with only some keys. Fail loudly on missing critical config, warn on suboptimal config.

---

PART 3: DATABASE ENGINE

Update src/models/database.py to support both local Postgres and Neon.

Think about what changes when the DB is remote and serverless:
- SSL is required (Neon rejects non-SSL connections)
- Connections are expensive (Neon charges per connection-minute)
- Idle connections get killed (Neon's idle timeout is ~5 min)
- The pool must be small (free tier caps total connections)

Modify create_async_engine to:
- Build a connect_args dict. When the URL starts with "postgresql", add {"ssl": settings.database_ssl_mode} unless ssl_mode is "disable"
- If settings.database_use_nullpool is True, use poolclass=NullPool and skip pool_size/max_overflow entirely — NullPool opens a fresh connection per request, which is correct for serverless
- Otherwise use pool_size=settings.database_pool_size and max_overflow=settings.database_max_overflow
- Pass pool_timeout=settings.database_connect_timeout
- Keep pool_pre_ping=True (essential — Neon kills idle connections and stale connections produce confusing errors)
- Keep pool_recycle=1800

Handle sqlite URLs for tests: skip SSL and pool args entirely, since sqlite doesn't support them.

---

PART 4: QDRANT CLOUD SUPPORT

Update src/services/retrieval.py to support both local Qdrant and Qdrant Cloud.

Extract a module-level helper:
```python
def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
        )
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        timeout=settings.qdrant_timeout,
    )
Replace every direct QdrantClient instantiation inside the class with a call to get_qdrant_client(). This ensures the same connection logic is used by the main service and by scripts.

Update Redis client creation to pass max_connections=settings.redis_max_connections and socket_connect_timeout=settings.redis_socket_timeout. Detect rediss:// scheme and pass ssl_cert_reqs="required".

Add HTTP 429 detection: Qdrant Cloud rate-limits free-tier clusters. When a 429 is received, log it, sleep with exponential backoff, and retry before tripping the circuit breaker. Cap retries at 3.

PART 5: REDIS CLIENT

Update src/services/embedding.py to match the retrieval.py Redis configuration. Both clients must use the same max_connections, socket_connect_timeout, and SSL settings. If they drift, you'll get hard-to-debug connection leaks.

PART 6: LLM SERVICE

This is the most important change. Update src/services/llm.py to support four providers (gemini, openai, anthropic, openrouter) with automatic fallback.

Create separate circuit breakers per provider: _gemini_breaker, _openai_breaker, _anthropic_breaker, _openrouter_breaker. Shared breakers mask the real failure: if Gemini is down and OpenRouter is healthy, you want OpenRouter to keep serving traffic, not get blocked by Gemini's breaker state.

Add a get_llm(provider: str) function that returns the correct LangChain client:

"gemini" → ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key)

"openai" → ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, base_url=settings.openai_base_url)

"anthropic" → ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, base_url=settings.anthropic_base_url)

"openrouter" → ChatOpenAI(model=settings.openrouter_model, api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)

Add a call_llm_with_fallback(prompt, **kwargs) function:

Try settings.llm_provider first

On any exception (timeout, rate limit, auth failure, network error), catch it, log it with full context (provider, error type, elapsed time), increment a metric (llm_fallback_used_total{from_provider="gemini"}), and try the fallback

If fallback succeeds, return its response

If fallback also fails, raise RuntimeError with both error messages concatenated

Add a WARNING-level log when fallback is used — this is not normal operation and should be visible

Update classification.py and drafting.py to call call_llm_with_fallback instead of the old single-provider path.

PART 7: SCRIPTS

Update scripts/seed_db.py and scripts/ingest_tickets.py to import get_qdrant_client from src.services.retrieval and use it everywhere. This ensures scripts and the main service use identical connection logic — critical for reproducibility.

PART 8: API HEALTH AND STARTUP

Update src/api/main.py.

In the lifespan context manager:

Call settings.validate() explicitly

Log at INFO: environment, DB host (redact password), Qdrant URL or host:port, Redis host (redact password), active LLM provider, fallback provider

If environment == "production" and any critical check fails, raise an exception to prevent startup. It's better to fail at boot than to fail on the first customer ticket.

Update /health to:

Include "environment": settings.environment

Make a real LLM API call to verify Gemini connectivity. Use a cheap call (list models or a 1-token completion). Cache the result for 60 seconds to avoid burning quota on every health check

If fallback is configured, verify the fallback provider is reachable too

Verify the Qdrant collection exists and has the correct vector size (settings.embedding_vector_size). A collection with wrong dimensions is worse than no collection — search returns garbage with no error

Return "healthy" only if all critical checks pass; "degraded" if non-critical checks fail (e.g., fallback provider down); "unhealthy" if DB or Redis is down

PART 9: OBSERVABILITY

Update src/utils/observability.py. Replace the hardcoded "deployment.environment": "development" with settings.environment. Import settings inside the function to avoid circular imports.

PART 10: ENV TEMPLATE

Rewrite .env.example with all keys grouped by service, with comments explaining each one and where to obtain it. Include comments for the actual Neon, Upstash, Qdrant Cloud, Gemini, and OpenRouter signup pages.

Add to .gitignore: .env.production, .env.staging, .env.*.local

PART 11: CONFIG VALIDATOR

Create scripts/validate_config.py. This script must:

Import Settings, call settings.validate()

Test DB connectivity with SELECT 1

Test Redis connectivity with PING

Test Qdrant connectivity with get_collections()

Test Gemini connectivity with a models.list call

Test OpenRouter connectivity with a /models call

Print a table: Service | Status | Latency | Notes

Exit 1 if any critical service fails, 0 otherwise

Support a --strict flag that also fails on warnings

Add a Makefile target: validate

PART 12: ALEMBIC

Remove the hardcoded sqlalchemy.url from alembic.ini (line 6). Verify env.py reads the URL from Settings and uses NullPool.

PART 13: DOCKER

Keep docker-compose.yml for local dev with postgres, qdrant, redis, migrate, app, streamlit.

Create docker-compose.prod.yml:

No postgres, qdrant, or redis services

Only app, streamlit, migrate

Uses env_file: .env.production

restart: unless-stopped on all services

PART 14: REMOVE HARDCODED MODEL NAMES

Remove hardcoded model names in classification.py and drafting.py — read from settings. Any hardcoded timeouts should also come from settings.

PART 15: README

Update README.md with:

Configuration section listing every env var, purpose, example, and signup link

Deploy to Production section with step-by-step provisioning for each service

Free Tier Setup section explaining the whole stack costs $0 for demo traffic

PART 16: EMBEDDING DIMENSIONS

Gemini embeddings (gemini-embedding-001) default to 3072 dimensions, but our schema uses 1536. Configure Gemini to output 1536 via output_dimensionality. Verify the Qdrant collection is created with size=1536. Do not change EMBEDDING_VECTOR_SIZE — match OpenAI's size for compatibility.

PART 17: VERIFICATION

Run these in order. Report honest results for each — including failures.

pip install -e ".[dev]"

ruff check src/ tests/ scripts/

mypy src/

python scripts/validate_config.py — all five services must show green

alembic upgrade head — must apply cleanly against Neon

python scripts/seed_db.py — must create Qdrant collection and seed initial data

uvicorn src.api.main:app --port 8000 — must start and log the environment

curl localhost:8000/health — must return status healthy with all subsystems green

POST a test ticket, trigger triage, poll status, fetch trace

Simulate Gemini failure (invalid key) — verify OpenRouter fallback fires

pytest tests/ -v --cov=src

python -m scripts.eval_run

streamlit run dashboard/app.py

Produce a final report: Fix Applied | Files Changed | Test Result | Notes. If any verification fails, debug it before moving on. If a failure is environmental (e.g., network blip), document it and retry.

TONE AND STANDARDS

Write this code the way you'd write it if you were the only engineer on call for it at 2 AM. Think about:

What happens when Neon suspends the DB after 5 minutes idle?

What happens when Upstash hits its 500K command limit mid-month?

What happens when Qdrant Cloud rate-limits a batch ingest?

What happens when Gemini returns a safety filter response instead of a classification?

What happens when OpenRouter's free model is temporarily unavailable?

Every one of those cases should either recover gracefully or fail loudly with a clear error. Never silently return wrong results. Never loop forever. Never leak credentials into logs.

Report back when done. I want to see the results of every verification step.




**Summary:** This prompt frames Claude Code as a senior backend engineer, instructing it to carefully wire the Support-Ticket Triage Agent to five live services (Neon, Upstash, Qdrant Cloud, Gemini, OpenRouter) across 17 parts — dependencies, configuration, database engine, Qdrant Cloud auth, Redis pooling, LLM fallback, scripts, health checks, observability, env templates, config validator, Alembic, Docker, cleanup, README, embedding dimensions — and run the full verification suite, thinking through real failure modes at every step rather than just mechanically applying changes.