COMPREHENSIVE LIVE-READINESS AUDIT REPORT

---

AUDIT SECTION 1: CURRENT DATABASE CONFIGURATION

Answers

1. DATABASE_URL read from environment or hardcoded?
   READ FROM ENVIRONMENT via pydantic-settings. The field database_url at src/config.py:28-31 has a hardcoded default of postgresql+asyncpg://triage_user:triage_pass@localhost:5432/triage_db, but it is overridden by the DATABASE_URL env var (line 19-24, env_file=".env").
2. Does the engine handle SSL connections (sslmode=require)?
   NO. The create_async_engine call at src/models/database.py:52-60 does NOT pass any connect_args with sslmode="require" or ssl=True. A hosted Postgres (Neon, Supabase, Railway) typically requires sslmode=require. The DATABASE_URL itself would need ?sslmode=require appended, but the engine does not ensure this. There is also no connect_args dict being passed.
3. pool_size and max_overflow?
   At src/models/database.py:55-56:
   - pool_size=settings.database_pool_size → default 20 (src/config.py:32-36)
   - max_overflow=settings.database_max_overflow → default 10 (src/config.py:37-43)

   Not suitable for Neon/Supabase free tier. Neon free tier allows ~20-100 concurrent connections total. A pool_size of 20 + max_overflow 10 = 30 connections per worker × 4 workers (Dockerfile CMD) = 120 connections. This will exceed Neon's limits. Needs to be reduced to pool_size=5, max_overflow=2 for serverless Postgres.
4. Is pool_pre_ping enabled?
   YES. src/models/database.py:57: pool_pre_ping=True.
5. Is pool_recycle set?
   YES. src/models/database.py:58: pool_recycle=1800 (30 minutes). Good for serverless Postgres that kills idle connections.
6. Does the code use NullPool anywhere (needed for serverless)?
   Only in alembic/env.py:52 (poolclass=pool.NullPool). The main engine in database.py uses the default QueuePool. For serverless deployments (Render, Railway), NullPool may be needed if the process is ephemeral. This is not currently configurable via Settings.
7. Are migrations run automatically on startup?
   Partially. In src/api/main.py:57-63 (lifespan), Base.metadata.create_all runs on startup as a "dev convenience fallback." The comment on line 59 says "In production, run alembic upgrade head before starting the app." The docker-compose.yml has a separate migrate service. But there is NO startup validation that checks if the DB schema is current.
8. Does alembic/env.py read DATABASE_URL from Settings?
   YES. alembic/env.py:21-22: settings = get_settings() then config.set_main_option("sqlalchemy.url", settings.database_url). However, alembic.ini:6 still has a hardcoded fallback URL: sqlalchemy.url = postgresql+asyncpg://triage_user:triage_pass@localhost:5432/triage_db.
9. Places where DB connection would fail against a remote host?
   - No SSL/TLS handling in database.py:52-60 — connections to hosted Postgres will fail without sslmode=require
   - pool_size=20 with 4 workers exceeds hosted Postgres connection limits
   - No connection timeout configuration for the engine

Files and Lines Requiring Changes

┌────────────────────────┬───────┬───────────────────────────────────────────────────────────────┐
│          File          │ Lines │                         Change Needed                         │
├────────────────────────┼───────┼───────────────────────────────────────────────────────────────┤
│ src/config.py          │ 28-43 │ Add database_ssl_mode field, reduce defaults for serverless   │
├────────────────────────┼───────┼───────────────────────────────────────────────────────────────┤
│ src/models/database.py │ 52-60 │ Add connect_args={"sslmode": settings.database_ssl_mode} or   │
│                        │       │ conditional SSL; reduce pool defaults; add NullPool option    │
├────────────────────────┼───────┼───────────────────────────────────────────────────────────────┤
│ alembic.ini            │ 6     │ Remove hardcoded fallback URL (env.py overrides it, but       │
│                        │       │ confusing)                                                    │
├────────────────────────┼───────┼───────────────────────────────────────────────────────────────┤
│ src/api/main.py        │ 57-63 │ Add startup validation that checks DB schema version          │
└────────────────────────┴───────┴───────────────────────────────────────────────────────────────┘

---

AUDIT SECTION 2: CURRENT REDIS CONFIGURATION

Answers

1. REDIS_URL read from environment?
   YES. src/config.py:105-108: redis_url field with default redis://localhost:6379/0.
2. Does Redis client handle TLS (rediss://)?
   PARTIALLY. The validator at src/config.py:231-235 accepts both redis:// and rediss:// schemes. However, the Redis clients created in src/services/embedding.py:313-317 and src/services/retrieval.py:82-86 use aioredis.from_url() which should handle rediss:// automatically. Missing: No ssl_cert_reqs, ssl_ca_certs, or ssl_certfile parameters are passed, which some hosted Redis providers require.
3. Connection pooling configured?
   NO. The Redis clients at embedding.py:313-317 and retrieval.py:82-86 use from_url() with only decode_responses=True and socket_connect_timeout=5. No max_connections parameter is set. For Upstash or hosted Redis, connection pooling is important to avoid exhausting limits.
4. Health check for Redis?
   YES. In src/api/main.py:290-310, check_redis() does a client.ping(). Also in scripts/seed_db.py:211-221.
5. Redis used only for caching, or also queues/sessions?
   Caching only. Used in EmbeddingCache (embedding.py:293-389) and QueryCache (retrieval.py:64-134). No queue or session usage found. The README mentions "Inngest + Redis" for background jobs, but no Inngest code is present.
6. Does the code fail gracefully if Redis is unavailable?
   YES. Both EmbeddingCache and QueryCache catch exceptions and log warnings, returning None on failure. The health check marks Redis as "degraded" not "unhealthy" (main.py:309).
7. Hardcoded localhost:6379 references?
   Only in src/config.py:106 as the default value for the redis_url field. No hardcoded references in service code.

Files and Lines Requiring Changes

┌───────────────────────────┬─────────┬──────────────────────────────────────────────────────────┐
│           File            │  Lines  │                      Change Needed                       │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/config.py             │ 105-108 │ Add redis_max_connections field, redis_ssl_cert_reqs     │
│                           │         │ field                                                    │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/services/embedding.py │ 313-317 │ Pass max_connections, SSL params to from_url()           │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/services/retrieval.py │ 82-86   │ Pass max_connections, SSL params to from_url()           │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/api/main.py           │ 290-310 │ Health check should handle TLS errors specifically       │
└───────────────────────────┴─────────┴──────────────────────────────────────────────────────────┘

---

AUDIT SECTION 3: CURRENT QDRANT CONFIGURATION

Answers

1. Qdrant configured via host/port only, or URL + API key?
   HOST/PORT ONLY. src/config.py:47-68 defines qdrant_host, qdrant_port, qdrant_grpc_port. The QdrantClient is instantiated with host= and port= in:
   - src/services/retrieval.py:200-205
   - scripts/seed_db.py:165-169
   - scripts/ingest_tickets.py:398-402
   - src/api/main.py:271-273 (AsyncQdrantClient in health check)

   No url or api_key parameter. Qdrant Cloud requires url="https://..." and api_key="...".
2. Is there a QDRANT_API_KEY setting?
   NO. Missing entirely from src/config.py.
3. Does the client handle HTTPS connections?
   NO. The QdrantClient is always instantiated with host= + port=, not url=. There is no HTTPS support. Qdrant Cloud uses https://xxx.qdrant.io:6333 URLs.
4. Is the collection created automatically or manually?
   Both. TicketRetriever.ensure_collection_exists() (retrieval.py:232-267) creates it lazily. scripts/seed_db.py:163-205 creates it explicitly. scripts/ingest_tickets.py:406-418 also creates it.
5. Hardcoded localhost or 6333 references?
   - src/config.py:48: default qdrant_host = "localhost"
   - src/config.py:52: default qdrant_port = 6333
   - src/config.py:58: default qdrant_grpc_port = 6334

   No hardcoded references in service code — all use settings.qdrant_host.
6. Does the code handle Qdrant Cloud rate limits and timeouts?
   Partial. The TicketRetriever has a circuit breaker (retrieval.py:52-56) and retry with exponential backoff (retrieval.py:271-294). However, there is no Qdrant-specific rate limit handling (HTTP 429).
7. Health check that verifies collection existence?
   NO. The health check at main.py:265-288 only calls get_collections() — it doesn't verify the specific collection exists or has the correct vector size.

Files and Lines Requiring Changes

┌───────────────────────────┬─────────┬──────────────────────────────────────────────────────────┐
│           File            │  Lines  │                      Change Needed                       │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/config.py             │ 47-78   │ Add qdrant_url and qdrant_api_key fields                 │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/services/retrieval.py │ 200-205 │ Support url= + api_key= for Qdrant Cloud                 │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ scripts/seed_db.py        │ 165-169 │ Support url= + api_key= for Qdrant Cloud                 │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ scripts/ingest_tickets.py │ 398-402 │ Support url= + api_key= for Qdrant Cloud                 │
├───────────────────────────┼─────────┼──────────────────────────────────────────────────────────┤
│ src/api/main.py           │ 271-273 │ Health check should use same connection params as main   │
│                           │         │ client                                                   │
└───────────────────────────┴─────────┴──────────────────────────────────────────────────────────┘

---

AUDIT SECTION 4: CURRENT LLM CONFIGURATION

Answers

1. OpenAI and Anthropic clients instantiated directly or through abstraction?
   Through abstraction. src/services/llm.py:32-56 uses LangChain wrappers (ChatOpenAI, ChatAnthropic). Clients are NOT instantiated directly — LangChain handles the underlying HTTP client.
2. Fallback provider mechanism?
   NO. llm_provider is a single-choice Literal (src/config.py:82-85). If OpenAI fails, there is no automatic fallback to Anthropic. The circuit breaker (llm.py:25-29) just blocks calls when open.
3. API keys read from environment?
   YES. src/config.py:86-101: openai_api_key and anthropic_api_key fields, both default to "".
4. base_url setting for Azure OpenAI or proxies?
   NO. There is no openai_base_url or anthropic_base_url field. Cannot use Azure OpenAI or local proxies.
5. Per-provider circuit breakers?
   YES. src/services/llm.py:25-29: a single _llm_breaker for the active provider. But it's shared — if you switch providers at runtime, the breaker state carries over. There are no separate breakers per provider.
6. Token usage and cost tracked per provider?
   YES. src/services/classification.py:179-190 and src/services/drafting.py:199-210 have MODEL_COST_RATES dicts with pricing for both OpenAI and Anthropic models. Token counting uses tiktoken.
7. Startup validation that ensures at least one provider is configured?
   NO. The require_openai_key() and require_anthropic_key() methods (config.py:254-264) only validate when called. They are NOT called at startup. The app will start without any API key and fail at runtime.
8. Health check that pings both providers?
   NO. main.py:312-332 only checks the configured provider, and it only verifies the client can be instantiated — it doesn't make an actual API call.
9. LLM calls subject to timeouts and retries?
   YES. llm.py:111-114 uses asyncio.wait_for() with configurable timeout. Both classification.py:358-460 and drafting.py:451-551 have full retry loops with exponential backoff and jitter.
10. Hardcoded API keys or model names?
    No leaked keys. The grep found only placeholder patterns in .env.example:30 (sk-ant-...) and test files. However, model names are hardcoded as defaults:
    - src/config.py:92: openai_model = "gpt-4o"
    - src/config.py:99: anthropic_model = "claude-sonnet-4-20250514"
    - src/services/classification.py:233: model_name: str = "claude-sonnet-4-20250514" (hardcoded default in constructor)
    - src/services/drafting.py:241: model_name: str = "claude-sonnet-4-20250514" (same)

Files and Lines Requiring Changes

┌─────────────────────┬─────────┬────────────────────────────────────────────────────────────────┐
│        File         │  Lines  │                         Change Needed                          │
├─────────────────────┼─────────┼────────────────────────────────────────────────────────────────┤
│ src/config.py       │ 82-101  │ Add openai_base_url, anthropic_base_url; change llm_provider   │
│                     │         │ to support fallback order                                      │
├─────────────────────┼─────────┼────────────────────────────────────────────────────────────────┤
│ src/config.py       │ —       │ Add startup validation in get_settings() or a dedicated        │
│                     │         │ validate() method                                              │
├─────────────────────┼─────────┼────────────────────────────────────────────────────────────────┤
│ src/services/llm.py │ 32-56   │ Add fallback provider logic; pass base_url to LangChain        │
│                     │         │ clients                                                        │
├─────────────────────┼─────────┼────────────────────────────────────────────────────────────────┤
│ src/services/llm.py │ 25-29   │ Separate circuit breakers per provider                         │
├─────────────────────┼─────────┼────────────────────────────────────────────────────────────────┤
│ src/api/main.py     │ 312-332 │ Health check should make a lightweight test call, not just     │
│                     │         │ instantiate                                                    │
└─────────────────────┴─────────┴────────────────────────────────────────────────────────────────┘

---

AUDIT SECTION 5: ENVIRONMENT AND SECRETS MANAGEMENT

Answers

1. Does .env.example list every env var the code reads?
   NEARLY. Cross-referencing settings.* references vs .env.example:

| Missing from .env.example   | Where it's read                                    |
|-----------------------------|----------------------------------------------------|
| EMBEDDING_PROVIDER          | config.py:117                                      |
| EMBEDDING_MODEL             | config.py:125                                      |
| EMBEDDING_CACHE_TTL_SECONDS | config.py:129                                      |
| RETRIEVER_CACHE_TTL_SECONDS | config.py:109                                      |
| QDRANT_TIMEOUT              | config.py:63                                       |
| QDRANT_GRPC_PORT            | config.py:57 (IS in .env.example but undocumented) |
| EMBEDDING_VECTOR_SIZE       | config.py:73                                       |
   All present in .env.example: DATABASE_URL, DATABASE_POOL_SIZE, DATABASE_MAX_OVERFLOW, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, REDIS_URL, CONFIDENCE_THRESHOLD, MAX_LOOP_RETRIES, TRIAGE_TIMEOUT_SECONDS, MAX_CONCURRENT_TRIAGES, CLASSIFICATION_TIMEOUT_SECONDS, DRAFTING_TIMEOUT_SECONDS, RETRIEVAL_TIMEOUT_SECONDS, OTLP_ENDPOINT, LOG_LEVEL, EVAL_TICKETS_PATH, DASHBOARD_USERNAME, DASHBOARD_PASSWORD.
2. Env vars read but not documented?
   QDRANT_GRPC_PORT is in .env.example but not commented. EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_CACHE_TTL_SECONDS, RETRIEVER_CACHE_TTL_SECONDS, QDRANT_TIMEOUT, EMBEDDING_VECTOR_SIZE are NOT in .env.example at all.
3. Documented but not used?
   All documented env vars in .env.example are used.
4. Does .gitignore exclude .env files?
   YES. .gitignore:25-27 excludes .env, .env.local, .env.*.local. However, it does NOT exclude .env.production or .env.staging — these would be tracked by git if created.
5. Does the README explain how to obtain each key?
   NO. The README at line 44 says "An OpenAI or Anthropic API key" but provides no signup links, pricing info, or instructions.
6. Is there a script to validate config before startup?
   NO. No scripts/validate_config.py exists.
7. Is there an ENVIRONMENT variable to switch between local/staging/production?
   NO. The only reference is src/utils/observability.py:54 which hardcodes "deployment.environment": "development". There is no ENVIRONMENT settings field.

Missing Items

- Missing from .env.example: EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_CACHE_TTL_SECONDS, RETRIEVER_CACHE_TTL_SECONDS, QDRANT_TIMEOUT, EMBEDDING_VECTOR_SIZE
- .gitignore does not exclude .env.production, .env.staging
- No ENVIRONMENT field in Settings
- No config validation script
- No provider signup links in README

---

AUDIT SECTION 6: DOCKER AND DEPLOYMENT READINESS

Answers

1. Does docker-compose.yml define app and streamlit services?
   YES. Defines: postgres, qdrant, redis, migrate, app, streamlit — a full local development stack.
2. Are env vars passed from .env to containers?
   YES. All app services use env_file: .env and override specific vars for container networking (e.g., QDRANT_HOST: qdrant instead of localhost).
3. Is there a separate prod compose file?
   NO. No docker-compose.prod.yml or similar exists.
4. Is there a deployment config for a free-tier host?
   NO. No render.yaml, fly.toml, railway.json, or Procfile.
5. Is there a deploy workflow in GitHub Actions?
   NO. ci.yml builds and pushes a Docker image to GHCR on main push, but there is no deployment step (no deploy.yml, no Render/Fly/Railway deploy action).
6. Does the Dockerfile install all dependencies?
   MOSTLY. Dockerfile:24: pip install --no-cache-dir ".[dev]" installs all project deps. However, opentelemetry-instrumentation-fastapi is in pyproject.toml:52 and will be installed. The Dockerfile.streamlit:24 installs a minimal subset — it does NOT install opentelemetry-instrumentation-fastapi.

Files to Create/Modify

┌──────────────────────────────┬────────┬────────────────────────────────────────────────────────┐
│             File             │ Action │                        Purpose                         │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ docker-compose.prod.yml      │ CREATE │ Production compose using hosted services (no local     │
│                              │        │ Postgres/Qdrant/Redis)                                 │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ render.yaml or fly.toml      │ CREATE │ Platform deployment config                             │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ .github/workflows/deploy.yml │ CREATE │ Automated deployment workflow                          │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ .env.production.example      │ CREATE │ Production env var template                            │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ Dockerfile                   │ MODIFY │ Consider multi-stage build for smaller image           │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────┤
│ Dockerfile.streamlit         │ MODIFY │ Add missing dependencies                               │
└──────────────────────────────┴────────┴────────────────────────────────────────────────────────┘

---

AUDIT SECTION 7: HARDCODED VALUES AND LOCALHOST REFERENCES

Grep Results

localhost / 127.0.0.1 (all in src/config.py defaults):

┌───────────────┬──────┬────────────────┬─────────────────────────────────────────┐
│     File      │ Line │     Value      │                Should Be                │
├───────────────┼──────┼────────────────┼─────────────────────────────────────────┤
│ src/config.py │ 29   │ localhost:5432 │ Settings default only — fine as default │
├───────────────┼──────┼────────────────┼─────────────────────────────────────────┤
│ src/config.py │ 48   │ localhost      │ Settings default only — fine as default │
├───────────────┼──────┼────────────────┼─────────────────────────────────────────┤
│ src/config.py │ 106  │ localhost:6379 │ Settings default only — fine as default │
├───────────────┼──────┼────────────────┼─────────────────────────────────────────┤
│ src/config.py │ 186  │ localhost:4317 │ Settings default only — fine as default │
└───────────────┴──────┴────────────────┴─────────────────────────────────────────┘

Port numbers (all defaults in src/config.py):

┌──────┬──────┬─────────────────────┐
│ Line │ Port │       Context       │
├──────┼──────┼─────────────────────┤
│ 29   │ 5432 │ PostgreSQL default  │
├──────┼──────┼─────────────────────┤
│ 52   │ 6333 │ Qdrant REST default │
├──────┼──────┼─────────────────────┤
│ 58   │ 6334 │ Qdrant gRPC default │
├──────┼──────┼─────────────────────┤
│ 106  │ 6379 │ Redis default       │
└──────┴──────┴─────────────────────┘

pool_size / max_overflow (in src/models/database.py):

┌──────┬───────────────────────────────────────────────────────┬────────────────────────────────┐
│ Line │                         Value                         │            Concern             │
├──────┼───────────────────────────────────────────────────────┼────────────────────────────────┤
│ 55   │ pool_size=settings.database_pool_size (default 20)    │ Too high for serverless        │
│      │                                                       │ Postgres                       │
├──────┼───────────────────────────────────────────────────────┼────────────────────────────────┤
│ 56   │ max_overflow=settings.database_max_overflow (default  │ Combined 30 connections per    │
│      │ 10)                                                   │ worker                         │
└──────┴───────────────────────────────────────────────────────┴────────────────────────────────┘

NullPool (in alembic/env.py):

┌──────┬─────────────────────────┬──────────────────────────────────────────────────────────┐
│ Line │          Value          │                           Note                           │
├──────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
│ 52   │ poolclass=pool.NullPool │ Correct for migrations, not configurable for main engine │
└──────┴─────────────────────────┴──────────────────────────────────────────────────────────┘

Hardcoded timeouts (not configurable via Settings):

┌───────────────────────────┬────────┬─────────────────────────────────┬────────────────────────┐
│           File            │  Line  │              Value              │       Should Be        │
├───────────────────────────┼────────┼─────────────────────────────────┼────────────────────────┤
│ src/models/database.py    │ 59     │ pool_timeout=30                 │ Settings field         │
├───────────────────────────┼────────┼─────────────────────────────────┼────────────────────────┤
│ src/services/llm.py       │ 64     │ timeout_seconds: float = 10.0   │ Already configurable   │
│                           │        │                                 │ via params             │
├───────────────────────────┼────────┼─────────────────────────────────┼────────────────────────┤
│ src/services/retrieval.py │ 85,    │ socket_connect_timeout=5,       │ Settings fields        │
│                           │ 335    │ timeout=10.0                    │                        │
├───────────────────────────┼────────┼─────────────────────────────────┼────────────────────────┤
│ src/services/embedding.py │ 185,   │ timeout=30.0,                   │ Settings fields        │
│                           │ 316    │ socket_connect_timeout=5        │                        │
└───────────────────────────┴────────┴─────────────────────────────────┴────────────────────────┘

Semaphore (in src/agent/graph.py):

┌──────┬────────────────────────────────────────────────────┬──────────────────────────────────┐
│ Line │                       Value                        │               Note               │
├──────┼────────────────────────────────────────────────────┼──────────────────────────────────┤
│ 56   │ asyncio.Semaphore(settings.max_concurrent_triages) │ Configurable via Settings — good │
└──────┴────────────────────────────────────────────────────┴──────────────────────────────────┘

Leaked API keys: NONE FOUND. Only placeholder patterns in .env.example and test files.

Mock/fake references in production code:

┌───────────────────────┬──────┬────────────────────────────────────────────┬───────────────────┐
│         File          │ Line │                  Content                   │      Concern      │
├───────────────────────┼──────┼────────────────────────────────────────────┼───────────────────┤
│ src/api/tickets.py    │ 481  │ "and mocks sending the reply"              │ Just a comment    │
│                       │      │                                            │ string            │
├───────────────────────┼──────┼────────────────────────────────────────────┼───────────────────┤
│                       │      │                                            │ Real concern —    │
│ src/api/tickets.py    │ 514  │ # Mock sending reply to customer           │ indicates         │
│                       │      │                                            │ unimplemented     │
│                       │      │                                            │ functionality     │
├───────────────────────┼──────┼────────────────────────────────────────────┼───────────────────┤
│ src/tools/registry.py │ 16   │ registry.register_tool("classify_ticket",  │ Test helper, not  │
│                       │      │ mock_classify_fn)                          │ production        │
└───────────────────────┴──────┴────────────────────────────────────────────┴───────────────────┘

---

AUDIT SECTION 8: MOCK VS REAL SERVICE PATHS

Answers

1. MockLLMProvider in production code?
   NO. src/services/llm.py only creates real ChatOpenAI or ChatAnthropic instances. No mock provider exists in src/.
2. Logic that defaults to mocks when API keys missing?
   PARTIALLY. In src/services/embedding.py:477-494, when openai_api_key is empty and no embedding provider is explicitly set, it falls back to SentenceTransformerProvider (local). This is NOT a mock — it's a legitimate local fallback. However, in llm.py:41, api_key=settings.openai_api_key or None — passing None to LangChain will cause a runtime error (no fallback).
3. Clear separation between test mocks and runtime fallbacks?
   YES. Tests are in tests/ with proper mocking. The SentenceTransformerProvider in src/services/embedding.py is a documented local fallback, not a mock.
4. Mock fallback producing realistic output?
   src/services/classification.py:537-601 has a _heuristic_fallback that produces keyword-based classifications when the LLM fails. This IS used in production (called on LLM exception at line 354). It produces reasonable but lower-confidence results.
5. Debug flags for mock mode?
   NO. No DEBUG, MOCK_LLM, or similar flags exist.

Places Where Mock Behavior Might Run in Production

- src/services/classification.py:346-354: The heuristic fallback runs automatically on any LLM failure. This is by design but could mask problems.
- src/services/embedding.py:488-494: Falls back to local SentenceTransformer when OpenAI key is missing. No warning at startup.
- src/api/tickets.py:514: The "Mock sending reply" comment suggests the reply-sending step is unimplemented.

---

AUDIT SECTION 9: HEALTH CHECK AND STARTUP VALIDATION

Answers

1. Does /health check DB, Redis, Qdrant, OpenAI, and Anthropic?
   YES for all four (DB, Redis, Qdrant, LLM). src/api/main.py:231-360 runs all four checks concurrently. However:
   - The LLM check (line 312-332) only verifies client instantiation, NOT an actual API call.
   - It only checks the configured provider, not both OpenAI and Anthropic.
2. Does /health return which environment is active?
   NO. Returns version: "0.1.0" but no environment name.
3. Startup validation that fails fast if critical config is missing?
   NO. The app starts without any API key validation. require_openai_key() and require_anthropic_key() are only called when the LLM is actually invoked.
4. Script to validate config before starting the app?
   NO. No scripts/validate_config.py.
5. Does the app log which services it connects to at startup?
   PARTIALLY. main.py:64: logger.info("api.startup", db=settings.database_url.split("@")[-1]) logs the DB host (redacted password). No logging of Qdrant or Redis connection info at startup.

Checks That Should Exist But Don't

- Startup validation that at least one LLM API key is configured
- Startup validation that DATABASE_URL is reachable
- Startup validation that Qdrant is reachable and the collection exists with correct vector size
- Startup validation that Redis is reachable
- /health should report the active environment (dev/staging/prod)
- /health should make a lightweight LLM API call (e.g., list models)
- Config validation script (scripts/validate_config.py)
- Log all service endpoints at startup (with secrets redacted)

---

AUDIT SECTION 10: FINAL REPORT

A. CURRENT STATE SUMMARY

Database (PostgreSQL): The project uses async SQLAlchemy with a well-configured engine (pool_pre_ping, pool_recycle, retry logic). Database URL is configurable via env var. Alembic is properly integrated and reads from Settings. However, the engine lacks SSL support, pool sizes are too large for serverless Postgres, and there is no NullPool option for ephemeral processes. Migrations can run via the migrate Docker service, but there is no startup validation.

Redis: Used exclusively for caching (embeddings and retriever results). The Redis URL is configurable and supports the rediss:// TLS scheme. Both cache implementations gracefully degrade if Redis is unavailable. Missing: connection pooling configuration, SSL certificate parameters, and max_connections limits.

Qdrant (Vector DB): Fully integrated with a class-based retriever featuring circuit breakers, retry logic, Redis-backed query caching, and automatic collection creation. However, it only supports host/port connections — there is no support for Qdrant Cloud's URL + API key authentication model. No HTTPS handling.

LLM Providers: Clean abstraction via LangChain wrappers. Supports OpenAI and Anthropic with configurable models. Has circuit breakers, retries with exponential backoff, token counting, and cost tracking. Missing: fallback provider mechanism, base_url for Azure/proxies, startup API key validation, and separate per-provider circuit breakers.

Secrets/Environment: pydantic-settings reads all config from env vars with validation. .env.example covers most but not all fields. .gitignore excludes .env but not .env.production. No ENVIRONMENT variable for mode switching. No config validation script. README lacks provider signup instructions.

Docker/Deployment: Complete local development stack via docker-compose.yml (Postgres + Qdrant + Redis + App + Streamlit + Migrate). Docker image builds and pushes to GHCR. Missing: production compose file, platform deployment configs (Render/Fly/Railway), deploy workflow, and production env template.

B. GAPS BY PRIORITY

┌────────────────────────┬───────────────┬──────────┬───────────────────────────────┬───────────┐
│          Gap           │   Subsystem   │ Priority │        Files to Change        │ Estimated │
│                        │               │          │                               │   Effort  │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No SSL/TLS for         │ DB            │ HIGH     │ config.py, database.py        │ 30 min    │
│ PostgreSQL             │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ pool_size too large    │               │          │                               │           │
│ for serverless         │ DB            │ HIGH     │ config.py, database.py        │ 15 min    │
│ Postgres               │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No Qdrant Cloud URL +  │               │          │ config.py, retrieval.py,      │           │
│ API key support        │ Qdrant        │ HIGH     │ seed_db.py,                   │ 2 hrs     │
│                        │               │          │ ingest_tickets.py, main.py    │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No LLM fallback        │ LLM           │ HIGH     │ config.py, llm.py             │ 2 hrs     │
│ provider mechanism     │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No startup config      │               │          │ New                           │           │
│ validation             │ All           │ HIGH     │ scripts/validate_config.py,   │ 1.5 hrs   │
│                        │               │          │ main.py                       │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No ENVIRONMENT mode    │ Config        │ MEDIUM   │ config.py, observability.py   │ 30 min    │
│ switching              │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No LLM base_url for    │ LLM           │ MEDIUM   │ config.py, llm.py             │ 30 min    │
│ Azure/proxies          │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ Redis connection       │ Redis         │ MEDIUM   │ config.py, embedding.py,      │ 30 min    │
│ pooling missing        │               │          │ retrieval.py                  │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No production          │ Deploy        │ MEDIUM   │ New docker-compose.prod.yml   │ 1 hr      │
│ docker-compose         │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No deployment config   │ Deploy        │ MEDIUM   │ New render.yaml or fly.toml   │ 1 hr      │
│ (Render/Fly)           │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No deploy workflow in  │ Deploy        │ MEDIUM   │ New                           │ 1 hr      │
│ CI/CD                  │               │          │ .github/workflows/deploy.yml  │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ .env.example missing   │ Config        │ LOW      │ .env.example                  │ 20 min    │
│ fields                 │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ .gitignore missing     │ Config        │ LOW      │ .gitignore                    │ 5 min     │
│ .env.production        │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ Health check doesn't   │               │          │                               │           │
│ verify LLM             │ Health        │ MEDIUM   │ main.py                       │ 30 min    │
│ connectivity           │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No startup service     │ Health        │ LOW      │ main.py                       │ 15 min    │
│ logging                │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ Hardcoded              │ Observability │ LOW      │ observability.py              │ 10 min    │
│ deployment.environment │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ README missing         │ Docs          │ LOW      │ README.md                     │ 30 min    │
│ provider signup links  │               │          │                               │           │
├────────────────────────┼───────────────┼──────────┼───────────────────────────────┼───────────┤
│ No NullPool option for │ DB            │ LOW      │ config.py, database.py        │ 20 min    │
│  serverless            │               │          │                               │           │
└────────────────────────┴───────────────┴──────────┴───────────────────────────────┴───────────┘

C. FILE-BY-FILE CHANGE LIST

File: src/config.py
What Needs to Change: Add fields: database_ssl_mode, database_use_nullpool, qdrant_url,
qdrant_api_key, openai_base_url, anthropic_base_url, redis_max_connections, environment,
llm_fallback_provider. Reduce database_pool_size default to 5, database_max_overflow default to 2.
Add startup validate() method.
Why: All live-service connection requirements
────────────────────────────────────────
File: src/models/database.py
What Needs to Change: Add conditional SSL (connect_args), conditional NullPool, make pool_timeout
configurable.
Why: Serverless Postgres compatibility
────────────────────────────────────────
File: src/services/llm.py
What Needs to Change: Add fallback provider logic (try primary, catch, try fallback). Pass base_url
to LangChain clients. Add per-provider circuit breakers.
Why: Live LLM reliability
────────────────────────────────────────
File: src/services/embedding.py
What Needs to Change: Pass max_connections and SSL params to Redis from_url().
Why: Hosted Redis compatibility
────────────────────────────────────────
File: src/services/retrieval.py
What Needs to Change: Support url= + api_key= for QdrantClient. Pass Redis SSL params.
Why: Qdrant Cloud compatibility
────────────────────────────────────────
File: scripts/seed_db.py
What Needs to Change: Support url= + api_key= for QdrantClient.
Why: Qdrant Cloud compatibility
────────────────────────────────────────
File: scripts/ingest_tickets.py
What Needs to Change: Support url= + api_key= for QdrantClient.
Why: Qdrant Cloud compatibility
────────────────────────────────────────
File: src/api/main.py
What Needs to Change: Add startup validation calls. Improve health check to verify LLM connectivity.
Log service endpoints at startup. Report environment in /health.
Why: Production readiness
────────────────────────────────────────
File: src/utils/observability.py
What Needs to Change: Read deployment.environment from Settings instead of hardcoding "development".
Why: Environment-aware tracing
────────────────────────────────────────
File: .env.example
What Needs to Change: Add missing fields: EMBEDDING_PROVIDER, EMBEDDING_MODEL,
EMBEDDING_CACHE_TTL_SECONDS, RETRIEVER_CACHE_TTL_SECONDS, QDRANT_TIMEOUT, EMBEDDING_VECTOR_SIZE,
QDRANT_URL, QDRANT_API_KEY, OPENAI_BASE_URL, ANTHROPIC_BASE_URL, REDIS_MAX_CONNECTIONS,
DATABASE_SSL_MODE, ENVIRONMENT
Why: Complete documentation
────────────────────────────────────────
File: .gitignore
What Needs to Change: Add .env.production, .env.staging
Why: Prevent accidental secret commits
────────────────────────────────────────
File: alembic.ini
What Needs to Change: Remove hardcoded sqlalchemy.url (line 6) — env.py overrides it but the
hardcoded value is misleading
Why: Clean config

D. NEW FILES TO CREATE

┌──────────────────────────────────────┬────────────────────────────────────────────────────────┐
│                 File                 │                        Purpose                         │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ scripts/validate_config.py           │ Validate all settings, test connectivity to all        │
│                                      │ services before startup                                │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ docker-compose.prod.yml              │ Production compose with only app + streamlit (no local │
│                                      │  infra)                                                │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ render.yaml                          │ Render.com deployment config (or fly.toml for Fly.io)  │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ .github/workflows/deploy.yml         │ Automated deployment on main push                      │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ .env.production.example              │ Production env var template with hosted service URLs   │
├──────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ src/services/llm_fallback.py         │ Dedicated fallback logic if the fallback is complex    │
│ (optional)                           │                                                        │
└──────────────────────────────────────┴────────────────────────────────────────────────────────┘

E. SETTINGS FIELDS TO ADD

Field: database_ssl_mode
Type: str
Default: "require"
Purpose: PostgreSQL SSL mode
Used In: database.py
────────────────────────────────────────
Field: database_use_nullpool
Type: bool
Default: False
Purpose: Use NullPool for serverless
Used In: database.py
────────────────────────────────────────
Field: qdrant_url
Type: str | None
Default: None
Purpose: Qdrant Cloud URL (overrides host/port)
Used In: retrieval.py, seed_db.py, ingest_tickets.py, main.py
────────────────────────────────────────
Field: qdrant_api_key
Type: str
Default: ""
Purpose: Qdrant Cloud API key
Used In: retrieval.py, seed_db.py, ingest_tickets.py
────────────────────────────────────────
Field: openai_base_url
Type: str | None
Default: None
Purpose: Custom OpenAI API base (Azure, proxy)
Used In: llm.py
────────────────────────────────────────
Field: anthropic_base_url
Type: str | None
Default: None
Purpose: Custom Anthropic API base
Used In: llm.py
────────────────────────────────────────
Field: redis_max_connections
Type: int
Default: 10
Purpose: Redis connection pool limit
Used In: embedding.py, retrieval.py
────────────────────────────────────────
Field: environment
Type: Literal["development","staging","production"]
Default: "development"
Purpose: Runtime environment
Used In: observability.py, main.py
────────────────────────────────────────
Field: llm_fallback_provider
Type: Literal["openai","anthropic"] | None
Default: None
Purpose: Fallback if primary LLM fails
Used In: llm.py

F. ENV VARS TO ADD

┌────────────────────────────┬──────────────────────────────────┬──────────────┬───────────────┐
│            Var             │             Example              │   Required   │    Purpose    │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │              │ PostgreSQL    │
│ DATABASE_SSL_MODE          │ require                          │ No (default: │ SSL mode for  │
│                            │                                  │  require)    │ hosted        │
│                            │                                  │              │ providers     │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ Only for     │ Qdrant Cloud  │
│ QDRANT_URL                 │ https://xxx.qdrant.io:6333       │ Qdrant Cloud │ connection    │
│                            │                                  │              │ URL           │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ Only for     │ Qdrant Cloud  │
│ QDRANT_API_KEY             │ eyJhbG...                        │ Qdrant Cloud │ authenticatio │
│                            │                                  │              │ n             │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │ https://my-azure.openai.azure.co │              │ Azure OpenAI  │
│ OPENAI_BASE_URL            │ m/                               │ No           │ or proxy      │
│                            │                                  │              │ endpoint      │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │              │ Custom        │
│ ANTHROPIC_BASE_URL         │ https://my-proxy.example.com/    │ No           │ Anthropic     │
│                            │                                  │              │ endpoint      │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ No (default: │ Redis         │
│ REDIS_MAX_CONNECTIONS      │ 10                               │  10)         │ connection    │
│                            │                                  │              │ pool limit    │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ No (default: │ Runtime       │
│ ENVIRONMENT                │ production                       │              │ environment   │
│                            │                                  │ development) │ mode          │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │              │ Fallback LLM  │
│ LLM_FALLBACK_PROVIDER      │ anthropic                        │ No           │ when primary  │
│                            │                                  │              │ fails         │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ No (auto-det │ Embedding     │
│ EMBEDDING_PROVIDER         │ openai                           │ ect)         │ backend       │
│                            │                                  │              │ selection     │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │              │ Embedding     │
│ EMBEDDING_MODEL            │ text-embedding-ada-002           │ No           │ model         │
│                            │                                  │              │ identifier    │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│ EMBEDDING_CACHE_TTL_SECOND │ 86400                            │ No (default: │ Embedding     │
│ S                          │                                  │  86400)      │ cache TTL     │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│ RETRIEVER_CACHE_TTL_SECOND │ 300                              │ No (default: │ Retriever     │
│ S                          │                                  │  300)        │ cache TTL     │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│                            │                                  │ No (default: │ Qdrant        │
│ QDRANT_TIMEOUT             │ 60                               │  60)         │ operation     │
│                            │                                  │              │ timeout       │
├────────────────────────────┼──────────────────────────────────┼──────────────┼───────────────┤
│ EMBEDDING_VECTOR_SIZE      │ 1536                             │ No (default: │ Vector dimens │
│                            │                                  │  1536)       │ ionality      │
└────────────────────────────┴──────────────────────────────────┴──────────────┴───────────────┘

G. RISKS AND GOTCHAS

1. Neon/Supabase connection limits: Neon free tier allows ~20-100 connections. With pool_size=20 and 4 uvicorn workers, the default config opens up to 120 connections. Mitigation: Reduce pool_size to 5, max_overflow to 2, and consider NullPool for serverless.
2. Upstash TLS requirements: Upstash requires rediss:// scheme. The validator accepts it, but the Redis client may need explicit ssl_cert_reqs=ssl.CERT_NONE for some providers. Mitigation: Add configurable SSL params.
3. Qdrant Cloud rate limits: Free tier has rate limits. The circuit breaker helps but doesn't parse 429 responses. Mitigation: Add Qdrant-specific rate limit handling.
4. OpenAI/Anthropic rate limits and cost: GPT-4o costs $2.50/$10 per 1M tokens. With 10 sample tickets and multiple LLM calls per triage, costs can accumulate. Mitigation: Token budget tracking exists but no hard limit enforcement.
5. Cold start issues on serverless: Render/Railway free tier apps spin down after inactivity. First request after idle can take 30-60s. Mitigation: Add health ping cron or upgrade to paid tier.
6. IPv6 issues with hosted Postgres: Some providers (Neon) use IPv6. asyncpg supports it, but DNS resolution may fail on some platforms. Mitigation: Test with target provider before deploying.
7. SSL certificate requirements: Hosted Postgres may need custom CA certificates. The current code has no mechanism to provide them. Mitigation: Add database_ssl_ca setting.
8. Qdrant Cloud URL parsing: The qdrant-client library handles url= parameter differently from host= + port=. The API key must be passed separately. Mitigation: Test with Qdrant Cloud before deploying.

H. RECOMMENDED ORDER OF WORK

1. Add new Settings fields (config.py) — foundation for everything else
2. Update .env.example and .gitignore — document all env vars
3. Add SSL support to database engine (database.py) — required for any hosted Postgres
4. Reduce pool defaults (config.py) — prevent connection exhaustion
5. Add Qdrant Cloud support (config.py, retrieval.py, seed_db.py, ingest_tickets.py, main.py)
6. Add LLM fallback and base_url (config.py, llm.py)
7. Add Redis connection pooling and SSL params (config.py, embedding.py, retrieval.py)
8. Add ENVIRONMENT mode (config.py, observability.py)
9. Create scripts/validate_config.py — startup validation
10. Update health checks (main.py) — proper LLM verification, environment reporting
11. Create docker-compose.prod.yml — production compose
12. Create deployment config (render.yaml or fly.toml)
13. Create deploy workflow (.github/workflows/deploy.yml)
14. Update README — provider signup links, production deployment guide

I. VERIFICATION CHECKLIST

- [ ] Set DATABASE_URL to a Neon/Supabase connection string with ?sslmode=require
- [ ] Verify app starts and migrations run against hosted Postgres
- [ ] Set QDRANT_URL and QDRANT_API_KEY to Qdrant Cloud credentials
- [ ] Verify collection creation and vector search work against Qdrant Cloud
- [ ] Set REDIS_URL to an Upstash rediss:// URL
- [ ] Verify embedding cache and retriever cache work against hosted Redis
- [ ] Set OPENAI_API_KEY and verify classification + drafting work
- [ ] Set ANTHROPIC_API_KEY as fallback and verify failover works
- [ ] Run python scripts/validate_config.py — all checks pass
- [ ] curl localhost:8000/health returns {"status": "healthy", ...} with all checks green
- [ ] Verify /health reports correct environment
- [ ] Verify circuit breakers trip and recover correctly under load
- [ ] Deploy to Render/Fly/Railway and verify end-to-end triage flow
- [ ] Verify graceful shutdown (SIGTERM) closes all connections cleanly
- [ ] Verify .env.production is NOT committed to git
- [ ] Run docker compose -f docker-compose.prod.yml up — app starts without local infra
- [ ] Verify costs stay within budget after 100 ticket triages















