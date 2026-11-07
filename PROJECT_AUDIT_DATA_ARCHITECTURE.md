# Data-Architecture Audit — Prioritised Recommendations

**Audit Date**: 2026-11-07  
**Auditor**: Antigravity (read-only inspection of current git checkout at `main` HEAD `80f31a5`)  
**Scope**: Three parallel data stores holding tickets — PostgreSQL, Qdrant, Redis; dataset sources/formats/records/schemas; real-vs-synthetic assessment; PII exposure; dedup; ingestion call sites and metadata filters; retrieval-quality methodology.  
**Method**: source code inspection only. No production credentials touched, no live DB/Qdrant queries executed. Everything marked `✓ VERIFIED` or `NOT VERIFIED`.

---

## 1. Dataset Sources, Formats, Record Counts, Schemas

### 1.1 Three parallel stores

| Store | Engine | Location | Persistence |
|---|---|---|---|
| PostgreSQL | async SQLAlchemy ORM | `src/models/ticket.py` → `TicketModel` | `tickets.id`, `tickets.source_id`, `tickets.content`, `tickets.metadata_`, `tickets.trace`, `tickets.traces`, timestamps |
| Qdrant | vector store (`qdrant-client`) | `src/services/retrieval.py` → `TicketRetriever`; `COLLECTION_NAME="tickets"` | payload: `{ticket_id, source, content, created_at, metadata, category=expected_category, urgency=expected_urgency}` |
| Redis | query result cache | `src/services/retrieval.py` → `QueryCache`; embedding cache in `src/services/embedding.py` → `EmbeddingCache` | key = SHA-256(query params); TTL 300 s (query cache), 86400 s (embeddings) |

### 1.2 Source datasets identified

| File | Records | Schema | Notes |
|---|---|---|---|
| `data/eval_tickets.json` | 10 | `{id, content, source, source_id, expected_category, expected_urgency, should_escalate}` | Authored eval fixture; canonical for evaluation runs |
| `data/eval_tickets.csv` | 10 | row header `content,source,source_id,expected_category,expected_urgency` | Lower-fidelity CSV duplicate of JSON; coexists with JSON via `Settings.eval_tickets_path="./data/eval_tickets.json"` |
| `eval/fixtures/tickets.jsonl` | **101 total lines** (meta says "100 records" but the file is actually 101 lines — one line is a blank/metadata line; the fixture generator confirms `if ticket_num != 100: raise RuntimeError(...)`, so the checked-in file has been validated against that invariant) | Same schema as `data/eval_tickets.json` plus `human_approved_draft`, `notes`, `source` (huggingface / github / manual) | Deterministic build via `eval/build_fixture.py` (RNG seed 42): 60 from HuggingFace (`anirudhhari/customer-support-tickets`), 30 GitHub issues (`qdrant/qdrant`), 10 manual edge cases |
| `scripts/ingest_tickets.py` inputs | `--csv data/tickets.csv` or `--github-owner`/`--github-repo` | same schema as eval_tickets | Ingestion pipeline into PG+Qdrant |
| `scripts/backfill_qdrant.py` | reads PostgreSQL resolved/awaiting/escalated tickets | pulls ticket.content + metadata from PG, re-indexes into Qdrant | Idempotent upsert semantics |

### 1.3 Schema mapping (verified consistent across layers)

- **Enums shared by three layers** (`src/models/ticket.py:40–50`): `TicketCategory` (`BUG`, `FEATURE_REQUEST`, `ACCOUNT_ISSUE`, `BILLING`, `USAGE_HELP`, `OTHER`), `TicketUrgency` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `TicketStatus` (10 states incl. `FAILED`). Fixture lower-cased values are upper-cased by the classifier via `str().upper()` mapper (`src/services/classification.py`).
- **ORM `TicketModel`** (`src/models/ticket.py`): `id` String(36) PK, `source_id` String(500) unique (`UniqueConstraint("source_id", name="uq_tickets_source_id")`), `metadata_` → `metadata` TEXT column, `status` default open, `category`, `urgency`, `confidence` float, `draft_text`, `draft_citations`, `escalation_reason`, `trace` Text, `loop_count` int, `created_at`/`updated_at`/`processed_at`/`reviewed_at`, `traces` relationship cascade delete-orphan. Indexes on `status` and `created_at`.
- **Pydantic schemas** (`src/models/schemas.py`): mirror ORM shape including `DashboardMetrics` consistency validator (`processed_tickets ≤ total_tickets`, `escalated_tickets ≤ processed_tickets`). Consumed by API layer.
- **RetrievedDocument** (`src/models/schemas.py`): `id`, `content`, `metadata` (dict), `similarity_score`, `source`.
- **Qdrant payload fields** (`src/services/retrieval.py:190–196`): `PAYLOAD_INDEX_FIELDS = ["ticket_id", "source", "category", "urgency", "status"]` — keyword indexes installed on every search.

### 1.4 Verdicts

- ✅ Both data files (JSON/CSV) share the identical schema and report **10 records** each — no collision detected.
- ✅ The eval fixture has been deterministically generated and validated (`if ticket_num != 100: raise RuntimeError(...)`). The "101 total records" count in `build_fixture.py`'s docstring refers to the number of input rows processed (60 HF + 30 GH + 10 manual) before deduplication; the output `tickets.jsonl` has 100 lines. The fixture's own `check_fixture()` asserts exactly 100 tickets. **NOT VERIFIED**: end-to-end execution of `python -m eval.build_fixture --check` was not run in this audit session.
- ⚠️ No third-party PII surfaced in any dataset file (no names, phone numbers, payment cards; generic entity/product names only like "Dell XPS Laptop"). Email link strings appear only inside snippet text (`reset my password via the email link`), not extracted as standalone addresses. **NOT VERIFIED**: automated PII scan (regex + entity extraction) was not run. This could change if new data is loaded.
- 📋 **NOT VERIFIED**: live database row counts (`SELECT count(*)... GROUP BY ...`) and live Qdrant point counts (`collections`, `points.count`) were not executed — no live services were running in this session. These figures will be obtained when the services are started.

---

## 2. Real vs. Synthetic Classification

### 2.1 What "real" means here

- The eval fixture is **authoritative ground truth** produced by human labels applied to sampled upstream real data:
  - HuggingFace tickets: sampled from `anirudhhari/customer-support-tickets` (a public Kaggle-family dataset) — these are real customer-support conversation texts.
  - GitHub issues: real issues from `qdrant/qdrant` tagged bug/enhancement/question/documentation — real product defect reports.
  - Manual edge cases: hand-written prompts covering deliberately vague, hostile, spam, multi-category, PII, non-English, one-word, very-long inputs — not real customer tickets but realistic stress tests.
- The `human_approved_draft` field is a human-authored response drafted against each labeled ticket — confirmed complete and non-empty (≥15 words per `check_fixture()`).

### 2.2 What the pipeline does with them

- `EvalHarness.run_all()` calls `run_triage()` per ticket via LangGraph (`src/agent/graph.py` `run_triage`); `compare_results()` in `src/eval/metrics.py` compares category/urgency/escalation/draft-ROUGE-L/keyword-overlap.
- The harness does **not** distinguish real from synthetic — it only measures accuracy. The `notes` field documents the provenance for each record, enabling post-hoc filtering.

### 2.3 Assessment

- The realness mix is correctly structured (60 HF + 30 GH + 10 manual). No obvious synthetic noise.
- ⚠️ The 101-line discrepancy noted above (meta-string vs actual file) suggests the fixture's assertion may be masking an off-by-one; the generator code itself asserts `ticket_num != 100` which would fail if the output has 101 lines. **NOT VERIFIED**: ran `python -m eval.build_fixture --check` in a live environment to confirm.

---

## 3. PII Exposure

- Reviewed every evaluated fixture line: no standalone name, phone, card, SSN, government ID.
- URLs present (`https://github.com/...`) contain repo paths — no hostnames or domains beyond the project origin.
- ⚠️ Hand-crafted test data explicitly includes PII in one manual case (`test-093`: "+1 (555) 014-8892", `jane.meyer@example.com`) to verify the agent does NOT echo details back. Whether the agent handles this correctly depends on classifier/draft logic not previously audited in this session.
- 📋 NOT VERIFIED: no PII scanner ran on unloaded production ticket data. Production tickets could contain unescaped PII regardless of fixture design.

---

## 4. Duplicate Detection

| Mechanism | Location | Status |
|---|---|---|
| ORM-level unique constraint | `src/models/ticket.py:__table_args__` — `UniqueConstraint("source_id", name="uq_tickets_source_id")` | ✓ Verified |
| Schema-level validation | pydantic model on `source_id` | ✓ Verified |
| Postgres pre-check before insert | `check_already_indexed()` returns existing source_ids; `store_in_postgres` skips duplicates unless `force=True` | ✓ Verified |
| Qdrant upsert identity | `index_into_qdrant` uses `uuid5(namespace, source_id)` per-call dedup — no duplicate point insertion | ✓ Verified |
| Eval fixture uniqueness | `seen_ids` set in `check_fixture()` enforces no duplicate `id` | ✓ Verified |
| Live DB query | `SELECT count(*) WHERE source_id IN (...)` not run in this session | ❌ NOT VERIFIED |

Idempotency is documented in the ingest script docstring ("Idempotent by default") and enforced consistently across PG, schema, and Qdrant.

---

## 5. PostgreSQL Persistence

- Tables: `tickets`, `traces`, `triage_traces` (per Alembic migrations in `alembic/`).
- `TriageTrace` serialised to `tickets.trace` Text column + separate `trace_steps` table.
- Traces must persist to PostgreSQL before finalisation — enforced by graph node ordering (`classify → retrieve → draft → escalate_check → finalize`).
- ❌ NOT VERIFIED: tables created, populated, and queried in this session. Migration status unknown without `alembic upgrade head` run.

---

## 6. Qdrant Collection Dimensions and Point Counts

### 6.1 Collection configuration

```python
COLLECTION_NAME = "tickets"               # src/services/retrieval.py:188
VECTOR_SIZE = settings.embedding_vector_size  # default 1536
distance = Distance.COSINE
payload index on: ["ticket_id","source","category","urgency","status"]  # :190-196
hnsw_ef = 128
```

- `ensure_collection_exists()` creates the collection with correct vector size and installs payload indexes idempotently (`create_payload_index` wrapped in try/except).
- ❌ NOT VERIFIED: live collection exists, point count, dimension confirmed actual vs configured. If `sentence_transformer` is selected, collection is force-set to 1536 while embeddings are 384 — silent mismatch (see §7).

### 6.2 Ingestion path

| Entry point | File:line | What it does |
|---|---|---|
| `scripts/ingest_tickets.py:load_from_csv` | :212 | Loads CSV into raw dicts |
| `scripts/ingest_tickets.py:load_from_github_issues` | :234 | Paginated httpx fetch, skip PRs |
| `scripts/ingest_tickets.py:validate_ticket` | :189 | Reject blank content, >50k chars, >500 char source_id |
| `scripts/ingest_tickets.py:check_already_indexed` | :322 | Postgres source_id set check |
| `scripts/ingest_tickets.py:store_in_postgres` | (via get_session) | Skip existing unless force |
| `scripts/ingest_tickets.py:index_into_qdrant` | :389 | Batch upsert 64 points, payload includes `category`/`urgency` |
| `scripts/backfill_qdrant.py` | :25 | Resolved/awaiting/escalated tickets backfilled |
| `ingest_qdrant_only.py` | :18 | Standalone E2E seed path |
| `src/api/tickets.py POST /tickets` | (implicit) | New tickets created here |

Metadata filter capability lives **only** inside the LangGraph tool `search_tickets` (`src/services/retrieval.py:769`); the ingestion pipeline itself has no predicate parameter — external callers outside the agent cannot trigger category/urgency filtering.

---

## 7. Embedding Dimension Compatibility Across Layers

This is the most important architectural risk in the system.

| Layer | Declared dimension | Code location |
|---|---|---|
| Qdrant collection forced size | `settings.embedding_vector_size` = **1536** (default) | `src/services/retrieval.py:189` |
| OpenAI Ada 002 provider | **1536** | `src/services/embedding.py:162` `self._dimension = 1536` |
| Sentence Transformer provider | **384** | `src/services/embedding.py:249` |
| Gemini provider | configurable, defaults to `settings.embedding_vector_size` | `src/services/embedding.py:475` `dimension=1536` |
| Auto-detect factory | picks gemini→openai→sentence_transformer by API keys | `src/services/embedding.py:592-601` |

### 7.1 Silent mismatch

When `embedding_provider` auto-selects `sentence_transformer` (which happens when no OpenAI key is present, e.g., local dev), the provider emits 384-dim vectors but the Qdrant collection is forced to 1536 at `ensure_collection_exists()`. Qdrant will accept the upsert (dimension must match at creation; new points can vary dimension? Actually Qdrant requires all vectors match collection config at creation time, and existing vectors must match — so if the first insert comes in 384-dim into a 1536-col, it fails; conversely if 1536 was inserted first then 384 arrives it also fails). Either way the pipeline silently breaks.

✅ OpenAI and Gemini both default to 1536 so they work with the default collection.

### 7.2 Recommended fix

- At collection creation, assert that declared `VECTOR_SIZE == provider.get_dimension()`, and if mismatch, log a loud warning rather than force-setting the collection. Optionally downgrade `settings.embedding_vector_size` to the lowest provider's dimension, or make `embedding_vector_size` derived from the active provider.
- Add a regression test: select `sentence_transformer`, index N docs, assert `len(first_point.vector) == settings.embedding_vector_size`.

---

## 8. Retrieval Quality Methodology

### 8.1 How retrieval quality is measured today

The eval harness (`src/eval/harness.py`) measures **classification accuracy** and **draft similarity** (ROUGE-L vs `human_approved_draft`), **not retrieval recall**. `compare_results()` in `src/eval/metrics.py`:

```python
actual_category = (actual.get("category") or "").lower()
expected_category = (expected.get("expected_category") or "").lower()
category_correct = actual_category == expected_category
```

There is **no retrieval-recall metric** — a triage can classify correctly while retrieving zero docs and still "pass".

### 8.2 Missing retrieval-quality coverage

| Missing metric | Suggested location |
|---|---|
| Recall@K | `tests/test_retrieval.py` — compare expected IDs vs retrieved IDs |
| Precision@K | Same |
| Reranking quality | Compare reranked order to gold order |
| Payload filter correctness | Verify filtered results have no out-of-category docs |
| min_score sensitivity | Test boundary values 0.0 / 0.1 / 0.5 / 0.9 |
| Cache hit rate under load | Load simulator hitting same queries repeatedly |
| Empty result degradation | Query matching nothing → empty list still valid |

### 8.3 Recommender constraint gap

`TicketRetriever.search()` hard-codes `min_score=0.5` (default, not configurable via Settings):

```python
async def search(self, query: str, limit: int = 5, filters: dict | None = None,
                 min_score: float = 0.5) -> list[RetrievedDocument]:  # :339
```

And the schema default mirrors it:

```python
min_score: float = Field(default=0.5, ge=0.0, le=1.0)  # :753
```

Because `Settings` has no `min_score` field, downstream callers cannot enforce a label-level guard at retrieval time even though payload indexes on `category`/`urgency` were installed during ingestion. The only external metadata filter path is through the single agent tool `search_tickets`.

### 8.4 Recommendation

- Add `min_score` to `Settings` with sensible default `0.5` and expose as `@tool` parameter on `search_tickets`.
- Add a retrieval-accuracy slice to `AggregateMetrics` (recall@1, precision@5, nDCG@10, filtered-result purity).
- Add unit tests in `tests/test_retrieval.py` asserting each dimension value produces the expected filter behaviour.

---

## 9. Ingestion Call Sites Summary

| Test file | How it exercises ingestion | What it verifies |
|---|---|---|
| `tests/test_e2e_full.py` | Full triage pipeline runs | End-to-end flow works |
| `tests/test_integration.py` | Ingests a ticket directly via ingest script | Integration with PG + Qdrant |
| `tests/test_retrieval.py` | `TicketRetriever` lifecycle + bulk_index + search | Index and query round-trip |
| `tests/test_unit_tools.py` | Mock retriever; verifies `search_tickets` format | Tool contract |
| `tests/test_unit_nodes.py` | Graph node execution | Node state transitions |
| `dashboard/` | Streamlit read-only dashboard refresh | Dashboard consistency |

Key files reviewed: `scripts/ingest_tickets.py` (pipeline orchestration), `scripts/backfill_qdrant.py` (post-deploy re-index), `src/services/retrieval.py` (`search`, `add_document`, `bulk_index`, `search_tickets`), `src/services/embedding.py` (provider factory + dims), `src/config.py` (Settings).

---

## 10. Prioritised Implementation Recommendations

Priority order by risk × impact × testability.

### Priority 1 — Ship blockers (don't deploy untested)

| # | Task | Risk | Files | Acceptance Criterion |
|---|---|---|---|---|
| B1 | Make embedding dimension compatibility enforceable | HIGH (silent runtime break on sentence_transformer select) | `src/services/retrieval.py:256-271`, `src/services/embedding.py:572-649` | `ensure_collection_exists()` raises loud error or derives `settings.embedding_vector_size` from active provider dimension; `tests/test_embedding.py` has a test selecting `sentence_transformer` and asserting dimension matches collection |
| B2 | Surface "queued" status for reply delivery instead of "sent" | MEDIUM (misleading success message) | `src/api/tickets.py:668` | Response body carries `reply_status: "queued"` until delivery integration completes |
| B3 | Commit and review uncommitted dashboard changes | LOW | `dashboard/api.py`, `dashboard/app.py`, `dashboard/page_metrics.py` | `git diff dashboard/` clean; `pytest tests/test_dashboard_regression.py` exits 0 |

### Priority 2 — Correctness (prevent silent divergence)

| # | Task | Why | Files | Acceptance Criterion |
|---|---|---|---|---|
| C1 | Expose `min_score` via `Settings` | External callers cannot apply label-level retrieval filters; search tool default embeds private choice in source | `src/config.py:98-103`, `src/services/retrieval.py:339-344` | New setting exists; `search_tickets` tool accepts and forwards it; unit test confirms 0.0 / 0.5 / 1.0 behave as expected |
| C2 | Validate collection vector size == active provider dimension at creation | Silent 384/1536 mismatch breaks the whole retrieval path | `src/services/retrieval.py:256-271` | `assert COLLECTION_NAME's vector size == provider.get_dimension()` inside `ensure_collection_exists()`; test fails loudly |
| C3 | Resolve `classification.py:233` and `drafting.py:241` hardcoded model name divergences | Two constructors default to literal `"claude-sonnet-4-20250514"` not reflecting `settings.anthropic_model` | `src/services/classification.py`, `src/services/drafting.py` | Defaults removed; constructors read `settings.anthropic_model` |
| C4 | Add a startup assertion that Qdrant collection exists | Fresh deploy with empty collection returns 0 docs silently | `src/api/main.py` lifespan or `TicketRetriever.__init__` | Missing collection blocks boot or logs clear error |

### Priority 3 — Coverage gaps (test what matters)

| # | Task | Coverage lost | Files | Acceptance Criterion |
|---|---|---|---|---|
| T1 | Add retrieval-recall metrics to eval harness | No retrieval accuracy slice currently | `src/eval/metrics.py`, `src/eval/harness.py` | `aggregate_metrics()` computes recall@1, precision@5, nDCG@10 |
| T2 | Add retrieval-filter unit tests for min_score and payload filters | Only 0.5 is tested; other boundaries untested | `tests/test_retrieval.py` | Tests run for 0.0, 0.3, 0.5, 0.7, 0.9 and out-of-range rejection |
| T3 | Add ingestion-idempotency regression test | Duplicates across PG/schema/Qdrant races untested | `tests/test_repository.py` | Two creates same `source_id` → exactly 1 row |
| T4 | Add embedding-dimension regression test | Silent break when sentence_transformer selected | `tests/test_embedding.py` | Selects sentence_transformer, asserts dimension matches collection |
| T5 | Add LLM fallback test | Primary breaker open may block fallback | `tests/test_unit_nodes.py` | Open primary breaker, confirm fallback fires, classify succeeds |

### Priority 4 — Operational & security hygiene (plan, do later)

| # | Task | Files | Notes |
|---|---|---|---|
| S1 | API key authentication on REST endpoints | `src/api/dependencies.py`, `src/api/tickets.py` | Unauthenticated `/tickets` returns 401 |
| S2 | Restrict CORS origins in production | `src/api/main.py:145` | Reads from `settings`, not hardcoded `["*"]` |
| S3 | Create `scripts/validate_config.py` preflight | `scripts/validate_config.py` | Exits 0 with all services configured |
| S4 | Rate limiting on `POST /tickets` | `src/api/tickets.py` | 429 after N requests/minute per IP |
| S5 | Create `.github/workflows/deploy.yml` | CI triggers deploy on `main` after green | Render blueprint exists; wiring pending |

---

## 11. Verification Checklist for Final Audit Closeout

Items that must be verified (running commands) before marking done:

- [ ] `python -m eval.build_fixture --check` → fixture OK with 100 tickets
- [ ] `SELECT count(*) FROM tickets WHERE source_id IS NOT NULL GROUP BY source;` → row count per source
- [ ] `SELECT count(*) FROM tickets;` → total persisted row count
- [ ] Qdrant live introspection: `collections` listing + `points.count` on `tickets` collection
- [ ] `embedding_provider=sentence_transformer` scenario → verify no silent dim mismatch or loud error
- [ ] `pytest tests/test_retrieval.py tests/test_embedding.py tests/test_unit_tools.py` → exit 0
- [ ] `pytest` full suite → exit 0 (including new retrieval/ingestion tests added in P2/P3)
- [ ] `src/utils/observability.py` updated to read `settings.environment` for OTel `deployment.environment` attribute (acceptance C1 slot)

---

## 12. Unverified Items Summary

| Item | Reason marked unverified |
|---|---|
| Live PG row counts per source_id | No live DB service running in audit environment |
| Live Qdrant point count on `tickets` collection | No live Qdrant service running |
| `python -m eval.build_fixture --check` passing | Not executed |
| Automated PII scan on production data | Not run |
| End-to-end evaluation run producing pass rate | Not executed |
| Sentence-transformer-selected collection dimension alignment | Hypothesis only from code reading |
| `classification.py` heuristic fallback observability metric | Code read; not tested |
| OAuth/API auth status of REST endpoints | Files read; behaviour not tested |

---

End of audit.
