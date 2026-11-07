# Support Ticket Triage Agent — CLAUDE.md

## Inherited Global Instructions

@SUPERAGENT.md

---

## PROJECT SKILL ROUTING AND EXECUTION RULES

### Installed Skills

Skills are discoverable at **`.claude/skills/<name>/SKILL.md`** (the canonical Claude Code discovery path).  
Each entry is a symlink or junction pointing to the upstream source at `.agents/skills/<name>/SKILL.md`.  
Both paths read identically; prefer `.claude/skills/` when referencing skill files.

| Skill Name | Claude Code Path | Purpose | Install Source |
|---|---|---|---|
| `python-development` | `.claude/skills/python-development/SKILL.md` | Python idioms, type hints, packaging | laurigates/claude-plugins |
| `fastapi` | `.claude/skills/fastapi/SKILL.md` | FastAPI endpoints, Pydantic, dependencies, SSE | fastapi/fastapi (official) |
| `langgraph` | `.claude/skills/langgraph/SKILL.md` | Agent graph, state, checkpoints, human approval | langchain-ai/docs |
| `qdrant-search-quality` | `.claude/skills/qdrant-search-quality/SKILL.md` | Vector indexing, metadata, hybrid search, retrieval | qdrant/skills (official) |
| `postgres-pro` | `.claude/skills/postgres-pro/SKILL.md` | SQL, migrations, indexes, transactions | jeffallan/claude-skills |
| `Testing` | `.claude/skills/Testing/SKILL.md` | Pytest unit/integration/E2E testing for async code | local (h4vzz/awesome-ai-agent-skills was private) |
| `rag-retrieval-testing` | `.claude/skills/rag-retrieval-testing/SKILL.md` | Retrieval relevance, citation accuracy, RAG quality | local (naodeng/awesome-qa-skills failed on Windows path-length) |
| `security-testing` | `.claude/skills/security-testing/SKILL.md` | Auth, authorization, secrets, OWASP testing | petrkindlmann/qa-skills |

### Skill Routing Guide

Use the relevant skill(s) for **every** implementation task. Read the SKILL.md at the listed path before writing code.

| Task area | Read this file |
|---|---|
| Python implementation/debugging | `.claude/skills/python-development/SKILL.md` |
| API endpoints, Pydantic models, validation, dependencies | `.claude/skills/fastapi/SKILL.md` |
| Agent graph, state machine, checkpoints, human approval node | `.claude/skills/langgraph/SKILL.md` |
| Vector indexing, metadata payloads, retrieval quality | `.claude/skills/qdrant-search-quality/SKILL.md` |
| SQL queries, Alembic migrations, indexes, transactions | `.claude/skills/postgres-pro/SKILL.md` |
| Unit, integration, regression, and E2E tests | `.claude/skills/Testing/SKILL.md` |
| Retrieval relevance, RAG quality, citation accuracy | `.claude/skills/rag-retrieval-testing/SKILL.md` |
| Authentication, authorization, secrets, security scanning | `.claude/skills/security-testing/SKILL.md` |

### Mandatory Rules for Every Implementation Request

1. **Understand the task** — restate what is being changed and define a "done" check.
2. **Identify relevant skills** — list which skill(s) apply before writing code.
3. **Read the SKILL.md** — open and read the complete `.claude/skills/<name>/SKILL.md` (canonical path) before implementing. Do not claim a skill is loaded unless it was actually read this session.
4. **Read referenced docs** — follow skill `references/` links when the task touches that area.
5. **Use multiple skills** — whenever a task crosses domains (e.g., a new FastAPI endpoint that queries Postgres and Qdrant), read all relevant skills.
6. **Do not replace architecture** — never replace existing patterns merely because a skill recommends a different approach. Skills are guidance, not mandates.
7. **Preserve conventions** — follow the project's existing Python 3.11+, FastAPI, LangGraph, SQLAlchemy/Alembic, and pytest patterns.
8. **Verify with tests and runtime** — run `pytest` after every change. Report real pytest output, not assumed results. Skipped tests do not count.
9. **Do not fabricate metrics** — never report accuracy numbers, test counts, or pass rates from unexecuted checks.

### Session Reload Note

Skills are discovered per session. After restarting Claude Code, you must re-read relevant SKILL.md files before implementing — do not assume they are already loaded from a prior session.

---

## Project Architecture (authoritative summary)

**Stack**: Python 3.11+ · FastAPI · LangGraph · PostgreSQL + SQLAlchemy (async) + Alembic · Qdrant · Redis · Gemini/OpenAI/Anthropic (configurable) · Sentence Transformers (local fallback) · Streamlit · Pytest · OpenTelemetry

**Agent graph** (`src/agent/graph.py`): `classify → retrieve → draft → escalate_check → (escalate?) → finalize`. Six nodes wrapped with tenacity retry (1 s → 2 s → 4 s, 3 attempts). PostgresSaver checkpointer for persistence; MemorySaver fallback for tests.

**Key files**:
- Config: `src/config.py` (pydantic-settings, all env vars)
- DB models: `src/models/` (SQLAlchemy ORM + Pydantic schemas)
- Agent nodes: `src/agent/nodes.py`, `src/agent/state.py`
- Services: `src/services/` (llm, classification, retrieval, embedding, drafting)
- API: `src/api/main.py`, `src/api/tickets.py`, `src/api/dashboard.py`
- Evaluation: `src/eval/`, `src/evaluation/`
- Dashboard: `dashboard/`
- Migrations: `alembic/`

**Important invariants**:
- Every ticket must be idempotent on re-triage (no duplicate DB rows or Qdrant points).
- Human approval is required before any customer-facing reply is sent (FR4/FR3 constraint).
- Loop detection triggers escalation at > 3 failed tool calls or > `max_loop_retries` loop count.
- All traces must persist to PostgreSQL via `TriageTrace` before finalization.
- Embedding vector size must match Qdrant collection at startup.
